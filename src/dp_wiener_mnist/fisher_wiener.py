"""Data-free, post-DP Fisher-Wiener; extracted from ExpV1."""

import torch
import math
from .utils import require_finite, RNGStream
from .types import CovariancePair
import torch.nn.functional as F
from .model import SimpleCNN
from .utils import digest
from .pink_noise import generate_pink_noise
from .recorder import KFACRecorder
from .covariance import compute_covariances, accumulate_covariances

LAYERS = ("conv1", "conv2", "fc1", "fc2")


def synthetic_samples(c: dict, dev: torch.device, rng: RNGStream) -> tuple:
    """Generate equal pink-noise batches with random labels in an isolated stream."""
    if c["M_syn"] <= 0 or c["batch_size"] <= 0 or c["M_syn"] % c["batch_size"]:
        raise ValueError("Invalid synthetic batch budget")
    before = rng.audit()
    with rng.use():
        batches = [
            (
                generate_pink_noise(c["batch_size"], (1, 28, 28), dev),
                torch.randint(0, 10, (c["batch_size"],), device=dev),
            )
            for _ in range(c["M_syn"] // c["batch_size"])
        ]
    x, y = (torch.cat([b[j] for b in batches]) for j in (0, 1))
    return (x, y), dict(
        rng_before=before,
        rng_after=rng.audit(),
        samples_hash=digest([x]),
        labels_hash=digest([y]),
        count=len(y),
    )


def build_covariances(
    model_state: dict, samples: tuple, c: dict, dev: torch.device
) -> CovariancePair:
    """Use synthetic probes and a separate model; mean CE matches source KFAC scaling."""
    x, y = samples
    if x.shape != (c["M_syn"], 1, 28, 28) or y.shape != (c["M_syn"],):
        raise ValueError("Invalid synthetic sample dimensions")
    require_finite(x, *model_state.values())
    if len(x) != c["M_syn"] or len(x) % c["batch_size"]:
        raise ValueError("Expected equal whole synthetic batches")
    # Separate model; creation cannot advance any training random stream.
    with torch.random.fork_rng(devices=[dev.index or 0] if dev.type == "cuda" else []):
        model = SimpleCNN().to(dev)
        model.load_state_dict(model_state)
        recorder = KFACRecorder(model)
        recorder.enable()
        factors = []
        try:
            for start in range(0, len(x), c["batch_size"]):
                model.zero_grad(set_to_none=True)
                F.cross_entropy(
                    model(x[start : start + c["batch_size"]]),
                    y[start : start + c["batch_size"]],
                ).backward()
                factors.append(
                    compute_covariances(model, recorder.activations, recorder.backprops)
                )
                recorder.clear()
            return accumulate_covariances(factors)
        finally:
            recorder.remove()


@torch.no_grad()
def build_fisher_state(
    covariances: CovariancePair, sigma: float, max_grad_norm: float, batch_size: int
) -> dict:
    """Build the FP32 active state required by the Fisher Wiener filter."""
    if (
        batch_size <= 0
        or max_grad_norm <= 0
        or sigma < 0
        or not all(math.isfinite(v) for v in (sigma, max_grad_norm, batch_size))
    ):
        raise ValueError("Invalid Fisher noise parameters")
    if not covariances.A or covariances.A.keys() != covariances.G.keys():
        raise ValueError("Missing covariance factors")
    r = (sigma * max_grad_norm / batch_size) ** 2
    result = {}
    for name, A in covariances.A.items():
        A, G = A.float(), covariances.G[name].float()
        if (
            A.ndim != 2
            or G.ndim != 2
            or A.shape[0] != A.shape[1]
            or G.shape[0] != G.shape[1]
        ):
            raise ValueError("Covariance factors must be square")
        require_finite(A, G)
        lambda_a, q_a = torch.linalg.eigh(A)
        lambda_g, q_g = torch.linalg.eigh(G)
        lambda_a = lambda_a.clamp_min(0)
        lambda_g = lambda_g.clamp_min(0)
        lambda_f = lambda_g[:, None] * lambda_a[None, :]
        H = torch.where(
            lambda_f + r > 0, lambda_f / (lambda_f + r), torch.zeros_like(lambda_f)
        )
        result[name] = dict(
            Q_A=q_a.float(),
            lambda_A=lambda_a.float(),
            Q_G=q_g.float(),
            lambda_G=lambda_g.float(),
            H=H.float(),
        )
    return result


def pack_layer_gradient(layer: torch.nn.Module, source: str = "grad") -> torch.Tensor:
    """Pack output rows, flattened weight columns, then the bias column."""
    weight = getattr(layer.weight, source).detach().reshape(layer.weight.shape[0], -1)
    return (
        torch.cat([weight, getattr(layer.bias, source).detach()[:, None]], 1)
        if layer.bias is not None
        else weight.clone()
    )


@torch.no_grad()
def unpack_layer_gradient(
    layer: torch.nn.Module, matrix: torch.Tensor, target: str = "grad"
) -> None:
    """Restore weights and bias without changing flattening order."""
    if matrix.shape != (
        layer.weight.shape[0],
        layer.weight[0].numel() + int(layer.bias is not None),
    ):
        raise ValueError("Gradient matrix shape mismatch")
    require_finite(matrix)
    width = layer.weight[0].numel()
    setattr(layer.weight, target, matrix[:, :width].reshape_as(layer.weight).clone())
    if layer.bias is not None:
        setattr(layer.bias, target, matrix[:, width].clone())


def apply_fisher_matrix(matrix: torch.Tensor, layer_state: dict) -> torch.Tensor:
    """Apply Q_G [H * (Q_G.T M Q_A)] Q_A.T in source operation order."""
    qa, qg, h = (layer_state[k] for k in ("Q_A", "Q_G", "H"))
    if (
        matrix.shape != h.shape
        or qa.shape != (h.shape[1], h.shape[1])
        or qg.shape != (h.shape[0], h.shape[0])
    ):
        raise ValueError("Fisher matrix shape mismatch")
    require_finite(matrix, qa, qg, h)
    result = qg @ (h * (qg.T @ matrix @ qa)) @ qa.T
    require_finite(result)
    return result


def state_bytes(state: object) -> int:
    """Count tensor bytes in the FP32 active state."""
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(state_bytes(v) for v in state.values())
    return 0


@torch.no_grad()
def apply_fisher_wiener(model: torch.nn.Module, state: dict) -> None:
    """Transform only the already-clipped, already-noised gradients."""
    base = getattr(model, "_module", model)
    for name in LAYERS:
        layer = getattr(base, name)
        unpack_layer_gradient(
            layer, apply_fisher_matrix(pack_layer_gradient(layer), state[name])
        )
