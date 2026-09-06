"""Matched synthetic refresh; KFAC mathematics are entirely upstream."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from opacus import GradSampleModule
from exp3.common import SimpleCNN, LAYERS, generate_pink_noise, digest, layer_gradient
from dp_kfac.recorder import KFACRecorder
from dp_kfac.covariance import compute_covariances, compute_inverse_sqrt, accumulate_covariances
from dp_kfac.precondition import precondition_per_sample_gradients


@contextmanager
def diagnostic_model(state, dev):
    # Neither model creation nor any diagnostics advance training RNG streams.
    with torch.random.fork_rng(devices=[dev.index or 0] if dev.type == "cuda" else []):
        model = GradSampleModule(SimpleCNN().to(dev), loss_reduction="sum")
        model._module.load_state_dict(state)
        model.train()
        try:
            yield model
        finally:
            model.remove_hooks()


def synthetic_samples(c, dev, rng, *, budget=None):
    budget = c["M_syn"] if budget is None else budget
    before = rng.audit()
    with rng.use():
        batches = []
        for start in range(0, budget, c["batch_size"]):
            b = min(c["batch_size"], budget-start)
            batches.append((generate_pink_noise(b, (1, 28, 28), dev),
                            torch.randint(0, 10, (b,), device=dev)))
    x, y = (torch.cat([b[j] for b in batches]) for j in (0, 1))
    return (x, y), dict(rng_before=before, rng_after=rng.audit(), samples_hash=digest([x]), labels_hash=digest([y]), count=len(y))


def make_p(v, damping):
    return {n: 1 / (x.sqrt() + damping) for n, x in v.items()}


def state_bytes(state):
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(state_bytes(x) for x in state.values())
    if isinstance(state, (tuple, list)):
        return sum(state_bytes(x) for x in state)
    return 0


def apply(model, active):
    if active is None:
        return
    kind, value = active
    if kind == "dp_kfc":
        precondition_per_sample_gradients(model, *value)
    elif kind == "syn_diag":
        for name, p in value.items():
            layer = getattr(model._module, name)
            p = p.reshape(layer.weight.shape[0], -1)
            layer.weight.grad_sample.mul_(p[:, :-1].reshape_as(layer.weight).unsqueeze(0))
            layer.bias.grad_sample.mul_(p[:, -1].unsqueeze(0))
    else:
        raise ValueError(kind)


def refresh(state, samples, c, dev, method, *, return_covariances=False):
    x, y = samples
    with diagnostic_model(state, dev) as model:
        if method == "syn_diag":
            sums = {}
            for start in range(0, len(x), c["analysis_batch_size"]):
                model.zero_grad(set_to_none=True)
                F.cross_entropy(model(x[start:start+c["analysis_batch_size"]].to(dev)),
                                y[start:start+c["analysis_batch_size"]].to(dev), reduction="sum").backward()
                for n in LAYERS:
                    value = layer_gradient(getattr(model._module, n)).detach().double().square().sum(0)
                    sums[n] = sums.get(n, 0) + value
            # Active P is actually FP32, the same dtype used in training.
            p = make_p({n: v / len(x) for n, v in sums.items()}, c["lambda"])
            return method, {n: v.float() for n, v in p.items()}
        if method != "dp_kfc":
            raise ValueError(method)
        recorder = KFACRecorder(model)
        recorder.enable()
        factors = []
        try:
            for start in range(0, len(x), c["batch_size"]):
                model.zero_grad(set_to_none=True)
                F.cross_entropy(model(x[start:start+c["batch_size"]].to(dev)),
                                y[start:start+c["batch_size"]].to(dev)).backward()
                # MEAN loss is the upstream recorder convention. Do not rescale
                # backprops or conflate covariance ridge with root damping.
                # Equal 256-sample microbatches; retain upstream covariance ridge.
                factors.append(compute_covariances(model, recorder.activations, recorder.backprops))
                recorder.clear()
            cov = accumulate_covariances(factors)
            active = method, compute_inverse_sqrt(cov, damping=c["damping"])
            return (active, cov) if return_covariances else active
        finally:
            recorder.remove()
