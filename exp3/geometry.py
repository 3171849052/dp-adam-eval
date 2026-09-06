"""Float64, parameter-chunked Gram spectra with disk-backed sample gradients."""
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from exp3.common import LAYERS, layer_gradient
from exp3.preconditioners import diagnostic_model, apply


def spread(v, eps=1e-12):
    v = torch.as_tensor(v, dtype=torch.float64)
    if not torch.isfinite(v).all() or (v < 0).any():
        raise FloatingPointError("Invalid nonnegative geometry input")
    z = torch.log10(v + eps)
    return float(torch.quantile(z, .95) - torch.quantile(z, .05))


def positive_eigenvalues(gram, dimension):
    gram = (gram.double() + gram.double().T) / 2
    if not torch.isfinite(gram).all():
        raise FloatingPointError("Nonfinite Gram")
    eig = torch.linalg.eigvalsh(gram)
    tolerance = float(eig[-1].clamp_min(0)) * max(dimension, len(eig)) * torch.finfo(torch.float64).eps
    if float(eig[0]) < -max(tolerance * 10, 1e-15):
        raise FloatingPointError("Gram is not numerically PSD")
    return eig[eig > tolerance], tolerance


def gram_spectrum(h, chunk=2048, dev=torch.device("cpu")):
    m, d = h.shape
    gram = torch.zeros((m, m), dtype=torch.float64, device=dev)
    for start in range(0, d, chunk):
        a = torch.as_tensor(np.array(h[:, start:start+chunk], copy=True), device=dev, dtype=torch.float64)
        gram.addmm_(a, a.T)
    return positive_eigenvalues(gram / m, d)


def geometry(h, c, dev):
    v = np.zeros(h.shape[1], dtype=np.float64)
    for start in range(0, h.shape[1], c["gram_chunk"]):
        a = np.array(h[:, start:start+c["gram_chunk"]], dtype=np.float64)
        v[start:start+a.shape[1]] = np.mean(a*a, axis=0)
    eig, tol = gram_spectrum(h, c["gram_chunk"], dev)
    if len(eig) < 2:
        raise FloatingPointError("Fisher has fewer than two nonzero eigenvalues; spectral spread undefined")
    return dict(A=spread(v, c["eps_num"]), S=spread(eig.cpu(), c["eps_num"]), rank=len(eig), eigen_tolerance=tol)


def ratio(pre, raw):
    if raw <= 0 or pre <= 0:
        raise FloatingPointError("Zero Fisher spread: positive ratio is undefined")
    return pre / raw


def compare(raw, pre):
    return dict(R_diag=ratio(pre["A"], raw["A"]), A_diag_raw=raw["A"], A_diag_pre=pre["A"],
                R_full=ratio(pre["S"], raw["S"]), S_full_raw=raw["S"], S_full_pre=pre["S"],
                rank_raw=raw["rank"], rank_pre=pre["rank"],
                eigen_tolerance_raw=raw["eigen_tolerance"], eigen_tolerance_pre=pre["eigen_tolerance"])


def stale(old, new):
    return dict(A_old_diag=None if old is None else old["A"], A_new_diag=new["A"],
                delta_stale_diag=None if old is None else old["A"]-new["A"],
                S_old_full=None if old is None else old["S"], S_new_full=new["S"],
                delta_stale_full=None if old is None else old["S"]-new["S"])


def oracle_compare(raw, new, old=None):
    # Legacy R/A_pre/S_pre columns are exact aliases for fresh/new geometry.
    row = compare(raw, new)
    for label, value in (("new", new), ("old", old)):
        row.update({f"R_diag_{label}": None if value is None else ratio(value["A"], raw["A"]),
                    f"R_full_{label}": None if value is None else ratio(value["S"], raw["S"]),
                    f"A_diag_{label}": None if value is None else value["A"],
                    f"S_full_{label}": None if value is None else value["S"],
                    f"rank_{label}": None if value is None else value["rank"],
                    f"eigen_tolerance_{label}": None if value is None else value["eigen_tolerance"]})
    row.update(private_delta_stale_diag=None if old is None else old["A"]-new["A"],
               private_delta_stale_full=None if old is None else old["S"]-new["S"])
    return row


def diagnose(state, samples, transforms, c, dev, scratch):
    """Apply all transforms to exactly the same raw gradients, on a separate model.

    Temporary FP32 matrices are removed on success/failure; eigensolvers use
    float64. No giant parameter-by-parameter Fisher is ever allocated.
    """
    x, y = samples
    with tempfile.TemporaryDirectory(prefix="geometry_", dir=scratch) as temp:
        maps = {}
        try:
            with diagnostic_model(state, dev) as model:
                for label in transforms:
                    for n in LAYERS:
                        layer = getattr(model._module, n)
                        d = layer.weight.numel() + layer.bias.numel()
                        maps[label, n] = np.memmap(Path(temp) / f"{label}_{n}.bin", mode="w+", dtype="float32", shape=(len(x), d))
                for start in range(0, len(x), c["analysis_batch_size"]):
                    stop = min(len(x), start+c["analysis_batch_size"])
                    model.zero_grad(set_to_none=True)
                    F.cross_entropy(model(x[start:stop].to(dev)), y[start:stop].to(dev), reduction="sum").backward()
                    original = [p.grad_sample.detach().clone() for p in model.parameters()]
                    for label, active in transforms.items():
                        for p, g in zip(model.parameters(), original):
                            p.grad_sample = g.clone()
                        apply(model, active)
                        for n in LAYERS:
                            maps[label, n][start:stop] = layer_gradient(getattr(model._module, n)).detach().cpu().numpy()
            return {label: {n: geometry(maps[label, n], c, dev) for n in LAYERS} for label in transforms}
        finally:
            for mmap in maps.values():
                mmap._mmap.close()
