"""Adaptive-beta gain construction on top of the ExpV1 eigensystem."""

import math

import torch

from expv1.fisher_wiener import build_fisher_state


def _validate_beta(beta):
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("training beta must be positive and finite")
    return beta


@torch.no_grad()
def rebuild_H_with_beta(base_state, beta_by_layer, r):
    """Return a Fisher state with only its eigenvalue gain rescaled.

    ``base_state`` is the state from ExpV1's ``build_fisher_state``.  The
    eigenvectors and eigenvalues are reused verbatim.  The beta=1 branch
    returns ExpV1's already-computed H to preserve bitwise equivalence.
    """
    result = {}
    for layer, state in base_state.items():
        beta = _validate_beta(beta_by_layer[layer])
        if beta == 1.0:
            h = state["H"].clone()
        else:
            lambda_f = state["lambda_G"][:, None] * state["lambda_A"][None, :]
            scaled_lambda_f = beta * lambda_f
            h = torch.where(
                scaled_lambda_f + r > 0,
                scaled_lambda_f / (scaled_lambda_f + r),
                torch.zeros_like(scaled_lambda_f),
            ).to(dtype=state["H"].dtype)
        if not bool(torch.isfinite(h).all()) or bool((h < 0).any()) or bool((h > 1).any()):
            raise FloatingPointError(f"invalid adaptive Wiener gain for {layer}")
        result[layer] = dict(state, H=h)
    return result


@torch.no_grad()
def build_adaptive_fisher_state(covariances, sigma, max_grad_norm, batch_size, beta_by_layer):
    """Build ExpV1's eigensystem once and rescale its gains by layer beta."""
    base = build_fisher_state(covariances, sigma, max_grad_norm, batch_size)
    r = (float(sigma) * float(max_grad_norm) / float(batch_size)) ** 2
    return rebuild_H_with_beta(base, beta_by_layer, r)


def h_stats(h):
    values = h.detach().double().flatten()
    return {
        "H_mean": float(values.mean()), "H_std": float(values.std(unbiased=False)),
        "H_min": float(values.min()), "H_max": float(values.max()),
        "H_q10": float(values.quantile(0.10)), "H_q25": float(values.quantile(0.25)),
        "H_q50": float(values.quantile(0.50)), "H_q75": float(values.quantile(0.75)),
        "H_q90": float(values.quantile(0.90)),
    }


def h_hash(h):
    import hashlib
    value = h.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def stats_digest(row, prefix="H"):
    import hashlib
    import json
    keys = [f"{prefix}_{name}" for name in ("mean", "std", "min", "max", "q10", "q25", "q50", "q75", "q90")]
    # CSV round-tripping can change the final binary float bits.  The
    # certificate is a consistency digest, so use a stable decimal precision.
    payload = json.dumps([round(float(row[key]), 12) for key in keys], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def selected_stats_digest(row, prefix="H"):
    """Digest the compact quantiles retained for the beta=1 certificate."""
    import hashlib
    import json
    keys = [f"{prefix}_{name}" for name in ("mean", "q10", "q50", "q90")]
    payload = json.dumps([round(float(row[key]), 12) for key in keys], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
