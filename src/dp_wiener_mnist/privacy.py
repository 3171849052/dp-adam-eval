"""Source DP-KFC global clipping and summed-space Gaussian mechanism."""

import math
from .utils import require_finite
from typing import List
import torch
import torch.nn as nn

from .types import Tensor


def clip_and_noise_gradients(
    model: nn.Module,
    noise_multiplier: float,
    max_grad_norm: float,
    sample_count: int,
    expected_batch_size: int | None = None,
    store_summed_grad: bool = False,
) -> None:
    """Apply global clipping and the summed-space Gaussian mechanism.

    ``sample_count`` is the actual size of the current Poisson batch.  The
    normalization denominator is always ``expected_batch_size`` (the logical
    configured batch size), including when the batch is empty.
    """
    if expected_batch_size is None:
        # This preserves the fixed-batch helper's old calling convention while
        # making the two concepts explicit for Poisson callers.
        expected_batch_size = sample_count
    if (
        type(sample_count) is not int
        or sample_count < 0
        or type(expected_batch_size) is not int
        or expected_batch_size <= 0
        or not math.isfinite(noise_multiplier)
        or noise_multiplier < 0
        or not math.isfinite(max_grad_norm)
        or max_grad_norm <= 0
    ):
        raise ValueError("Invalid DP parameters")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("Model has no trainable parameters")

    if sample_count == 0:
        # An empty Poisson draw is still one Gaussian mechanism.  There is no
        # private backward pass in this case, so construct its zero summed
        # gradient directly and consume exactly one noise tensor per parameter.
        for p in trainable:
            summed = torch.zeros_like(p)
            if store_summed_grad:
                p.summed_grad = summed / expected_batch_size
            noise = torch.randn_like(summed) * noise_multiplier * max_grad_norm
            p.grad = (summed + noise) / expected_batch_size
            if hasattr(p, "grad_sample"):
                p.grad_sample = None
        return

    params = trainable
    if any(not hasattr(p, "grad_sample") or p.grad_sample is None for p in params):
        raise ValueError("Missing per-example gradients")
    for p in params:
        sample = _get_grad_sample(p)
        if sample.shape != (sample_count, *p.shape):
            raise ValueError("Invalid per-example gradient shape")
        require_finite(sample)

    device = params[0].device
    total_norm_sq = _compute_per_sample_norms_squared(params, sample_count, device)
    require_finite(total_norm_sq)
    clip_factors = _compute_clip_factors(total_norm_sq, max_grad_norm)

    for p in params:
        grad_sample = _get_grad_sample(p)
        grad_sample = grad_sample.contiguous().view(sample_count, -1)

        clipped = grad_sample * clip_factors.unsqueeze(1)
        summed = clipped.sum(dim=0)

        if store_summed_grad:
            p.summed_grad = (summed / expected_batch_size).view_as(p)

        noise = torch.randn_like(summed) * noise_multiplier * max_grad_norm

        p.grad = ((summed + noise) / expected_batch_size).view_as(p)
        p.grad_sample = None


def _get_params_with_grad_sample(model: nn.Module) -> List[nn.Parameter]:
    return [
        p
        for p in model.parameters()
        if hasattr(p, "grad_sample") and p.grad_sample is not None
    ]


def _get_grad_sample(param: nn.Parameter) -> Tensor:
    grad_sample = param.grad_sample
    if isinstance(grad_sample, list):
        if len(grad_sample) != 1:
            raise ValueError("Accumulated grad_sample batches are unsupported")
        grad_sample = grad_sample[-1]
    return grad_sample


def _compute_per_sample_norms_squared(
    params: List[nn.Parameter],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    total_norm_sq = torch.zeros(batch_size, device=device)
    for p in params:
        grad_sample = _get_grad_sample(p)
        grad_sample = grad_sample.contiguous().view(batch_size, -1)
        total_norm_sq += grad_sample.norm(2, dim=1).pow(2)
    return total_norm_sq


def _compute_clip_factors(
    total_norm_sq: Tensor,
    max_grad_norm: float,
) -> Tensor:
    total_norm = total_norm_sq.sqrt()
    return (max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0)
