"""Training diagnostics using exactly the upstream clipping factors."""
import math
import torch
from exp3.common import LAYERS, layer_gradient
from dp_kfac.privacy import _compute_per_sample_norms_squared, _compute_clip_factors


def before_clip(model, c, b):
    params = list(model.parameters())
    sq = _compute_per_sample_norms_squared(params, b, params[0].device)
    norms = sq.sqrt()
    coeff = _compute_clip_factors(sq, c['max_grad_norm'])
    row = {'clip_rate': float((norms > c['max_grad_norm']).float().mean()), 'coefficient_mean': float(coeff.mean())}
    row['coefficient_std'] = float(coeff.double().std(unbiased=False))
    row['coefficient_cv'] = row['coefficient_std'] / (float(coeff.double().mean()) + c['eps_num'])
    for prefix, value, qs in [('norm', norms, (10, 50, 90, 99)), ('coefficient', coeff, (10, 50, 90))]:
        for q in qs:
            row[f'{prefix}_q{q}'] = float(torch.quantile(value, q/100))
    dot = raw_sq = clipped_sq = diff_sq = 0.
    aggregates = []
    for name in LAYERS:
        g = layer_gradient(getattr(model._module, name)).detach()
        raw, clipped = g.mean(0), (g * coeff[:, None]).mean(0)
        aggregates.append((raw.double(), clipped.double()))
        dot += float((raw.double()*clipped).sum())
        raw_sq += float(raw.double().square().sum())
        clipped_sq += float(clipped.double().square().sum())
        diff_sq += float((raw.double()-clipped).square().sum())
        row[f'contrib_{name}'] = float((g.square().sum(1) / sq.clamp_min(c['eps_num'])).mean())
    row['aggregate_cosine'] = dot / math.sqrt(raw_sq * clipped_sq) if raw_sq * clipped_sq > 0 else None
    row['relative_distortion'] = math.sqrt(diff_sq) / (math.sqrt(raw_sq) + c['eps_num'])
    # One scalar for the full aggregate, not independently fitted layer scalars.
    alpha = dot / (raw_sq + c['eps_num'])
    # Direct float64 residual avoids cancellation in ||clip||² - dot²/||raw||².
    residual_sq = sum(float((clipped - alpha * raw).square().sum())
                      for raw, clipped in aggregates)
    row['clipping_alpha_star'] = alpha
    row['clipping_shape_error'] = math.sqrt(residual_sq) / (math.sqrt(clipped_sq) + c['eps_num'])
    row['clipped_aggregate_norm'] = math.sqrt(clipped_sq)  # mean, not sum
    return row


def after_noise(model, sigma, c, b):
    params = list(model.parameters())
    noise_sq = sum(float((p.grad.double() - p.summed_grad.double()).square().sum()) for p in params)
    grad_sq = sum(float(p.grad.double().square().sum()) for p in params)
    expected = sigma * c['max_grad_norm'] / b * math.sqrt(sum(p.numel() for p in params))
    return {'actual_noise_norm': math.sqrt(noise_sq), 'expected_noise_norm': expected,
            'noisy_update_norm': math.sqrt(grad_sq), 'update_norm': c['learning_rate'] * math.sqrt(grad_sq)}


def norm_metrics(norms, eps=1e-12):
    x = norms.detach().double()
    mean, std = float(x.mean()), float(x.std(unbiased=False))
    q = torch.quantile(x, torch.tensor([.1, .5, .9, .99], dtype=x.dtype, device=x.device))
    return dict(norm_mean=mean, norm_std=std, norm_cv=std/(mean+eps),
                norm_q10=float(q[0]), norm_q50=float(q[1]), norm_q90=float(q[2]), norm_q99=float(q[3]),
                norm_q90_over_q10=float(q[2])/(float(q[0])+eps))
