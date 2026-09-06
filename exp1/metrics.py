"""Alignment metrics with Private as oracle; no dense Fisher construction."""
import math
import numpy as np
import torch
from scipy.stats import spearmanr


def alignment(inner, source_sq, private_sq, source_trace, private_trace):
    if source_sq <= 0 or private_sq <= 0 or private_trace <= 0:
        return dict.fromkeys(('cosine', 'relative_frobenius_error', 'trace_ratio', 'alpha_star', 'shape_error'), None)
    cosine = min(1., max(0., inner / math.sqrt(source_sq * private_sq)))
    return {'cosine': cosine,
            'relative_frobenius_error': math.sqrt(max(0., source_sq + private_sq - 2*inner) / private_sq),
            'trace_ratio': source_trace / private_trace,
            'alpha_star': inner / source_sq,
            'shape_error': math.sqrt(max(0., 1 - cosine*cosine))}


def matrix_alignment(source, private):
    return alignment(float((source*private).sum()), float(source.square().sum()),
                     float(private.square().sum()), float(source.trace()), float(private.trace()))


def kfac_alignment(source, private):
    a, g, pa, pg = source['A'], source['G'], private['A'], private['G']
    return alignment(float((a*pa).sum() * (g*pg).sum()),
                     float(a.square().sum() * g.square().sum()),
                     float(pa.square().sum() * pg.square().sum()),
                     float(a.trace() * g.trace()), float(pa.trace() * pg.trace()))


def vector_alignment(source, private):
    return alignment(float((source*private).sum()), float(source.square().sum()),
                     float(private.square().sum()), float(source.sum()), float(private.sum()))


def spread(log_values):
    return float(np.quantile(log_values, .95) - np.quantile(log_values, .05))


def diagonal_metrics(source, private, eps):
    src, priv = source.double().numpy(), private.double().numpy()
    ls, lp = np.log10(src + eps), np.log10(priv + eps)
    valid = np.std(ls) > 0 and np.std(lp) > 0
    raw = spread(lp)
    residual = spread(np.log10(priv / (src + eps) + eps))
    return {'log_pearson': float(np.corrcoef(ls, lp)[0, 1]) if valid else None,
            'log_spearman': float(spearmanr(ls, lp).statistic) if valid else None,
            'log_ratio_spread': spread(lp-ls), 'A_raw': raw, 'A_residual': residual,
            'R': residual / raw if raw > 0 else None}
