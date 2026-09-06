"""Read-only research diagnostics; private outputs are not DP releases."""
import numpy as np
import torch
from common import LAYERS, diagonal_metrics, spread


def anisotropy(x, eps):
    return spread(torch.log10(x.double().cpu() + eps).numpy())


def centered(v, eps):
    z = (v + eps).log()
    return z - z.mean()


def attenuation(v, damping):
    """Coordinate attenuation of the current synthetic inverse-root geometry."""
    s = v.sqrt()
    return s / (s + damping)


def refresh_rows(v, halves, p, old_v, old_p, c):
    rows = []
    eps = c['eps_num']
    for name in LAYERS:
        s = v[name].sqrt()
        row = {'layer': name}
        for prefix, values in [('v_syn', v[name]), ('sqrt_v', s), ('P', p[name])]:
            for q in (1, 5, 50, 95, 99):
                row[f'{prefix}_q{q:02d}'] = float(torch.quantile(values, q/100))
        a = attenuation(v[name], c['lambda'])
        if not torch.isfinite(a).all() or not ((a >= 0) & (a <= 1)).all():
            raise FloatingPointError('Invalid synthetic attenuation')
        for q in (5, 50, 95):
            row[f'attenuation_q{q:02d}'] = float(torch.quantile(a, q/100))
        med = float(torch.quantile(s, .5))
        row['lambda_over_median'] = c['lambda'] / med if med else None
        row['fraction_below_lambda'] = float((s < c['lambda']).double().mean())
        row['D_sample'] = float((centered(halves[0][name], eps) - centered(halves[1][name], eps)).square().mean().sqrt())
        row['D_time'] = None if old_v is None else float((centered(v[name], eps) - centered(old_v[name], eps)).square().mean().sqrt())
        row['A_new'] = anisotropy(p[name].square() * v[name], eps)
        row['A_old'] = None if old_p is None else anisotropy(old_p[name].square() * v[name], eps)
        row['stale_ratio'] = None if old_p is None else row['A_old'] / (row['A_new'] + eps)
        rows.append(row)
    return rows


def oracle_rows(v, private, p, c):
    rows = []
    for name in LAYERS:
        row = diagonal_metrics(v[name], private[name], c['eps_num'])
        row['A_residual'] = anisotropy(p[name].square() * private[name], c['eps_num'])
        row['R'] = row['A_residual'] / row['A_raw'] if row['A_raw'] > 0 else None
        rows.append({'layer': name, **row})
    return rows
