"""Research-only scalar diagnostics. No result is consumed by training."""
import math
import torch
from expv1.common import LAYERS
from expv1.fisher_wiener import pack_layer_gradient, apply_filter_to_copy

EPS = 1e-12


def reconstruction(s, y, hat, wg, wn, identity=False):
    s, y, hat, wg, wn = [v.double().flatten() for v in (s, y, hat, wg, wn)]
    n = y-s
    sp, np = float(s.square().sum()), float(n.square().sum())
    gp, wp = float(wg.square().sum()), float(wn.square().sum())
    err = float((hat-s).square().sum())
    def cosine(v):
        den = float(v.norm()*s.norm())
        return float((v*s).sum())/den if den > 0 else 0.
    sin, sout = sp/(np+EPS), gp/(wp+EPS)
    row = dict(clean_clipped_norm=math.sqrt(sp), actual_noise_norm=math.sqrt(np),
        noisy_gradient_norm=float(y.norm()), filtered_gradient_norm=float(hat.norm()),
        relmse_noisy=np/(sp+EPS), relmse_filtered=err/(sp+EPS),
        cosine_noisy=cosine(y), cosine_filtered=cosine(hat),
        signal_retention=gp/(sp+EPS), noise_retention=wp/(np+EPS),
        snr_in=sin, snr_out=sout, snr_gain_db=10*math.log10((sout+EPS)/(sin+EPS)),
        mse_reduction=1-err/(np+EPS))
    if identity:
        row.update(signal_retention=1., noise_retention=1., snr_gain_db=0., mse_reduction=0.,
                   relmse_filtered=row['relmse_noisy'], cosine_filtered=row['cosine_noisy'])
    return row


def gain_stats(active_layer_state, diagnostic_layer_state, method):
    if active_layer_state is None:
        return dict(H_mean=1., H_std=0., H_min=1., H_max=1.,
                    **{f'H_q{q}': 1. for q in (10,25,50,75,90)})
    h = (active_layer_state['H'].double().flatten()
         if method == 'dp_fisher_wiener'
         else active_layer_state['scalar_h'].double().reshape(1))
    row = dict(H_mean=float(h.mean()), H_std=float(h.std(unbiased=False)), H_min=float(h.min()), H_max=float(h.max()))
    row.update({f'H_q{q}': float(h.quantile(q/100)) for q in (10,25,50,75,90)})
    diagnostic = diagnostic_layer_state
    if diagnostic is None:
        return row
    row.update(lambdaF_mean=diagnostic['lambdaF_mean'], lambdaF_median=diagnostic['lambdaF_median'])
    row.update({f'lambdaF_q{q}': diagnostic[f'lambdaF_q{q}'] for q in (10,90)})
    row.update({k: diagnostic[k] for k in ('trace_A','trace_G','trace_F')})
    return row


def eigenbins(s, n, state):
    qa, qg = state['Q_A'], state['Q_G']
    sp = (qg.T @ s @ qa).double().flatten()
    np = (qg.T @ n @ qa).double().flatten()
    h = state['H'].double().flatten()
    lf = (state['lambda_G'][:,None]*state['lambda_A'][None,:]).double().flatten()
    rows = []
    for b, ids in enumerate(torch.tensor_split(lf.argsort(stable=True), 10)):
        l, s1, n1, h1 = lf[ids], sp[ids], np[ids], h[ids]
        signal, noise = float(s1.square().mean()), float(n1.square().mean())
        after = float((h1*(s1+n1)-s1).square().mean())
        rows.append(dict(bin=b, count=len(ids), lambda_min=float(l.min()), lambda_max=float(l.max()),
            lambda_mean=float(l.mean()), log10_lambda_mean=math.log10(max(float(l.mean()),EPS)),
            signal_power=signal, noise_power=noise, empirical_snr=signal/(noise+EPS), H_mean=float(h1.mean()),
            mse_before=noise, mse_after=after, mse_reduction=1-after/(noise+EPS),
            signal_retention=float((h1*s1).square().mean())/(signal+EPS),
            noise_retention=float((h1*n1).square().mean())/(noise+EPS)))
    return rows


@torch.no_grad()
def diagnose(model, noisy, active_state, diagnostic_state, method,
             refresh=False, eigen_budget=None):
    base = getattr(model, '_module', model)
    layers, bins, all_values = [], [], []
    for name in LAYERS:
        layer = getattr(base, name)
        s = pack_layer_gradient(layer, 'summed_grad')
        y = noisy[name]
        hat = pack_layer_gradient(layer)
        active = None if active_state is None else active_state[name]
        diagnostic = None if diagnostic_state is None else diagnostic_state[name]
        wg = apply_filter_to_copy(s, active, method)
        wn = apply_filter_to_copy(y-s, active, method)
        values = (s, y, hat, wg, wn)
        row = dict(layer=name, **reconstruction(*values, identity=method=='dp_sgd'),
                   **gain_stats(active, diagnostic, method))
        row['kappa'] = (float(s.double().square().sum()) /
                        (diagnostic['trace_F']+EPS) if diagnostic is not None else None)
        layers.append(row)
        all_values.append(values)
        if refresh and method == 'dp_fisher_wiener' and (eigen_budget is None or LAYERS.index(name) < eigen_budget):
            bins.extend(dict(layer=name, **r) for r in eigenbins(s, y-s, active))
    global_row = reconstruction(*(torch.cat([v[i].flatten() for v in all_values]) for i in range(5)), identity=method=='dp_sgd')
    return global_row, layers, bins
