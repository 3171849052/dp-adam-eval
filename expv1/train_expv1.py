"""Paired fixed-protocol runs. Formal experiments only by explicit CLI invocation."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from expv1.common import (ROOT, METHODS, LAYERS, read_config, check_config, fingerprint,
    provenance, require_pinned, output_path, save_json, write_csv, datasets,
    set_seed, SimpleCNN, RNGStream, digest)
from expv1.fisher_wiener import (synthetic_samples, build_covariances,
    build_scalar_state, build_fisher_state, build_diagnostic_state,
    pack_layer_gradient, apply_fisher_wiener, apply_scalar_wiener, state_bytes)
from expv1.metrics import diagnose
from dp_kfac.privacy import clip_and_noise_gradients, _compute_per_sample_norms_squared


def sync(model):
    dev = next(model.parameters()).device
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)


def private_update(model, active, optimizer, rng, sigma, c, b, method,
                   diagnostics=True, refresh=False, eigen_budget=None,
                   diagnostic_state=None):
    if method not in METHODS:
        raise ValueError(method)
    diagnostic_seconds = 0.
    clip_rate = None
    if diagnostics:
        sync(model)
        t = time.perf_counter()
        sq = _compute_per_sample_norms_squared(list(model.parameters()), b, next(model.parameters()).device)
        clip_rate = float((sq.sqrt() > c['max_grad_norm']).float().mean())
        sync(model)
        diagnostic_seconds += time.perf_counter()-t
    audit = dict(noise_rng_before=rng.audit())
    # Raw per-example gradients -> global clipping -> Gaussian noise.
    with rng.use():
        clip_and_noise_gradients(model, noise_multiplier=sigma,
            max_grad_norm=c['max_grad_norm'], batch_size=b, store_summed_grad=True)
    audit['noise_rng_after'] = rng.audit()
    if diagnostics:
        sync(model)
        t = time.perf_counter()
        noisy = {n: pack_layer_gradient(getattr(model._module, n)) for n in LAYERS}
        sync(model)
        diagnostic_seconds += time.perf_counter()-t
    sync(model)
    t = time.perf_counter()
    # Active filters access only p.grad and synthetic state; never summed_grad.
    if method == 'dp_fisher_wiener':
        apply_fisher_wiener(model, active)
    elif method == 'dp_scalar_wiener':
        apply_scalar_wiener(model, active)
    elif method == 'dp_sgd':
        pass
    sync(model)
    filter_time = time.perf_counter()-t if method != 'dp_sgd' else 0.
    optimizer.step()
    rows, layers, bins = {}, [], []
    if diagnostics:
        sync(model)
        t = time.perf_counter()
        rows, layers, bins = diagnose(model, noisy, active, diagnostic_state,
                                      method, refresh, eigen_budget)
        del noisy
        sync(model)
        diagnostic_seconds += time.perf_counter()-t
    rows.update(clip_rate=clip_rate, wiener_filter_time=filter_time)
    return rows, layers, bins, audit, diagnostic_seconds


class Indexed(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        x, y = self.data[i]
        return x, y, i


def make_optimizer(model, c):
    if c["optimizer"] != "SGD" or c["learning_rate"] != .1 or c["momentum"] != 0:
        raise ValueError("ExpV1 requires plain SGD(lr=0.1, momentum=0)")
    return torch.optim.SGD(model.parameters(), lr=.1, momentum=0)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    loss = correct = count = 0
    for x, y in loader:
        y = y.to(dev)
        out = model(x.to(dev))
        loss += float(F.cross_entropy(out, y, reduction="sum"))
        correct += int((out.argmax(1) == y).sum())
        count += len(y)
    model.train()
    return dict(test_loss=loss/count, test_accuracy=correct/count)


def train(c, seed, method, output, data_override=None, *, diagnostics=True, eigen_budget=None):
    check_config(c)
    if method not in METHODS or seed not in c['seeds']:
        raise ValueError('Unknown method/seed')
    prov = provenance()
    require_pinned(prov, c['smoke'])
    root = output_path(output) / f'seed{seed}' / method
    root.mkdir(parents=True, exist_ok=False)
    save_json(root/'config.json', c)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if c['device']=='auto' else torch.device(c['device'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    source, test = datasets(c) if data_override is None else data_override
    n = c['train_subset'] or len(source)
    if not c['batch_size'] <= n <= len(source):
        raise ValueError('Invalid training subset')
    data = Subset(source, range(n))
    if c['test_subset'] is not None:
        if not 1 <= c['test_subset'] <= len(test):
            raise ValueError('Invalid test subset')
        test = Subset(test, range(c['test_subset']))
    loader = DataLoader(Indexed(data), batch_size=c['batch_size'], shuffle=True, drop_last=True,
        num_workers=0, generator=torch.Generator().manual_seed(seed+1))
    test_loader = DataLoader(test, batch_size=c['batch_size'], num_workers=0,
        generator=torch.Generator().manual_seed(seed+2))
    total, q = len(loader)*c['epochs'], c['batch_size']/n
    sigma = get_noise_multiplier(target_epsilon=c['epsilon'], target_delta=c['delta'],
        sample_rate=q, steps=total, accountant='rdp')
    set_seed(seed)
    model = GradSampleModule(SimpleCNN().to(dev), loss_reduction='sum')
    optimizer = make_optimizer(model, c)
    accountant = RDPAccountant()
    syn_rng, noise_rng = RNGStream(seed+3, dev), RNGStream(seed+4, dev)
    meta = dict(seed=seed, method=method, fingerprint=fingerprint(c), provenance=prov,
        device=str(dev), noise_multiplier=sigma, sample_rate=q, total_steps=total,
        initial_model_hash=digest(model.parameters()), beta=1, covariance_ridge=1e-5,
        rng_seeds=dict(init=seed, loader=seed+1, test=seed+2, synthetic=seed+3, noise=seed+4),
        sampling='fixed shuffle/drop_last; inherited RDP convention, not Poisson',
        diagnostics_enabled=diagnostics, eigen_budget=eigen_budget,
        diagnostics='Non-DP research statistics, never consumed by training', output=str(root))
    save_json(root/'metadata.json', meta)
    rows, layers, bins, refreshes, audits, syn_audits = [], [], [], [], [], []
    active, diagnostic_state, step, diagnostic_seconds = None, None, 0, 0.
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)
    sync(model)
    started = time.perf_counter()
    try:
        for epoch in range(1, c['epochs']+1):
            iterator = iter(loader)
            for _ in range(len(loader)):
                refreshed = step % c['K'] == 0 and method != 'dp_sgd'
                refresh_time = 0.
                diagnostic_spectrum_time = 0.
                # Refresh precedes next(iterator): no current private batch access.
                if refreshed:
                    sync(model)
                    t = time.perf_counter()
                    samples, sa = synthetic_samples(c, dev, syn_rng)
                    cov = build_covariances(model._module.state_dict(), samples, c, dev)
                    if method == 'dp_scalar_wiener':
                        active = build_scalar_state(cov, sigma, c['max_grad_norm'], c['batch_size'])
                    elif method == 'dp_fisher_wiener':
                        active = build_fisher_state(cov, sigma, c['max_grad_norm'], c['batch_size'])
                    else:
                        raise ValueError(method)
                    sync(model)
                    refresh_time = time.perf_counter()-t
                    if diagnostics:
                        sync(model)
                        t = time.perf_counter()
                        diagnostic_state = build_diagnostic_state(cov, active, method)
                        sync(model)
                        diagnostic_spectrum_time = time.perf_counter()-t
                        diagnostic_seconds += diagnostic_spectrum_time
                    else:
                        diagnostic_state = None
                    del samples, cov
                    refreshes.append(dict(step=step, refresh_time=refresh_time,
                        diagnostic_spectrum_time=diagnostic_spectrum_time,
                        active_state_bytes=state_bytes(active)))
                    syn_audits.append(dict(step=step, **sa))
                x, y, indices = next(iterator)
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x.to(dev)), y.to(dev), reduction='sum')
                loss.backward()
                row, lr, er, audit, dt = private_update(model, active, optimizer, noise_rng, sigma, c,
                    len(x), method, diagnostics, refreshed, eigen_budget,
                    diagnostic_state)
                diagnostic_seconds += dt
                accountant.step(noise_multiplier=sigma, sample_rate=q)
                row.update(step=step, epoch=epoch, train_loss=float(loss.detach())/len(x),
                    refresh_time=refresh_time, diagnostic_spectrum_time=diagnostic_spectrum_time,
                    active_state_bytes=state_bytes(active), test_loss=None, test_accuracy=None)
                if (step+1) % c['eval_interval'] == 0 or step+1 == total:
                    row.update(evaluate(model, test_loader, dev))
                rows.append(row)
                layers.extend(dict(step=step, **r) for r in lr)
                bins.extend(dict(step=step, **r) for r in er)
                finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
                if not finite:
                    raise FloatingPointError('Nonfinite model')
                audits.append(dict(step=step, batch_indices=indices.tolist(), model_hash=digest(model.parameters()),
                    parameters_finite=finite, **audit))
                step += 1
        sync(model)
        wall = time.perf_counter()-started
        ev = [r for r in rows if r['test_accuracy'] is not None]
        acc = [r['test_accuracy'] for r in ev]
        progress = np.array([(r['step']+1)/total for r in ev])
        auc = float(np.trapezoid(acc, progress)/(progress[-1]-progress[0])) if len(ev)>1 else acc[0]
        late = [r['test_accuracy'] for r in ev if r['step']+1 > total/2]
        rt = sum(r['refresh_time'] for r in refreshes)
        dst = sum(r['diagnostic_spectrum_time'] for r in refreshes)
        ft = sum(r['wiener_filter_time'] for r in rows)
        summary = dict(seed=seed, method=method, fingerprint=fingerprint(c), completed_steps=step,
            epsilon_spent=accountant.get_epsilon(c['delta']), noise_multiplier=sigma,
            final_accuracy=acc[-1], best_accuracy=max(acc), late_mean_accuracy=float(np.mean(late)),
            accuracy_auc=auc, final_test_loss=ev[-1]['test_loss'], final_model_hash=digest(model.parameters()),
            parameters_finite=True, wall_time=wall, diagnostic_seconds=diagnostic_seconds,
            core_training_runtime=wall-diagnostic_seconds, total_refresh_time=rt,
            mean_refresh_time=rt/len(refreshes) if refreshes else 0., number_of_refreshes=len(refreshes),
            total_diagnostic_spectrum_time=dst,
            mean_diagnostic_spectrum_time=dst/len(refreshes) if refreshes else 0.,
            total_wiener_filter_time=ft, mean_wiener_filter_time=ft/total,
            active_state_bytes=state_bytes(active), peak_cuda_allocated_memory=torch.cuda.max_memory_allocated(dev) if dev.type=='cuda' else 0)
        save_json(root/'summary.json', summary)
        meta.update(summary, complete=True)
        save_json(root/'metadata.json', meta)
    finally:
        model.remove_hooks()
        for name, values in [('train',rows),('layer',layers),('eigenbin',bins),('refresh',refreshes)]:
            empty_fields = (['seed','method','config_fingerprint','step','refresh_time',
                             'diagnostic_spectrum_time','active_state_bytes']
                            if name == 'refresh' else ['seed','method','config_fingerprint','step','layer'])
            write_csv(root/f'{name}_metrics.csv', [dict(seed=seed, method=method, config_fingerprint=fingerprint(c), **r) for r in values],
                fields=None if values else empty_fields)
        save_json(root/'pairing.json', dict(private=audits, synthetic=syn_audits))
    print(f'Completed seed={seed} {method}: {step} steps, accuracy={acc[-1]:.4f}', flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/full.json'))
    parser.add_argument('--output', default=str(ROOT/'runs'))
    parser.add_argument('--method', choices=[*METHODS,'all'], default='all')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    c = read_config(args.config)
    for seed in c['seeds'] if args.seed is None else [args.seed]:
        for method in METHODS if args.method=='all' else [args.method]:
            train(c, seed, method, args.output)


if __name__ == '__main__':
    main()
