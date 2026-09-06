"""Paired, block-wise synthetic diagonal DP-SGD. No adaptive controls."""
import csv
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from common import (parse, run_root, fingerprint, provenance, save_json, save_torch,
                    datasets, device, set_seed, SimpleCNN, RNGStream, digest)
from synthetic_preconditioner import synthetic, estimate, make_p, apply_p
from diagnostics import refresh_rows, oracle_rows
from metrics import before_clip, after_noise
from dp_kfac.privacy import DPGradientAccumulator


class Indexed(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        x, y = self.data[i]
        return x, y, i


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    loss = correct = count = 0
    for x, y in loader:
        out = model(x.to(dev))
        y = y.to(dev)
        loss += float(F.cross_entropy(out, y, reduction='sum'))
        correct += int((out.argmax(1) == y).sum())
        count += len(y)
    model.train()
    return {'test_loss': loss/count, 'test_accuracy': correct/count}


def train(c, method):
    if method != 'syn_diag':
        raise ValueError('Exp2b runs only syn_diag')
    root = run_root(c)
    marker = root / f'{method}.started.json'
    if marker.exists():
        raise ValueError(f'{method} output already exists; choose a fresh config output directory')
    dev = device(c)
    set_seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, test = datasets(c)
    n = c['train_subset'] if c['train_subset'] is not None else len(source)
    if not c['batch_size'] <= n <= len(source):
        raise ValueError('train_subset must contain at least one batch and fit MNIST')
    data = Subset(source, range(n))
    if c['test_subset'] is not None:
        if not 1 <= c['test_subset'] <= len(test):
            raise ValueError('Invalid test_subset')
        test = Subset(test, range(c['test_subset']))
    loader_rng = torch.Generator().manual_seed(c['seed'] + 1)
    # num_workers=0 is deliberate: no private prefetch across a refresh boundary.
    loader = DataLoader(Indexed(data), batch_size=c['batch_size'], shuffle=True,
                        drop_last=True, num_workers=0, generator=loader_rng)
    test_loader = DataLoader(test, batch_size=c['batch_size'], shuffle=False,
                             num_workers=0, generator=torch.Generator().manual_seed(c['seed'] + 2))
    total = len(loader) * c['epochs']
    q = c['batch_size'] / n
    sigma = get_noise_multiplier(target_epsilon=c['epsilon'], target_delta=c['delta'],
                                 sample_rate=q, steps=total, accountant='rdp')
    # Dataset setup cannot change model initialization pairing.
    set_seed(c['seed'])
    model = GradSampleModule(SimpleCNN().to(dev), batch_first=True, loss_reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=c['learning_rate'])
    accountant = RDPAccountant()
    syn_rng, noise_rng = RNGStream(c['seed'] + 3, dev), RNGStream(c['seed'] + 4, dev)
    oracle_ids = torch.randperm(n, generator=torch.Generator().manual_seed(c['seed'] + 5))[:c['M_oracle']]
    if len(oracle_ids) != c['M_oracle']:
        raise ValueError('M_oracle exceeds training dataset')
    metadata = {'config': c, 'method': method, 'fingerprint': fingerprint(c), 'upstream': provenance(),
                'torch': str(torch.__version__), 'device': str(dev), 'noise_multiplier': sigma,
                'sample_rate': q, 'total_steps': total, 'accountant': 'rdp',
                'sampling': 'shuffle, drop_last=True; upstream RDP convention, not Poisson sampling',
                'clipping': 'upstream C/(norm+1e-6), capped at 1',
                'private_diagnostics': 'not DP releases; never used for training or adaptive decisions',
                'parameter_order': list(dict(model._module.named_parameters())),
                'layer_vector_order': 'Exp1 output-unit rows, weight followed by bias',
                'oracle_indices': oracle_ids.tolist(), 'initial_model_hash': digest(model.parameters()),
                'rng_seeds': {'init': c['seed'], 'loader': c['seed']+1, 'test': c['seed']+2,
                              'synthetic': c['seed']+3, 'noise': c['seed']+4, 'oracle': c['seed']+5},
                'train_num_workers': 0}
    save_json(marker, metadata)
    save_json(root / f'config_{method}.json', c)
    train_rows, refresh_log, oracle_log, audits = [], [], [], []
    old_v = old_p = active_p = None
    started = time.perf_counter()
    step = 0

    def sync():
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)

    def snapshot(label):
        save_torch(root / 'checkpoints' / f'{method}_{label}.pt', {
            'model': {k: v.detach().cpu().clone() for k, v in model._module.state_dict().items()},
            'P': active_p, 'step': step, 'accountant': accountant.state_dict(),
            'fingerprint': fingerprint(c)})

    snapshot('initial')
    try:
        for epoch in range(1, c['epochs']+1):
            iterator = iter(loader)
            for _ in range(len(loader)):
                # theta_step contains only previous privatized updates.
                # No next(iterator), no worker prefetch until P is frozen.
                if step % c['K'] == 0:
                    sync()
                    refresh_start = time.perf_counter()
                    state = model._module.state_dict()
                    v, halves = synthetic(state, c, dev, syn_rng)
                    candidate = make_p(v, c['lambda'])
                    if method == 'syn_diag':
                        active_p = candidate
                    # dp_sgd records candidate synthetic geometry but applies I.
                    for row in refresh_rows(v, halves, candidate, old_v, old_p, c):
                        refresh_log.append({'method': method, 'step': step, 'epoch': epoch, **row})
                    if c['oracle_enabled']:
                        # Only after P is frozen, access a fixed private subset.
                        samples = [data[int(i)] for i in oracle_ids]
                        private_samples = (torch.stack([s[0] for s in samples]), torch.tensor([s[1] for s in samples]))
                        private, _ = estimate(state, private_samples, c, dev)
                        used_p = candidate if method == 'syn_diag' else {n: torch.ones_like(x) for n, x in v.items()}
                        for row in oracle_rows(v, private, used_p, c):
                            oracle_log.append({'method': method, 'step': step, 'epoch': epoch, **row})
                    sync()
                    elapsed = time.perf_counter() - refresh_start
                    for row in refresh_log[-4:]:
                        row['refresh_seconds'] = elapsed
                    old_v, old_p = v, candidate
                    print(f'{method} refresh step={step}, {elapsed:.2f}s', flush=True)
                x, y, indices = next(iterator)
                model.train()
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x.to(dev)), y.to(dev), reduction='sum')
                loss.backward()
                if active_p is not None:
                    apply_p(model, active_p)
                row = before_clip(model, c, len(x))
                accumulator = DPGradientAccumulator()
                accumulator.accumulate(model, c['max_grad_norm'], len(x))
                noise_state_hash = digest([noise_rng.cpu] + ([noise_rng.cuda] if noise_rng.cuda is not None else []))
                with noise_rng.use():
                    accumulator.finalize(sigma, c['max_grad_norm'], store_summed_grad=True)
                row.update(after_noise(model, sigma, c, len(x)))
                row['diagnostic_snr'] = row['clipped_aggregate_norm'] / (row['expected_noise_norm'] + c['eps_num'])
                optimizer.step()
                accountant.step(noise_multiplier=sigma, sample_rate=q)
                step += 1
                row.update({'method': method, 'step': step, 'epoch': epoch, 'train_loss': float(loss.detach())/len(x),
                            'test_loss': None, 'test_accuracy': None})
                if step % c['eval_interval'] == 0 or step == total:
                    row.update(evaluate(model, test_loader, dev))
                sync()
                row['wall_seconds'] = time.perf_counter() - started
                train_rows.append(row)
                audits.append({'step': step, 'batch_indices': indices.tolist(), 'noise_rng_before': noise_state_hash})
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite model parameters')
        snapshot('final')
    finally:
        model.remove_hooks()
        for filename, rows in [('train_metrics', train_rows), ('refresh_metrics', refresh_log), ('oracle_metrics', oracle_log)]:
            write_csv(root / 'results' / f'{filename}_{method}.csv',
                      [dict(row, config_fingerprint=fingerprint(c)) for row in rows])
    metadata.update({'epsilon_spent': accountant.get_epsilon(c['delta']), 'completed_steps': step,
                     'final_model_hash': digest(model.parameters()), 'wall_seconds': time.perf_counter()-started})
    save_json(root / f'metadata_{method}.json', metadata)
    save_json(root / 'results' / f'pairing_{method}.json', audits)
    # Rebuild joint files from completed methods, so invocation order is irrelevant.
    for filename in ('train_metrics', 'refresh_metrics', 'oracle_metrics'):
        combined = []
        for m in ('dp_sgd', 'syn_diag'):
            path = root / 'results' / f'{filename}_{m}.csv'
            if path.exists():
                with path.open() as f:
                    combined.extend(csv.DictReader(f))
        write_csv(root / 'results' / f'{filename}.csv', combined)
    summary = {}
    for m in ('dp_sgd', 'syn_diag'):
        path = root / f'metadata_{m}.json'
        if path.exists():
            summary[m] = json.loads(path.read_text())
    save_json(root / 'results/summary.json', summary)
    print(f'{method}: {step} steps, epsilon={metadata["epsilon_spent"]:.5f}, final accuracy={train_rows[-1]["test_accuracy"]:.4f}', flush=True)
    return metadata


if __name__ == '__main__':
    train(*parse(method=True))
