"""Two reference trajectories with persistent Adam/DP-Adam/SynDiag shadows."""
import argparse
import copy
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from exp4.common import (ROOT, TRAJECTORIES, read_config, fingerprint, provenance,
                         save_json, write_csv, datasets, set_seed, SimpleCNN, RNGStream, digest, adam)
from exp4.dynamics import AdamDirection, aggregate, metrics, flat, counterfactual
from exp3.preconditioners import synthetic_samples, refresh
from exp3.train_exp3 import Indexed, evaluate


def train(c, seed, trajectory, output, data_override=None, diagnostics=True, event_hook=None):
    if trajectory not in TRAJECTORIES or seed not in c['seeds']:
        raise ValueError('Unknown trajectory/seed')
    root = Path(output).resolve() / f'seed{seed}' / trajectory
    root.mkdir(parents=True, exist_ok=False)
    save_json(root/'config.json', c)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if c['device'] == 'auto' else torch.device(c['device'])
    torch.set_num_threads(c['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    source, test = datasets(c) if data_override is None else data_override
    n = c['train_subset'] or len(source)
    if not c['batch_size'] <= n <= len(source):
        raise ValueError('Invalid training subset')
    if c['test_subset'] is not None:
        if not 1 <= c['test_subset'] <= len(test):
            raise ValueError('Invalid test subset')
        test = Subset(test, range(c['test_subset']))
    loader_rng = torch.Generator().manual_seed(seed+1)
    loader = DataLoader(Indexed(Subset(source, range(n))), batch_size=c['batch_size'],
                        shuffle=True, drop_last=True, num_workers=0, generator=loader_rng)
    test_loader = DataLoader(test, batch_size=c['batch_size'], shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(seed+2))
    total = len(loader)*c['epochs']
    q = c['batch_size']/n
    sigma = get_noise_multiplier(target_epsilon=c['epsilon'], target_delta=c['delta'],
                                 sample_rate=q, steps=total, accountant='rdp')
    set_seed(seed)
    model = GradSampleModule(SimpleCNN().to(dev), loss_reduction='sum')
    optimizer = adam(model.parameters(), c)
    shadow_a = AdamDirection(model.parameters(), c) if diagnostics else None
    shadow_d = AdamDirection(model.parameters(), c) if diagnostics else None
    accountant = RDPAccountant()
    syn_rng, noise_rng = RNGStream(seed+3, dev), RNGStream(seed+4, dev)
    meta = dict(seed=seed, trajectory=trajectory, config_fingerprint=fingerprint(c),
                provenance=provenance(), device=str(dev), torch=str(torch.__version__),
                noise_multiplier=sigma, sample_rate=q, total_steps=total,
                initial_model_hash=digest(model.parameters()), diagnostics_enabled=diagnostics,
                parameter_order=list(dict(model._module.named_parameters())),
                rng_seeds=dict(init=seed, loader=seed+1, test=seed+2, synthetic=seed+3, noise=seed+4),
                sampling='shuffle/drop_last; Exp1 RDP calibration convention, not Poisson',
                clipping='upstream min(1,C/(norm+1e-6)); global per-sample norm',
                current_noise_off='DP-Adam retains historical noisy m/v; only this step has z=0',
                diagnostic_scope='private mechanism diagnostics, not DP releases', complete=False)
    save_json(root/'metadata.json', meta)
    rows, evals, audits, synthetic = [], [], [], []
    active = None
    step = 0
    start_time = time.perf_counter()
    def event(kind):
        if event_hook is not None:
            event_hook(kind, step, active)
    try:
        for _ in range(c['epochs']):
            iterator = iter(loader)  # num_workers=0: no private sample prefetch
            for _ in range(len(loader)):
                is_refresh = step % c['K'] == 0
                if diagnostics and is_refresh:
                    event('refresh_start')
                    samples, audit = synthetic_samples(c, dev, syn_rng)
                    active = refresh(model._module.state_dict(), samples, c, dev, 'syn_diag')
                    synthetic.append(dict(step=step, preconditioner_hash=digest(active[1].values()), **audit))
                    del samples
                    event('refresh_end')
                event('before_batch')
                x, y, indices = next(iterator)
                x, y = x.to(dev), y.to(dev)
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y, reduction='sum')
                loss.backward()
                raw = [p.grad.detach().clone()/len(x) for p in model.parameters()]
                samples = [p.grad_sample.detach().clone() for p in model.parameters()] if diagnostics else None
                syn_noise_rng = copy.deepcopy(noise_rng)
                dp = aggregate(model, None, noise_rng, sigma, c, len(x))
                directions = None
                if diagnostics:
                    # The noise-off DP candidate MUST precede the noisy commit.
                    ud0 = shadow_d.peek(dp['clean'])
                    ua = shadow_a.advance(raw)
                    ud = shadow_d.advance(dp['noisy'])
                    for p, gs in zip(model.parameters(), samples):
                        p.grad_sample = gs
                    syn = aggregate(model, active, syn_noise_rng, sigma, c, len(x))
                    if syn['noise_hash'] != dp['noise_hash']:
                        raise AssertionError('DP-Adam and SynDiag did not share standard Gaussian z')
                    row = metrics(ua, ud, syn['noisy'], ud0, syn['clean'], dp, syn)
                    directions = dict(adam=ua, dp=ud, syn=syn['noisy'])
                    diagnostic_step = step == 0 or (step+1) % c['diagnostic_interval'] == 0 or step+1 == total
                    row.update({f'{prefix}_{name}': None for prefix in ('loss_progress', 'candidate_loss') for name in directions})
                    if diagnostic_step:
                        row.update(counterfactual(model, x, y, directions, c))
                    row.update(seed=seed, trajectory=trajectory, step=step+1,
                               syn_refresh=is_refresh, batch_hash=digest([indices]), noise_hash=dp['noise_hash'])
                reference_gradients = raw if trajectory == 'adam' else dp['noisy']
                previous = [p.detach().clone() for p in model.parameters()]
                for p, g in zip(model.parameters(), reference_gradients):
                    p.grad = g
                optimizer.step()
                step += 1
                if trajectory == 'dp_adam':
                    accountant.step(noise_multiplier=sigma, sample_rate=q)
                if diagnostics:
                    row['reference_update_norm'] = float(flat([p-old for p, old in zip(model.parameters(), previous)]).norm())
                    rows.append(row)
                result = dict(trajectory=trajectory, seed=seed, step=step,
                              train_loss=float(loss.detach())/len(x), test_loss=None, test_accuracy=None)
                if step % c['eval_interval'] == 0 or step == total:
                    result.update(evaluate(model, test_loader, dev))
                    print(f'seed={seed} {trajectory} step={step}/{total} loss={result["train_loss"]:.4f} accuracy={result["test_accuracy"]:.4f}', flush=True)
                evals.append(result)
                audits.append(dict(step=step, batch_hash=digest([indices]), noise_hash=dp['noise_hash'],
                                   syn_noise_hash=syn['noise_hash'] if diagnostics else None,
                                   noise_rng_before=dp['noise_rng_before'], noise_rng_after=noise_rng.audit(),
                                   loader_rng_hash=digest([loader_rng.get_state()]),
                                   reference_rng_hash=digest([torch.get_rng_state()] + ([torch.cuda.get_rng_state(dev)] if dev.type == 'cuda' else [])),
                                   model_hash=digest(model.parameters())))
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite reference parameters')
        meta.update(complete=True, completed_steps=step, wall_seconds=time.perf_counter()-start_time,
                    final_model_hash=digest(model.parameters()), final_test_loss=evals[-1]['test_loss'],
                    final_test_accuracy=evals[-1]['test_accuracy'],
                    epsilon_spent=accountant.get_epsilon(c['delta']) if trajectory == 'dp_adam' else None)
        torch.save(dict(model=model._module.state_dict(), optimizer=optimizer.state_dict(),
                        shadow_adam=shadow_a.optimizer.state_dict() if diagnostics else None,
                        shadow_dp_adam=shadow_d.optimizer.state_dict() if diagnostics else None), root/'final_state.pt')
        save_json(root/'metadata.json', meta)
    finally:
        model.remove_hooks()
        write_csv(root/'shadow_metrics.csv', rows, fields=None if rows else ['seed', 'trajectory', 'step'])
        write_csv(root/'eval_metrics.csv', evals)
        save_json(root/'pairing.json', dict(private=audits, synthetic=synthetic))
    return meta


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/full.json'))
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--trajectory', choices=[*TRAJECTORIES, 'all'], default='all')
    a = p.parse_args()
    c = read_config(a.config)
    for seed in c['seeds'] if a.seed is None else [a.seed]:
        for trajectory in TRAJECTORIES if a.trajectory == 'all' else [a.trajectory]:
            train(c, seed, trajectory, a.output)


if __name__ == '__main__':
    main()
