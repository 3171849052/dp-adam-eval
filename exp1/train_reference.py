"""One fixed DP trajectory using DP-KFC's actual Adam/grad-sample/RDP path."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from common import (parse, run_root, fingerprint, provenance, save_json,
                    save_torch, loaders, device, set_seed, SimpleCNN)
from dp_kfac.privacy import DPGradientAccumulator


def train(c):
    root = run_root(c)
    manifest_path = root / 'checkpoints/manifest.json'
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text())
        if manifest['fingerprint'] != fingerprint(c) or manifest['upstream'] != provenance():
            raise ValueError('Existing trajectory config/source differs; use a fresh output directory')
        if all((root / 'checkpoints' / x['file']).exists() for x in manifest['checkpoints']):
            print(f'All 11 checkpoints exist; skipping training: {manifest_path}', flush=True)
            return
        raise RuntimeError('Incomplete saved trajectory; use a fresh output directory')
    set_seed(c['seed'])
    loader, _, _ = loaders(c)
    if c['train_subset'] is not None:
        if c['train_subset'] > len(loader.dataset):
            raise ValueError('train_subset exceeds dataset')
        loader = DataLoader(Subset(loader.dataset, range(c['train_subset'])),
                            batch_size=c['batch_size'], shuffle=True, drop_last=True,
                            num_workers=c['num_workers'])
    steps = len(loader) * c['epochs']
    if steps < 10:
        raise ValueError('Need at least 10 optimizer steps for 11 distinct checkpoints')
    sample_rate = c['batch_size'] / len(loader.dataset)
    sigma = get_noise_multiplier(target_epsilon=c['epsilon'], target_delta=c['delta'],
                                 sample_rate=sample_rate, steps=steps, accountant='rdp')
    model = GradSampleModule(SimpleCNN().to(device(c)), batch_first=True, loss_reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'])
    accountant = RDPAccountant()
    accumulator = DPGradientAccumulator()
    marks = {round(steps * percent / 100): percent for percent in range(0, 101, 10)}
    checkpoints = []

    def snapshot(step):
        percent = marks[step]
        filename = f'checkpoint_{percent:03d}.pt'
        epsilon = accountant.get_epsilon(c['delta']) if step else 0.0
        save_torch(root / 'checkpoints' / filename, {
            'model': {k: v.detach().cpu() for k, v in model._module.state_dict().items()},
            'step': step, 'percent': percent, 'epsilon_spent': epsilon,
            'fingerprint': fingerprint(c)})
        checkpoints.append({'file': filename, 'step': step, 'percent': percent, 'epsilon_spent': epsilon})
        print(f'checkpoint {percent:3d}% step={step}/{steps} epsilon={epsilon:.5f}', flush=True)

    snapshot(0)
    step = 0
    for _ in range(c['epochs']):
        model.train()
        for x, y in loader:
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x.to(device(c))), y.to(device(c)), reduction='sum').backward()
            accumulator.accumulate(model, c['max_grad_norm'], len(x))
            accumulator.finalize(sigma, c['max_grad_norm'])
            optimizer.step()
            accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
            step += 1
            if step in marks:
                snapshot(step)
    model.remove_hooks()
    save_json(manifest_path, {'config': c, 'fingerprint': fingerprint(c), 'upstream': provenance(),
                             'torch': str(torch.__version__), 'noise_multiplier': sigma,
                             'sample_rate': sample_rate, 'total_steps': steps,
                             'accountant': 'rdp', 'sampling': 'shuffle, drop_last=True (upstream)',
                             'checkpoints': checkpoints})


if __name__ == '__main__':
    train(parse(__doc__))
