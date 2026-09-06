"""Unclipped per-sample Fisher diagnostics; disk-backed, parameter-chunked Gram."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from opacus import GradSampleModule
from common import LAYERS, SimpleCNN
from dp_kfac.recorder import KFACRecorder
from kfac_utils import FactorAccumulator, diagonal


def layer_gradient(layer):
    w = layer.weight.grad_sample
    b = layer.bias.grad_sample
    if isinstance(w, list) or isinstance(b, list):
        raise RuntimeError('Unexpected accumulated grad_sample')
    return torch.cat((w.flatten(2), b.unsqueeze(-1)), dim=-1).flatten(1)


def collect(state, samples, c, dev, full_dir=None):
    # TF32 rounding can change ReLU/MaxPool branches and distort small Fisher
    # coordinates. Diagnostics use FP32 kernels; training keeps repo defaults.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = GradSampleModule(SimpleCNN().to(dev), batch_first=True, loss_reduction='sum')
    model._module.load_state_dict(state)
    # Opacus hooks collect only in training mode; SimpleCNN has no stochastic
    # or running-stat layers, so train/eval give identical outputs here.
    model.train()
    recorder = KFACRecorder(model)
    recorder.enable()
    factors = FactorAccumulator(c['batch_size'])
    sums, maps, paths = {}, {}, {}
    x, y = samples
    if full_dir is not None:
        Path(full_dir).mkdir(parents=True, exist_ok=True)
        for name in LAYERS:
            layer = getattr(model._module, name)
            d = layer.weight.numel() + layer.bias.numel()
            path = Path(full_dir) / f'{name}.npy'
            maps[name] = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(c['m_full'], d))
            paths[name] = path
    try:
        for start in range(0, len(x), c['analysis_batch_size']):
            stop = min(start + c['analysis_batch_size'], len(x))
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[start:stop].to(dev)), y[start:stop].to(dev), reduction='sum').backward()
            factors.add(model, recorder, stop - start)
            for name in LAYERS:
                grad = layer_gradient(getattr(model._module, name)).detach()
                value = grad.double().square().sum(0).cpu()
                sums[name] = sums.get(name, torch.zeros_like(value)) + value
                end = min(stop, c['m_full'])
                if name in maps and start < end:
                    maps[name][start:end] = grad[:end-start].cpu().numpy()
            recorder.clear()
        fac = factors.finish()
        stats = {name: {'v_direct': sums[name] / len(x), 'factors': fac[name],
                        'v_kfac_raw': diagonal(fac[name]['raw']),
                        'v_kfac_repo': diagonal(fac[name]['repo'])} for name in LAYERS}
        return stats, paths
    finally:
        for mmap in maps.values():
            mmap.flush()
            mmap._mmap.close()
        recorder.remove()
        model.remove_hooks()


def gram(x, y=None, chunk=4096):
    """Gx Gy^T / sqrt(Mx My); at most two M x chunk float64 copies.

    Inputs can be mmap arrays. Only the small Mx x My Gram is in memory.
    """
    y = x if y is None else y
    if x.shape[1] != y.shape[1]:
        raise ValueError('Gradient dimensions differ')
    result = torch.zeros((len(x), len(y)), dtype=torch.float64)
    for start in range(0, x.shape[1], chunk):
        a = torch.from_numpy(np.array(x[:, start:start+chunk], dtype=np.float64, copy=True))
        b = torch.from_numpy(np.array(y[:, start:start+chunk], dtype=np.float64, copy=True))
        result.addmm_(a, b.T)
    return result / np.sqrt(len(x) * len(y))


def empirical_summary(path, chunk):
    x = np.load(path, mmap_mode='r')
    try:
        k = gram(x, chunk=chunk)
        eig = torch.linalg.eigvalsh(k).clamp_min(0).flip(0)
        return {'spectrum': eig, 'norm_sq': float(k.square().sum()), 'trace': float(k.trace()),
                'rank_tolerance': float(float(eig[0]) * max(x.shape) * np.finfo(np.float64).eps)}
    finally:
        x._mmap.close()


def empirical_inner(x_path, y_path, chunk):
    x, y = np.load(x_path, mmap_mode='r'), np.load(y_path, mmap_mode='r')
    try:
        return float(gram(x, y, chunk).square().sum())
    finally:
        x._mmap.close()
        y._mmap.close()
