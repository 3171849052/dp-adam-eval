"""Block-fixed diagonals in Exp1's output-unit, weight-then-bias order."""
import torch
import torch.nn.functional as F
from opacus import GradSampleModule
from common import LAYERS, SimpleCNN, layer_gradient, generate_pink_noise


def estimate(state, samples, c, dev, split=None):
    # Model construction is RNG-isolated even when this function is used alone.
    devices = [dev.index or 0] if dev.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        model = GradSampleModule(SimpleCNN().to(dev), loss_reduction='sum')
    model._module.load_state_dict(state)
    model.train()
    sums, halves = {}, [{}, {}]
    x, y = samples
    try:
        for start in range(0, len(x), c['analysis_batch_size']):
            stop = min(start + c['analysis_batch_size'], len(x))
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[start:stop].to(dev)), y[start:stop].to(dev), reduction='sum').backward()
            for name in LAYERS:
                g2 = layer_gradient(getattr(model._module, name)).detach().double().square()
                value = g2.sum(0).cpu()
                sums[name] = sums.get(name, torch.zeros_like(value)) + value
                if split is not None:
                    for j in range(2):
                        part = g2[(split[start:stop] == j).to(dev)].sum(0).cpu()
                        halves[j][name] = halves[j].get(name, torch.zeros_like(part)) + part
        v = {n: s / len(x) for n, s in sums.items()}
        if split is not None:
            halves = [{n: s / int((split == j).sum()) for n, s in h.items()} for j, h in enumerate(halves)]
        return v, halves
    finally:
        model.remove_hooks()


def synthetic(state, c, dev, rng):
    with rng.use():
        batches = []
        for start in range(0, c['M_syn'], c['batch_size']):
            b = min(c['batch_size'], c['M_syn'] - start)
            batches.append((generate_pink_noise(b, (1, 28, 28), dev), torch.randint(0, 10, (b,), device=dev)))
        samples = tuple(torch.cat([b[j] for b in batches]) for j in (0, 1))
        split = torch.ones(c['M_syn'], dtype=torch.long)
        split[torch.randperm(c['M_syn'])[:c['M_syn']//2]] = 0
        return estimate(state, samples, c, dev, split)


def make_p(v, damping):
    return {name: 1 / (value.sqrt() + damping) for name, value in v.items()}


def parameter_p(model, p):
    result = {}
    for name in LAYERS:
        layer = getattr(model._module, name)
        block = p[name].to(device=layer.weight.device, dtype=layer.weight.dtype)
        if block.numel() != layer.weight.numel() + layer.bias.numel():
            raise ValueError(f'P dimension mismatch: {name}')
        block = block.reshape(layer.weight.shape[0], -1)
        result[f'{name}.weight'] = block[:, :-1].reshape_as(layer.weight)
        result[f'{name}.bias'] = block[:, -1].reshape_as(layer.bias)
    if list(result) != list(dict(model._module.named_parameters())):
        raise ValueError('Unexpected parameter order')
    return result


def apply_p(model, p):
    for name, scale in parameter_p(model, p).items():
        param = dict(model._module.named_parameters())[name]
        if not isinstance(param.grad_sample, torch.Tensor) or param.grad_sample.shape[1:] != scale.shape:
            raise ValueError(f'Unexpected grad_sample shape: {name}')
        param.grad_sample.mul_(scale.unsqueeze(0))
