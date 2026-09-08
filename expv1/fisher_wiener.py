"""Data-free covariance construction and beta=1 post-noise linear filters."""
import torch
import torch.nn.functional as F
from expv1.common import SimpleCNN, LAYERS, generate_pink_noise, digest
from dp_kfac.recorder import KFACRecorder
from dp_kfac.covariance import compute_covariances, accumulate_covariances


def synthetic_samples(c, dev, rng):
    before = rng.audit()
    with rng.use():
        batches = [(generate_pink_noise(c['batch_size'], (1, 28, 28), dev),
                    torch.randint(0, 10, (c['batch_size'],), device=dev))
                   for _ in range(c['M_syn'] // c['batch_size'])]
    x, y = (torch.cat([b[j] for b in batches]) for j in (0, 1))
    return (x, y), dict(rng_before=before, rng_after=rng.audit(),
        samples_hash=digest([x]), labels_hash=digest([y]), count=len(y))


def build_covariances(model_state, samples, c, dev):
    x, y = samples
    if len(x) != c['M_syn'] or len(x) % c['batch_size']:
        raise ValueError('Expected equal whole synthetic batches')
    if not c['smoke'] and (len(x), c['batch_size']) != (2560, 256):
        raise ValueError('Formal refresh requires 10 x 256')
    # Separate model; creation cannot advance any training random stream.
    with torch.random.fork_rng(devices=[dev.index or 0] if dev.type == 'cuda' else []):
        model = SimpleCNN().to(dev)
        model.load_state_dict(model_state)
        recorder = KFACRecorder(model)
        recorder.enable()
        factors = []
        try:
            for start in range(0, len(x), c['batch_size']):
                model.zero_grad(set_to_none=True)
                F.cross_entropy(model(x[start:start+c['batch_size']]), y[start:start+c['batch_size']]).backward()
                factors.append(compute_covariances(model, recorder.activations, recorder.backprops))
                recorder.clear()
            return accumulate_covariances(factors)
        finally:
            recorder.remove()


@torch.no_grad()
def build_wiener_state(covariances, sigma, max_grad_norm, batch_size):
    r = (sigma * max_grad_norm / batch_size)**2
    result = {}
    for name, A in covariances.A.items():
        A, G = A.float(), covariances.G[name].float()
        la, qa = torch.linalg.eigh(A)
        lg, qg = torch.linalg.eigh(G)
        la, lg = la.clamp_min(0), lg.clamp_min(0)
        lf = lg[:, None] * la[None, :]
        trace_a, trace_g = A.trace(), G.trace()
        mean = trace_a * trace_g / lf.numel()
        result[name] = dict(Q_A=qa, lambda_A=la, Q_G=qg, lambda_G=lg,
            H=torch.where(lf+r > 0, lf/(lf+r), torch.zeros_like(lf)),
            scalar_h=mean/(mean+r), trace_A=trace_a, trace_G=trace_g,
            trace_F=trace_a*trace_g)
    return result


def pack_layer_gradient(layer, source='grad'):
    weight = getattr(layer.weight, source).detach().reshape(layer.weight.shape[0], -1)
    return torch.cat([weight, getattr(layer.bias, source).detach()[:, None]], 1) if layer.bias is not None else weight.clone()


@torch.no_grad()
def unpack_layer_gradient(layer, matrix, target='grad'):
    width = layer.weight[0].numel()
    setattr(layer.weight, target, matrix[:, :width].reshape_as(layer.weight).clone())
    if layer.bias is not None:
        setattr(layer.bias, target, matrix[:, width].clone())


def apply_fisher_matrix(matrix, layer_state):
    qa, qg, h = (layer_state[k] for k in ('Q_A', 'Q_G', 'H'))
    return qg @ (h * (qg.T @ matrix @ qa)) @ qa.T


def apply_scalar_matrix(matrix, layer_state):
    return layer_state['scalar_h'] * matrix


def apply_filter_to_copy(matrix, layer_state, method):
    if method == 'dp_sgd':
        return matrix.clone()
    if method == 'dp_scalar_wiener':
        return apply_scalar_matrix(matrix, layer_state)
    if method == 'dp_fisher_wiener':
        return apply_fisher_matrix(matrix, layer_state)
    raise ValueError(method)


@torch.no_grad()
def _apply(model, state, method):
    base = getattr(model, '_module', model)
    for name in LAYERS:
        layer = getattr(base, name)
        unpack_layer_gradient(layer, apply_filter_to_copy(pack_layer_gradient(layer), state[name], method))


def apply_fisher_wiener(model, state):
    _apply(model, state, 'dp_fisher_wiener')


def apply_scalar_wiener(model, state):
    _apply(model, state, 'dp_scalar_wiener')


def state_bytes(state):
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(state_bytes(v) for v in state.values())
    return 0
