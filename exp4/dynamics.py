"""Persistent Adam candidates, upstream clipping, and isolated diagnostics."""
import copy
import torch
import torch.nn.functional as F
from exp4.common import adam, digest
from exp3.preconditioners import apply
from dp_kfac.privacy import clip_and_noise_gradients


class AdamDirection:
    """Use actual torch Adam at lr=1 and zero scratch parameters to read u.

    Only parameters are zeroed: step, exp_avg and exp_avg_sq persist. With
    weight_decay=0, Adam's state is independent of its scratch parameters.
    """
    def __init__(self, params, c):
        self.params = [torch.nn.Parameter(torch.zeros_like(p)) for p in params]
        self.optimizer = adam(self.params, c, lr=1.)

    @torch.no_grad()
    def advance(self, gradients):
        for p, g in zip(self.params, gradients):
            p.zero_()
            p.grad = g.detach().clone()
        self.optimizer.step()
        return [-p.detach().clone() for p in self.params]

    def peek(self, gradients):
        # A disposable copy preserves all historical noise, but never commits
        # the diagnostic step to the official shadow state.
        branch = copy.deepcopy(self)
        return branch.advance(gradients)


@torch.no_grad()
def aggregate(model, active, rng, sigma, c, b):
    """Destructively consume grad_sample using the unchanged upstream API.

    Replay only the noise stream to audit the exact standard-normal draws.
    Upstream draws flat tensors in parameter order; do not change that layout.
    """
    apply(model, active)
    params = list(model.parameters())
    mean = [p.grad_sample.mean(0).clone() for p in params]
    replay = copy.deepcopy(rng)
    with replay.use():
        z = [torch.randn_like(p.grad_sample.contiguous().view(b, -1).sum(0)) for p in params]
    noise_hash = digest(z)
    before = rng.audit()
    with rng.use():
        clip_and_noise_gradients(model, sigma, c['max_grad_norm'], b, store_summed_grad=True)
    if replay.audit() != rng.audit():
        raise AssertionError('Upstream Gaussian draw layout changed')
    return dict(mean=mean, clean=[p.summed_grad.detach().clone() for p in params],
                noisy=[p.grad.detach().clone() for p in params], noise_hash=noise_hash,
                noise_rng_before=before, noise_rng_after=rng.audit())


def flat(values):
    return torch.cat([v.detach().reshape(-1) for v in values]).double()


def cosine(a, b):
    den = a.norm() * b.norm()
    # Undefined zero-vector cosine is missing, never fabricated as alignment.
    return float((a @ b / den).clamp(-1, 1)) if den > 0 else None


def difference(a, b):
    return None if a is None or b is None else a-b


def metrics(ua, udp, usyn, udp0, usyn0, dp, syn):
    a, d, s, d0, s0 = map(flat, (ua, udp, usyn, udp0, usyn0))
    row = {}
    for suffix, dv, sv in [('noisy', d, s), ('current_noise_off', d0, s0)]:
        row[f'cos_dp_adam_{suffix}'] = cosine(dv, a)
        row[f'cos_syn_adam_{suffix}'] = cosine(sv, a)
        row[f'cos_syn_dp_{suffix}'] = cosine(sv, dv)
        row[f'alignment_advantage_{suffix}'] = difference(cosine(sv, a), cosine(dv, a))
    row.update(norm_ratio_dp_adam=float(d.norm()/a.norm()) if a.norm() > 0 else None,
               norm_ratio_syn_adam=float(s.norm()/a.norm()) if a.norm() > 0 else None,
               noise_degradation_dp=difference(cosine(d0, a), cosine(d, a)),
               noise_degradation_syn=difference(cosine(s0, a), cosine(s, a)),
               clip_cosine_dp=cosine(flat(dp['mean']), flat(dp['clean'])),
               clip_cosine_syn=cosine(flat(syn['mean']), flat(syn['clean'])),
               adam_direction_norm=float(a.norm()), dp_direction_norm=float(d.norm()),
               syn_direction_norm=float(s.norm()))
    return row


@torch.no_grad()
def counterfactual(model, x, y, directions, c):
    # Copy the plain module; no hooks, parameter edits, or RNG consumption in
    # the reference. This model has no stochastic layers or mutable buffers.
    probe = copy.deepcopy(model._module)
    probe.eval()
    original = {n: p.detach().clone() for n, p in model._module.named_parameters()}
    base = float(F.cross_entropy(probe(x), y))
    result = {}
    for name, direction in directions.items():
        lr = c['syn_lr'] if name == 'syn' else c['adam_lr']
        for (n, p), u in zip(probe.named_parameters(), direction):
            p.copy_(original[n] - lr*u)
        loss = float(F.cross_entropy(probe(x), y))
        distance = float(flat([p-original[n] for n, p in probe.named_parameters()]).norm())
        # Positive means lower loss per unit actual parameter movement.
        result[f'loss_progress_{name}'] = (base-loss)/(distance+c['eps_num'])
        result[f'candidate_loss_{name}'] = loss
    return result
