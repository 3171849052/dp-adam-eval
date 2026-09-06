"""DP-KFC factors, retaining upstream convolution spatial averaging."""
import torch
from common import LAYERS
from dp_kfac.covariance import compute_covariances


class FactorAccumulator:
    def __init__(self, reference_batch_size):
        self.batch_size = reference_batch_size
        self.count = 0
        self.sums = {}

    def add(self, model, recorder, count):
        # Recorder sees SUM-loss backprops: single-example deltas, independent
        # of the diagnostic microbatch size. Reuse exact upstream covariances.
        cov = compute_covariances(model, recorder.activations, recorder.backprops, eps=0.0)
        for name in LAYERS:
            pair = (cov.A[name].detach().cpu().double(), cov.G[name].detach().cpu().double())
            if name not in self.sums:
                self.sums[name] = [torch.zeros_like(x) for x in pair]
            for dest, value in zip(self.sums[name], pair):
                dest.add_(value, alpha=count)
        self.count += count

    def finish(self):
        result = {}
        for name, (a, g) in self.sums.items():
            a, g = a / self.count, g / self.count
            # Same result as averaging upstream mean-loss batches of size B.
            # eps=1e-5 is upstream covariance damping, distinct from the
            # extra 1e-3 used ONLY during inverse-square-root preconditioning.
            repo_a = a + 1e-5 * torch.eye(len(a), dtype=a.dtype)
            repo_g = g / self.batch_size**2 + 1e-5 * torch.eye(len(g), dtype=g.dtype)
            result[name] = {'raw': {'A': a, 'G': g}, 'repo': {'A': repo_a, 'G': repo_g}}
        return result


def diagonal(factors):
    # Parameter order: [weight row, bias] for each output channel/unit.
    # This is a permutation of column-vectorized A kron G (i.e. G kron A).
    return (factors['G'].diag()[:, None] * factors['A'].diag()[None, :]).flatten()


def spectrum(factors):
    a = torch.linalg.eigvalsh(factors['A']).clamp_min(0)
    g = torch.linalg.eigvalsh(factors['G']).clamp_min(0)
    return (a[:, None] * g[None, :]).flatten().sort(descending=True).values
