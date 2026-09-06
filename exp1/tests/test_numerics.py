"""Small exact numerical checks; loops below are independent test oracles only."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SimpleCNN, LAYERS, read_config, set_seed, generate_pink_noise
from fisher_utils import collect, gram, empirical_summary, empirical_inner
from kfac_utils import diagonal
from metrics import alignment, kfac_alignment, diagonal_metrics
from dp_kfac.recorder import KFACRecorder
from dp_kfac.covariance import compute_covariances


class Numerics(unittest.TestCase):
    def setUp(self):
        self.c = read_config(Path(__file__).resolve().parents[1] / 'configs/smoke.json')
        set_seed(42)

    def test_four_layers_against_independent_backward_and_repo(self):
        model = SimpleCNN()
        state = copy.deepcopy(model.state_dict())
        x = generate_pink_noise(3, (1, 28, 28), torch.device('cpu'))
        y = torch.tensor([0, 1, 2])
        reference = {name: [] for name in LAYERS}
        for i in range(3):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[i:i+1]), y[i:i+1]).backward()
            for name in LAYERS:
                layer = getattr(model, name)
                reference[name].append(torch.cat([layer.weight.grad.flatten(1), layer.bias.grad[:, None]], 1).flatten())
        self.c['m_full'] = 3
        self.c['analysis_batch_size'] = 2  # Includes a final partial microbatch.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            stats, paths = collect(state, (x, y), self.c, torch.device('cpu'), temp)
            recorder = KFACRecorder(model)
            recorder.enable()
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y, reduction='sum').backward()
            raw = compute_covariances(model, recorder.activations, recorder.backprops, eps=0)
            # Upstream mean-loss at nominal B, independent of microbatch size.
            repo = compute_covariances(model, recorder.activations,
                                       {k: v/self.c['batch_size'] for k, v in recorder.backprops.items()})
            recorder.remove()
            for name in LAYERS:
                g = torch.stack(reference[name]).double()
                np.testing.assert_allclose(np.load(paths[name]), g.numpy(), rtol=1e-4, atol=2e-7)
                torch.testing.assert_close(stats[name]['v_direct'], g.square().mean(0), rtol=1e-4, atol=1e-9)
                for version, oracle in [('raw', raw), ('repo', repo)]:
                    for factor, values in [('A', oracle.A), ('G', oracle.G)]:
                        torch.testing.assert_close(stats[name]['factors'][version][factor], values[name].double(), rtol=2e-5, atol=1e-7)
                self.assertEqual(stats[name]['v_kfac_raw'].shape, stats[name]['v_direct'].shape)
                np.testing.assert_allclose(gram(g.numpy(), chunk=57).numpy(), (g@g.T/3).numpy(), atol=1e-10)
                summary = empirical_summary(paths[name], 257)
                # Result artifacts must support safe torch.load(weights_only=True).
                summary_path = Path(temp) / 'summary.pt'
                torch.save(summary, summary_path)
                loaded = torch.load(summary_path, weights_only=True)
                self.assertIs(type(loaded['rank_tolerance']), float)
                self.assertAlmostEqual(summary['trace'], float(g.square().sum()/3), places=5)
                self.assertAlmostEqual(empirical_inner(paths[name], paths[name], 257), summary['norm_sq'], places=8)
            # Changing diagnostic microbatch size must preserve raw AND repo factors.
            self.c['analysis_batch_size'] = 3
            other, _ = collect(state, (x, y), self.c, torch.device('cpu'))
            for name in LAYERS:
                torch.testing.assert_close(stats[name]['v_direct'], other[name]['v_direct'], rtol=1e-4, atol=1e-9)

    def test_small_dense_fisher_identities(self):
        x, y = torch.randn(4, 7, dtype=torch.double), torch.randn(5, 7, dtype=torch.double)
        fx, fy = x.T@x/4, y.T@y/5
        cross = gram(x.numpy(), y.numpy(), chunk=3).square().sum()
        torch.testing.assert_close(cross, (fx*fy).sum())
        result = alignment(float(cross), float(fy.square().sum()), float(fx.square().sum()), float(fy.trace()), float(fx.trace()))
        self.assertAlmostEqual(result['shape_error'], float((result['alpha_star']*fy-fx).norm()/fx.norm()))
        self.assertAlmostEqual(result['relative_frobenius_error'], float((fy-fx).norm()/fx.norm()))
        a, g = torch.randn(3, 3, dtype=torch.double), torch.randn(2, 2, dtype=torch.double)
        factors = {'A': a@a.T, 'G': g@g.T}
        dense = torch.kron(factors['G'], factors['A'])
        torch.testing.assert_close(diagonal(factors), dense.diag())
        oracle = {'A': torch.eye(3, dtype=torch.double), 'G': torch.eye(2, dtype=torch.double)}
        result = kfac_alignment(factors, oracle)
        self.assertAlmostEqual(result['cosine'], float(dense.trace()/(dense.norm()*np.sqrt(6))))
        v = torch.logspace(-4, 2, 100, dtype=torch.double)
        d = diagonal_metrics(v, v, 1e-12)
        self.assertLess(d['A_residual'], 1e-7)
        self.assertAlmostEqual(d['log_pearson'], 1)
        self.assertAlmostEqual(d['log_spearman'], 1)
        self.assertAlmostEqual(d['log_ratio_spread'], 0)
        self.assertIsNone(diagonal_metrics(torch.zeros(3), torch.zeros(3), 1e-12)['R'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_diagnostic_precision(self):
        state = SimpleCNN().state_dict()
        x = generate_pink_noise(3, (1, 28, 28), torch.device('cpu'))
        samples = x, torch.tensor([0, 1, 2])
        cpu, _ = collect(state, samples, self.c, torch.device('cpu'))
        gpu, _ = collect(state, samples, self.c, torch.device('cuda'))
        for name in LAYERS:
            torch.testing.assert_close(cpu[name]['v_direct'], gpu[name]['v_direct'], rtol=1e-4, atol=1e-8)
            for factor in ('A', 'G'):
                torch.testing.assert_close(cpu[name]['factors']['raw'][factor],
                                           gpu[name]['factors']['raw'][factor], rtol=1e-4, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
