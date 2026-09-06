"""Numerical and causal checks using the actual upstream privacy mechanism."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from opacus import GradSampleModule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import (SimpleCNN, LAYERS, RNGStream, layer_gradient, read_config, digest)
from synthetic_preconditioner import apply_p, parameter_p, make_p, synthetic, estimate
from diagnostics import refresh_rows, oracle_rows
from dp_kfac.privacy import DPGradientAccumulator
from metrics import before_clip
import pandas as pd
from summarize_exp2b import validate_contributions
import train_exp2b as training


class Exp2Tests(unittest.TestCase):
    def setUp(self):
        self.c = read_config(ROOT / 'smoke_configs/lambda1e4_K2.json')
        torch.manual_seed(123)

    def toy_metrics(self, gradients):
        """Eight explicit coordinates: weight/bias for each of four layers; no RNG."""
        g = torch.tensor(gradients, dtype=torch.float64)
        model = torch.nn.Module()
        model._module = torch.nn.Module()
        for i, name in enumerate(LAYERS):
            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float64))
            layer.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
            layer.weight.grad_sample = g[:, 2*i].reshape(-1, 1, 1).clone()
            layer.bias.grad_sample = g[:, 2*i+1].reshape(-1, 1).clone()
            model._module.add_module(name, layer)
        return before_clip(model, self.c, len(g))

    def test_contribution_sum_with_zero_and_tiny_gradients(self):
        for gradients in ([[0]*8], [[1e-8]+[0]*7],
                          [[0]*8, [2]+[0]*7], [[2]+[0]*7]):
            with self.subTest(gradients=gradients):
                row = self.toy_metrics(gradients)
                sq = torch.tensor(gradients, dtype=torch.float64).square().sum(1)
                expected = float((sq / sq.clamp_min(self.c['eps_num'])).mean())
                self.assertAlmostEqual(sum(row[f'contrib_{n}'] for n in LAYERS), expected, places=7)
                validate_contributions(pd.DataFrame([dict(row, step=1)]))

    def test_pure_scalar_clipping_preserves_shape(self):
        row = self.toy_metrics([[3, 0, 4, 0, 0, 0, 0, 0]] * 2)
        self.assertEqual(row['clip_rate'], 1.)
        self.assertAlmostEqual(row['aggregate_cosine'], 1., places=12)
        self.assertLess(row['clipping_shape_error'], 1e-12)
        self.assertGreater(row['relative_distortion'], .7)
        # Upstream accumulates norms in FP32 even for float64 grad_sample.
        expected_coeff = float(1/(torch.tensor(5., dtype=torch.float32)+1e-6))
        self.assertAlmostEqual(row['clipping_alpha_star'], expected_coeff, places=12)
        self.assertAlmostEqual(row['coefficient_cv'], 0., places=12)

    def test_heterogeneous_clipping_and_global_shape_reference(self):
        # Different layers contain orthogonal samples; per-layer fitting would
        # incorrectly report zero shape error instead of the global residual.
        g = torch.tensor([[10, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float64)
        row = self.toy_metrics(g.tolist())
        coeff = (1/(g.float().norm(dim=1)+1e-6)).clamp(max=1).double()
        std = float(coeff.std(unbiased=False))
        self.assertGreater(row['coefficient_std'], 0.)
        self.assertGreater(row['coefficient_cv'], 0.)
        self.assertAlmostEqual(row['coefficient_std'], std, places=12)
        self.assertAlmostEqual(row['coefficient_cv'], std/(float(coeff.mean())+self.c['eps_num']), places=12)
        raw, clipped = g.mean(0), (g*coeff[:, None]).mean(0)
        alpha = float(torch.dot(clipped, raw)/(raw.square().sum()+self.c['eps_num']))
        error = float((clipped-alpha*raw).norm()/(clipped.norm()+self.c['eps_num']))
        self.assertAlmostEqual(row['clipping_alpha_star'], alpha, places=12)
        self.assertAlmostEqual(row['clipping_shape_error'], error, places=12)
        self.assertGreater(error, .5)
        equal = self.toy_metrics([[10, 0, 0, 0, 0, 0, 0, 0]] * 2)
        self.assertAlmostEqual(equal['coefficient_std'], 0., places=12)
        self.assertAlmostEqual(equal['coefficient_cv'], 0., places=12)

    def test_new_clipping_metrics_finite_for_zero_and_tiny_aggregates(self):
        for gradients in ([[0]*8], [[1e-20]*8], [[2]*8, [-2]*8]):
            with self.subTest(gradients=gradients):
                row = self.toy_metrics(gradients)
                for key in ('coefficient_std', 'coefficient_cv', 'clipping_alpha_star', 'clipping_shape_error'):
                    self.assertTrue(np.isfinite(row[key]), key)

    def test_exp1_layout_round_trip_and_grad_sample_order(self):
        model = GradSampleModule(SimpleCNN(), loss_reduction='sum')
        self.addCleanup(model.remove_hooks)
        scales, originals = {}, {}
        for name in LAYERS:
            layer = getattr(model._module, name)
            for param in (layer.weight, layer.bias):
                param.grad_sample = torch.randn(2, *param.shape)
            originals[name] = layer_gradient(layer).clone()
            scales[name] = torch.linspace(.25, 2, originals[name].shape[1], dtype=torch.double)
        mapped = parameter_p(model, scales)
        self.assertEqual(list(mapped), list(dict(model._module.named_parameters())))
        apply_p(model, scales)
        for name in LAYERS:
            torch.testing.assert_close(layer_gradient(getattr(model._module, name)),
                                       originals[name] * scales[name].float())
        with self.assertRaises(ValueError):
            parameter_p(model, {**scales, 'fc2': torch.ones(3)})

    def test_global_clip_noise_and_sgd_against_reference(self):
        model = torch.nn.Linear(2, 1)
        model.weight.grad_sample = torch.tensor([[[3., 0.]], [[0., 0.]]])
        model.bias.grad_sample = torch.tensor([[4.], [0.]])
        initial = [p.detach().clone() for p in model.parameters()]
        # Joint weight+bias norm is 5, so both use the SAME factor.
        coeff = 1/(5+1e-6)
        sums = [torch.tensor([3*coeff, 0.]), torch.tensor([4*coeff])]
        sigma, lr = .7, .03
        ref_rng = RNGStream(321, 'cpu')
        with ref_rng.use():
            expected = [(s + torch.randn_like(s)*sigma)/2 for s in sums]
        accumulator = DPGradientAccumulator()
        accumulator.accumulate(model, 1., 2)
        with RNGStream(321, 'cpu').use():
            accumulator.finalize(sigma, 1., store_summed_grad=True)
        for p, target in zip(model.parameters(), expected):
            torch.testing.assert_close(p.grad.flatten(), target)
        torch.optim.SGD(model.parameters(), lr=lr).step()
        for p, old, target in zip(model.parameters(), initial, expected):
            torch.testing.assert_close(p, old-lr*target.reshape_as(p))

    def test_synthetic_rng_isolation_and_repeatability(self):
        model = SimpleCNN()
        loader = torch.Generator().manual_seed(101)
        noise = RNGStream(102, 'cpu')
        global_state, loader_state = torch.get_rng_state().clone(), loader.get_state().clone()
        noise_state = noise.cpu.clone()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        v, halves = synthetic(model.state_dict(), self.c, torch.device('cpu'), RNGStream(103, 'cpu'))
        self.assertTrue(torch.equal(global_state, torch.get_rng_state()))
        self.assertTrue(torch.equal(loader_state, loader.get_state()))
        self.assertTrue(torch.equal(noise_state, noise.cpu))
        self.assertEqual(python_state, random.getstate())
        np.testing.assert_equal(numpy_state, np.random.get_state())
        v2, halves2 = synthetic(model.state_dict(), self.c, torch.device('cpu'), RNGStream(103, 'cpu'))
        for n in LAYERS:
            torch.testing.assert_close(v[n], v2[n], rtol=0, atol=0)
            torch.testing.assert_close(v[n], (halves[0][n]+halves[1][n])/2)
        p = make_p(v, self.c['lambda'])
        for n in LAYERS:
            torch.testing.assert_close(p[n], 1/(v[n].sqrt()+self.c['lambda']))
        rows = refresh_rows(v, halves, p, v, p, self.c)
        self.assertTrue(all(r['D_time'] == 0 for r in rows))
        self.assertTrue(all(r['A_old'] == r['A_new'] for r in rows))
        # Actual P residual must be used, including the fixed lambda.
        oracle = oracle_rows(v, v, p, self.c)
        self.assertTrue(all(np.isfinite(r['R']) for r in oracle))

    def test_estimate_matches_single_example_autograd(self):
        model = SimpleCNN()
        x, y = torch.randn(2, 1, 28, 28), torch.tensor([1, 2])
        v, _ = estimate(model.state_dict(), (x, y), self.c, torch.device('cpu'))
        ref = {n: [] for n in LAYERS}
        for i in range(2):
            model.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(x[i:i+1]), y[i:i+1]).backward()
            for n in LAYERS:
                layer = getattr(model, n)
                grad = torch.cat([layer.weight.grad.flatten(1), layer.bias.grad[:, None]], 1).flatten()
                ref[n].append(grad.double().square())
        for n in LAYERS:
            torch.testing.assert_close(v[n], torch.stack(ref[n]).mean(0), rtol=2e-4, atol=1e-10)

    def test_oracle_cannot_change_trajectory_and_block_causality(self):
        (ROOT / 'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='unit_') as tmp:
            c = copy.deepcopy(self.c)
            c['output'] = str(Path(tmp).relative_to(ROOT) / 'oracle')
            events, applied = [], []
            real_synthetic, real_getitem, real_apply = training.synthetic, training.Indexed.__getitem__, training.apply_p
            def syn(*args, **kwargs):
                events.append('refresh')
                return real_synthetic(*args, **kwargs)
            def getitem(obj, i):
                events.append('private_batch_access')
                return real_getitem(obj, i)
            def apply(model, p):
                applied.append(digest(p.values()))
                return real_apply(model, p)
            def poison(state, samples, config, dev):
                v, halves = estimate(state, samples, config, dev)
                return {n: value.flip(0)*1e12 for n, value in v.items()}, halves
            with patch.object(training, 'synthetic', syn), patch.object(training.Indexed, '__getitem__', getitem), patch.object(training, 'apply_p', apply), patch.object(training, 'estimate', poison):
                a = training.train(c, 'syn_diag')
            self.assertEqual(events[0], 'refresh')
            self.assertEqual(events.count('refresh'), 2)
            self.assertEqual(events.index('refresh', 1), 1+c['K']*c['batch_size'])
            self.assertEqual(applied[0], applied[1])
            self.assertEqual(applied[2], applied[3])
            self.assertNotEqual(applied[0], applied[2])
            c['output'] = str(Path(tmp).relative_to(ROOT) / 'no_oracle')
            c['oracle_enabled'] = False
            b = training.train(c, 'syn_diag')
            self.assertEqual(a['final_model_hash'], b['final_model_hash'])
            self.assertEqual(a['initial_model_hash'], b['initial_model_hash'])


if __name__ == '__main__':
    unittest.main()
