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
import train_exp2 as training


class Exp2Tests(unittest.TestCase):
    def setUp(self):
        self.c = read_config(ROOT / 'configs/smoke.json')
        torch.manual_seed(123)

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
