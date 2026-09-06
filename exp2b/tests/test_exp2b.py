"""Factorial-grid, attenuation, aggregation and validation regression checks."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics import attenuation, refresh_rows
from synthetic_preconditioner import make_p
from common import LAYERS, fingerprint
from summarize_exp2b import (load_grid, expected_refresh_steps, aggregate, validate_run,
                            TRAIN_REQUIRED, REFRESH_REQUIRED, ORACLE_REQUIRED, validate_contributions)


class Exp2bTests(unittest.TestCase):
    def test_attenuation_formula_and_bounds(self):
        v = torch.tensor([0., 1., 4., 1e30], dtype=torch.float64)
        a = attenuation(v, 2.)
        torch.testing.assert_close(a, torch.tensor([0., 1/3, 1/2, 1e15/(1e15+2)], dtype=torch.float64))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(((a >= 0) & (a <= 1)).all())
        self.assertGreater(a[-1], .999999)
        vs = dict.fromkeys(LAYERS, v)
        rows = refresh_rows(vs, [vs, vs], make_p(vs, 2.), None, None, {'lambda': 2., 'eps_num': 1e-12})
        for row in rows:
            for q in (5, 50, 95):
                self.assertEqual(row[f'attenuation_q{q:02d}'], float(torch.quantile(a, q/100)))

    def test_invalid_contributions_rejected(self):
        for values in ([-.1, .3, .4, .4], [0., 0., 0., 1.01],
                       [.3]*4, [np.nan, 0., 0., 0.], [np.inf, 0., 0., 0.]):
            with self.subTest(values=values):
                frame = pd.DataFrame([{'step': 7, **dict(zip([f'contrib_{n}' for n in LAYERS], values))}])
                with self.assertRaises(ValueError):
                    validate_contributions(frame)
        frame = pd.DataFrame([{'step': 7, **dict.fromkeys([f'contrib_{n}' for n in LAYERS], .25000001)}])
        validate_contributions(frame)  # FP32 roundoff near one remains valid.

    def test_grid(self):
        configs = load_grid(ROOT / 'configs')
        self.assertEqual(len({(c['lambda'], c['K']) for c in configs}), 9)
        self.assertEqual(len(load_grid(ROOT / 'smoke_configs', smoke=True)), 2)

    def test_bad_grid_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for p in (ROOT / 'configs').glob('*.json'):
                Path(tmp, p.name).write_text(p.read_text())
            p = Path(tmp, 'lambda1e4_K50.json')
            c = json.loads(p.read_text()); c['K'] = 100; p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, 'mismatch|duplicate'):
                load_grid(tmp)
            p.unlink()
            with self.assertRaisesRegex(ValueError, 'exactly 9'):
                load_grid(tmp)

    def test_refresh_schedule(self):
        for k, count, last in [(50, 24, 1150), (100, 12, 1100), (200, 6, 1000)]:
            expected = [i*k for i in range(count)]
            self.assertEqual(expected_refresh_steps(1170, k), expected)
            self.assertEqual(expected[-1], last)
        self.assertEqual(expected_refresh_steps(4, 2), [0, 2])
        self.assertEqual(expected_refresh_steps(4, 4), [0])

    def fixture(self, root):
        c = copy.deepcopy(load_grid(ROOT / 'smoke_configs', True)[0])
        c.update(output=str(root.relative_to(ROOT)), K=2, eval_interval=1)
        train = pd.DataFrame([{**dict.fromkeys(TRAIN_REQUIRED, 1.), 'step': s,
                    'test_accuracy': acc, 'test_loss': 2-s/10,
                    **{f'contrib_{n}': .25 for n in LAYERS}} for s, acc in enumerate([.2, .9, .4, .6], 1)])
        refresh = pd.DataFrame([{**dict.fromkeys(REFRESH_REQUIRED, .5), 'step': s, 'layer': n,
                    'A_new': s+i+1., 'A_old': s+i+2. if s else np.nan,
                    'D_time': .1 if s else np.nan, 'stale_ratio': 1. if s else np.nan}
                    for s in (0, 2) for i, n in enumerate(LAYERS)])
        oracle = pd.DataFrame([{**dict.fromkeys(ORACLE_REQUIRED, .5), 'step': s, 'layer': n,
                    'R': s+i+.25} for s in (0, 2) for i, n in enumerate(LAYERS)])
        results = root / 'results'; results.mkdir()
        for key, frame in [('train', train), ('refresh', refresh), ('oracle', oracle)]:
            frame['method'] = 'syn_diag'; frame['config_fingerprint'] = fingerprint(c)
            frame.to_csv(results / f'{key}_metrics_syn_diag.csv', index=False)
            frame.to_csv(results / f'{key}_metrics.csv', index=False)
        meta = dict(config=c, method='syn_diag', fingerprint=fingerprint(c), total_steps=4, completed_steps=4)
        (root / 'config_syn_diag.json').write_text(json.dumps(c))
        (root / 'metadata_syn_diag.json').write_text(json.dumps(meta))
        (results / 'summary.json').write_text(json.dumps({'syn_diag': meta}))
        return c

    def test_csv_summary_and_validation_failures(self):
        (ROOT / 'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='fixture_') as tmp:
            c = self.fixture(Path(tmp))
            tables, total = validate_run(c)
            row = aggregate(c, tables, total)
            self.assertEqual(row['final_test_accuracy'], .6)
            self.assertEqual(row['best_test_accuracy'], .9)
            self.assertEqual(row['late_mean_accuracy'], .5)
            self.assertEqual(row['final_test_loss'], 1.6)
            for key in ('coefficient_cv', 'aggregate_cosine', 'clipping_shape_error', 'diagnostic_snr'):
                self.assertEqual(row[f'final_{key}'], 1.)
            for i, n in enumerate(LAYERS):
                self.assertEqual(row[f'final_R_{n}'], 2+i+.25)
                self.assertEqual(row[f'final_A_old_{n}'], 4+i)
                self.assertEqual(row[f'final_A_new_{n}'], 3+i)
                self.assertEqual(row[f'final_attenuation_q50_{n}'], .5)
            for kind, mutation in [
                ('train', lambda f: f.iloc[:-1]),
                ('train', lambda f: f.assign(config_fingerprint='wrong')),
                ('train', lambda f: f.assign(diagnostic_snr=np.inf)),
                ('refresh', lambda f: f.assign(attenuation_q50=1.1)),
                ('refresh', lambda f: f.assign(step=0)),
                ('oracle', lambda f: f[f.layer != LAYERS[0]]),
                ('oracle', lambda f: f.assign(R=np.nan))]:
                with self.subTest(kind=kind, mutation=mutation):
                    original = tables[kind]
                    for suffix in ('_syn_diag', ''):
                        mutation(original.copy()).to_csv(Path(tmp)/'results'/f'{kind}_metrics{suffix}.csv', index=False)
                    with self.assertRaises(ValueError):
                        validate_run(c)
                    for suffix in ('_syn_diag', ''):
                        original.to_csv(Path(tmp)/'results'/f'{kind}_metrics{suffix}.csv', index=False)
            (Path(tmp)/'results/summary.json').unlink()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                validate_run(c)

    def test_original_training_path_and_preconditioner_unchanged(self):
        # Preserve the actual loop, optimizer/accountant/noise, and full synthetic implementation.
        upstream = ROOT.parent / 'exp2'
        self.assertEqual((ROOT/'synthetic_preconditioner.py').read_bytes(), (upstream/'synthetic_preconditioner.py').read_bytes())
        def core(path):
            tree = ast.parse(path.read_text())
            train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
            start = next(i for i, n in enumerate(train.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'dev' for t in n.targets))
            stop = next(i for i, n in enumerate(train.body) if isinstance(n, ast.Try))
            return [ast.dump(n) for n in train.body[start:stop]], ast.dump(train.body[stop].body[0])
        self.assertEqual(core(ROOT/'train_exp2b.py'), core(upstream/'train_exp2.py'))


if __name__ == '__main__':
    unittest.main()
