"""Core protocol, pairing, optimizer input, and lag checks for research runs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from expv3.adaptive_fisher_wiener import h_hash, h_stats
from expv3.common import expected_total_steps, save_json
from expv5.train_expv5 import ADAM_CONFIG
from expv6.train_expv6 import check_config
from expv6.summarize_expv6 import sweep_roots


def load(path):
    return json.loads(Path(path).read_text())


def validate_certificates(certificates, layers, config, alpha, device):
    for certificate in certificates:
        assert certificate['alpha'] == alpha
        lambda_a = torch.tensor(certificate['lambda_A'], dtype=torch.float32, device=device)
        lambda_g = torch.tensor(certificate['lambda_G'], dtype=torch.float32, device=device)
        scaled = certificate['beta_train'] * (lambda_g[:, None] * lambda_a[None, :])
        h_beta = scaled / (scaled + certificate['r'])
        h_alpha = (1-alpha) + alpha*h_beta
        assert h_hash(h_beta) == certificate['H_beta_hash']
        assert h_hash(h_alpha) == certificate['H_alpha_hash']
        observed = layers[(layers.layer == certificate['layer']) &
                          (layers.step // config['K'] == certificate['interval_index'])]
        assert len(observed) > 0
        np.testing.assert_allclose(observed.beta_train, certificate['beta_train'])
        for field, expected in h_stats(h_alpha).items():
            np.testing.assert_allclose(observed[field], expected, rtol=1e-6, atol=1e-8)


def validate(config, runs, expv5_runs, output):
    check_config(config)
    bases, checked = {}, set()
    protocol = [*ADAM_CONFIG, 'epochs', 'batch_size', 'epsilon', 'delta', 'max_grad_norm',
                'K', 'beta_window', 'M_syn', 'eval_interval', 'beta_initial', 'beta_update_rule',
                'train_subset', 'test_subset']
    for labels, root in sweep_roots(config, runs, expv5_runs):
        if root in checked:
            continue
        checked.add(root)
        cfg, meta, summary, pair = [load(root / f'{name}.json') for name in ('config', 'metadata', 'summary', 'pairing')]
        assert all(cfg[k] == config[k] for k in protocol), f'Protocol mismatch: {root}'
        assert summary['status'] == 'completed'
        assert summary['completed_steps'] == expected_total_steps(config)
        assert not meta['gamma_influences_training']
        train = pd.read_csv(root / 'train_metrics.csv')
        layers = pd.read_csv(root / 'layer_metrics.csv')
        assert train.diagnostics_finite.all() and train.oracle_diagnostics_finite.all()
        np.testing.assert_allclose(layers.gradient_norm_into_adam, layers.filtered_gradient_norm)
        evaluations = train.loc[train.test_accuracy.notna(), 'step'].tolist()
        seed = labels['seed']
        if seed not in bases:
            bases[seed] = (meta, pair, evaluations)
        base_meta, base_pair, base_eval = bases[seed]
        assert meta['rng_seeds'] == base_meta['rng_seeds']
        assert evaluations == base_eval
        # Compare the existing RNG states and batch indices; no new hashes.
        for kind, fields in [('private', ('step', 'batch_indices', 'noise_rng_before', 'noise_rng_after')),
                             ('synthetic', ('step', 'rng_before', 'rng_after'))]:
            assert [[r[k] for k in fields] for r in pair[kind]] == [[r[k] for k in fields] for r in base_pair[kind]]
        adaptive = labels['family'] == 'adaptive_beta' and labels['alpha'] != 0
        assert meta['beta_algorithm_active'] == adaptive
        if 0 < labels['alpha'] < 1:
            validate_certificates(load(root / 'h_certificates.json'), layers, config,
                                  labels['alpha'], meta['device'])
            np.testing.assert_allclose(layers.alpha, labels['alpha'])
            assert np.isfinite(layers.clean_gradient_distortion).all()
            assert (layers.H_min >= 1-labels['alpha']-1e-6).all()
            assert (layers.H_max <= 1+1e-6).all()
        if labels['alpha'] == 0:
            continue
        control = pd.read_csv(root / 'beta_controller_metrics.csv')
        for layer, values in control.groupby('layer'):
            previous = 1.
            for c in values.itertuples():
                expected = 1.
                if adaptive and c.interval_index > 0:
                    raw = c.numerator_previous/c.denominator_previous
                    expected = raw if np.isfinite(raw) and raw > 0 else previous
                    assert c.beta_source_interval == c.interval_index-1
                assert np.isclose(c.beta_train, expected)
                observed = layers[(layers.layer == layer) & (layers.step // config['K'] == c.interval_index)]
                np.testing.assert_allclose(observed.beta_train, expected)
                previous = expected
    result = dict(passed=True, pairing_passed=True, unique_runs=len(checked))
    save_json(Path(output) / 'validation.json', result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for arg in ('config', 'runs', 'expv5-runs', 'output'):
        parser.add_argument('--' + arg, required=True)
    args = parser.parse_args()
    validate(load(args.config), args.runs, args.expv5_runs, args.output)


if __name__ == '__main__':
    main()
