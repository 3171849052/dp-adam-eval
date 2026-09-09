import json

import pandas as pd
import pytest
import torch
from opacus.accountants import RDPAccountant

import expv3.train_expv3 as trainer
import expv3.validate_expv3 as validator
from expv3.common import fingerprint, provenance, run_specs
from expv3.tests.test_diagnostic_isolation import _trajectory

RUN = 'dp_fisher_wiener_adaptive_beta_lr0p50'


def expected_epsilon(summary, count, delta):
    accountant = RDPAccountant()
    for _ in range(count):
        accountant.step(noise_multiplier=summary['noise_multiplier'], sample_rate=summary['sample_rate'])
    return accountant.get_epsilon(delta)


@pytest.fixture
def filtered(config, tiny_data, tmp_path, monkeypatch):
    original = trainer.apply_fisher_wiener
    calls = 0
    def fail(model, active):
        nonlocal calls
        original(model, active)
        if calls == config['K']:
            next(model.parameters()).grad.fill_(float('nan'))
        calls += 1
    monkeypatch.setattr(trainer, 'apply_fisher_wiener', fail)
    summary = trainer.train(config, 42, RUN, tmp_path, tiny_data)
    return summary, tmp_path / 'seed42' / RUN


def test_filtered_gradient_divergence_consumes_privacy_step(filtered, config):
    summary, root = filtered
    assert summary['epsilon_spent'] == pytest.approx(expected_epsilon(summary, 3, config['delta']))
    assert summary['privacy_steps'] == summary['completed_steps'] + 1 == 3


def test_filtered_gradient_divergence_preserves_beta_observation(filtered):
    summary, root = filtered
    steps = pd.read_csv(root / 'beta_step_metrics.csv')
    assert len(steps) == 12
    assert summary['beta_observation_steps'] == 3
    intervals = pd.read_csv(root / 'beta_interval_metrics.csv')
    assert intervals.loc[intervals.interval_index == 1, 'n_steps'].eq(1).all()
    assert not intervals.applied_to_training.any()


def test_formal_diverged_anchor_regression():
    # Exercise formal accounting without launching formal training.
    summary = dict(noise_multiplier=1.068115234375, sample_rate=256/60000,
                   privacy_steps=401, status='diverged')
    summary['epsilon_spent'] = expected_epsilon(summary, 401, 1e-5)
    config = dict(smoke=False, batch_size=256, train_subset=None, epsilon=1., delta=1e-5)
    validator._validate_privacy_budget(config, summary)
    summary["epsilon_spent"] = 0.995693195331761
    with pytest.raises(AssertionError):
        validator._validate_privacy_budget(config, summary)


def test_oracle_exception_does_not_change_training(config, tiny_data, tmp_path, monkeypatch):
    normal = trainer.train(config, 42, RUN, tmp_path / 'normal', tiny_data)
    def throwing(*args):
        raise RuntimeError('research only')
    monkeypatch.setattr(trainer, 'capture_oracle_beta_diagnostics', throwing)
    broken = trainer.train(config, 42, RUN, tmp_path / 'throwing', tiny_data)
    assert _trajectory(tmp_path / 'normal') == _trajectory(tmp_path / 'throwing')
    assert broken['status'] == 'completed'
    assert broken['oracle_diagnostic_error_steps'] == list(range(4))
    validate_one(config, tmp_path / 'throwing' / 'seed42' / RUN)
    print('ORACLE HASH', normal['final_model_hash'], broken['final_model_hash'])


def validate_one(config, root):
    spec = next(value for value in run_specs(config) if value['run_id'] == RUN)
    return validator._validate_run(config, root.parent.parent, dict(spec, seed=42), fingerprint(config), provenance())


def report_stage(summary):
    print('DIVERGENCE', summary['divergence_stage'],
          {key: summary[key] for key in ('completed_steps', 'privacy_steps', 'beta_observation_steps', 'epsilon_spent')})


def test_diverged_epsilon_matches_privacy_steps(filtered, config):
    summary, root = filtered
    assert summary['epsilon_spent'] == pytest.approx(expected_epsilon(summary, summary['privacy_steps'], config['delta']))
    validate_one(config, root)
    report_stage(summary)


@pytest.mark.parametrize('failure_step', [0, 2])
def test_loss_divergence_does_not_consume_privacy_step(config, tiny_data, tmp_path, monkeypatch, failure_step):
    original = trainer.F.cross_entropy
    calls = 0
    def loss(*args, **kwargs):
        nonlocal calls
        value = original(*args, **kwargs)
        if kwargs.get('reduction') == 'sum' and torch.is_grad_enabled():
            if calls == failure_step:
                value = value * float('nan')
            calls += 1
        return value
    monkeypatch.setattr(trainer.F, 'cross_entropy', loss)
    summary = trainer.train(config, 42, RUN, tmp_path, tiny_data)
    assert summary['divergence_stage'] == 'loss'
    assert summary['privacy_steps'] == summary['completed_steps'] == summary['beta_observation_steps'] == failure_step
    validate_one(config, tmp_path / 'seed42' / RUN)
    report_stage(summary)


def test_noisy_gradient_divergence_consumes_privacy_step(config, tiny_data, tmp_path, monkeypatch):
    original = trainer.clip_and_noise_gradients
    calls = 0
    def noisy(model, **kwargs):
        nonlocal calls
        original(model, **kwargs)
        if calls == 2:
            next(model.parameters()).grad.fill_(float('nan'))
        calls += 1
    monkeypatch.setattr(trainer, 'clip_and_noise_gradients', noisy)
    summary = trainer.train(config, 42, RUN, tmp_path, tiny_data)
    assert summary['divergence_stage'] == 'noisy_gradient'
    assert summary['privacy_steps'] == 3
    assert summary['completed_steps'] == summary['beta_observation_steps'] == 2
    assert summary['epsilon_spent'] == pytest.approx(expected_epsilon(summary, 3, config['delta']))
    validate_one(config, tmp_path / 'seed42' / RUN)
    report_stage(summary)


@pytest.mark.parametrize('stage', ['parameters', 'optimizer_exception'])
def test_optimizer_divergence_preserves_consumed_events(config, tiny_data, tmp_path, monkeypatch, stage):
    original = trainer.make_optimizer
    def make(*args):
        optimizer = original(*args)
        update = optimizer.step
        calls = 0
        def fail():
            nonlocal calls
            if calls == 2 and stage == 'optimizer_exception':
                raise RuntimeError('injected optimizer failure')
            update()
            if calls == 2:
                with torch.no_grad():
                    optimizer.param_groups[0]['params'][0].fill_(float('inf'))
            calls += 1
        optimizer.step = fail
        return optimizer
    monkeypatch.setattr(trainer, 'make_optimizer', make)
    summary = trainer.train(config, 42, RUN, tmp_path, tiny_data)
    assert summary['divergence_stage'] == stage
    assert summary['completed_steps'] == 2
    assert summary['privacy_steps'] == summary['beta_observation_steps'] == 3
    assert summary['parameters_finite'] is (stage != 'parameters')
    validate_one(config, tmp_path / 'seed42' / RUN)


def test_dp_sgd_synthetic_measurement_training_isolation(config, tiny_data, tmp_path, monkeypatch):
    run = 'dp_sgd_lr0p50'
    normal = trainer.train(config, 42, run, tmp_path / 'on', tiny_data)
    def forbidden(*args, **kwargs):
        raise AssertionError('synthetic measurement was not disabled')
    for name in ('synthetic_samples', 'build_covariances', 'build_trace_state'):
        monkeypatch.setattr(trainer, name, forbidden)
    off = trainer.train(config, 42, run, tmp_path / 'off', tiny_data,
                        synthetic_measurement=False, beta_measurement=False)
    left, right = _trajectory(tmp_path / 'on'), _trajectory(tmp_path / 'off')
    for key in ('private', 'privacy', 'model', 'final'):
        assert left[key] == right[key]
    assert left['synthetic'] and not right['synthetic']
    assert normal['number_of_refreshes'] > 0 and off['number_of_refreshes'] == 0
    print('SYNTHETIC ON/OFF HASH', normal['final_model_hash'], off['final_model_hash'],
          'REFRESH', normal['number_of_refreshes'], off['number_of_refreshes'])


def test_synthetic_measurement_requires_dp_sgd_without_beta(config, tiny_data, tmp_path):
    with pytest.raises(ValueError):
        trainer.train(config, 42, RUN, tmp_path, tiny_data, synthetic_measurement=False, beta_measurement=False)
    with pytest.raises(ValueError):
        trainer.train(config, 42, 'dp_sgd_lr0p50', tmp_path, tiny_data, synthetic_measurement=False)


def test_raw_beta_statistics_use_interval_rows():
    from expv3.summarize_expv3 import raw_beta_statistics
    intervals = pd.DataFrame({'beta_dp_raw': [-1., -2., 3., float('nan'), float('inf')]})
    assert raw_beta_statistics(intervals) == (-1., 2/3)
    assert raw_beta_statistics(intervals.iloc[:0]) == (None, None)


@pytest.mark.parametrize('mutation', ['count', 'audit', 'stage', 'epsilon', 'applied'])
def test_validator_rejects_partial_privacy_corruption(filtered, config, mutation):
    summary, root = filtered
    validate_one(config, root)
    if mutation == 'audit':
        path = root / 'pairing.json'
        pair = json.loads(path.read_text())
        pair['privacy'][-1]['noise_rng_before'] = 'corrupt'
        path.write_text(json.dumps(pair))
    elif mutation == 'applied':
        path = root / 'beta_interval_metrics.csv'
        frame = pd.read_csv(path)
        frame.loc[frame.interval_index == 0, ['applied_to_training', 'was_used_by_next_interval']] = True
        frame.to_csv(path, index=False)
    else:
        for name in ('summary.json', 'metadata.json'):
            path = root / name
            value = json.loads(path.read_text())
            if mutation == 'count':
                value['privacy_steps'] -= 1
            elif mutation == 'stage':
                value['divergence_stage'] = 'loss'
            else:
                value['epsilon_spent'] = expected_epsilon(value, 2, config['delta'])
            path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        validate_one(config, root)


def test_completed_formal_privacy_anchor():
    config = dict(smoke=False, batch_size=256, train_subset=None, epsilon=1., delta=1e-5)
    summary = dict(status='completed', privacy_steps=1170, noise_multiplier=1.068115234375,
                   sample_rate=256/60000, epsilon_spent=0.995693195331761)
    validator._validate_privacy_budget(config, summary)
    summary['privacy_steps'] = 401
    with pytest.raises(AssertionError):
        validator._validate_privacy_budget(config, summary)


@pytest.mark.parametrize('target', ['capture_deployable_beta_observations', 'clip_and_noise_gradients'])
def test_algorithmic_exceptions_are_not_swallowed(config, tiny_data, tmp_path, monkeypatch, target):
    def broken(*args, **kwargs):
        raise RuntimeError('algorithmic failure')
    monkeypatch.setattr(trainer, target, broken)
    with pytest.raises(RuntimeError, match='algorithmic failure'):
        trainer.train(config, 42, RUN, tmp_path, tiny_data)


def test_oracle_exception_bookkeeping_without_beta_measurement(config, tiny_data, tmp_path, monkeypatch):
    def throwing(*args):
        raise RuntimeError('research only')
    monkeypatch.setattr(trainer, 'capture_oracle_beta_diagnostics', throwing)
    run = 'dp_sgd_lr0p50'
    summary = trainer.train(config, 42, run, tmp_path, tiny_data, beta_measurement=False)
    assert summary['status'] == 'completed'
    assert summary['oracle_diagnostic_error_steps'] == list(range(4))
    assert summary['oracle_diagnostics_all_valid'] is False
    spec = next(value for value in run_specs(config) if value['run_id'] == run)
    validator._validate_run(config, tmp_path, dict(spec, seed=42), fingerprint(config), provenance())
