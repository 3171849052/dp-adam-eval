import copy
import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from opacus import GradSampleModule
from exp4.common import DEFAULT, SimpleCNN, RNGStream, digest, adam
from exp4.dynamics import AdamDirection, aggregate, flat, metrics
from exp4.train_exp4 import train
from exp3.preconditioners import apply, refresh, synthetic_samples
from dp_kfac.privacy import clip_and_noise_gradients

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


@pytest.mark.parametrize('dev', DEVICES)
@pytest.mark.parametrize('trajectory', ['adam', 'dp_adam'])
def test_adam_equivalence_and_current_noise_off_isolation(dev, trajectory):
    c = DEFAULT
    actual = [torch.nn.Parameter(torch.randn(3, 4, device=dev)), torch.nn.Parameter(torch.randn(4, device=dev))]
    opt = torch.optim.Adam(actual, lr=c['adam_lr'], betas=(c['beta1'], c['beta2']), eps=c['adam_eps'], weight_decay=0.)
    shadow = AdamDirection(actual, c)
    for _ in range(5):
        clean = [torch.randn_like(p) for p in actual]
        g = [v+torch.randn_like(v)*.5 for v in clean] if trajectory == 'dp_adam' else clean
        before = copy.deepcopy(shadow.optimizer.state_dict())
        rng = digest([torch.get_rng_state()] + ([torch.cuda.get_rng_state()] if dev == 'cuda' else []))
        model_hash = digest(actual)
        branch_result = shadow.peek(clean)
        assert model_hash == digest(actual)
        assert rng == digest([torch.get_rng_state()] + ([torch.cuda.get_rng_state()] if dev == 'cuda' else []))
        after = shadow.optimizer.state_dict()
        assert before['param_groups'] == after['param_groups']
        for key, state in before['state'].items():
            for n, value in state.items():
                assert torch.equal(value, after['state'][key][n])
        # Place an independent copy of the shadow on theta, at the real lr.
        # Exact equality verifies the next step, including persisted moments.
        candidate = copy.deepcopy(shadow)
        candidate.optimizer.param_groups[0]['lr'] = c['adam_lr']
        for p, theta, grad in zip(candidate.params, actual, g):
            p.data.copy_(theta)
            p.grad = grad.clone()
        candidate.optimizer.step()
        old = [p.detach().clone() for p in actual]
        for p, grad in zip(actual, g):
            p.grad = grad.clone()
        opt.step()
        u = shadow.advance(g)
        assert all(torch.equal(a, b) for a, b in zip(actual, candidate.params))
        for p, theta, direction in zip(actual, old, u):
            torch.testing.assert_close(p, theta-c['adam_lr']*direction, rtol=1e-6, atol=2e-7)
        if trajectory == 'adam':
            assert all(torch.equal(a, b) for a, b in zip(branch_result, u))
        for a, b in zip(opt.state.values(), shadow.optimizer.state.values()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                assert torch.equal(a[key], b[key])


@pytest.mark.parametrize('dev', DEVICES)
def test_syndiag_equivalence_and_shared_noise(dev):
    device = torch.device(dev)
    c = dict(DEFAULT, batch_size=4, M_syn=8, analysis_batch_size=4)
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction='sum')
    x, y = torch.randn(4, 1, 28, 28, device=device), torch.arange(4, device=device)
    synthetic, _ = synthetic_samples(c, device, RNGStream(45, device))
    active = refresh(model._module.state_dict(), synthetic, c, device, 'syn_diag')
    # Independently recover v = mean per-example gradient squared.
    oracle = SimpleCNN().to(device)
    oracle.load_state_dict(model._module.state_dict())
    sums = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in oracle.named_parameters()}
    for xx, yy in zip(*synthetic):
        gs = torch.autograd.grad(F.cross_entropy(oracle(xx[None]), yy[None]), tuple(oracle.parameters()))
        for (n, _), g in zip(oracle.named_parameters(), gs):
            sums[n] += g.double().square()
    for name, value in active[1].items():
        v = torch.cat([sums[name+'.weight'].flatten(1), sums[name+'.bias'][:, None]], 1).flatten()/8
        torch.testing.assert_close(value, (1/(v.sqrt()+c['lambda'])).float(), rtol=2e-5, atol=1e-5)
    F.cross_entropy(model(x), y, reduction='sum').backward()
    raw = [p.grad_sample.clone() for p in model.parameters()]
    rng = RNGStream(46, device)
    dp = aggregate(model, None, copy.deepcopy(rng), 1.3, c, 4)
    for p, gs in zip(model.parameters(), raw):
        p.grad_sample = gs.clone()
    syn = aggregate(model, active, copy.deepcopy(rng), 1.3, c, 4)
    assert dp['noise_hash'] == syn['noise_hash']
    old = [p.detach().clone() for p in model.parameters()]
    for p, gs in zip(model.parameters(), raw):
        p.grad_sample = gs.clone()
    apply(model, active)
    with rng.use():
        clip_and_noise_gradients(model, 1.3, 1., 4, store_summed_grad=True)
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), syn['noisy']))
    torch.optim.SGD(model.parameters(), lr=.1).step()
    for p, theta, u in zip(model.parameters(), old, syn['noisy']):
        torch.testing.assert_close(p, theta-.1*u, rtol=1e-6, atol=1e-7)
    model.remove_hooks()


@pytest.mark.parametrize('dev', DEVICES)
def test_diagnostic_isolation_pairing_and_refresh_causality(tmp_path, dev):
    c = dict(DEFAULT, smoke=True, seeds=[42], device=dev, batch_size=4, epochs=2,
             M_syn=8, K=3, analysis_batch_size=4, train_subset=8, test_subset=8,
             eval_interval=2, diagnostic_interval=2, threads=2)
    data = TensorDataset(torch.randn(8, 1, 28, 28), torch.arange(8))
    pairs = []
    for trajectory in ('adam', 'dp_adam'):
        events = []
        def hook(kind, step, active):
            events.append((kind, step, digest(active[1].values()) if active else None))
        enabled = train(c, 42, trajectory, tmp_path/'on', (data, data), event_hook=hook)
        disabled = train(c, 42, trajectory, tmp_path/'off', (data, data), diagnostics=False)
        assert enabled['final_model_hash'] == disabled['final_model_hash']
        saved = torch.load(tmp_path/'on'/f'seed42/{trajectory}/final_state.pt', weights_only=True)
        for ref_state, shadow_state in zip(saved['optimizer']['state'].values(), saved['shadow_'+trajectory]['state'].values()):
            assert all(torch.equal(v, shadow_state[k]) for k, v in ref_state.items())
        on = json.loads((tmp_path/'on'/f'seed42/{trajectory}/pairing.json').read_text())
        off = json.loads((tmp_path/'off'/f'seed42/{trajectory}/pairing.json').read_text())
        for a, b in zip(on['private'], off['private']):
            assert {k:v for k,v in a.items() if k != 'syn_noise_hash'} == {k:v for k,v in b.items() if k != 'syn_noise_hash'}
            assert a['noise_hash'] == a['syn_noise_hash']
        before = [e for e in events if e[0] == 'before_batch']
        assert before[0][2] == before[1][2] == before[2][2]
        assert before[3][2] != before[2][2]
        for step in (0, 3):
            assert events.index(next(e for e in events if e[:2] == ('refresh_end', step))) < events.index(next(e for e in events if e[:2] == ('before_batch', step)))
        pairs.append(on)
    for a, b in zip(pairs[0]['private'], pairs[1]['private']):
        for key in ('batch_hash', 'noise_hash', 'noise_rng_before', 'noise_rng_after', 'loader_rng_hash'):
            assert a[key] == b[key]
    from exp4.analyze_exp4 import load_runs
    frame, evaluations, meta, current = load_runs(c, tmp_path/'on')
    assert len(frame) == 8 and len(evaluations) == 8 and len(meta) == 2
    # A paired comparison with tampered noise must fail before aggregation.
    path = tmp_path/'on/seed42/dp_adam/pairing.json'
    corrupt = json.loads(path.read_text())
    corrupt['private'][0]['noise_hash'] = 'bad'
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match='Shared noise mismatch'):
        load_runs(c, tmp_path/'on')
    for a, b in zip(pairs[0]['synthetic'], pairs[1]['synthetic']):
        assert {k:v for k,v in a.items() if k != 'preconditioner_hash'} == {k:v for k,v in b.items() if k != 'preconditioner_hash'}


def test_metric_signs_and_lr_independence():
    a, d, s = [torch.tensor([1., 0.])], [torch.tensor([0., 2.])], [torch.tensor([3., 0.])]
    agg = dict(mean=a, clean=a)
    row = metrics(a, d, s, a, s, agg, agg)
    assert row['alignment_advantage_noisy'] == 1
    assert row['noise_degradation_dp'] == 1
    assert row['noise_degradation_syn'] == 0
    assert row['norm_ratio_dp_adam'] == 2
    assert row['norm_ratio_syn_adam'] == 3


def test_summary_uses_late_seed_medians_then_sample_sd():
    import pandas as pd
    from exp4.analyze_exp4 import summarize, SUMMARY_METRICS
    rows = []
    for seed, values in [(42, [1000, 1000, 1, 3]), (7, [-1000, -1000, 7, 9]), (91, [999, 999, 4, 6])]:
        for step, value in enumerate(values, 1):
            rows.append(dict(seed=seed, trajectory='adam', step=step, **{key: value for key in SUMMARY_METRICS}))
    per_seed, summary = summarize(pd.DataFrame(rows))
    assert per_seed.groupby('seed').late_median.first().to_dict() == {7: 8., 42: 2., 91: 5.}
    assert (summary['mean'] == 5.).all()
    assert (summary['std'] == 3.).all()
