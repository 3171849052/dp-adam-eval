from expv3.common import RNGStream


def test_paired_rng():
    left, right = RNGStream(46, "cpu"), RNGStream(46, "cpu")
    with left.use():
        import torch
        a = torch.randn(16)
    with right.use():
        import torch
        b = torch.randn(16)
    assert left.audit() == right.audit()
    assert a.equal(b)


def test_determinism():
    a, b = RNGStream(45, "cpu"), RNGStream(45, "cpu")
    with a.use():
        import torch
        x = torch.randint(0, 10, (32,))
    with b.use():
        import torch
        y = torch.randint(0, 10, (32,))
    assert x.equal(y) and a.audit() == b.audit()


def test_training_determinism(config, tiny_data, tmp_path):
    from expv3.tests.test_diagnostic_isolation import _trajectory
    from expv3.train_expv3 import train

    for run in ("dp_sgd_lr0p50", "dp_fisher_wiener_adaptive_beta_lr0p50"):
        train(config, 42, run, tmp_path / f"a_{run}", tiny_data)
        train(config, 42, run, tmp_path / f"b_{run}", tiny_data)
        assert _trajectory(tmp_path / f"a_{run}") == _trajectory(tmp_path / f"b_{run}")
