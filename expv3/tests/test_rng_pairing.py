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

