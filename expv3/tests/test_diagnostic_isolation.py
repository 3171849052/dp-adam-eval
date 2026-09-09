from expv3.beta_controller import AdaptiveBetaController


def test_oracle_isolation():
    first = AdaptiveBetaController(("layer",))
    second = AdaptiveBetaController(("layer",))
    for controller in (first, second):
        controller.observe("layer", 12.0, 10.0, 10.0)
    # Oracle values are deliberately different but are not accepted by observe.
    oracle_a, oracle_b = 1e-30, 1e30
    assert oracle_a != oracle_b
    assert first.finalize_all(0)["layer"] == second.finalize_all(0)["layer"]


def test_dp_sgd_measurement_isolation():
    from expv3.common import DP_METHOD, FISHER_METHOD
    assert DP_METHOD == "dp_sgd"
    assert FISHER_METHOD != DP_METHOD

