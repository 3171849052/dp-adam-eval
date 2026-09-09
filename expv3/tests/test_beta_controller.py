import math

from expv3.beta_controller import AdaptiveBetaController


def test_initial_beta_is_one():
    controller = AdaptiveBetaController(("layer",))
    assert controller.active_beta("layer") == 1.0


def test_positive_previous_interval_updates_next_beta():
    controller = AdaptiveBetaController(("layer",))
    controller.observe("layer", 12.0, 10.0, 10.0)
    result = controller.finalize_all(0)["layer"]
    assert result["beta_raw"] == 0.2
    assert controller.active_beta("layer") == 0.2


def test_negative_previous_interval_holds_beta():
    controller = AdaptiveBetaController(("layer",))
    controller.observe("layer", 12.0, 10.0, 10.0)
    controller.finalize_all(0)
    controller.observe("layer", 9.0, 10.0, 10.0)
    result = controller.finalize_all(1)["layer"]
    assert result["beta_raw"] == -0.1
    assert controller.active_beta("layer") == 0.2
    assert result["fallback_reason"] == "negative"


def test_nonfinite_previous_interval_holds_beta():
    controller = AdaptiveBetaController(("layer",))
    controller.observe("layer", float("nan"), 0.0, 10.0)
    result = controller.finalize_all(0)["layer"]
    assert math.isnan(result["beta_raw"])
    assert controller.active_beta("layer") == 1.0
    assert result["fallback_reason"] == "nonfinite"


def test_ratio_of_sums_not_mean_of_ratios():
    controller = AdaptiveBetaController(("layer",))
    controller.observe("layer", 11.0, 10.0, 1.0)
    controller.observe("layer", 30.0, 10.0, 100.0)
    result = controller.finalize_all(0)["layer"]
    assert result["beta_raw"] == 21.0 / 101.0
    assert result["beta_raw"] != (1.0 + 20.0 / 100.0) / 2


def test_partial_interval_finalize():
    controller = AdaptiveBetaController(("layer",))
    controller.observe("layer", 13.0, 10.0, 10.0)
    result = controller.finalize_all(0, apply=False)["layer"]
    assert result["observation_count"] == 1
    assert result["numerator_sum"] == 3.0
    assert result["denominator_sum"] == 10.0
    assert result["applied_to_training"] is False


def test_reset_after_finalize():
    controller = AdaptiveBetaController(("layer",))
    controller.observe("layer", 12.0, 10.0, 10.0)
    controller.finalize_all(0)
    assert controller.snapshot("layer")["observation_count"] == 0


def test_layer_states_independent():
    controller = AdaptiveBetaController(("a", "b"))
    controller.observe("a", 12.0, 10.0, 10.0)
    controller.finalize_all(0)
    assert controller.active_beta("a") == 0.2
    assert controller.active_beta("b") == 1.0

