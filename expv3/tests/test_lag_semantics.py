import torch

from expv3.adaptive_fisher_wiener import rebuild_H_with_beta
from expv3.beta_controller import AdaptiveBetaController


def test_no_current_batch_leakage():
    controller = AdaptiveBetaController(("layer",))
    before = controller.active_beta("layer")
    active_before = rebuild_H_with_beta(
        {"layer": {"lambda_A": torch.ones(1), "lambda_G": torch.ones(1), "H": torch.ones(1),
                    "Q_A": torch.eye(1), "Q_G": torch.eye(1)}}, {"layer": before}, 1.0)
    # A current batch is not an argument to either refresh construction or beta.
    controller.observe("layer", 1e9, 0.0, 1.0)
    active_after_observe = rebuild_H_with_beta(
        {"layer": {"lambda_A": torch.ones(1), "lambda_G": torch.ones(1), "H": torch.ones(1),
                    "Q_A": torch.eye(1), "Q_G": torch.eye(1)}}, {"layer": before}, 1.0)
    assert torch.equal(active_before["layer"]["H"], active_after_observe["layer"]["H"])


def test_observation_only_affects_future_interval():
    controller = AdaptiveBetaController(("layer",))
    assert controller.active_beta("layer") == 1.0
    controller.observe("layer", 12.0, 10.0, 10.0)
    assert controller.active_beta("layer") == 1.0
    controller.finalize_all(0)
    assert controller.active_beta("layer") == 0.2

