import inspect

from expv3 import train_expv3


def test_pre_wiener_y():
    source = inspect.getsource(train_expv3.private_update)
    assert source.index("beta_capture") < source.index("apply_fisher_wiener")
    assert source.index("controller.observe") < source.index("apply_fisher_wiener")


def test_actual_noise_excluded():
    signature = inspect.signature(train_expv3.capture_deployable_beta_observations)
    forbidden = {"actual_noise", "noise_tensor", "clean_signal_energy", "summed_grad"}
    assert not forbidden.intersection(signature.parameters)
    assert "summed_grad" not in inspect.getsource(
        train_expv3.capture_deployable_beta_observations
    )
    assert not forbidden.intersection(inspect.signature(train_expv3.beta_capture).parameters)
    controller_signature = inspect.signature(__import__("expv3.beta_controller", fromlist=["AdaptiveBetaController"]).AdaptiveBetaController.observe)
    assert list(controller_signature.parameters) == [
        "self", "layer", "noisy_energy", "expected_noise_energy", "trace_F"
    ]
