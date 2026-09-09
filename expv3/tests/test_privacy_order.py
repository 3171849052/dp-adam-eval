import inspect

from expv3 import train_expv3


def test_pre_wiener_y():
    source = inspect.getsource(train_expv3.private_update)
    assert source.index("beta_capture") < source.index("apply_fisher_wiener")
    assert source.index("controller.observe") < source.index("apply_fisher_wiener")


def test_actual_noise_excluded():
    signature = inspect.signature(train_expv3.beta_capture)
    forbidden = {"actual_noise", "noise_tensor", "clean_signal_energy", "summed_grad"}
    assert not forbidden.intersection(signature.parameters)
    controller_signature = inspect.signature(__import__("expv3.beta_controller", fromlist=["AdaptiveBetaController"]).AdaptiveBetaController.observe)
    assert not forbidden.intersection(controller_signature.parameters)
