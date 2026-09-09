import pytest
from dp_wiener_mnist.config import Config, load_config
from pathlib import Path


@pytest.mark.parametrize("path", list(Path("config").rglob("*.yaml")))
def test_configs(path):
    load_config(path).validate()


@pytest.mark.parametrize(
    "raw",
    [
        {"algorithm": "dp_adam"},
        {"bad": 1},
        {"seed": True},
        {"data": {"batch_size": 0}},
        {"data": {"dataset": "qnli"}},
        {"model": {"name": "bert"}},
        {"training": {"momentum": 0.1}},
        {"training": {"weight_decay": 0.1}},
        {"training": {"learning_rate": float("nan")}},
        {"privacy": {"delta": 1}},
        {"privacy": {"poisson_sampling": False}},
        {"privacy": {"sampling": "fixed_shuffle_drop_last"}},
        {"privacy": {"accounting_convention": "inherited_rdp_sample_rate_convention"}},
        {"logging": {"eval_interval": 100}},
        {"privacy": {"accountant": "gdp"}},
        {"wiener": {"beta": 2}},
        {"wiener": {"synthetic_samples": 3}},
        {"wiener": {"covariance_ridge": 0.1}},
        {"runtime": {"deterministic": False}},
    ],
)
def test_reject(raw):
    with pytest.raises(ValueError):
        Config.from_dict(raw)


def test_poisson_defaults_are_strict():
    c = Config()
    c.validate()
    assert c.privacy.sampling == "poisson"
    assert c.privacy.accounting_convention == "poisson_rdp"
    assert c.privacy.poisson_sampling is True
