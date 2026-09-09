import pytest

from expv3.common import threshold_step
from expv3.train_expv3 import DivergenceError


def test_divergence_semantics():
    error = DivergenceError(7, "filtered gradient")
    assert error.step == 7 and error.stage == "filtered gradient"
    assert threshold_step([{"step": 0, "test_accuracy": 0.5}], 0.5) == 1
    with pytest.raises(ValueError):
        raise ValueError("research diagnostics must not be divergence")

