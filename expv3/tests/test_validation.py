import pandas as pd
import pytest
import torch

from expv3.adaptive_fisher_wiener import h_hash, selected_stats_digest, stats_digest
from expv3.common import FISHER_METHOD, LAYERS
from expv3.validate_expv3 import _validate_controller


def _controller_frame():
    rows = []
    for interval in (0, 1):
        for layer in LAYERS:
            h = .5 if interval == 0 else 1. / 3.
            stats = stats_digest({"H_mean": h, "H_std": 0., "H_min": h, "H_max": h,
                                  "H_q10": h, "H_q25": h, "H_q50": h, "H_q75": h, "H_q90": h})
            compact = selected_stats_digest({"H_mean": .5, "H_q10": .5, "H_q50": .5, "H_q90": .5})
            rows.append({
                "refresh_step": interval * 2, "interval_index": interval, "layer": layer,
                "beta_train": 1.0 if interval == 0 else .5,
                "beta_source_interval": None if interval == 0 else 0,
                "beta_raw_previous": None if interval == 0 else .5,
                "beta_update_accepted": interval == 1,
                "beta_fallback_used": False, "beta_fallback_reason": "initial" if interval == 0 else "none",
                "previous_beta_train": 1.0 if interval == 0 else 1.0,
                "numerator_previous": None if interval == 0 else 1.,
                "denominator_previous": None if interval == 0 else 2.,
                "previous_interval_n_steps": 0 if interval == 0 else 2,
                "accepted_update_count": 0 if interval == 0 else 1, "fallback_count": 0,
                "trace_A": 1., "trace_G": 1., "trace_F": 1.,
                "beta_scaled_trace_F": 1. if interval == 0 else .5,
                "H_mean": h, "H_std": 0., "H_min": h, "H_max": h,
                "H_q10": h, "H_q25": h, "H_q50": h, "H_q75": h, "H_q90": h,
                "H_beta1_mean": .5, "H_beta1_q10": .5, "H_beta1_q50": .5, "H_beta1_q90": .5,
                "H_fro_ratio_vs_beta1": 1., "H_hash": h_hash(torch.tensor([[h]])),
                "H_hash_copy": h_hash(torch.tensor([[h]])),
                "H_beta1_hash": h_hash(torch.tensor([[.5]])),
                "H_beta1_hash_copy": h_hash(torch.tensor([[.5]])),
                "H_stats_digest": stats, "H_beta1_stats_digest": compact,
            })
    return pd.DataFrame(rows)


def _valid_inputs():
    frame = _controller_frame()
    interval = pd.DataFrame([{"interval_index": i, "layer": layer, "beta_dp_raw": .5, "n_steps": 2}
                             for i in (0, 1) for layer in LAYERS])
    refresh = pd.DataFrame([{"refresh_step": 0, "refresh_index": 0, "measurement_only": False},
                            {"refresh_step": 2, "refresh_index": 1, "measurement_only": False}])
    certificates = [{"interval_index": i, "layer": layer,
                     "H_hash": h_hash(torch.tensor([[.5 if i == 0 else 1. / 3.]])),
                     "H_beta1_hash": h_hash(torch.tensor([[.5]])),
                     "lambda_A": [1.], "lambda_G": [1.], "r": 1.,
                     "beta_train": 1. if i == 0 else .5}
                    for i in (0, 1) for layer in LAYERS]
    return frame, interval, refresh, certificates


def test_validator_lag_corruption():
    frame, interval, refresh, certificates = _valid_inputs()
    _validate_controller({}, frame, interval, FISHER_METHOD, refresh, certificates)
    frame.loc[(frame.interval_index == 1), "beta_source_interval"] = 1
    with pytest.raises(AssertionError):
        _validate_controller({}, frame, interval, FISHER_METHOD, refresh, certificates)


def test_validator_h_corruption():
    frame, interval, refresh, certificates = _valid_inputs()
    _validate_controller({}, frame, interval, FISHER_METHOD, refresh, certificates)
    frame.loc[0, "H_hash"] = "corrupt"
    with pytest.raises(AssertionError):
        _validate_controller({}, frame, interval, FISHER_METHOD, refresh, certificates)
