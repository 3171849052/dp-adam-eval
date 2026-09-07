import json

import numpy as np
import pandas as pd
import pytest

from exp3b.common import (
    CONTRIB_COLUMNS,
    LAYERS,
    METHODS,
    SEEDS,
    across_seed_mean_std,
    add_derived_train_metrics,
    correlation_summary,
    derive_geometry,
    geometry_auc_summary,
    geometric_mean,
    paired_deltas,
    require,
    spearman,
    validate_contributions,
    validate_alpha_cosine_consistency,
    alpha_negative_summary,
    validate_no_inferred_layer_snr,
    validate_source,
    window_mask,
    window_summary,
)


def train_frame(steps=(1, 2, 200, 201, 585, 586, 1170)):
    rows = []
    for step in steps:
        row = dict(method="dp_kfc", seed=42, step=step, coefficient_mean=2., clipping_alpha_star=1.,
                   coefficient_cv=.5, aggregate_cosine=.8, clipping_shape_error=.2,
                   relative_distortion=.2, clipped_aggregate_norm=1., diagnostic_snr=.1,
                   norm_mean=2., norm_cv=.5)
        row.update(dict(zip(CONTRIB_COLUMNS, [.1, .2, .3, .4])))
        rows.append(row)
    return pd.DataFrame(rows)


def geometry_frame():
    rows = []
    values = {"conv1": .5, "conv2": .5, "fc1": .25, "fc2": .25}
    for layer in LAYERS:
        rows.append(dict(method="dp_kfc", seed=42, step=0, layer=layer,
                         R_diag_new=values[layer], R_full_new=values[layer] * 2))
    return pd.DataFrame(rows)


def test_contribution_sum_validation():
    mass = validate_contributions(train_frame())
    assert np.allclose(mass, 1.)
    bad = train_frame()
    bad.loc[0, "contrib_fc1"] = .9
    with pytest.raises(ValueError, match="exceeds one"):
        validate_contributions(bad)


def test_contribution_sum_below_one_is_legal():
    frame = train_frame()
    frame[CONTRIB_COLUMNS] *= .5
    assert np.allclose(validate_contributions(frame), .5)


def test_fc_share():
    result = add_derived_train_metrics(train_frame())
    assert result.loc[0, "conv_share"] == pytest.approx(.3)
    assert result.loc[0, "fc_share"] == pytest.approx(.7)


def test_fc_to_conv():
    frame = train_frame()
    frame[CONTRIB_COLUMNS] *= .5
    result = add_derived_train_metrics(frame)
    assert result.loc[0, "fc_to_conv"] == pytest.approx(7 / 3)


def test_normalized_share_sum_and_conv_fc_shares():
    frame = train_frame()
    frame[CONTRIB_COLUMNS] *= .5
    result = add_derived_train_metrics(frame)
    assert result.loc[0, "contrib_mass"] == pytest.approx(.5)
    assert result.loc[0, ["share_conv1", "share_conv2", "share_fc1", "share_fc2"]].sum() == pytest.approx(1.)
    assert result.loc[0, "conv_share"] == pytest.approx(.3)
    assert result.loc[0, "fc_share"] == pytest.approx(.7)


def test_hhi():
    frame = train_frame()
    frame[CONTRIB_COLUMNS] *= .5
    result = add_derived_train_metrics(frame)
    assert result.loc[0, "layer_hhi"] == pytest.approx(.3)


def test_normalized_entropy():
    frame = train_frame()
    frame[CONTRIB_COLUMNS] *= .5
    result = add_derived_train_metrics(frame)
    expected = -sum(x * np.log(x + 1e-12) for x in [.1, .2, .3, .4]) / np.log(4)
    assert result.loc[0, "layer_entropy"] == pytest.approx(expected)


def test_geometry_layer_macros():
    result = derive_geometry(geometry_frame())
    assert result.loc[0, "G_diag_conv"] == pytest.approx(.5)
    assert result.loc[0, "G_diag_fc"] == pytest.approx(.25)
    assert result.loc[0, "G_full_conv"] == pytest.approx(1.)
    assert result.loc[0, "G_full_fc"] == pytest.approx(.5)


def test_geometric_mean():
    assert geometric_mean([1, 4]) == pytest.approx(2)
    with pytest.raises(ValueError):
        geometric_mean([0, 1])


def test_deep_gap():
    result = derive_geometry(geometry_frame())
    assert result.loc[0, "diag_deep_gap"] == pytest.approx(np.log(.5))
    assert result.loc[0, "diag_fc_over_conv"] == pytest.approx(.5)


def test_window_boundaries():
    frame = train_frame()
    assert frame.loc[window_mask(frame, "early"), "step"].tolist() == [1, 2, 200]
    assert frame.loc[window_mask(frame, "mid"), "step"].tolist() == [201, 585]
    assert frame.loc[window_mask(frame, "late"), "step"].tolist() == [586, 1170]


def test_log_auc():
    frame = pd.concat([geometry_frame().assign(step=step) for step in range(0, 1170, 50)], ignore_index=True)
    result = geometry_auc_summary(derive_geometry(frame))
    value = result[(result.window == "all") & (result.metric == "full_log_auc")].iloc[0]["median"]
    assert value == pytest.approx(np.log(2 ** -0.5))


def test_beneficial_fraction():
    raw = pd.concat([geometry_frame().assign(step=step) for step in range(0, 1170, 50)], ignore_index=True)
    frame = derive_geometry(raw)
    result = geometry_auc_summary(frame)
    value = result[(result.window == "early") & (result.metric == "full_beneficial_fraction")].iloc[0]["median"]
    assert value == pytest.approx(1.)


def test_alpha_ratio():
    result = add_derived_train_metrics(train_frame())
    assert result.loc[0, "alpha_over_mean_coeff"] == pytest.approx(.5)


@pytest.mark.parametrize("alpha", [1., 0., -1.])
def test_signed_alpha_is_legal(alpha):
    frame = train_frame()
    frame["clipping_alpha_star"] = alpha
    frame["aggregate_cosine"] = np.sign(alpha) * .8 if alpha else 0.
    result = add_derived_train_metrics(frame)
    assert np.allclose(result["alpha_over_mean_coeff"], alpha / 2.)


@pytest.mark.parametrize("alpha", [np.nan, np.inf, -np.inf])
def test_nonfinite_alpha_rejected(alpha):
    frame = train_frame()
    frame.loc[0, "clipping_alpha_star"] = alpha
    with pytest.raises(ValueError, match="Nonfinite clipping_alpha_star"):
        add_derived_train_metrics(frame)


@pytest.mark.parametrize("alpha,cosine,passes", [
    (-.5, -.2, True), (-.5, .2, False), (.5, -.2, False),
])
def test_alpha_cosine_sign_consistency(alpha, cosine, passes):
    frame = train_frame(steps=(1,))
    frame.loc[0, "clipping_alpha_star"] = alpha
    frame.loc[0, "aggregate_cosine"] = cosine
    if passes:
        validate_alpha_cosine_consistency(frame)
    else:
        with pytest.raises(ValueError, match="signs disagree"):
            validate_alpha_cosine_consistency(frame)


def test_alpha_negative_fraction_and_na_conditional_summary():
    frame = train_frame(steps=(1, 201, 586, 1170))
    frame["clipping_alpha_star"] = [0., 1., 0., 0.]
    frame["aggregate_cosine"] = [0., .8, 0., 0.]
    result = alpha_negative_summary(frame)
    row = result[(result.window == "early")].iloc[0]
    assert row.alpha_negative_fraction == pytest.approx(0.)
    assert row.alpha_negative_count == 0
    assert np.isnan(row.negative_aggregate_cosine_median)


def test_nearzero_mass_deficit_and_exact_mass():
    exact = add_derived_train_metrics(train_frame())
    assert exact.loc[0, "nearzero_mass_deficit"] == pytest.approx(0.)
    floored = train_frame()
    floored[CONTRIB_COLUMNS] *= .2
    result = add_derived_train_metrics(floored)
    assert result.loc[0, "nearzero_mass_deficit"] == pytest.approx(.8)


def test_zero_mass_rejected():
    frame = train_frame()
    frame[CONTRIB_COLUMNS] = 0.
    with pytest.raises(ValueError, match="positive"):
        add_derived_train_metrics(frame)


def test_window_median_iqr():
    frame = pd.DataFrame(dict(method="dp_kfc", seed=42, step=list(range(1, 1171)),
                              x=np.linspace(1., 6., 1170)))
    result = window_summary(frame, ["x"])
    row = result[(result.window == "all") & (result.metric == "x")].iloc[0]
    assert row["median"] == pytest.approx(3.5)
    assert row.iqr == pytest.approx(2.5)


def test_paired_deltas():
    rows = [{"method": method, "seed": seed, "x": value}
            for seed in SEEDS for method, value in zip(METHODS, [1., 2., 4.])]
    result = paired_deltas(pd.DataFrame(rows), ["x"])
    row = result[(result.pair == "dp_kfc - dp_sgd") & (result.seed == "all")].iloc[0]
    assert row["mean"] == pytest.approx(3.)
    assert row["std"] == pytest.approx(0.)
    assert row["n"] == 3


def test_spearman():
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.)
    assert np.isnan(spearman([1, 1], [1, 2]))


def test_nearzero_temporal_correlation():
    frame = train_frame(steps=range(1, 8))
    frame["nearzero_mass_deficit"] = np.arange(len(frame), dtype=float)
    frame["norm_cv"] = np.arange(len(frame), dtype=float)
    result = correlation_summary(frame, ("nearzero_mass_deficit",), ("norm_cv",))
    assert result.iloc[0]["rho"] == pytest.approx(1.)


def test_n3_mean_std():
    frame = pd.DataFrame(dict(method="dp_kfc", window="late", metric="x",
                              median=[1., 2., 3.], q25=0., q75=0., iqr=0., n=1, seed=SEEDS))
    result = across_seed_mean_std(frame).iloc[0]
    assert result["mean"] == pytest.approx(2.)
    assert result["std"] == pytest.approx(1.)
    assert result["n"] == 3


def test_missing_nan_rejection():
    bad = train_frame()
    bad.loc[0, "contrib_fc1"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        add_derived_train_metrics(bad)


def test_source_validation_rejection(tmp_path):
    (tmp_path / "validation.json").write_text(json.dumps({"passed": False, "runs": 9, "steps_per_run": 1170}))
    with pytest.raises(ValueError, match="did not pass"):
        validate_source(tmp_path)


def test_no_layerwise_snr_inferred():
    validate_no_inferred_layer_snr(["fc_share", "diagnostic_snr"])
    with pytest.raises(ValueError, match="layer-wise SNR"):
        validate_no_inferred_layer_snr(["contrib_fc1_layer_snr"])


def test_deterministic_outputs():
    first = add_derived_train_metrics(train_frame())
    second = add_derived_train_metrics(train_frame())
    pd.testing.assert_frame_equal(first, second)
