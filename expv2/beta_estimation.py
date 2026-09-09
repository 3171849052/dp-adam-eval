"""Pure scalar beta-estimation algebra.

Nothing in this module knows about models, optimizers, hooks, or training
control flow.  It only consumes already aggregated scalar measurements.
"""

import math
from collections import defaultdict


EPS = 1e-12

# These are derived research diagnostics. Core measurements are validated
# separately by the artifact validator and remain required to be finite.
BETA_STEP_DIAGNOSTIC_FIELDS = (
    "clean_signal_energy",
    "noisy_gradient_energy",
    "expected_noise_energy",
    "noise_debiased_energy_raw",
    "beta_oracle_step",
    "beta_dp_step_raw",
    "beta_dp_step_positive",
)


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def beta_step_diagnostic_is_finite(row):
    """Return the canonical finiteness flag for one beta step row."""
    return all(_finite(row.get(field)) for field in BETA_STEP_DIAGNOSTIC_FIELDS)


def safe_ratio(numerator, denominator, eps=EPS):
    denominator = float(denominator)
    # Keep ordinary finite ratios exact for auditable toy algebra; the epsilon
    # is only needed for the zero-denominator diagnostic edge case.
    return float(numerator) / (denominator if denominator != 0 else eps)


def noise_variance(sigma, max_grad_norm, batch_size):
    """Per-packed-coordinate variance after batch averaging."""
    return (float(sigma) * float(max_grad_norm) / float(batch_size)) ** 2


def single_step_beta(clean_signal_energy, noisy_gradient_energy, dimension, trace_F, r, eps=EPS):
    """Return oracle and noise-debiased single-step beta diagnostics."""
    expected_noise_energy = float(dimension) * float(r)
    debiased = float(noisy_gradient_energy) - expected_noise_energy
    oracle = safe_ratio(clean_signal_energy, trace_F, eps)
    raw = safe_ratio(debiased, trace_F, eps)
    result = dict(
        clean_signal_energy=float(clean_signal_energy),
        noisy_gradient_energy=float(noisy_gradient_energy),
        expected_noise_energy=expected_noise_energy,
        noise_debiased_energy_raw=debiased,
        beta_oracle_step=oracle,
        beta_dp_step_raw=raw,
        beta_dp_step_positive=max(raw, 0.0),
        beta_dp_step_negative=raw < 0,
    )
    result["diagnostic_valid"] = beta_step_diagnostic_is_finite(result)
    return result


def theoretical_conditional_variance(r, signal_energy, dimension):
    """Var(||s+n||²-d*r | s) for n~N(0,r I)."""
    return 4.0 * float(r) * float(signal_energy) + 2.0 * float(dimension) * float(r) ** 2


def _aggregate(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("cannot aggregate an empty window")
    q_oracle = sum(float(row["clean_signal_energy"]) for row in rows)
    q_noisy = sum(float(row["noisy_gradient_energy"]) for row in rows)
    q_expected = sum(float(row["expected_noise_energy"]) for row in rows)
    q_debiased = sum(float(row["noise_debiased_energy_raw"]) for row in rows)
    denominator = sum(float(row["trace_F"]) for row in rows)
    variance = sum(
        theoretical_conditional_variance(row["r"], row["clean_signal_energy"], row["dimension"])
        for row in rows
    )
    beta_oracle = safe_ratio(q_oracle, denominator)
    beta_dp = safe_ratio(q_debiased, denominator)
    return dict(
        trace_F=denominator,
        oracle_signal_energy_sum=q_oracle,
        noisy_energy_sum=q_noisy,
        expected_noise_energy_sum=q_expected,
        noise_debiased_energy_sum=q_debiased,
        beta_oracle=beta_oracle,
        beta_dp_raw=beta_dp,
        beta_dp_positive=max(beta_dp, 0.0),
        beta_dp_negative=beta_dp < 0,
        estimated_signal_energy_positive=max(q_debiased, 0.0),
        estimated_energy_to_noise_ratio=max(q_debiased, 0.0) / (q_expected + EPS),
        oracle_estimator_sd=math.sqrt(max(variance, 0.0)) / (denominator + EPS),
        oracle_observability_snr=q_oracle / (math.sqrt(max(variance, 0.0)) + EPS),
        conditional_variance_sum=variance,
        sample_count=len(rows),
    )


def aggregate_beta_window(rows):
    """Pool a window using ratio-of-sums, never a mean of beta ratios."""
    return _aggregate(rows)


def _group_step_rows(step_rows):
    groups = defaultdict(list)
    for row in step_rows:
        key = (row["seed"], row["run_id"], row["method"], float(row["learning_rate"]), row["layer"])
        groups[key].append(row)
    return groups


def _chunks(rows, window):
    rows = sorted(rows, key=lambda row: int(row["step"]))
    for start in range(0, len(rows), int(window)):
        chunk = rows[start:start + int(window)]
        if chunk:
            yield start, chunk


def build_interval_rows(step_rows, window):
    """Build per-layer non-overlapping pooled windows from step scalar rows."""
    result = []
    groups = _group_step_rows(step_rows)
    for (seed, run_id, method, learning_rate, layer), rows in groups.items():
        for interval_index, (_, chunk) in enumerate(_chunks(rows, window)):
            aggregate = _aggregate(chunk)
            first, last = int(chunk[0]["step"]), int(chunk[-1]["step"])
            result.append(dict(
                seed=seed, run_id=run_id, method=method, learning_rate=learning_rate,
                interval_index=interval_index, start_step=first, end_step=last,
                n_steps=len(chunk), layer=layer, **aggregate,
            ))
    return sorted(result, key=lambda row: (row["seed"], row["run_id"], row["layer"], row["interval_index"]))


def _interval_groups(interval_rows):
    groups = defaultdict(list)
    for row in interval_rows:
        key = (row["seed"], row["run_id"], row["method"], float(row["learning_rate"]), row["layer"])
        groups[key].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["interval_index"]))
    return groups


def _log_error(prediction, target):
    if prediction > 0 and target > 0 and math.isfinite(prediction) and math.isfinite(target):
        return abs(math.log10(prediction / target))
    return None


def build_lagged_rows(interval_rows):
    """Use interval m's raw DP estimate to predict interval m+1's oracle beta."""
    result = []
    for key, rows in _interval_groups(interval_rows).items():
        for previous, current in zip(rows, rows[1:]):
            raw = previous["beta_dp_raw"]
            oracle = current["beta_oracle"]
            result.append(dict(
                seed=key[0], run_id=key[1], method=key[2], learning_rate=key[3], layer=key[4],
                source_interval=previous["interval_index"], target_interval=current["interval_index"],
                beta_dp_previous_raw=raw, beta_oracle_current=oracle,
                lag_prediction_positive=raw > 0,
                lag_ratio=(raw / oracle if raw > 0 and oracle > 0 else None),
                lag_log10_abs_ratio_error=_log_error(raw, oracle),
            ))
    return sorted(result, key=lambda row: (row["seed"], row["run_id"], row["layer"], row["target_interval"]))


def build_hold_predictor(interval_rows):
    """Offline hold-last-positive predictor for the next interval."""
    result = []
    for key, rows in _interval_groups(interval_rows).items():
        beta_hold = 1.0
        for previous, current in zip(rows, rows[1:]):
            raw = previous["beta_dp_raw"]
            if raw > 0:
                beta_hold = raw
            oracle = current["beta_oracle"]
            result.append(dict(
                seed=key[0], run_id=key[1], method=key[2], learning_rate=key[3], layer=key[4],
                source_interval=previous["interval_index"], target_interval=current["interval_index"],
                beta_hold_predictor=beta_hold, beta_oracle_current=oracle,
                hold_prediction_positive=beta_hold > 0,
                hold_ratio=(beta_hold / oracle if beta_hold > 0 and oracle > 0 else None),
                hold_log10_abs_ratio_error=_log_error(beta_hold, oracle),
            ))
    return sorted(result, key=lambda row: (row["seed"], row["run_id"], row["layer"], row["target_interval"]))


def window_sensitivity(step_rows, window_sizes):
    """Offline recomputation of pooled beta for non-overlapping windows."""
    result = []
    for window in window_sizes:
        for row in build_interval_rows(step_rows, window):
            result.append(dict(
                window_size=int(window), seed=row["seed"], run_id=row["run_id"],
                method=row["method"], learning_rate=row["learning_rate"], layer=row["layer"],
                window_index=row["interval_index"], n_steps=row["n_steps"],
                beta_oracle=row["beta_oracle"], beta_dp_raw=row["beta_dp_raw"],
                negative=row["beta_dp_negative"],
                log_error=_log_error(row["beta_dp_raw"], row["beta_oracle"]),
                trace_F=row["trace_F"], oracle_observability_snr=row["oracle_observability_snr"],
            ))
    return result
