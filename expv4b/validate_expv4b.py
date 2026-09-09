"""Check only experiment-defining math, timing, completion, and inherited pairing."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from expv3.common import expected_total_steps, save_json
from expv3.validate_expv3 import _compare_pairing
from expv4a.gamma_estimators import model_gamma_from_eigenvalues
from expv4b.train_expv4b import ARMS, check_config


def load(path):
    return json.loads(Path(path).read_text())


def validate(config, runs, output):
    check_config(config)
    statuses = []
    for seed in config["seeds"]:
        records = []
        for arm in ARMS:
            root = Path(runs) / f"seed{seed}" / arm
            meta, summary, pair = [load(root / name) for name in
                                   ("metadata.json", "summary.json", "pairing.json")]
            records.append((meta, pair))
            assert meta["beta_algorithm_active"] == summary["beta_algorithm_active"] == (arm == ARMS[2])
            statuses.append(dict(seed=seed, run_id=arm, status=summary["status"],
                                 divergence_stage=summary["divergence_stage"]))
            if config["smoke"]:
                assert summary["status"] == "completed", (seed, arm, summary["divergence_stage"])
            if summary["status"] == "completed":
                assert summary["completed_steps"] == expected_total_steps(config)
            else:
                assert summary["divergence_stage"] in (
                    "loss", "noisy_gradient", "filtered_gradient", "compensated_gradient", "parameters")
            train = pd.read_csv(root / "train_metrics.csv")
            if len(train):
                assert train.diagnostics_finite.all() and train.oracle_diagnostics_finite.all()
            if arm == ARMS[0]:
                assert not meta["gamma_influences_training"]
                continue
            assert meta["gamma_influences_training"]
            refresh = pd.read_csv(root / "gamma_refresh_metrics.csv")
            control = pd.read_csv(root / "beta_controller_metrics.csv")
            layers = pd.read_csv(root / "layer_metrics.csv")
            certs = load(root / "h_certificates.json")
            assert len(refresh) == len(control) == len(certs)
            if config["smoke"]:
                assert refresh.interval_index.nunique() >= 2
            for cert in certs:
                interval, layer = cert["interval_index"], cert["layer"]
                r = refresh[(refresh.interval_index == interval) & (refresh.layer == layer)].iloc[0]
                c = control[(control.interval_index == interval) & (control.layer == layer)].iloc[0]
                assert r.beta_train == c.beta_train == cert["beta_train"]
                if interval == 0 or arm == ARMS[1]:
                    assert c.beta_train == 1.0
                else:
                    prev = control[(control.interval_index == interval-1) & (control.layer == layer)].iloc[0]
                    raw = c.numerator_previous / c.denominator_previous
                    expected = raw if math.isfinite(raw) and raw > 0 else prev.beta_train
                    assert np.isclose(c.beta_train, expected)
                    assert c.beta_source_interval == interval - 1
                expected_gamma = model_gamma_from_eigenvalues(
                    cert["lambda_A"], cert["lambda_G"], cert["beta_train"], cert["r"])["gamma_model_raw"]
                assert np.isclose(r.gamma_model, expected_gamma, rtol=2e-6)
                values = layers[(layers.layer == layer) & (layers.step // config["K"] == interval)]
                np.testing.assert_allclose(values.beta_train, r.beta_train)
                np.testing.assert_allclose(values.gamma_model, r.gamma_model)
                np.testing.assert_allclose(values.gamma_update_ratio, r.gamma_model, rtol=2e-6)
                np.testing.assert_allclose(values.signal_retention_after_gamma,
                                           values.signal_retention * r.gamma_model**2)
                np.testing.assert_allclose(values.noise_retention_after_gamma,
                                           values.noise_retention * r.gamma_model**2)
            if len(layers):
                assert np.isfinite(layers[["gamma_model", "compensated_gradient_norm"]]).all().all()
                np.testing.assert_allclose(layers.optimizer_update_norm,
                                           0.5 * layers.compensated_gradient_norm)
        base_meta, base_pair = records[0]
        for meta, pair in records[1:]:
            assert base_meta["initial_model_hash"] == meta["initial_model_hash"]
            for kind in ("private", "privacy"):
                common = min(len(base_pair[kind]), len(pair[kind]))
                _compare_pairing(base_pair[kind][:common], pair[kind][:common],
                                 ("step", "batch_indices", "noise_rng_before", "noise_rng_after"))
            common = min(len(base_pair["synthetic"]), len(pair["synthetic"]))
            _compare_pairing(base_pair["synthetic"][:common], pair["synthetic"][:common],
                             ("step", "samples_hash", "labels_hash", "rng_before", "rng_after"))
    result = dict(passed=True, pairing_passed=True, runs=statuses)
    save_json(Path(output) / "validation.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for arg in ("config", "runs", "output"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    validate(load(args.config), args.runs, args.output)


if __name__ == "__main__":
    main()
