"""Minimal completion, beta timing, Adam metrics, and inherited pairing checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from expv3.common import expected_total_steps, save_json
from expv3.validate_expv3 import _compare_pairing
from expv5.train_expv5 import ARMS, ADAM_CONFIG, ADAM_FIELDS, check_config


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
            train = pd.read_csv(root / "train_metrics.csv")
            layers = pd.read_csv(root / "layer_metrics.csv")
            assert summary["status"] == "completed"
            assert len(train) == summary["completed_steps"] == expected_total_steps(config)
            assert all(meta[k] == v for k, v in ADAM_CONFIG.items())
            assert not meta["gamma_influences_training"]
            assert meta["beta_algorithm_active"] == (arm == ARMS[2])
            assert train.diagnostics_finite.all() and train.oracle_diagnostics_finite.all()
            assert np.isfinite(layers[ADAM_FIELDS]).all().all()
            np.testing.assert_allclose(layers.gradient_norm_into_adam, layers.filtered_gradient_norm)
            evaluations = train.loc[train.test_accuracy.notna(), "step"].tolist()
            records.append((meta, pair, evaluations))
            statuses.append(dict(seed=seed, run_id=arm, status=summary["status"]))
            if arm == ARMS[0]:
                np.testing.assert_allclose(layers.gradient_norm_into_adam, layers.noisy_gradient_norm)
                continue
            np.testing.assert_allclose(layers.filter_gradient_norm_ratio,
                                       layers.filtered_gradient_norm / layers.noisy_gradient_norm)
            control = pd.read_csv(root / "beta_controller_metrics.csv")
            for layer, values in control.groupby("layer"):
                previous = 1.0
                for c in values.itertuples():
                    if c.interval_index == 0 or arm == ARMS[1]:
                        expected = 1.0
                    else:
                        raw = c.numerator_previous / c.denominator_previous
                        expected = raw if np.isfinite(raw) and raw > 0 else previous
                        assert c.beta_source_interval == c.interval_index - 1
                    assert np.isclose(c.beta_train, expected)
                    observed = layers[(layers.layer == layer) & (layers.step // config["K"] == c.interval_index)]
                    np.testing.assert_allclose(observed.beta_train, expected)
                    previous = expected
        base_meta, base_pair, base_eval = records[0]
        for meta, pair, evaluations in records[1:]:
            assert base_meta["initial_model_hash"] == meta["initial_model_hash"]
            assert base_eval == evaluations
            for kind in ("private", "privacy"):
                _compare_pairing(base_pair[kind], pair[kind],
                                 ("step", "batch_indices", "noise_rng_before", "noise_rng_after"))
            _compare_pairing(base_pair["synthetic"], pair["synthetic"],
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
