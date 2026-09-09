"""Summarize ExpV4a gamma diagnostics without using accuracy as a criterion."""

import argparse
import json
from pathlib import Path

import pandas as pd

from expv3.common import save_json, write_csv

from expv4a.train_expv4a import check_config, run_specs
from expv4a.validate_expv4a import validate


METRICS = (
    "gamma_model", "gamma_oracle", "model_multiplicative_error",
    "model_to_oracle_ratio",
    "signal_retention_oracle", "signal_retention_after_model_gamma",
    "noise_retention_after_model_gamma",
)


def _stats(frame, scope):
    row = {"scope": scope, "n_intervals": int(len(frame))}
    for metric in METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        row[f"{metric}_median"] = float(values.median()) if len(values) else None
        row[f"{metric}_q10"] = float(values.quantile(.10)) if len(values) else None
        row[f"{metric}_q90"] = float(values.quantile(.90)) if len(values) else None
    valid = frame.gamma_dp_valid.astype(str).str.lower().isin(("true", "1"))
    row["gamma_dp_valid_rate"] = float(valid.mean()) if len(frame) else None
    return row


def summarize(config, runs, output, require_tests=True):
    check_config(config)
    runs, output = Path(runs), Path(output)
    validate(config, runs, output, require_tests=require_tests)
    layer_rows, overall_frames = [], []
    for seed in config["seeds"]:
        for spec in run_specs(config):
            path = runs / f"seed{seed}" / spec["run_id"] / "gamma_interval_metrics.csv"
            frame = pd.read_csv(path)
            overall_frames.append(frame)
            for layer, group in frame.groupby("layer", sort=True):
                row = _stats(group, layer)
                row.update(seed=seed, run_id=spec["run_id"])
                layer_rows.append(row)
    overall = _stats(pd.concat(overall_frames, ignore_index=True), "overall")
    write_csv(output / "summary_gamma_layers.csv", layer_rows)
    write_csv(output / "summary_gamma_overall.csv", [overall])
    result = {"layers": layer_rows, "overall": overall}
    save_json(output / "summary_gamma.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = check_config(json.loads(Path(args.config).read_text()))
    result = summarize(config, args.runs, args.output)
    print(json.dumps(result["overall"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
