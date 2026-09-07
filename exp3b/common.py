"""Offline helpers for Exp3b; this module never writes to Exp3 inputs."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT.parent / "exp3" / "runs" / "formal_20260906_200824"
DEFAULT_RESULTS = ROOT / "results"
SEEDS = (42, 7, 91)
METHODS = ("dp_sgd", "syn_diag", "dp_kfc")
LAYERS = ("conv1", "conv2", "fc1", "fc2")
WINDOWS = {
    "early": (1, 200),
    "mid": (201, 585),
    "late": (586, 1170),
    "all": (1, 1170),
}
ORACLE_WINDOWS = {
    "early": (0, 200),
    "mid": (201, 585),
    "late": (586, 1170),
    "all": (0, 1170),
}
CONTRIB_COLUMNS = [f"contrib_{layer}" for layer in LAYERS]
SHARE_COLUMNS = [f"share_{layer}" for layer in LAYERS]
ENERGY_METRICS = CONTRIB_COLUMNS + ["contrib_mass", "nearzero_mass_deficit"] + SHARE_COLUMNS + [
    "conv_share", "fc_share", "fc_to_conv", "fc2_to_fc1", "layer_hhi", "layer_entropy"
]
CLIPPING_METRICS = [
    "coefficient_mean", "coefficient_cv", "clipping_alpha_star", "alpha_over_mean_coeff",
    "alpha_negative", "aggregate_cosine", "clipping_shape_error",
    "relative_distortion", "clipped_aggregate_norm", "diagnostic_snr", "norm_mean", "norm_cv",
]
GEOMETRY_METRICS = [
    "G_diag_new", "G_full_new", "G_diag_conv", "G_diag_fc", "G_full_conv", "G_full_fc",
    "diag_deep_gap", "full_deep_gap", "diag_fc_over_conv", "full_fc_over_conv",
]
GEOMETRY_AUC_METRICS = ["full_beneficial_fraction", "full_log_auc", "diag_log_auc"]
EPS = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_manifest(source):
    """Return content hashes for the source tree, for a no-mutation check."""
    source = Path(source)
    files = {}
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = path.relative_to(source).as_posix()
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def source_manifest_digest(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"Missing source file: {path}") from exc


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise ValueError(f"Missing source file: {path}") from exc


def _check_tagged_frame(frame, seed, method, fingerprint, name):
    require(set(("seed", "method", "config_fingerprint")) <= set(frame.columns),
            f"{name} missing provenance columns")
    if len(frame):
        require(set(frame["seed"]) == {seed}, f"{name} seed mismatch")
        require(set(frame["method"]) == {method}, f"{name} method mismatch")
        require(set(frame["config_fingerprint"]) == {fingerprint}, f"{name} fingerprint mismatch")


def validate_contributions(frame):
    """Validate epsilon-floored per-example contribution averages."""
    require(set(CONTRIB_COLUMNS) <= set(frame.columns), "Missing contribution columns")
    values = frame[CONTRIB_COLUMNS].to_numpy(dtype=float)
    require(np.isfinite(values).all(), "Contribution values must be finite")
    require((values >= 0).all(), "Contribution values must be nonnegative")
    mass = values.sum(axis=1)
    require((mass > 0).all(), "Contribution mass must be positive")
    require((mass <= 1 + 1e-5).all(), "Contribution mass exceeds one beyond tolerance")
    return mass


def validate_clipping_inputs(frame):
    required = ("coefficient_mean", "clipping_alpha_star", "aggregate_cosine", "clipping_shape_error")
    require(set(required) <= set(frame.columns), "Missing clipping inputs")
    for column in required:
        values = frame[column].to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"Nonfinite {column}")
    require((frame["coefficient_mean"] > 0).all(), "coefficient_mean must be positive")
    require((frame["clipping_shape_error"] >= 0).all(), "clipping_shape_error must be nonnegative")
    require((frame["aggregate_cosine"].abs() <= 1 + 1e-12).all(), "aggregate_cosine outside [-1,1]")


def validate_alpha_cosine_consistency(frame, tol=1e-12):
    """Check signed projection and cosine signs without imposing alpha >= 0."""
    alpha = frame["clipping_alpha_star"].to_numpy(dtype=float)
    cosine = frame["aggregate_cosine"].to_numpy(dtype=float)
    mask = (np.abs(alpha) > tol) & (np.abs(cosine) > tol)
    require(np.all(np.sign(alpha[mask]) == np.sign(cosine[mask])),
            "alpha_star and aggregate_cosine signs disagree")


def validate_no_inferred_layer_snr(columns):
    forbidden = [c for c in columns if "layer" in c.lower() and "snr" in c.lower()]
    require(not forbidden, f"Exp3b must not infer layer-wise SNR: {forbidden}")


def validate_source(source):
    """Load and strictly validate the immutable formal Exp3 source."""
    source = Path(source).resolve()
    source_validation = _read_json(source / "validation.json")
    require(source_validation.get("passed") is True, "Source Exp3 validation did not pass")
    require(source_validation.get("runs") == 9, "Source Exp3 must contain 9 runs")
    require(source_validation.get("steps_per_run") == 1170, "Source Exp3 must contain 1170 steps per run")

    summary_by_seed = _read_csv(source / "summary_by_seed.csv")
    _read_csv(source / "summary_mean_std.csv")
    _read_csv(source / "paired_deltas.csv")
    require(len(summary_by_seed) == 9, "Source summary_by_seed must contain 9 rows")
    require(set(summary_by_seed.seed) == set(SEEDS), "Source summary seeds mismatch")
    require(set(summary_by_seed.method) == set(METHODS), "Source summary methods mismatch")

    runs = []
    configs = []
    fingerprints = set()
    provenances = set()
    oracle_steps = None
    for seed in SEEDS:
        seed_root = source / f"seed{seed}"
        require(seed_root.is_dir(), f"Missing source seed directory: {seed_root}")
        require({p.name for p in seed_root.iterdir() if p.is_dir()} == set(METHODS),
                f"Source method directories mismatch for seed {seed}")
        for method in METHODS:
            root = seed_root / method
            config = _read_json(root / "config.json")
            metadata = _read_json(root / "metadata.json")
            summary = _read_json(root / "summary.json")
            train = _read_csv(root / "train_metrics.csv")
            oracle = _read_csv(root / "oracle_metrics.csv")
            refresh = _read_csv(root / "refresh_metrics.csv")
            fingerprint = metadata.get("fingerprint")
            require(fingerprint and summary.get("fingerprint") == fingerprint, f"Fingerprint mismatch: {root}")
            require(metadata.get("seed") == seed and metadata.get("method") == method and metadata.get("complete") is True,
                    f"Metadata mismatch: {root}")
            configs.append(json.dumps(config, sort_keys=True))
            fingerprints.add(fingerprint)
            provenances.add(json.dumps(metadata.get("provenance"), sort_keys=True))
            _check_tagged_frame(train, seed, method, fingerprint, f"{root}/train_metrics.csv")
            require(len(train) == 1170 and train.step.tolist() == list(range(1, 1171)),
                    f"Train steps mismatch: {root}")
            validate_contributions(train)
            validate_clipping_inputs(train)
            validate_alpha_cosine_consistency(train)
            required_oracle = {"step", "layer", "R_diag_new", "R_full_new"}
            require(required_oracle <= set(oracle.columns), f"Oracle columns missing: {root}")
            _check_tagged_frame(oracle, seed, method, fingerprint, f"{root}/oracle_metrics.csv")
            require(set(oracle.layer) == set(LAYERS), f"Oracle layers mismatch: {root}")
            steps = sorted(set(oracle.step))
            require(steps == list(range(0, 1170, 50)), f"Oracle steps mismatch: {root}")
            require(len(oracle) == 96, f"Oracle row count mismatch: {root}")
            require(np.isfinite(oracle[["R_diag_new", "R_full_new"]].to_numpy(dtype=float)).all(),
                    f"Nonfinite oracle geometry: {root}")
            require((oracle[["R_diag_new", "R_full_new"]] > 0).all().all(),
                    f"Nonpositive oracle geometry: {root}")
            if oracle_steps is None:
                oracle_steps = steps
            require(steps == oracle_steps, f"Oracle step alignment mismatch: {root}")
            _check_tagged_frame(refresh, seed, method, fingerprint, f"{root}/refresh_metrics.csv")
            if method == "dp_sgd":
                require(refresh.empty, f"DP-SGD refresh must be empty: {root}")
            else:
                require(len(refresh) == 96 and sorted(set(refresh.step)) == oracle_steps,
                        f"Refresh schedule mismatch: {root}")
            runs.append({"seed": seed, "method": method, "root": root, "config": config,
                         "metadata": metadata, "summary": summary, "train": train,
                         "oracle": oracle, "refresh": refresh})

    require(len(set(configs)) == 1, "Source configs differ")
    require(len(fingerprints) == 1, "Source config fingerprints differ")
    require(len(provenances) == 1, "Source upstream provenance differs")
    validate_no_inferred_layer_snr(runs[0]["train"].columns)
    return {"source": source, "validation": source_validation, "runs": runs,
            "fingerprint": next(iter(fingerprints)), "provenance": json.loads(next(iter(provenances))),
            "config": runs[0]["config"]}


def add_derived_train_metrics(train):
    """Add epsilon-weighted layer composition and clipping diagnostics."""
    mass = validate_contributions(train)
    validate_clipping_inputs(train)
    validate_alpha_cosine_consistency(train)
    result = train.copy()
    result["contrib_mass"] = mass
    deficit = 1 - mass
    require((deficit >= -1e-5).all(), "nearzero mass deficit is below tolerance")
    result["nearzero_mass_deficit"] = np.where(deficit < 0, 0., deficit)
    for raw, share in zip(CONTRIB_COLUMNS, SHARE_COLUMNS):
        result[share] = result[raw] / result["contrib_mass"]
    require(np.isfinite(result[SHARE_COLUMNS].to_numpy(dtype=float)).all(), "Nonfinite normalized share")
    require((result[SHARE_COLUMNS] >= 0).all().all(), "Normalized shares must be nonnegative")
    require(np.all(np.abs(result[SHARE_COLUMNS].sum(axis=1) - 1) < 1e-8),
            "Normalized shares do not sum to one")
    result["conv_share"] = result["share_conv1"] + result["share_conv2"]
    result["fc_share"] = result["share_fc1"] + result["share_fc2"]
    result["fc_to_conv"] = result["fc_share"] / (result["conv_share"] + EPS)
    result["fc2_to_fc1"] = result["share_fc2"] / (result["share_fc1"] + EPS)
    result["layer_hhi"] = result[SHARE_COLUMNS].pow(2).sum(axis=1)
    result["layer_entropy"] = -(result[SHARE_COLUMNS] * np.log(result[SHARE_COLUMNS] + EPS)).sum(axis=1) / math.log(4)
    result["alpha_over_mean_coeff"] = result["clipping_alpha_star"] / (result["coefficient_mean"] + EPS)
    result["alpha_negative"] = (result["clipping_alpha_star"] < 0).astype(int)
    validate_no_inferred_layer_snr(result.columns)
    require(np.isfinite(result[ENERGY_METRICS + CLIPPING_METRICS].to_numpy(dtype=float)).all(),
            "Nonfinite derived train metric")
    return result


def geometric_mean(values):
    values = np.asarray(values, dtype=float)
    require(len(values) > 0 and np.isfinite(values).all() and (values > 0).all(),
            "Geometric mean requires finite positive values")
    return float(np.exp(np.log(values).mean()))


def derive_geometry(oracle):
    """Recompute layer and conv-vs-FC geometry from raw oracle rows."""
    rows = []
    for (method, seed, step), group in oracle.groupby(["method", "seed", "step"], sort=True):
        by_layer = {row.layer: row for row in group.itertuples(index=False)}
        require(set(by_layer) == set(LAYERS), f"Missing oracle layer at step {step}")
        diag = {layer: float(by_layer[layer].R_diag_new) for layer in LAYERS}
        full = {layer: float(by_layer[layer].R_full_new) for layer in LAYERS}
        values = np.asarray(list(diag.values()) + list(full.values()), dtype=float)
        require(np.isfinite(values).all(), "Nonfinite geometry")
        require((values > 0).all(), "Nonpositive geometry")
        g_diag_conv = geometric_mean([diag["conv1"], diag["conv2"]])
        g_diag_fc = geometric_mean([diag["fc1"], diag["fc2"]])
        g_full_conv = geometric_mean([full["conv1"], full["conv2"]])
        g_full_fc = geometric_mean([full["fc1"], full["fc2"]])
        row = {"method": method, "seed": int(seed), "step": int(step),
               "G_diag_new": geometric_mean(list(diag.values())),
               "G_full_new": geometric_mean(list(full.values())),
               "G_diag_conv": g_diag_conv, "G_diag_fc": g_diag_fc,
               "G_full_conv": g_full_conv, "G_full_fc": g_full_fc,
               "diag_deep_gap": math.log(g_diag_fc) - math.log(g_diag_conv),
               "full_deep_gap": math.log(g_full_fc) - math.log(g_full_conv),
               "diag_fc_over_conv": g_diag_fc / g_diag_conv,
               "full_fc_over_conv": g_full_fc / g_full_conv}
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["method", "seed", "step"]).reset_index(drop=True)
    require(np.isfinite(result[GEOMETRY_METRICS].to_numpy(dtype=float)).all(), "Nonfinite derived geometry")
    return result


def window_mask(frame, window, oracle=False):
    bounds = ORACLE_WINDOWS if oracle else WINDOWS
    lo, hi = bounds[window]
    return (frame["step"] >= lo) & (frame["step"] <= hi)


def window_summary(frame, metrics, oracle=False):
    rows = []
    for (method, seed), group in frame.groupby(["method", "seed"], sort=True):
        for window in ("early", "mid", "late", "all"):
            selected = group.loc[window_mask(group, window, oracle)]
            require(len(selected) > 0, f"Empty {window} window for {method}/{seed}")
            for metric in metrics:
                values = selected[metric].to_numpy(dtype=float)
                require(np.isfinite(values).all(), f"Nonfinite {metric} in {window}")
                q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
                rows.append({"method": method, "seed": int(seed), "window": window,
                             "metric": metric, "median": float(median), "q25": float(q25),
                             "q75": float(q75), "iqr": float(q75 - q25), "n": len(values)})
    return pd.DataFrame(rows)


def alpha_negative_summary(frame):
    """Summarize negative signed projections and conditional diagnostics."""
    rows = []
    for (method, seed), group in frame.groupby(["method", "seed"], sort=True):
        for window in ("early", "mid", "late", "all"):
            selected = group.loc[window_mask(group, window, oracle=False)]
            require(len(selected) > 0, f"Empty alpha window for {method}/{seed}/{window}")
            negative = selected[selected["clipping_alpha_star"] < 0]
            count = len(negative)
            row = {"method": method, "seed": int(seed), "window": window,
                   "alpha_negative_count": count,
                   "alpha_negative_fraction": float(count / len(selected)),
                   "negative_aggregate_cosine_median": np.nan,
                   "negative_clipping_shape_error_median": np.nan,
                   "negative_norm_cv_median": np.nan,
                   "negative_coefficient_cv_median": np.nan}
            if count:
                for column in ("aggregate_cosine", "clipping_shape_error", "norm_cv", "coefficient_cv"):
                    row[f"negative_{column}_median"] = float(negative[column].median())
            rows.append(row)
    return pd.DataFrame(rows)


def geometry_auc_summary(frame):
    """Summarize discrete oracle-point log AUCs and the beneficial fraction."""
    rows = []
    for (method, seed), group in frame.groupby(["method", "seed"], sort=True):
        for window in ("early", "mid", "late", "all"):
            selected = group.loc[window_mask(group, window, oracle=True)]
            require(len(selected) > 0, f"Empty geometry AUC window: {method}/{seed}/{window}")
            values = {
                "full_beneficial_fraction": float((selected["G_full_new"] < 1).mean()),
                "full_log_auc": float(np.log(selected["G_full_new"]).mean()),
                "diag_log_auc": float(np.log(selected["G_diag_new"]).mean()),
            }
            for metric, value in values.items():
                rows.append({"method": method, "seed": int(seed), "window": window,
                             "metric": metric, "median": value, "q25": value,
                             "q75": value, "iqr": 0.0, "n": len(selected)})
    return pd.DataFrame(rows)


def window_lookup(summary, method, seed, window, metric, field="median"):
    match = summary[(summary.method == method) & (summary.seed == seed) &
                    (summary.window == window) & (summary.metric == metric)]
    require(len(match) == 1, f"Missing summary lookup {method}/{seed}/{window}/{metric}")
    return float(match.iloc[0][field])


def across_seed_mean_std(summary):
    rows = []
    for (method, window, metric), group in summary.groupby(["method", "window", "metric"], sort=True):
        values = group["median"].to_numpy(dtype=float)
        rows.append({"method": method, "window": window, "metric": metric,
                     "mean": float(values.mean()),
                     "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                     "n": len(values)})
    return pd.DataFrame(rows)


def spearman(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or x.rank(method="average").nunique() < 2 or y.rank(method="average").nunique() < 2:
        return float("nan")
    return float(np.corrcoef(x.rank(method="average"), y.rank(method="average"))[0, 1])


def correlation_summary(trajectories, predictors, targets):
    rows = []
    for (method, seed), group in trajectories.groupby(["method", "seed"], sort=True):
        for predictor in predictors:
            for target in targets:
                rho = spearman(group[predictor], group[target])
                rows.append({"method": method, "seed": int(seed), "predictor": predictor,
                             "target": target, "rho": rho,
                             "n": int(group[[predictor, target]].dropna().shape[0])})
    raw = pd.DataFrame(rows)
    aggregates = []
    for (method, predictor, target), group in raw.groupby(["method", "predictor", "target"], sort=True):
        values = group.rho.dropna().to_numpy(dtype=float)
        aggregates.append({"method": method, "seed": "all", "predictor": predictor,
                           "target": target, "rho": float(values.mean()) if len(values) else float("nan"),
                           "rho_std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                           "n": len(values)})
    return pd.concat([raw, pd.DataFrame(aggregates)], ignore_index=True)


def paired_deltas(seed_metrics, metrics):
    pairs = [("dp_kfc", "syn_diag"), ("dp_kfc", "dp_sgd"), ("syn_diag", "dp_sgd")]
    rows = []
    for left, right in pairs:
        for metric in metrics:
            values = []
            for seed in SEEDS:
                a = float(seed_metrics[(seed_metrics.method == left) & (seed_metrics.seed == seed)][metric].iloc[0])
                b = float(seed_metrics[(seed_metrics.method == right) & (seed_metrics.seed == seed)][metric].iloc[0])
                delta = a - b
                values.append(delta)
                rows.append({"pair": f"{left} - {right}", "metric": metric, "seed": seed,
                             "delta": delta, "mean": np.nan, "std": np.nan, "n": np.nan})
            rows.append({"pair": f"{left} - {right}", "metric": metric, "seed": "all",
                         "delta": np.nan, "mean": float(np.mean(values)),
                         "std": float(np.std(values, ddof=1)), "n": len(values)})
    return pd.DataFrame(rows)
