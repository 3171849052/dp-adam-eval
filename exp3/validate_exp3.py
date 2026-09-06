"""Strict protocol, completeness, finite geometry, and cross-run pairing checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from exp3.common import ROOT, LAYERS, read_config, save_json
from exp3.audit_upstream import EXPECTED_UPSTREAM_REPO, canonicalize_repo_url
from exp3.summarize_exp3 import load_runs


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(root, c, output):
    runs = load_runs(root, c)
    require(len(runs) == 3*len(c["seeds"]), "Run count")
    if not c["smoke"]:
        require(len(runs) == 9, "Formal experiment requires 9 runs")
    audits = {}
    oracle_ids = []
    sigmas, epsilons = [], []
    for path, m, s, frames in runs:
        total = (c["train_subset"] or 60000)//c["batch_size"]*c["epochs"]
        require(m["total_steps"] == s["completed_steps"] == m["completed_steps"] == total, f"Steps: {path}")
        if not c["smoke"]:
            require(total == 1170, "Formal private steps")
        require(m["optimizer"] == dict(name="SGD", lr=.1, momentum=0), "Optimizer")
        require(m["accountant"] == "rdp", "Accountant")
        require(m["kfac_loss_reduction"] == "mean" and m["covariance_ridge"] == 1e-5 and m["inverse_root_damping"] == 1e-3, "KFAC convention")
        required_hashes = [f"upstream/{n}.py" for n in ("models", "data", "optimizer", "privacy", "covariance", "recorder", "precondition", "trainer", "types")]
        require(all(k in m["provenance"] for k in required_hashes), "Missing upstream dependency hashes")
        sigmas.append(s["noise_multiplier"])
        epsilons.append(s["epsilon_spent"])
        require(0 < s["epsilon_spent"] <= c["epsilon"]+1e-8, "Privacy budget")
        require(list(frames["train"].step) == list(range(1, total+1)), "Train step sequence")
        schedule = list(range(0, total, c["K"]))
        for name, frame in frames.items():
            if name == "refresh" and m["method"] == "dp_sgd":
                require(frame.empty, "DP-SGD must have no refresh")
                continue
            if name != "train":
                require(sorted(frame.step.unique()) == schedule, f"{name} schedule")
                require(len(frame) == len(schedule)*4 and not frame.duplicated(["step", "layer"]).any(), "Duplicate/missing layer rows")
                for _, group in frame.groupby("step"):
                    require(set(group.layer) == set(LAYERS), "Four layers required")
            required = {
                "train": ["norm_mean", "norm_std", "norm_cv", "coefficient_cv", "aggregate_cosine", "clipping_shape_error", "diagnostic_snr", "test_accuracy", "test_loss"],
                "oracle": ["R_diag_new", "R_full_new", "R_diag_old", "R_full_old", "A_diag_raw", "A_diag_new", "A_diag_old", "S_full_raw", "S_full_new", "S_full_old", "private_delta_stale_diag", "private_delta_stale_full"],
                "refresh": ["M_syn", "M_stale", "A_old_diag", "A_new_diag", "S_old_full", "S_new_full", "delta_stale_diag", "delta_stale_full"],
            }[name]
            require(set(required) <= set(frame.columns), f"Missing required {name} columns")
            for col in frame.columns.difference(["layer", "method", "seed", "config_fingerprint"]):
                values = frame[col]
                optional = np.zeros(len(frame), dtype=bool)
                if name == "train" and col in ("test_accuracy", "test_loss"):
                    optional = ~((frame.step % c["eval_interval"] == 0) | (frame.step == total))
                if name == "refresh" and col in ("A_old_diag", "S_old_full", "delta_stale_diag", "delta_stale_full"):
                    optional = frame.step == 0
                    require(values[optional].isna().all(), "Step zero old/delta must be empty")
                if name == "oracle" and (col.endswith("_old") or col.startswith("private_delta_stale")):
                    optional = frame.step == 0
                    require(values[optional].isna().all(), "Step zero oracle old/delta must be empty")
                require(np.isfinite(values[~optional].to_numpy(dtype=float)).all(), f"Nonfinite/missing {path}/{name}/{col}")
            if name == "oracle":
                for col in ("R_diag", "R_full", "R_diag_new", "R_diag_old", "R_full_new", "R_full_old"):
                    require((frame[col].dropna() > 0).all(), "Ratios must be positive")
                    if m["method"] == "dp_sgd":
                        require(np.allclose(frame[col].dropna(), 1, atol=1e-12), "Identity old/new geometry")
                for kind, prefix in (("diag", "A_diag"), ("full", "S_full")):
                    require(np.allclose(frame[f"R_{kind}"], frame[f"R_{kind}_new"]), "Legacy fresh alias mismatch")
                    for age in ("new", "old"):
                        f = frame if age == "new" else frame[frame.step > 0]
                        require(np.allclose(f[f"R_{kind}_{age}"], f[f"{prefix}_{age}"]/f[f"{prefix}_raw"]), "Oracle ratio mismatch")
                    f = frame[frame.step > 0]
                    require(np.allclose(f[f"private_delta_stale_{kind}"], f[f"{prefix}_old"]-f[f"{prefix}_new"]), "Private staleness subtraction")
                require((frame[["rank_raw", "rank_pre"]] >= 2).all().all(), "Invalid spectrum rank")
                if m["method"] == "dp_sgd":
                    require(np.allclose(frame[["R_diag", "R_full"]], 1, atol=1e-12), "Identity geometry")
            if name == "refresh":
                require((frame.M_syn == c["M_syn"]).all(), "Synthetic budget")
                require((frame.M_stale == c["M_stale"]).all(), "Independent stale probe budget")
                for kind, old, new in (("diag", "A_old_diag", "A_new_diag"), ("full", "S_old_full", "S_new_full")):
                    f = frame[frame.step > 0]
                    require(np.allclose(f[f"delta_stale_{kind}"], f[old]-f[new]), "Staleness subtraction")
        expected_refreshes = 0 if m["method"] == "dp_sgd" else len(schedule)
        require(s["number_of_refreshes"] == expected_refreshes, "Refresh count")
        for k in ("wall_time", "core_wall_time", "total_refresh_time", "mean_refresh_time", "diagnostic_seconds", "peak_cuda_memory_core", "peak_cuda_memory_overall", "peak_cuda_memory_allocated", "peak_cuda_memory_reserved", "preconditioner_state_bytes", "final_test_loss"):
            require(np.isfinite(s[k]) and s[k] >= 0, f"Invalid cost {k}")
        require(np.isclose(s["core_wall_time"], s["wall_time"]-s["diagnostic_seconds"]), "Core time accounting")
        require(s["peak_cuda_memory_core"] <= s["peak_cuda_memory_overall"] == s["peak_cuda_memory_allocated"], "Memory peak accounting")
        require(np.isclose(s["final_test_loss"], frames["train"].iloc[-1].test_loss), "Final test loss")
        if m["method"] == "dp_sgd":
            require(s["preconditioner_state_bytes"] == s["total_refresh_time"] == s["mean_refresh_time"] == 0, "Identity cost")
        else:
            require(s["preconditioner_state_bytes"] > 0, "Missing active state")
            times = frames["refresh"].groupby("step").refresh_seconds.first()
            require(np.isclose(times.sum(), s["total_refresh_time"]) and np.isclose(times.mean(), s["mean_refresh_time"]), "Refresh cost aggregation")
        a = json.loads((path / "pairing.json").read_text())
        require(len(a["private"]) == total and [v["step"] for v in a["private"]] == list(range(1,total+1)), "Private audit steps")
        require(all(len(v["batch_indices"]) == c["batch_size"] for v in a["private"]), "Audit batch size")
        require([v["step"] for v in a["synthetic"]] == ([] if m["method"] == "dp_sgd" else schedule), "Synthetic audit schedule")
        require(all(v["count"] == c["M_syn"] for v in a["synthetic"]), "Audit synthetic budget")
        require([v["step"] for v in a["stale"]] == ([] if m["method"] == "dp_sgd" else schedule), "Stale audit schedule")
        require(all(v["count"] == c["M_stale"] for v in a["stale"]), "Audit stale probe budget")
        require(m["rng_seeds"]["stale"] == m["seed"]+c["stale_seed_offset"] and
                len(set(m["rng_seeds"].values()) | {c["oracle_seed"]}) == len(m["rng_seeds"])+1, "RNG streams must be independent")
        for construction, probe in zip(a["synthetic"], a["stale"]):
            require(construction["rng_before"] != probe["rng_before"] and construction["samples_hash"] != probe["samples_hash"], "Stale probes reuse construction samples")
        for stream in (a["synthetic"], a["stale"]):
            for first, second in zip(stream, stream[1:]):
                require(first["rng_after"] == second["rng_before"], "Synthetic RNG continuity")
        require(a["oracle_indices"] == m["oracle_indices"] and len(set(a["oracle_indices"])) == c["M_oracle"], "Oracle audit")
        oracle_ids.append(a["oracle_indices"])
        audits[m["seed"], m["method"]] = (m, a)
    require(len(set(sigmas)) == len(set(epsilons)) == 1, "Privacy parameters differ")
    require(all(ids == oracle_ids[0] for ids in oracle_ids), "Oracle indices differ across seeds")
    for seed in c["seeds"]:
        reference, ref_a = audits[seed, "dp_sgd"]
        for method in ("syn_diag", "dp_kfc"):
            m, a = audits[seed, method]
            require(m["initial_model_hash"] == reference["initial_model_hash"], "Initial model pairing")
            require(m["device"] == reference["device"] and m["rng_seeds"] == reference["rng_seeds"], "Device/RNG mismatch")
            for x, y in zip(ref_a["private"], a["private"]):
                require(all(x[k] == y[k] for k in ("step", "batch_indices", "noise_rng_before", "noise_rng_after")), "Private/noise pairing")
        require(audits[seed, "syn_diag"][1]["synthetic"] == audits[seed, "dp_kfc"][1]["synthetic"], "Synthetic samples/labels/RNG pairing")
        require(audits[seed, "syn_diag"][1]["stale"] == audits[seed, "dp_kfc"][1]["stale"], "Independent stale probe pairing")
    result = dict(passed=True, runs=len(runs), steps_per_run=runs[0][1]["total_steps"], smoke=c["smoke"],
                  epsilon_spent=epsilons[0], noise_multiplier=sigmas[0], oracle_trajectory_isolation="covered by unit test for all 3 methods",
                  upstream_git_commit=runs[0][1]["upstream_git_commit"], upstream_git_dirty=runs[0][1]["upstream_git_dirty"],
                  upstream_origin=runs[0][1]["upstream_origin"],
                  origin_matches_expected=(canonicalize_repo_url(runs[0][1]["upstream_origin"]) == canonicalize_repo_url(EXPECTED_UPSTREAM_REPO)),
                  warnings=["Dirty upstream accepted for smoke only; formal training/aggregation requires clean checkout"] if runs[0][1]["upstream_git_dirty"] else [],
                  note="Smoke has no strictly late refresh/oracle point when total=4,K=2; corresponding summaries are N/A" if c["smoke"] else "")
    save_json(Path(output) / "validation.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/full.json"))
    p.add_argument("--runs", default=str(ROOT / "runs"))
    p.add_argument("--output", default=str(ROOT))
    a = p.parse_args()
    validate(a.runs, read_config(a.config), a.output)
