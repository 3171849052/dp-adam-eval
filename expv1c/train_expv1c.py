"""ExpV1c harness. Experiment-loop glue adapted from ExpV1b; math and updates are shared."""

import argparse
import time

import torch
import torch.nn.functional as F
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader, Subset

from expv1c.common import (
    ROOT,
    check_config,
    datasets,
    digest,
    expected_total_steps,
    fingerprint,
    output_path,
    provenance,
    require_pinned,
    run_spec,
    run_specs,
    save_json,
    set_seed,
    threshold_steps,
    write_csv,
    RNGStream,
    SimpleCNN,
    read_config,
)
from expv1.fisher_wiener import (
    build_covariances,
    build_diagnostic_state,
    build_fisher_state,
    state_bytes,
    synthetic_samples,
)
from expv1b.train_expv1b import (
    private_update, make_optimizer, evaluate, DivergenceError, Indexed, sync,
    _all_finite, _numeric_values_finite, _summary_from_rows,
    TRAIN_FIELDS, LAYER_FIELDS, REFRESH_FIELDS, EIGEN_FIELDS,
)

def train(c, run_id, output, data_override=None, *, diagnostics=True, eigen_budget=0):
    """Train one canonical run.  ``output`` is always confined to expv1c/runs."""
    check_config(c)
    spec = run_spec(run_id, c)
    output = output_path(output)
    root = output / f"seed{c['seed']}" / run_id
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / "config.json", c)

    current_provenance = provenance()
    require_pinned(current_provenance, c["smoke"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if c["device"] == "auto" else torch.device(c["device"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    source, test = datasets(c) if data_override is None else data_override
    n = c["train_subset"] or len(source)
    if not c["batch_size"] <= n <= len(source):
        raise ValueError("Invalid training subset")
    data = Subset(source, range(n))
    if c["test_subset"] is not None:
        if not 1 <= c["test_subset"] <= len(test):
            raise ValueError("Invalid test subset")
        test = Subset(test, range(c["test_subset"]))
    loader = DataLoader(
        Indexed(data), batch_size=c["batch_size"], shuffle=True, drop_last=True,
        num_workers=0, generator=torch.Generator().manual_seed(c["seed"] + 1),
    )
    test_loader = DataLoader(
        test, batch_size=c["batch_size"], num_workers=0,
        generator=torch.Generator().manual_seed(c["seed"] + 2),
    )
    total, q = len(loader) * c["epochs"], c["batch_size"] / n
    sigma = get_noise_multiplier(
        target_epsilon=c["epsilon"], target_delta=c["delta"], sample_rate=q,
        steps=total, accountant="rdp",
    )

    # Keep this initialization point identical to ExpV1 for trajectory regression.
    set_seed(c["seed"])
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    optimizer = make_optimizer(model, spec["learning_rate"], c)
    accountant = RDPAccountant()
    synthetic_rng = RNGStream(c["seed"] + 3, device)
    noise_rng = RNGStream(c["seed"] + 4, device)
    meta = dict(
        seed=c["seed"], run_id=run_id, method=spec["method"], learning_rate=spec["learning_rate"],
        fingerprint=fingerprint(c), provenance=current_provenance, device=str(device),
        noise_multiplier=sigma, sample_rate=q, total_steps=total,
        target_epsilon=c["epsilon"], target_delta=c["delta"],
        accountant="rdp", max_grad_norm=c["max_grad_norm"],
        initial_model_hash=digest(model.parameters()), beta=1, covariance_ridge=1e-5,
        rng_seeds=dict(init=c["seed"], loader=c["seed"] + 1, test=c["seed"] + 2,
                       synthetic=c["seed"] + 3, noise=c["seed"] + 4),
        sampling="fixed shuffle/drop_last; inherited RDP convention, not Poisson",
        diagnostics_enabled=diagnostics, eigen_budget=0, eigenmode_diagnostics=False,
        diagnostics="Non-DP research statistics, never consumed by training",
        output=str(root), optimizer="SGD", momentum=0,
        filter_position="after global clipping and DP Gaussian noise",
    )
    save_json(root / "metadata.json", meta)

    rows, layers, bins, refreshes, audits, synthetic_audits = [], [], [], [], [], []
    active = diagnostic_state = None
    step = 0
    diagnostic_seconds = 0.0
    status = "completed"
    diverged_step = None
    last_finite_accuracy = None
    last_finite_loss = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(model)
    started = time.perf_counter()
    try:
        for epoch in range(1, c["epochs"] + 1):
            iterator = iter(loader)
            for _ in range(len(loader)):
                refreshed = step % c["K"] == 0 and spec["method"] == "dp_fisher_wiener"
                refresh_time = 0.0
                diagnostic_spectrum_time = 0.0
                # Refresh happens before consuming the next private batch.
                if refreshed:
                    sync(model)
                    refresh_started = time.perf_counter()
                    samples, synthetic_audit = synthetic_samples(c, device, synthetic_rng)
                    covariances = build_covariances(model._module.state_dict(), samples, c, device)
                    active = build_fisher_state(covariances, sigma, c["max_grad_norm"], c["batch_size"])
                    sync(model)
                    refresh_time = time.perf_counter() - refresh_started
                    if diagnostics:
                        sync(model)
                        spectrum_started = time.perf_counter()
                        diagnostic_state = build_diagnostic_state(covariances, active, spec["method"])
                        sync(model)
                        diagnostic_spectrum_time = time.perf_counter() - spectrum_started
                        diagnostic_seconds += diagnostic_spectrum_time
                    else:
                        diagnostic_state = None
                    del samples, covariances
                    refreshes.append(dict(
                        step=step, refresh_time=refresh_time,
                        diagnostic_spectrum_time=diagnostic_spectrum_time,
                        active_state_bytes=state_bytes(active),
                    ))
                    synthetic_audits.append(dict(step=step, **synthetic_audit))

                x, y, indices = next(iterator)
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
                if not bool(torch.isfinite(loss)):
                    status, diverged_step = "diverged", step
                    break
                loss.backward()
                try:
                    row, layer_rows, eigen_rows, audit, elapsed = private_update(
                        model, active, optimizer, noise_rng, sigma, c, len(x), spec["method"],
                        spec["learning_rate"], diagnostics, refreshed, 0, diagnostic_state,
                    )
                except DivergenceError as exc:
                    status, diverged_step = "diverged", step
                    break
                diagnostic_seconds += elapsed
                accountant.step(noise_multiplier=sigma, sample_rate=q)
                row.update(
                    step=step, epoch=epoch, train_loss=float(loss.detach()) / len(x),
                    refresh_time=refresh_time, diagnostic_spectrum_time=diagnostic_spectrum_time,
                    active_state_bytes=state_bytes(active), test_loss=None, test_accuracy=None,
                )
                parameters_finite = _all_finite(model.parameters())
                diagnostics_finite = (
                    True if not diagnostics else _numeric_values_finite(row) and all(
                        _numeric_values_finite(layer_row) for layer_row in layer_rows
                    )
                )
                row["diagnostics_finite"] = diagnostics_finite
                if parameters_finite and (
                    (step + 1) % c["eval_interval"] == 0 or step + 1 == total
                ):
                    row.update(evaluate(model, test_loader, device))
                rows.append(row)
                layers.extend(
                    dict(step=step, **layer_row) for layer_row in layer_rows
                )
                bins.extend(dict(step=step, **eigen_row) for eigen_row in eigen_rows)
                audits.append(dict(
                    step=step, batch_indices=indices.tolist(),
                    model_hash=digest(model.parameters()), parameters_finite=parameters_finite,
                    **audit,
                ))
                step += 1
                if not parameters_finite:
                    status, diverged_step = "diverged", step - 1
                    break
                evaluations = [r for r in rows if r.get("test_accuracy") is not None]
                if evaluations:
                    last_finite_accuracy = evaluations[-1]["test_accuracy"]
                last_finite_loss = row["train_loss"]
            if status == "diverged":
                break
    finally:
        sync(model)

    wall = time.perf_counter() - started
    summary = _summary_from_rows(
        c, spec, model, rows, refreshes, accountant, sigma, step, wall,
        diagnostic_seconds, active, status, diverged_step,
        last_finite_accuracy, last_finite_loss,
    )
    summary["steps_to_acc_90"] = threshold_steps(
        [row for row in rows if row.get("test_accuracy") is not None], 0.90
    ) if status == "completed" else None
    save_json(root / "summary.json", summary)
    meta.update(summary, complete=status == "completed")
    save_json(root / "metadata.json", meta)
    try:
        save_json(root / "pairing.json", {"private": audits, "synthetic": synthetic_audits})
        write_csv(
            root / "train_metrics.csv",
            [dict(seed=c["seed"], method=spec["method"], config_fingerprint=fingerprint(c), **row) for row in rows],
            fields=TRAIN_FIELDS,
        )
        write_csv(
            root / "layer_metrics.csv",
            [dict(seed=c["seed"], method=spec["method"], config_fingerprint=fingerprint(c), **row) for row in layers],
            fields=LAYER_FIELDS,
        )
        write_csv(
            root / "refresh_metrics.csv",
            [dict(seed=c["seed"], method=spec["method"], config_fingerprint=fingerprint(c), **row) for row in refreshes],
            fields=REFRESH_FIELDS,
        )
        write_csv(
            root / "eigenbin_metrics.csv",
            [dict(seed=c["seed"], method=spec["method"], config_fingerprint=fingerprint(c), **row) for row in bins],
            fields=EIGEN_FIELDS,
        )
    finally:
        model.remove_hooks()
    print(
        f"{status} seed={c['seed']} {run_id}: {step}/{total} steps"
        + (f", accuracy={summary['final_accuracy']:.4f}" if summary["final_accuracy"] is not None else ""),
        flush=True,
    )
    return meta


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "full.json"))
    parser.add_argument("--output", default=str(ROOT / "runs"))
    parser.add_argument("--run", choices=("all", *[spec["run_id"] for spec in run_specs()]), default="all")
    args = parser.parse_args()
    config = read_config(args.config)
    selected = run_specs(config) if args.run == "all" else [run_spec(args.run, config)]
    for spec in selected:
        train(config, spec["run_id"], args.output)


if __name__ == "__main__":
    main()
