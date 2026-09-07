"""Train the six paired Exp5 methods; full runs require explicit invocation."""
import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader, Subset

from exp3.audit_upstream import require_pinned
from exp3.cost import CostTracker
from exp3.geometry import diagnose, oracle_compare, stale
from exp3.metrics import after_noise, before_clip
from exp3.preconditioners import (apply, diagnostic_model, refresh, state_bytes,
                                  synthetic_samples)
from exp4.dynamics import AdamDirection, aggregate
from exp5.common import (DEFAULT, LAYERS, METHODS, ROOT, SYNTHETIC_METHODS,
                         Indexed, SimpleCNN, RNGStream, datasets, digest,
                         evaluate, fingerprint, provenance, read_config,
                         save_json, set_seed, write_csv)
from exp5.optimizers import (FirstMomentState, SecondMomentState,
                             adam_candidate, adam_optimizer, apply_gradients,
                             beta2_diagnostics, state_bytes as tensor_state_bytes)


def method_flags(method):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return method in ("syn_diag_beta1", "syn_diag_beta12"), method in ("syn_diag_beta2", "syn_diag_beta12")


def synthetic_q(state, samples, c, dev):
    """Compute Exp3's diagonal q with the same SUM-loss and layer layout."""
    x, y = samples
    with diagnostic_model(state, dev) as model:
        sums = {}
        for start in range(0, len(x), c["analysis_batch_size"]):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[start:start + c["analysis_batch_size"]].to(dev)),
                            y[start:start + c["analysis_batch_size"]].to(dev), reduction="sum").backward()
            for name in LAYERS:
                layer = getattr(model._module, name)
                value = torch.cat([layer.weight.grad_sample.flatten(2),
                                   layer.bias.grad_sample.unsqueeze(-1)], -1).flatten(1)
                sums[name] = sums.get(name, 0) + value.detach().double().square().sum(0)
    return {name: value / len(x) for name, value in sums.items()}


def make_diagonal_p(q, c):
    return {name: (1. / (value.sqrt() + c["lambda"])).float() for name, value in q.items()}


def refresh_active(model_state, samples, c, dev, method, second_state=None, step=0):
    """Return active transform, serializable audit, and local beta2 payload."""
    _, beta2_on = method_flags(method)
    if method == "dp_kfc_adam":
        return refresh(model_state, samples, c, dev, "dp_kfc"), {"kind": "kfac"}, None
    if not beta2_on:
        active = refresh(model_state, samples, c, dev, "syn_diag")
        return active, {"kind": "q_current", "beta2_delta_t": None}, None
    q = synthetic_q(model_state, samples, c, dev)
    v, previous, delta_t = second_state.update(q, step)
    audit = {
        "kind": "q_ema", "beta2_delta_t": delta_t,
        "q_hash": digest(q.values()), "v_hash": digest(v.values()),
    }
    # These tensors are deliberately returned separately from the audit and
    # are consumed immediately by the diagnostics segment only.
    return ("syn_diag", make_diagonal_p(v, c)), audit, (previous, q, v)


def cosine(a, b):
    aa = torch.cat([x.detach().reshape(-1) for x in a]).double()
    bb = torch.cat([x.detach().reshape(-1) for x in b]).double()
    denominator = aa.norm() * bb.norm()
    return float((aa @ bb / denominator).clamp(-1, 1)) if denominator > 0 else None


def direction_norm(values):
    return float(torch.cat([v.detach().reshape(-1) for v in values]).double().norm())


def direction_metrics(adam_direction, noisy_direction, clean_direction, clip_row, method_name):
    return {
        f"cos_{method_name}_adam_noisy": cosine(noisy_direction, adam_direction),
        f"cos_{method_name}_adam_current_noise_off": cosine(clean_direction, adam_direction),
        f"noise_degradation_{method_name}": (
            cosine(clean_direction, adam_direction) - cosine(noisy_direction, adam_direction)
            if cosine(clean_direction, adam_direction) is not None and cosine(noisy_direction, adam_direction) is not None else None
        ),
        f"{method_name}_direction_norm": direction_norm(noisy_direction),
        f"{method_name}_direction_norm_current_noise_off": direction_norm(clean_direction),
        f"{method_name}_clip_cosine": cosine(clip_row["raw_direction"], clip_row["clean_direction"]),
    }


def counterfactual(model, x, y, directions, c):
    original = [p.detach().clone() for p in model._module.parameters()]
    with torch.no_grad():
        base = float(F.cross_entropy(model._module(x), y))
    result = {}
    for name, direction in directions.items():
        probe = copy.deepcopy(model._module)
        with torch.no_grad():
            for p, old, update in zip(probe.parameters(), original, direction):
                p.copy_(old - c["learning_rate"] * update)
        with torch.no_grad():
            loss = float(F.cross_entropy(probe(x), y))
        distance = direction_norm([p.detach() - old for p, old in zip(probe.parameters(), original)])
        result[f"loss_progress_{name}"] = (base - loss) / (distance + c["eps_num"])
        result[f"candidate_loss_{name}"] = loss
    return result


def train(c, seed, method, output, data_override=None, diagnostics=True, event_hook=None):
    if method not in METHODS or seed not in c["seeds"]:
        raise ValueError("Unknown method/seed")
    source_provenance = provenance()
    require_pinned(source_provenance, c["smoke"])
    root = Path(output).resolve() / f"seed{seed}" / method
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / "config.json", c)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if c["device"] == "auto" else torch.device(c["device"])
    torch.set_num_threads(c["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    source, test = datasets(c) if data_override is None else data_override
    n = c["train_subset"] or len(source)
    if not c["batch_size"] <= n <= len(source):
        raise ValueError("Invalid training subset")
    if c["test_subset"] is not None:
        test = Subset(test, range(c["test_subset"]))
    data = Subset(source, range(n))
    loader_rng = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(Indexed(data), batch_size=c["batch_size"], shuffle=True, drop_last=True,
                        num_workers=0, generator=loader_rng)
    test_loader = DataLoader(test, batch_size=c["batch_size"], shuffle=False, num_workers=0,
                             generator=torch.Generator().manual_seed(seed + 2))
    total = len(loader) * c["epochs"]
    sample_rate = c["batch_size"] / n
    sigma = get_noise_multiplier(target_epsilon=c["epsilon"], target_delta=c["delta"],
                                 sample_rate=sample_rate, steps=total, accountant="rdp")
    set_seed(seed)
    model = GradSampleModule(SimpleCNN().to(dev), loss_reduction="sum")
    is_syn = method in SYNTHETIC_METHODS
    beta1_on, beta2_on = method_flags(method) if method != "dp_adam" and method != "dp_kfc_adam" else (False, False)
    optimizer = adam_optimizer(model.parameters(), c) if method in ("dp_adam", "dp_kfc_adam") else None
    first_state = FirstMomentState(model.parameters(), c["beta1"]) if is_syn and beta1_on else None
    second_state = SecondMomentState(c["beta2"]) if is_syn and beta2_on else None
    shadow_adam = AdamDirection(model.parameters(), dict(c, adam_lr=c["learning_rate"])) if diagnostics else None
    accountant = RDPAccountant()
    noise_rng = RNGStream(seed + 4, dev)
    syn_rng = RNGStream(seed + 3, dev) if is_syn else None
    stale_rng = RNGStream(seed + c["stale_seed_offset"], dev)
    oracle_ids = torch.randperm(n, generator=torch.Generator().manual_seed(c["oracle_seed"]))[:c["M_oracle"]]
    meta = dict(seed=seed, method=method, fingerprint=fingerprint(c), provenance=source_provenance,
                device=str(dev), torch=str(torch.__version__), noise_multiplier=sigma,
                sample_rate=sample_rate, total_steps=total, initial_model_hash=digest(model.parameters()),
                rng_seeds=dict(init=seed, loader=seed + 1, test=seed + 2,
                               synthetic=None if syn_rng is None else seed + 3,
                               noise=seed + 4, stale=seed + c["stale_seed_offset"]),
                parameter_order=list(dict(model._module.named_parameters())),
                optimizer=(dict(name="Adam", lr=c["learning_rate"], betas=[c["beta1"], c["beta2"]],
                                eps=c["adam_eps"], weight_decay=c["weight_decay"]) if optimizer else
                          dict(name="ExplicitEMA", lr=c["learning_rate"], beta1_enabled=beta1_on,
                               beta2_enabled=beta2_on)),
                accountant="rdp", sampling="shuffle/drop_last; RDP calibration convention (not Poisson)",
                clipping="upstream global per-sample clipping after optional preconditioning",
                diagnostics_scope="non-DP research diagnostics; never fed into training",
                oracle_indices=oracle_ids.tolist(),
                current_noise_off="historical state retained; only current Gaussian draw removed",
                complete=False)
    save_json(root / "metadata.json", meta)
    rows, evals, audits, synthetic_audits, refresh_rows, oracle_rows = [], [], [], [], [], []
    active = None
    step = 0
    refresh_times = []
    tracker = CostTracker(dev)
    start_time = time.perf_counter()

    def event(kind):
        if event_hook is not None:
            event_hook(kind, step, active)

    try:
        for epoch in range(c["epochs"]):
            iterator = iter(loader)
            for _ in range(len(loader)):
                is_refresh = is_syn and step % c["K"] == 0
                old_active = active
                refresh_audit = None
                if is_refresh:
                    event("refresh_start")
                    # Refresh construction is algorithm work. Only the
                    # optional probes and geometry below belong to diagnostics.
                    tracker.sync()
                    started = time.perf_counter()
                    samples, sample_audit = synthetic_samples(c, dev, syn_rng)
                    active, refresh_audit, beta2_diag_payload = refresh_active(
                        model._module.state_dict(), samples, c, dev, method,
                        second_state=second_state, step=step)
                    tracker.sync()
                    elapsed = time.perf_counter() - started
                    refresh_times.append(elapsed)
                    synthetic_audits.append(dict(step=step, **sample_audit, **refresh_audit))
                    if diagnostics:
                        with tracker.diagnostics():
                            beta2_metrics = None
                            if beta2_diag_payload is not None:
                                beta2_metrics = beta2_diagnostics(
                                    *beta2_diag_payload, c["eps_num"])
                            transforms = {"new": active}
                            if old_active is not None:
                                transforms["old"] = old_active
                            probes, probe_audit = synthetic_samples(c, dev, stale_rng, budget=c["M_stale"])
                            geom = diagnose(model._module.state_dict(), probes, transforms, c, dev, root)
                            for layer in LAYERS:
                                beta2_fields = {}
                                if beta2_metrics is not None:
                                    beta2_fields = dict(beta2_delta_t=refresh_audit["beta2_delta_t"],
                                                        **beta2_metrics[layer])
                                refresh_rows.append(dict(step=step, layer=layer, M_syn=c["M_syn"], M_stale=c["M_stale"],
                                                         refresh_seconds=elapsed, preconditioner_state_bytes=state_bytes(active),
                                                         **beta2_fields,
                                                         **stale(geom.get("old", {}).get(layer), geom["new"][layer])))
                            del probes, probe_audit, geom, transforms
                    beta2_diag_payload = None
                    del samples
                    event("refresh_end")
                if diagnostics and c["oracle_enabled"] and step % c["K"] == 0:
                    values = [data[int(i)] for i in oracle_ids]
                    oracle_samples = (torch.stack([v[0] for v in values]), torch.tensor([v[1] for v in values]))
                    transforms = {"raw": None}
                    if active is not None:
                        transforms["new"] = active
                        if old_active is not None:
                            transforms["old"] = old_active
                    with tracker.diagnostics():
                        geom = diagnose(model._module.state_dict(), oracle_samples, transforms, c, dev, root)
                    raw_geom = geom["raw"]
                    new_geom = geom.get("new", raw_geom)
                    old_geom = geom.get("old")
                    for layer in LAYERS:
                        oracle_rows.append(dict(step=step, layer=layer,
                                                **oracle_compare(raw_geom[layer], new_geom[layer],
                                                                 None if old_geom is None else old_geom[layer])))
                    del values, oracle_samples, transforms, geom
                event("before_batch")
                x, y, indices = next(iterator)
                x, y = x.to(dev), y.to(dev)
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y, reduction="sum")
                loss.backward()
                raw = [p.grad.detach().clone() / len(x) for p in model.parameters()]
                if active is not None:
                    apply(model, active)
                clip_row = before_clip(model, c, len(x))
                clip_row["raw_direction"] = [p.grad_sample.mean(0).detach().clone() for p in model.parameters()]
                with_noise = aggregate(model, None, noise_rng, sigma, c, len(x))
                clip_row["clean_direction"] = with_noise["clean"]
                row = dict(clip_row)
                row.pop("raw_direction")
                row.pop("clean_direction")
                row.update(after_noise(model, sigma, c, len(x)))
                row["diagnostic_snr"] = row["clipped_aggregate_norm"] / (row["expected_noise_norm"] + c["eps_num"])
                if diagnostics:
                    # Disposable references and counterfactuals belong to the
                    # diagnostic segment and precede the official state commit.
                    with tracker.diagnostics():
                        adam_direction = shadow_adam.advance(raw)
                        if optimizer is not None:
                            noisy_direction = adam_candidate(list(model.parameters()), optimizer,
                                                             with_noise["noisy"], c["learning_rate"])
                            clean_direction = adam_candidate(list(model.parameters()), optimizer,
                                                             with_noise["clean"], c["learning_rate"])
                        elif first_state is not None:
                            noisy_direction = first_state.peek(with_noise["noisy"])
                            clean_direction = first_state.peek(with_noise["clean"])
                        else:
                            noisy_direction = with_noise["noisy"]
                            clean_direction = with_noise["clean"]
                        row.update(direction_metrics(
                            adam_direction, noisy_direction, clean_direction,
                            {"raw_direction": clip_row["raw_direction"],
                             "clean_direction": clip_row["clean_direction"]}, "method"))
                        diag_step = step == 0 or (step + 1) % c["diagnostic_interval"] == 0 or step + 1 == total
                        row.update({f"loss_progress_{name}": None for name in ("adam", "method", "current_noise_off")})
                        row.update({f"candidate_loss_{name}": None for name in ("adam", "method", "current_noise_off")})
                        if diag_step:
                            row.update(counterfactual(
                                model, x, y,
                                {"adam": adam_direction, "method": noisy_direction,
                                 "current_noise_off": clean_direction}, c))
                if optimizer is not None:
                    before = [p.detach().clone() for p in model.parameters()]
                    for p, g in zip(model.parameters(), with_noise["noisy"]):
                        p.grad = g
                    optimizer.step()
                else:
                    before = [p.detach().clone() for p in model.parameters()]
                    if first_state is not None:
                        actual_direction = first_state.update(with_noise["noisy"])
                    else:
                        actual_direction = with_noise["noisy"]
                    apply_gradients(model.parameters(), actual_direction, c["learning_rate"])
                update = direction_norm([p.detach() - old for p, old in zip(model.parameters(), before)])
                row.update(update_norm=update, method_update_norm=update)
                accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
                step += 1
                row.update(seed=seed, method=method, step=step, epoch=epoch + 1,
                           train_loss=float(loss.detach()) / len(x), test_loss=None, test_accuracy=None,
                           syn_refresh=is_refresh, batch_hash=digest(indices), noise_hash=with_noise["noise_hash"])
                if step % c["eval_interval"] == 0 or step == total:
                    row.update(evaluate(model, test_loader, dev))
                    print(f"seed={seed} {method} step={step}/{total} accuracy={row['test_accuracy']:.4f}", flush=True)
                rows.append(row)
                audits.append(dict(step=step, batch_indices=indices.tolist(), batch_hash=digest(indices),
                                   noise_hash=with_noise["noise_hash"], noise_rng_before=with_noise["noise_rng_before"],
                                   noise_rng_after=with_noise["noise_rng_after"], loader_rng_hash=digest([loader_rng.get_state()]),
                                   model_hash=digest(model.parameters())))
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite model parameters")
        tracker.sync()
        acc = [r["test_accuracy"] for r in rows if r["test_accuracy"] is not None]
        late = [r["test_accuracy"] for r in rows if r["step"] > total / 2 and r["test_accuracy"] is not None]
        preconditioner_state_bytes = state_bytes(active)
        temporal_state_bytes = tensor_state_bytes(first_state.m if first_state is not None else {})
        temporal_state_bytes += tensor_state_bytes(second_state.v if second_state is not None else {})
        optimizer_state_bytes = tensor_state_bytes(optimizer.state if optimizer is not None else {})
        total_algorithm_state_bytes = preconditioner_state_bytes + temporal_state_bytes + optimizer_state_bytes
        cost = dict(**tracker.finish(), total_refresh_time=sum(refresh_times),
                    mean_refresh_time=sum(refresh_times) / len(refresh_times) if refresh_times else 0.,
                    number_of_refreshes=len(refresh_times),
                    preconditioner_state_bytes=preconditioner_state_bytes,
                    temporal_state_bytes=temporal_state_bytes,
                    optimizer_state_bytes=optimizer_state_bytes,
                    total_algorithm_state_bytes=total_algorithm_state_bytes)
        summary = dict(**cost, final_accuracy=acc[-1], best_accuracy=max(acc),
                       late_mean_accuracy=sum(late) / len(late) if late else None,
                       final_test_loss=rows[-1]["test_loss"], epsilon_spent=accountant.get_epsilon(c["delta"]),
                       noise_multiplier=sigma, completed_steps=step, final_model_hash=digest(model.parameters()),
                       fingerprint=fingerprint(c), seed=seed, method=method)
        meta.update(summary, complete=True)
        save_json(root / "metadata.json", meta)
        save_json(root / "summary.json", summary)
        torch.save(dict(model=model._module.state_dict(), optimizer=None if optimizer is None else optimizer.state_dict(),
                        first_moment=None if first_state is None else first_state.state_dict(),
                        second_moment=None if second_state is None else {"beta2": second_state.beta2,
                                                                           "last_refresh_step": second_state.last_refresh_step,
                                                                           "v": second_state.v}), root / "final_state.pt")
    finally:
        model.remove_hooks()
        write_csv(root / "train_metrics.csv", rows, fields=None if rows else ["step", "seed", "method"])
        write_csv(root / "eval_metrics.csv", [r for r in rows if r["test_accuracy"] is not None])
        write_csv(root / "refresh_metrics.csv", refresh_rows, fields=None if refresh_rows else ["step", "layer"])
        write_csv(root / "oracle_metrics.csv", oracle_rows, fields=None if oracle_rows else ["step", "layer"])
        save_json(root / "pairing.json", dict(private=audits, synthetic=synthetic_audits,
                                               oracle_indices=oracle_ids.tolist()))
    return meta


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/full.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    args = parser.parse_args()
    config = read_config(args.config)
    for seed in config["seeds"] if args.seed is None else [args.seed]:
        for method in METHODS if args.method == "all" else [args.method]:
            train(config, seed, method, args.output)


if __name__ == "__main__":
    main()
