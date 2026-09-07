"""Train the six paired Exp5b methods; full runs require explicit invocation."""
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
from exp3.common import generate_pink_noise
from exp3.cost import CostTracker
from exp3.geometry import diagnose, oracle_compare, stale
from exp3.metrics import after_noise, before_clip
from exp3.preconditioners import apply, diagnostic_model, refresh
from exp4.dynamics import AdamDirection, aggregate
from exp5b.common import (
    DEFAULT, LAYERS, METHODS, MOMENTUM_METHODS, ROOT, SYNTHETIC_METHODS,
    Indexed, SimpleCNN, RNGStream, datasets, digest, evaluate, fingerprint,
    provenance, read_config, save_json, set_seed, write_csv,
)
from exp5b.optimizers import (
    FirstMomentState, SecondMomentState, apply_gradients, beta2_diagnostics,
    state_bytes as tensor_state_bytes,
)


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
            F.cross_entropy(
                model(x[start:start + c["analysis_batch_size"]].to(dev)),
                y[start:start + c["analysis_batch_size"]].to(dev),
                reduction="sum",
            ).backward()
            for name in LAYERS:
                layer = getattr(model._module, name)
                value = torch.cat([
                    layer.weight.grad_sample.flatten(2),
                    layer.bias.grad_sample.unsqueeze(-1),
                ], -1).flatten(1)
                sums[name] = sums.get(name, 0) + value.detach().double().square().sum(0)
    return {name: value / len(x) for name, value in sums.items()}


def synthetic_samples(c, dev, rng, *, budget=None):
    """Generate the paired pink-noise/uniform-label synthetic stream."""
    budget = c["M_syn"] if budget is None else budget
    before = rng.audit()
    with rng.use():
        batches = []
        for start in range(0, budget, c["batch_size"]):
            b = min(c["batch_size"], budget - start)
            batches.append((
                generate_pink_noise(b, (1, 28, 28), dev),
                torch.randint(0, 10, (b,), device=dev),
            ))
    x, y = (torch.cat([batch[index] for batch in batches]) for index in (0, 1))
    return (x, y), dict(rng_before=before, rng_after=rng.audit(), count=len(y))


def synthetic_sample_audit(samples):
    x, y = samples
    return dict(samples_hash=digest([x]), labels_hash=digest([y]))


def make_diagonal_p(q, c):
    return {name: (1. / (value.sqrt() + c["lambda"])).float() for name, value in q.items()}


def refresh_beta2_core_a(model_state, samples, c, dev, second_state, step):
    """Core A: synthetic generation, q construction, and true-time beta2 EMA."""
    q = synthetic_q(model_state, samples, c, dev)
    v, previous, delta_t = second_state.update(q, step)
    return q, v, previous, delta_t


def refresh_beta2_core_b(second_state, c):
    """Core B: construct P from persistent v after diagnostic tensors are gone."""
    return "syn_diag", make_diagonal_p(second_state.v, c)


def refresh_active(model_state, samples, c, dev, method):
    """Refresh a non-beta2 preconditioner using the upstream Exp3 implementation."""
    if method == "dp_kfc_momentum":
        return refresh(model_state, samples, c, dev, "dp_kfc")
    if method in ("syn_diag", "syn_diag_beta1"):
        return refresh(model_state, samples, c, dev, "syn_diag")
    raise ValueError(f"Unexpected non-beta2 method: {method}")


def cosine(a, b):
    aa = torch.cat([x.detach().reshape(-1) for x in a]).double()
    bb = torch.cat([x.detach().reshape(-1) for x in b]).double()
    denominator = aa.norm() * bb.norm()
    return float((aa @ bb / denominator).clamp(-1, 1)) if denominator > 0 else None


def direction_norm(values):
    return float(torch.cat([v.detach().reshape(-1) for v in values]).double().norm())


def direction_metrics(adam_direction, noisy_direction, clean_direction, clip_row):
    clean_cos = cosine(clean_direction, adam_direction)
    noisy_cos = cosine(noisy_direction, adam_direction)
    return {
        "cos_method_adam_noisy": noisy_cos,
        "cos_method_adam_current_noise_off": clean_cos,
        "noise_degradation_method": clean_cos - noisy_cos if clean_cos is not None and noisy_cos is not None else None,
        "method_direction_norm": direction_norm(noisy_direction),
        "method_direction_norm_current_noise_off": direction_norm(clean_direction),
        "method_clip_cosine": cosine(clip_row["raw_direction"], clip_row["clean_direction"]),
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


def _official_optimizer_metadata(method, beta1_on):
    if method in MOMENTUM_METHODS:
        return dict(
            name="ExplicitFirstMomentEMA",
            implementation="exp5b.optimizers.FirstMomentState",
            lr=.1,
            learning_rate=.1,
            beta1=.9,
            bias_correction=True,
            update_formula="m_t=beta1*m_(t-1)+(1-beta1)*tilde_g_t; update=m_t/(1-beta1^t)",
            update_timing="post-DP-noise",
        )
    return dict(
        name="DirectPrivatizedGradient",
        implementation="exp5b.optimizers.apply_gradients",
        lr=.1,
        learning_rate=.1,
        beta1_enabled=beta1_on,
        bias_correction=False,
        update_timing="post-DP-noise",
    )


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
        if not 1 <= c["test_subset"] <= len(test):
            raise ValueError("Invalid test subset")
        test = Subset(test, range(c["test_subset"]))
    data = Subset(source, range(n))
    loader_rng = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(
        Indexed(data), batch_size=c["batch_size"], shuffle=True, drop_last=True,
        num_workers=0, generator=loader_rng,
    )
    test_loader = DataLoader(
        test, batch_size=c["batch_size"], shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(seed + 2),
    )
    total = len(loader) * c["epochs"]
    if total != c["total_private_steps"]:
        raise ValueError(f"Computed private steps {total} != configured {c['total_private_steps']}")
    sample_rate = c["batch_size"] / n
    sigma = get_noise_multiplier(
        target_epsilon=c["epsilon"], target_delta=c["delta"], sample_rate=sample_rate,
        steps=total, accountant="rdp",
    )
    set_seed(seed)
    model = GradSampleModule(SimpleCNN().to(dev), loss_reduction="sum")
    is_syn = method in SYNTHETIC_METHODS
    beta1_on, beta2_on = method_flags(method)
    first_state = FirstMomentState(model.parameters(), c["beta1"]) if method in MOMENTUM_METHODS else None
    second_state = SecondMomentState(c["beta2"]) if beta2_on else None
    # AdamDirection is a disposable mechanistic reference only. It is never
    # used for an official update or included as an Exp5b baseline.
    shadow_adam = AdamDirection(model.parameters(), dict(c, adam_lr=c["learning_rate"])) if diagnostics else None
    accountant = RDPAccountant()
    noise_rng = RNGStream(seed + 4, dev)
    syn_rng = RNGStream(seed + 3, dev) if is_syn else None
    stale_rng = RNGStream(seed + c["stale_seed_offset"], dev)
    oracle_ids = torch.randperm(n, generator=torch.Generator().manual_seed(c["oracle_seed"]))[:c["M_oracle"]]
    meta = dict(
        seed=seed, method=method, fingerprint=fingerprint(c), provenance=source_provenance,
        device=str(dev), torch=str(torch.__version__), learning_rate=c["learning_rate"],
        beta1=c["beta1"], beta2=c["beta2"], noise_multiplier=sigma,
        sample_rate=sample_rate, total_steps=total, total_private_steps=c["total_private_steps"],
        initial_model_hash=digest(model.parameters()), diagnostics_enabled=diagnostics,
        rng_seeds=dict(
            init=seed, loader=seed + 1, test=seed + 2,
            synthetic=None if syn_rng is None else seed + 3,
            noise=seed + 4, stale=seed + c["stale_seed_offset"],
        ),
        parameter_order=list(dict(model._module.named_parameters())),
        optimizer=_official_optimizer_metadata(method, beta1_on),
        optimizer_is_official=False,
        adam_direction=dict(
            enabled=diagnostics,
            role="mechanistic diagnostic reference only; not an Exp5b competing optimizer",
        ),
        accountant="rdp", sampling="shuffle/drop_last; RDP calibration convention (not Poisson)",
        clipping="upstream global per-sample clipping after optional preconditioning",
        covariance_ridge=1e-5,
        inverse_root_damping=c["damping"],
        upstream_dp_kfc_commit="eb31b9aeb2280642684f4cedfa65cc02b76c76cd",
        diagnostics_scope="non-DP research diagnostics; never fed into training",
        state_bytes_definition=(
            "persistent P plus persistent m/v; temporary q, previous_v, diagnostics, hashes, "
            "oracle state, and candidate directions excluded"
        ),
        oracle_indices=oracle_ids.tolist(),
        current_noise_off="historical first-moment state retained; only current Gaussian draw removed",
        complete=False,
    )
    save_json(root / "metadata.json", meta)
    rows, audits, synthetic_audits, refresh_rows, oracle_rows = [], [], [], [], []
    active = None
    step = 0
    refresh_times = []
    tracker = CostTracker(dev)

    def event(kind):
        if event_hook is not None:
            event_hook(kind, step, active)

    try:
        for epoch in range(c["epochs"]):
            # num_workers=0 means no private-batch prefetch can cross refresh order.
            iterator = iter(loader)
            for _ in range(len(loader)):
                is_refresh = is_syn and step % c["K"] == 0
                old_active = active
                refresh_audit = None
                if is_refresh:
                    event("refresh_start")
                    tracker.sync()
                    core_a_started = time.perf_counter()
                    samples, sample_audit_core = synthetic_samples(c, dev, syn_rng)
                    if beta2_on:
                        q, v, previous, delta_t = refresh_beta2_core_a(
                            model._module.state_dict(), samples, c, dev, second_state, step,
                        )
                    else:
                        active = refresh_active(model._module.state_dict(), samples, c, dev, method)
                    tracker.sync()
                    core_a_seconds = time.perf_counter() - core_a_started
                    beta2_metrics = None

                    if beta2_on:
                        event("beta2_core_a_end")
                        # Hashes and D metrics are audit-only. They are kept in
                        # the diagnostic segment even when diagnostics=False.
                        with tracker.diagnostics():
                            sample_audit = dict(sample_audit_core, **synthetic_sample_audit(samples))
                            q_hash = digest(q.values())
                            v_hash = digest(v.values())
                            beta2_metrics = beta2_diagnostics(previous, q, v, c["eps_num"])
                        event("beta2_diag_end")
                        refresh_audit = {
                            "kind": "q_ema", "beta2_delta_t": delta_t,
                            "q_hash": q_hash, "v_hash": v_hash,
                        }
                        # Release diagnostic-only payload before constructing P.
                        samples = None
                        q = None
                        previous = None
                        v = None
                        del samples, q, previous, v
                        event("beta2_core_b_start")
                        tracker.sync()
                        core_b_started = time.perf_counter()
                        active = refresh_beta2_core_b(second_state, c)
                        tracker.sync()
                        core_b_seconds = time.perf_counter() - core_b_started
                        event("beta2_core_b_end")
                        refresh_elapsed = core_a_seconds + core_b_seconds
                    else:
                        refresh_audit = {"kind": "kfac" if method == "dp_kfc_momentum" else "q_current",
                                         "beta2_delta_t": None}
                        with tracker.diagnostics():
                            sample_audit = dict(sample_audit_core, **synthetic_sample_audit(samples))
                        samples = None
                        del samples
                        refresh_elapsed = core_a_seconds

                    # Stale/geometry work follows official refresh work and is
                    # wholly excluded from refresh_seconds and core time.
                    if diagnostics:
                        with tracker.diagnostics():
                            transforms = {"new": active}
                            if old_active is not None:
                                transforms["old"] = old_active
                            probes, _ = synthetic_samples(c, dev, stale_rng, budget=c["M_stale"])
                            geom = diagnose(model._module.state_dict(), probes, transforms, c, dev, root)
                            for layer in LAYERS:
                                beta2_fields = {}
                                if beta2_metrics is not None:
                                    beta2_fields = dict(
                                        beta2_delta_t=refresh_audit["beta2_delta_t"],
                                        **beta2_metrics[layer],
                                    )
                                refresh_rows.append(dict(
                                    step=step, layer=layer, M_syn=c["M_syn"], M_stale=c["M_stale"],
                                    refresh_seconds=refresh_elapsed,
                                    preconditioner_state_bytes=tensor_state_bytes(active),
                                    **beta2_fields,
                                    **stale(geom.get("old", {}).get(layer), geom["new"][layer]),
                                ))
                            del probes, geom, transforms
                    synthetic_audits.append(dict(step=step, **sample_audit, **refresh_audit))
                    refresh_times.append(refresh_elapsed)
                    event("refresh_end")

                if diagnostics and c["oracle_enabled"] and step % c["K"] == 0:
                    values = [data[int(i)] for i in oracle_ids]
                    oracle_samples = (
                        torch.stack([v[0] for v in values]),
                        torch.tensor([v[1] for v in values]),
                    )
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
                        oracle_rows.append(dict(
                            step=step, layer=layer,
                            **oracle_compare(raw_geom[layer], new_geom[layer],
                                             None if old_geom is None else old_geom[layer]),
                        ))
                    del values, oracle_samples, transforms, geom

                event("before_private_batch")
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
                    with tracker.diagnostics():
                        adam_direction = shadow_adam.advance(raw)
                        if first_state is not None:
                            noisy_direction = first_state.peek(with_noise["noisy"])
                            clean_direction = first_state.peek(with_noise["clean"])
                        else:
                            noisy_direction = with_noise["noisy"]
                            clean_direction = with_noise["clean"]
                        row.update(direction_metrics(
                            adam_direction, noisy_direction, clean_direction,
                            {"raw_direction": clip_row["raw_direction"],
                             "clean_direction": clip_row["clean_direction"]},
                        ))
                        row.update({f"loss_progress_{name}": None for name in ("adam", "method", "current_noise_off")})
                        row.update({f"candidate_loss_{name}": None for name in ("adam", "method", "current_noise_off")})
                        diag_step = step == 0 or (step + 1) % c["diagnostic_interval"] == 0 or step + 1 == total
                        if diag_step:
                            row.update(counterfactual(
                                model, x, y,
                                {"adam": adam_direction, "method": noisy_direction,
                                 "current_noise_off": clean_direction}, c,
                            ))

                before = [p.detach().clone() for p in model.parameters()]
                if first_state is not None:
                    # The shared helper consumes only the privatized noisy
                    # aggregate, then returns the bias-corrected direction.
                    actual_direction = first_state.update(with_noise["noisy"])
                else:
                    actual_direction = with_noise["noisy"]
                apply_gradients(model.parameters(), actual_direction, c["learning_rate"])
                update = direction_norm([p.detach() - old for p, old in zip(model.parameters(), before)])
                row.update(update_norm=update, method_update_norm=update)
                accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
                step += 1
                row.update(
                    seed=seed, method=method, step=step, epoch=epoch + 1,
                    train_loss=float(loss.detach()) / len(x), test_loss=None, test_accuracy=None,
                    syn_refresh=is_refresh, batch_hash=digest(indices), noise_hash=with_noise["noise_hash"],
                )
                if step % c["eval_interval"] == 0 or step == total:
                    row.update(evaluate(model, test_loader, dev))
                    print(f"seed={seed} {method} step={step}/{total} accuracy={row['test_accuracy']:.4f}", flush=True)
                rows.append(row)
                audits.append(dict(
                    step=step, batch_indices=indices.tolist(), batch_hash=digest(indices),
                    noise_hash=with_noise["noise_hash"], noise_rng_before=with_noise["noise_rng_before"],
                    noise_rng_after=with_noise["noise_rng_after"], loader_rng_hash=digest([loader_rng.get_state()]),
                    model_hash=digest(model.parameters()),
                ))
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite model parameters")

        tracker.sync()
        acc = [r["test_accuracy"] for r in rows if r["test_accuracy"] is not None]
        late = [r["test_accuracy"] for r in rows if r["step"] > total / 2 and r["test_accuracy"] is not None]
        preconditioner_state_bytes = tensor_state_bytes(active)
        temporal_state_bytes = tensor_state_bytes(first_state.m if first_state is not None else {})
        temporal_state_bytes += tensor_state_bytes(second_state.v if second_state is not None else {})
        optimizer_state_bytes = 0
        total_algorithm_state_bytes = preconditioner_state_bytes + temporal_state_bytes
        cost = dict(
            **tracker.finish(), total_refresh_time=sum(refresh_times),
            mean_refresh_time=sum(refresh_times) / len(refresh_times) if refresh_times else 0.,
            number_of_refreshes=len(refresh_times),
            preconditioner_state_bytes=preconditioner_state_bytes,
            temporal_state_bytes=temporal_state_bytes,
            optimizer_state_bytes=optimizer_state_bytes,
            total_algorithm_state_bytes=total_algorithm_state_bytes,
        )
        summary = dict(
            **cost, final_accuracy=acc[-1], best_accuracy=max(acc),
            late_mean_accuracy=sum(late) / len(late) if late else None,
            final_test_loss=rows[-1]["test_loss"], epsilon_spent=accountant.get_epsilon(c["delta"]),
            noise_multiplier=sigma, completed_steps=step,
            final_model_hash=digest(model.parameters()), fingerprint=fingerprint(c), seed=seed, method=method,
        )
        meta.update(summary, complete=True)
        save_json(root / "metadata.json", meta)
        save_json(root / "summary.json", summary)
        torch.save(dict(
            model=model._module.state_dict(), preconditioner=active, optimizer=None,
            first_moment=None if first_state is None else first_state.state_dict(),
            second_moment=None if second_state is None else {
                "beta2": second_state.beta2, "last_refresh_step": second_state.last_refresh_step,
                "v": second_state.v,
            },
        ), root / "final_state.pt")
    finally:
        model.remove_hooks()
        write_csv(root / "train_metrics.csv", rows, fields=None if rows else ["step", "seed", "method"])
        write_csv(root / "eval_metrics.csv", [r for r in rows if r["test_accuracy"] is not None])
        write_csv(root / "refresh_metrics.csv", refresh_rows, fields=None if refresh_rows else ["step", "layer"])
        write_csv(root / "oracle_metrics.csv", oracle_rows, fields=None if oracle_rows else ["step", "layer"])
        save_json(root / "pairing.json", dict(
            private=audits, synthetic=synthetic_audits, oracle_indices=oracle_ids.tolist(),
        ))
    return meta


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
