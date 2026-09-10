"""Train the three paired Exp6b methods."""
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

from exp6b.common import (DEFAULT, METHODS, ROOT, SYNTHETIC_METHODS, Indexed,
                          RNGStream, SimpleCNN, datasets, digest, evaluate,
                          fingerprint, generate_pink_noise, parameter_names,
                          read_config, save_json, set_seed, stats,
                          target_device, write_csv, log_rms_distance)
from exp6b.optimizers import FirstMomentState, SecondMomentState, ExplicitAdamState
from dp_kfac.privacy import (_compute_clip_factors,
                             _compute_per_sample_norms_squared,
                             clip_and_noise_gradients)
from exp3.preconditioners import diagnostic_model


def synthetic_samples(config, device, rng):
    """Generate the paired pink-noise/random-label synthetic set."""
    before = rng.audit()
    batches = []
    with rng.use():
        for start in range(0, config["M_syn"], config["batch_size"]):
            size = min(config["batch_size"], config["M_syn"] - start)
            batches.append((generate_pink_noise(size, (1, 28, 28), device),
                            torch.randint(0, 10, (size,), device=device)))
    samples = (torch.cat([batch[0] for batch in batches]),
               torch.cat([batch[1] for batch in batches]))
    return samples, dict(count=len(samples[1]), rng_before=before,
                         rng_after=rng.audit(),
                         samples_hash=digest([samples[0]]),
                         labels_hash=digest([samples[1]]))


def synthetic_q(state, samples, config, device):
    """Compute 1/M sum of per-sample synthetic gradients squared.

    The diagnostic model is disposable and never receives or writes the
    private model's ``grad_sample`` tensors.
    """
    x, y = samples
    names = list(state.keys())
    q_sum = {name: torch.zeros_like(value, dtype=torch.float64, device=device)
             for name, value in state.items() if value.is_floating_point()}
    with diagnostic_model(state, device) as model:
        named = dict(model._module.named_parameters())
        for start in range(0, len(x), config["analysis_batch_size"]):
            stop = start + config["analysis_batch_size"]
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[start:stop].to(device)),
                            y[start:stop].to(device), reduction="sum").backward()
            for name in names:
                param = named[name]
                grad_sample = param.grad_sample
                if isinstance(grad_sample, list):
                    grad_sample = grad_sample[-1]
                q_sum[name].add_(grad_sample.detach().double().square().sum(0))
    return {name: value / len(x) for name, value in q_sum.items()}


def _private_noise_hash(model, batch_size, rng):
    """Replay only the upstream Gaussian draw layout for the audit hash."""
    replay = copy.deepcopy(rng)
    z = []
    with replay.use():
        for param in model.parameters():
            shape = param.grad_sample.contiguous().view(batch_size, -1).sum(0).shape
            z.append(torch.randn(shape, device=param.device))
    return digest(z), replay


def clip_stats(model, config, batch_size):
    """Read-only statistics before the single shared clipping operation."""
    params = list(model.parameters())
    norms_sq = _compute_per_sample_norms_squared(params, batch_size, params[0].device)
    norms = norms_sq.sqrt()
    factors = _compute_clip_factors(norms_sq, config["max_grad_norm"])
    quantiles = torch.quantile(norms.double(), torch.tensor([.05, .50, .95],
                                                              dtype=torch.float64,
                                                              device=norms.device))
    return {
        "clip_rate": float((norms > config["max_grad_norm"]).double().mean()),
        "clip_factor_q05": float(torch.quantile(factors.double(), .05)),
        "clip_factor_q50": float(torch.quantile(factors.double(), .50)),
        "clip_factor_q95": float(torch.quantile(factors.double(), .95)),
        "private_norm_q05": float(quantiles[0]),
        "private_norm_q50": float(quantiles[1]),
        "private_norm_q95": float(quantiles[2]),
    }


@torch.no_grad()
def private_update(model, noise_rng, sigma, config, batch_size):
    """Run the one common private pipeline and return its privatized mean."""
    before = noise_rng.audit()
    noise_hash, replay = _private_noise_hash(model, batch_size, noise_rng)
    pre_clip = clip_stats(model, config, batch_size)
    with noise_rng.use():
        clip_and_noise_gradients(model, sigma, config["max_grad_norm"],
                                 batch_size, store_summed_grad=True)
    if replay.audit() != noise_rng.audit():
        raise AssertionError("DP Gaussian draw layout changed")
    params = list(model.parameters())
    clean = [param.summed_grad.detach().clone() for param in params]
    noisy = [param.grad.detach().clone() for param in params]
    noise_sq = sum((n.double() - c.double()).square().sum()
                   for n, c in zip(noisy, clean))
    noisy_sq = sum(n.double().square().sum() for n in noisy)
    clean_sq = sum(c.double().square().sum() for c in clean)
    expected = sigma * config["max_grad_norm"] / batch_size
    expected *= (sum(param.numel() for param in params) ** .5)
    pre_clip.update(
        actual_noise_norm=float(noise_sq.sqrt()),
        expected_noise_norm=expected,
        clipped_aggregate_norm=float(clean_sq.sqrt()),
        noisy_gradient_norm=float(noisy_sq.sqrt()),
        noise_rng_before=before,
        noise_rng_after=noise_rng.audit(),
        noise_hash=noise_hash,
    )
    pre_clip["clean"] = clean
    pre_clip["noisy"] = noisy
    return pre_clip


def _named_values(model, values):
    return {name: value for (name, _), value in
            zip(model._module.named_parameters(), values)}


def _parameter_stats(values, prefix, include_inverse=False):
    result = stats(values, prefix)
    if include_inverse:
        result[f"{prefix}max_inverse"] = float(1.0 / torch.cat(
            [value.detach().reshape(-1).double() for value in values]).min())
    return result


def _apply_update(model, direction, learning_rate):
    with torch.no_grad():
        for param, value in zip(model.parameters(), direction):
            param.add_(value, alpha=-learning_rate)


def _refresh_metrics(q, config, noise_variance_floor, previous_q, step):
    rows = []
    for name, value in q.items():
        sqrt_q = value.sqrt()
        effective = torch.sqrt(value + noise_variance_floor) + config["adam_eps"]
        row = dict(step=step, layer=name, parameter=name, M_syn=config["M_syn"],
                   noise_variance_floor=noise_variance_floor,
                   q_below_floor_probability=float((value < noise_variance_floor).double().mean()),
                   q_log_rms_drift=None if previous_q is None else
                   log_rms_distance({name: value}, {name: previous_q[name]}, config["eps_num"]))
        row.update(_parameter_stats([sqrt_q], "sqrt_q_"))
        row.update(_parameter_stats([value], "q_"))
        row.update(_parameter_stats([effective], "denominator_", include_inverse=True))
        row.update(_parameter_stats([effective], "effective_denominator_"))
        rows.append(row)
    return rows


def _direction_state(method, model, config):
    if method == "dp_adam":
        return ExplicitAdamState(list(model.parameters()), config["beta1"],
                                 config["beta2"], config["adam_eps"])
    return FirstMomentState(list(model.parameters()), config["beta1"]), (
        SecondMomentState(config["beta2"]) if method == "syn_adam_ema_floor" else None)


def train(config, seed, method, output, data_override=None, event_hook=None):
    if method not in METHODS or seed not in config["seeds"]:
        raise ValueError("Unknown method or seed")
    root = Path(output).resolve() / f"seed{seed}" / method
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / "config.json", config)
    device = target_device(config)
    torch.set_num_threads(config["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    source, test = datasets(config) if data_override is None else data_override
    n = config["train_subset"] or len(source)
    if not config["batch_size"] <= n <= len(source):
        raise ValueError("Invalid training subset")
    if config["test_subset"] is not None:
        test = Subset(test, range(config["test_subset"]))
    data = Subset(source, range(n))
    loader_rng = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(Indexed(data), batch_size=config["batch_size"], shuffle=True,
                        drop_last=True, num_workers=0, generator=loader_rng)
    test_loader = DataLoader(test, batch_size=config["batch_size"], shuffle=False,
                             num_workers=0)
    total = len(loader) * config["epochs"]
    sample_rate = config["batch_size"] / n
    sigma = get_noise_multiplier(target_epsilon=config["epsilon"],
                                 target_delta=config["delta"],
                                 sample_rate=sample_rate, steps=total,
                                 accountant="rdp")
    noise_std = sigma * config["max_grad_norm"] / config["batch_size"]
    noise_variance_floor = noise_std ** 2
    set_seed(seed)
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    names = parameter_names(model)
    first_state = None
    second_state = None
    adam_state = None
    if method == "dp_adam":
        adam_state = _direction_state(method, model, config)
    else:
        first_state, second_state = _direction_state(method, model, config)
    noise_rng = RNGStream(seed + 4, device)
    syn_rng = RNGStream(seed + 3, device) if method in SYNTHETIC_METHODS else None
    accountant = RDPAccountant()
    meta = dict(seed=seed, method=method, fingerprint=fingerprint(config),
                device=str(device), total_steps=total, noise_multiplier=sigma,
                noise_std=noise_std, noise_variance_floor=noise_variance_floor,
                max_grad_norm=config["max_grad_norm"], batch_size=config["batch_size"],
                sample_rate=sample_rate, initial_model_hash=digest(list(model.parameters())),
                parameter_order=names,
                rng_seeds=dict(init=seed, loader=seed + 1,
                               synthetic=None if syn_rng is None else seed + 3,
                               noise=seed + 4),
                dp_pipeline="per-sample global clip -> aggregate -> Gaussian noise -> privatized gradient",
                synthetic_is_post_dp_state=True,
                preclip_synthetic_modification=False,
                optimizer=("explicit_adam" if method == "dp_adam" else
                           "explicit_first_moment_plus_synthetic_second_moment"),
                complete=False)
    save_json(root / "metadata.json", meta)

    rows, private_audit, synthetic_audit, refresh_rows = [], [], [], []
    current_q = None
    previous_q = None
    step = 0
    started = time.perf_counter()
    try:
        for epoch in range(config["epochs"]):
            iterator = iter(loader)
            for _ in range(len(loader)):
                refresh_event = False
                refresh_record = None
                if method in SYNTHETIC_METHODS and step % config["K"] == 0:
                    refresh_event = True
                    if event_hook:
                        event_hook("refresh_start", step)
                    samples, audit = synthetic_samples(config, device, syn_rng)
                    previous_q = current_q
                    current_q = synthetic_q(model._module.state_dict(), samples, config, device)
                    synthetic_audit.append(dict(step=step, **audit))
                    refresh_record = _refresh_metrics(current_q, config,
                                                      noise_variance_floor,
                                                      previous_q, step)
                    refresh_rows.extend(refresh_record)
                    if event_hook:
                        event_hook("refresh_end", step)
                if event_hook:
                    event_hook("before_batch", step)
                x, y, indices = next(iterator)
                x, y = x.to(device), y.to(device)
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y, reduction="sum")
                loss.backward()
                mechanism = private_update(model, noise_rng, sigma, config, len(x))
                noisy = mechanism["noisy"]
                if method == "dp_adam":
                    direction, m_hat, v_hat_list, denominator = adam_state.update(noisy)
                    v_hat = _named_values(model, v_hat_list)
                    denominator_map = _named_values(model, denominator)
                else:
                    m_hat = first_state.update(noisy)
                    raw_v_hat = (current_q if method == "syn_adam_floor"
                                 else second_state.update(current_q))
                    denominator_map = {
                        name: (torch.sqrt(value + noise_variance_floor).to(m_hat[index].dtype)
                               + config["adam_eps"])
                        for index, (name, value) in enumerate(raw_v_hat.items())}
                    direction = [m / denominator_map[name]
                                 for m, name in zip(m_hat, denominator_map)]
                    v_hat = raw_v_hat
                    if refresh_record is not None and method == "syn_adam_ema_floor":
                        for record in refresh_record:
                            effective = torch.sqrt(
                                raw_v_hat[record["layer"]] + noise_variance_floor
                            ) + config["adam_eps"]
                            record.update(_parameter_stats([effective], "denominator_",
                                                            include_inverse=True))
                            record.update(_parameter_stats([effective],
                                                           "effective_denominator_"))
                before_parameters = [param.detach().clone() for param in model.parameters()]
                _apply_update(model, direction, config["learning_rate"])
                update_norm = float(torch.cat([
                    (param.detach() - before).reshape(-1).double()
                    for param, before in zip(model.parameters(), before_parameters)]).norm())
                moment_stats = _parameter_stats(m_hat, "first_moment_")
                denominator_values = list(denominator_map.values())
                denominator_stats = _parameter_stats(denominator_values, "denominator_",
                                                     include_inverse=True)
                second_stats = _parameter_stats(list(v_hat.values()), "second_moment_")
                q_distance = None
                if method == "syn_adam_ema_floor":
                    q_distance = log_rms_distance(v_hat, current_q, config["eps_num"])
                    if refresh_record is not None:
                        for record in refresh_record:
                            record["v_q_log_rms_distance"] = q_distance
                accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
                step += 1
                row = dict(seed=seed, method=method, step=step, epoch=epoch + 1,
                           train_loss=float(loss.detach()) / len(x), test_loss=None,
                           test_accuracy=None, syn_refresh=refresh_event,
                           epsilon_spent=accountant.get_epsilon(config["delta"]),
                           update_norm=update_norm, noise_std=noise_std,
                           noise_variance_floor=noise_variance_floor,
                           max_inverse_denominator=denominator_stats["denominator_max_inverse"],
                           q_log_rms_drift=None if previous_q is None or not refresh_event else
                           log_rms_distance(current_q, previous_q, config["eps_num"]),
                           v_q_log_rms_distance=q_distance,
                           **{key: value for key, value in mechanism.items()
                              if key not in ("clean", "noisy", "noise_rng_before",
                                             "noise_rng_after", "noise_hash")},
                           **moment_stats, **denominator_stats, **second_stats,
                           batch_hash=digest([indices]), noise_hash=mechanism["noise_hash"])
                if step % config["eval_interval"] == 0 or step == total:
                    row.update(evaluate(model, test_loader, device))
                    print(f"seed={seed} {method} step={step}/{total} accuracy={row['test_accuracy']:.4f}",
                          flush=True)
                rows.append(row)
                private_audit.append(dict(step=step, batch_indices=indices.tolist(),
                                          batch_hash=row["batch_hash"],
                                          noise_hash=mechanism["noise_hash"],
                                          noise_rng_before=mechanism["noise_rng_before"],
                                          noise_rng_after=mechanism["noise_rng_after"]))
                if not all(torch.isfinite(param).all() for param in model.parameters()):
                    raise FloatingPointError("Nonfinite model parameters")
        evaluated = [row["test_accuracy"] for row in rows if row["test_accuracy"] is not None]
        late = [row["test_accuracy"] for row in rows
                if row["step"] > total / 2 and row["test_accuracy"] is not None]
        first = rows[0]
        last = rows[-1]
        elapsed = time.perf_counter() - started
        summary = dict(status="completed", seed=seed, method=method,
                       completed_steps=step, planned_steps=total,
                       final_accuracy=evaluated[-1], best_accuracy=max(evaluated),
                       late_mean_accuracy=sum(late) / len(late),
                       final_test_loss=last["test_loss"],
                       epsilon_spent=accountant.get_epsilon(config["delta"]),
                       noise_multiplier=sigma, noise_std=noise_std,
                       noise_variance_floor=noise_variance_floor, total_time=elapsed,
                       first_update_norm=first["update_norm"],
                       first_denominator_q05=first["denominator_q05"],
                       first_denominator_q50=first["denominator_q50"],
                       first_denominator_q95=first["denominator_q95"],
                       first_max_inverse_denominator=first["denominator_max_inverse"],
                       final_update_norm=last["update_norm"],
                       final_first_moment_norm=last["first_moment_norm"],
                       final_denominator_q05=last["denominator_q05"],
                       final_denominator_q50=last["denominator_q50"],
                       final_denominator_q95=last["denominator_q95"],
                       final_max_inverse_denominator=last["denominator_max_inverse"],
                       final_model_hash=digest(list(model.parameters())),
                       fingerprint=fingerprint(config))
        meta.update(summary, complete=True)
        save_json(root / "metadata.json", meta)
        save_json(root / "summary.json", summary)
        torch.save({"model": model._module.state_dict()}, root / "final_state.pt")
    finally:
        model.remove_hooks()
        write_csv(root / "train_metrics.csv", rows,
                  fields=None if rows else ["step", "seed", "method"])
        write_csv(root / "eval_metrics.csv",
                  [row for row in rows if row["test_accuracy"] is not None])
        write_csv(root / "refresh_metrics.csv", refresh_rows,
                  fields=None if refresh_rows else ["step", "layer"])
        save_json(root / "pairing.json", dict(private=private_audit,
                                               synthetic=synthetic_audit))
    return meta


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/full.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    args = parser.parse_args()
    config = read_config(args.config)
    seeds = config["seeds"] if args.seed is None else [args.seed]
    methods = METHODS if args.method == "all" else [args.method]
    for seed in seeds:
        for method in methods:
            train(config, seed, method, args.output)


if __name__ == "__main__":
    main()
