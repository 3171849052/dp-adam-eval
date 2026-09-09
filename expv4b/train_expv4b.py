"""Three paired SGD trajectories with raw model gamma applied after Wiener."""
import argparse
import json
import sys
from pathlib import Path

import torch

from expv1.fisher_wiener import apply_filter_to_copy, pack_layer_gradient
from expv1.metrics import diagnose, reconstruction
from expv3.common import DP_METHOD, FISHER_METHOD, FULL_CONFIG as V3_FULL, LAYERS
from expv3.train_expv3 import train as train_v3
from expv4a.gamma_estimators import model_gamma_from_state, multiplicative_error, ratio_sqrt

ARMS = (
    "dp_sgd_lr0p50",
    "dp_fisher_wiener_beta1_gamma_lr0p50",
    "dp_fisher_wiener_adaptive_beta_gamma_lr0p50",
)
FULL_CONFIG = {k: v for k, v in V3_FULL.items()
               if k not in ("dp_sgd_learning_rate", "fisher_learning_rates")}
FULL_CONFIG.update(experiment="expv4b", seeds=[123, 456, 789, 1024, 2027], learning_rate=0.5)
SMOKE_CONFIG = dict(FULL_CONFIG, seeds=[42], beta_window=2, epochs=1, batch_size=4,
                    K=2, M_syn=8, eval_interval=1, train_subset=16, test_subset=32, smoke=True)
GAMMA_FIELDS = [
    "beta_train", "gamma_model", "filtered_gradient_norm_before_gamma",
    "compensated_gradient_norm", "gamma_update_ratio", "gamma_oracle",
    "model_to_oracle_ratio", "model_multiplicative_error",
    "signal_retention_after_gamma", "noise_retention_after_gamma",
    "relmse_after_gamma", "cosine_after_gamma", "snr_gain_db_after_gamma",
]


def check_config(config):
    assert config["experiment"] == "expv4b"
    assert config["learning_rate"] == 0.5
    assert config["beta_initial"] == 1.0
    assert config["beta_update_rule"] == "one_interval_lag_hold_last_positive"
    assert config["beta_window"] == config["K"]
    assert config["optimizer"] == "SGD" and config["momentum"] == 0
    torch.set_num_threads(config["threads"])
    return config


def run_specs(config):
    return [dict(run_id=name, method=DP_METHOD if i == 0 else FISHER_METHOD,
                 learning_rate=config["learning_rate"], adaptive_beta=i == 2, gamma=i > 0)
            for i, name in enumerate(ARMS)]


def refresh_gamma(active, betas, step, interval, spec):
    """Read only the currently active synthetic eigensystem, H, and beta."""
    gamma, rows = {}, []
    for name in LAYERS:
        stats = model_gamma_from_state(active[name], betas[name])
        gamma[name] = stats["gamma_model_raw"]
        rows.append(dict(seed=spec["seed"], run_id=spec["run_id"], layer=name,
                         step=step, interval_index=interval, beta_train=betas[name],
                         gamma_model=gamma[name], **stats))
    return gamma, rows


@torch.no_grad()
def compensate(model, gamma, betas):
    """Scale the optimizer's existing Wy in place. Gamma is entirely uncapped."""
    base = getattr(model, "_module", model)
    result = {}
    for name, value in gamma.items():
        layer = getattr(base, name)
        before = pack_layer_gradient(layer).clone()
        for parameter in layer.parameters():
            parameter.grad.mul_(value)
        before_norm = float(before.double().norm())
        after_norm = float(pack_layer_gradient(layer).double().norm())
        result[name] = dict(before=before, beta_train=betas[name], gamma_model=value,
                            filtered_gradient_norm_before_gamma=before_norm,
                            compensated_gradient_norm=after_norm,
                            gamma_update_ratio=after_norm / before_norm if before_norm else None)
    return result


@torch.no_grad()
def diagnose_compensated(model, noisy, active, diagnostic, method, refresh, eigen_budget,
                         *, gamma, gamma_metrics):
    """Oracle diagnostics run after optimizer.step and never return control values."""
    rows, layers, bins = diagnose(
        model, noisy, active, diagnostic, method, refresh, eigen_budget,
        filtered_gradients={name: value["before"] for name, value in gamma_metrics.items()})
    base = getattr(model, "_module", model)
    all_values = []
    for row in layers:
        name = row["layer"]
        g = gamma[name]
        s = pack_layer_gradient(getattr(base, name), "summed_grad")
        y = noisy[name]
        values = (s, y, pack_layer_gradient(getattr(base, name)),
                  g * apply_filter_to_copy(s, active[name], method),
                  g * apply_filter_to_copy(y - s, active[name], method))
        after = reconstruction(*values)
        all_values.append(values)
        oracle = ratio_sqrt(1.0, row["signal_retention"])
        row.update({k: v for k, v in gamma_metrics[name].items() if k != "before"})
        row.update(gamma_oracle=oracle,
                   model_to_oracle_ratio=g / oracle if oracle else None,
                   model_multiplicative_error=multiplicative_error(g, oracle),
                   signal_retention_after_gamma=g*g*row["signal_retention"],
                   noise_retention_after_gamma=g*g*row["noise_retention"],
                   relmse_after_gamma=after["relmse_filtered"],
                   cosine_after_gamma=after["cosine_filtered"],
                   snr_gain_db_after_gamma=after["snr_gain_db"])
    after = reconstruction(*(torch.cat([v[i].flatten() for v in all_values]) for i in range(5)))
    rows.update(filtered_gradient_norm_before_gamma=rows["filtered_gradient_norm"],
                compensated_gradient_norm=after["filtered_gradient_norm"],
                gamma_update_ratio=after["filtered_gradient_norm"] / rows["filtered_gradient_norm"],
                signal_retention_after_gamma=after["signal_retention"],
                noise_retention_after_gamma=after["noise_retention"],
                relmse_after_gamma=after["relmse_filtered"],
                cosine_after_gamma=after["cosine_filtered"],
                snr_gain_db_after_gamma=after["snr_gain_db"])
    return rows, layers, bins


def train(config, seed, run_name, output, data_override=None):
    return train_v3(config, seed, run_name, output, data_override=data_override,
                    experiment=sys.modules[__name__])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", choices=("all", *ARMS), default="all")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = check_config(json.loads(Path(args.config).read_text()))
    for seed in config["seeds"] if args.seed is None else [args.seed]:
        for name in ARMS if args.run == "all" else [args.run]:
            train(config, seed, name, args.output)


if __name__ == "__main__":
    main()
