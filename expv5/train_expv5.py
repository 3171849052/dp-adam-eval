"""Paired DP-Adam and Fisher-Wiener Adam research experiment."""
import argparse
import json
import sys
from pathlib import Path
import torch
from expv3.common import DP_METHOD, FISHER_METHOD, LAYERS
from expv3.train_expv3 import train as train_v3
from expv4a.gamma_estimators import model_gamma_from_state

ARMS = ("dp_adam", "dp_fisher_wiener_beta1_adam", "dp_fisher_wiener_adaptive_beta_adam")
ADAM_CONFIG = dict(optimizer="Adam", learning_rate=0.001, adam_beta1=0.9,
                   adam_beta2=0.999, adam_eps=1e-8, weight_decay=0)
ADAM_FIELDS = ["gradient_norm_into_adam", "adam_first_moment_norm",
               "adam_second_moment_sqrt_norm", "adam_normalized_update_norm", "parameter_update_norm"]
# Existing V3 writer extension point; gamma is only a diagnostic in V5.
GAMMA_FIELDS = ADAM_FIELDS + ["filter_gradient_norm_ratio", "beta_train", "gamma_model_raw"]


def check_config(config):
    assert config["experiment"] == "expv5"
    assert all(config[k] == v for k, v in ADAM_CONFIG.items())
    assert config["beta_initial"] == 1
    assert config["beta_update_rule"] == "one_interval_lag_hold_last_positive"
    assert config["beta_window"] == config["K"]
    torch.set_num_threads(config["threads"])
    return config


def run_specs(config):
    return [dict(run_id=name, method=DP_METHOD if i == 0 else FISHER_METHOD,
                 learning_rate=config["learning_rate"], adaptive_beta=i == 2, gamma=False)
            for i, name in enumerate(ARMS)]


def norm(values):
    return float(torch.cat([v.detach().double().flatten() for v in values]).norm())


class MeasuredAdam(torch.optim.Adam):
    """Ordinary Adam, with read-only measurements around its actual step."""
    def __init__(self, model, **kwargs):
        super().__init__(model.parameters(), **kwargs)
        base = getattr(model, "_module", model)
        self.layers = {name: list(getattr(base, name).parameters()) for name in LAYERS}

    @torch.no_grad()
    def step(self, closure=None):
        before = {p: p.detach().clone() for group in self.param_groups for p in group["params"]}
        result = super().step(closure)
        beta1, beta2 = self.param_groups[0]["betas"]
        eps = self.param_groups[0]["eps"]
        self.layer_metrics = {}
        for name, params in self.layers.items():
            states = [self.state[p] for p in params]
            normalized = [(s["exp_avg"] / (1 - beta1 ** float(s["step"]))) /
                          ((s["exp_avg_sq"] / (1 - beta2 ** float(s["step"]))).sqrt() + eps)
                          for s in states]
            self.layer_metrics[name] = dict(
                gradient_norm_into_adam=norm([p.grad for p in params]),
                adam_first_moment_norm=norm([s["exp_avg"] for s in states]),
                adam_second_moment_sqrt_norm=norm([s["exp_avg_sq"].sqrt() for s in states]),
                adam_normalized_update_norm=norm(normalized),
                parameter_update_norm=norm([p - before[p] for p in params]))
        return result


def make_optimizer(model, learning_rate, config):
    return MeasuredAdam(model, lr=learning_rate,
                        betas=(config["adam_beta1"], config["adam_beta2"]),
                        eps=config["adam_eps"], weight_decay=config["weight_decay"])


def decorate_metrics(rows, layers, optimizer, active, betas):
    # lr * gradient is not an Adam parameter update.
    for row in [rows, *layers]:
        for key in ("signal_amplitude_retention", "effective_signal_lr", "optimizer_update_norm",
                    "clean_reference_update_norm", "update_to_clean_reference_ratio",
                    "lr_compensation_to_dp_sgd_0p5"):
            row.pop(key, None)
    for row in layers:
        name = row["layer"]
        row.update(optimizer.layer_metrics[name])
        if active is not None:
            row.update(beta_train=betas[name],
                       filter_gradient_norm_ratio=row["filtered_gradient_norm"] / row["noisy_gradient_norm"],
                       gamma_model_raw=model_gamma_from_state(active[name], betas[name])["gamma_model_raw"])
    for field in ADAM_FIELDS:
        rows[field] = sum(v[field] ** 2 for v in optimizer.layer_metrics.values()) ** 0.5


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
        for arm in ARMS if args.run == "all" else [args.run]:
            train(config, seed, arm, args.output)


if __name__ == "__main__":
    main()
