"""Partial Fisher-Wiener filtering with the ExpV5 Adam protocol."""
import argparse
import json
import sys
from pathlib import Path
import torch
from expv1.fisher_wiener import pack_layer_gradient, apply_filter_to_copy
from expv3.common import FISHER_METHOD
from expv3.train_expv3 import train as train_v3
from expv5 import train_expv5 as v5

FAMILIES = ('beta1', 'adaptive_beta')
ALPHAS = (.25, .5, .75)
GAMMA_FIELDS = v5.ADAM_FIELDS + ['filter_gradient_norm_ratio', 'beta_train', 'alpha',
                                'clean_gradient_distortion']
make_optimizer = v5.make_optimizer


def arm_name(family, alpha):
    return f'dp_fisher_wiener_{family}_alpha{alpha:.2f}_adam'.replace('.', 'p')


ARMS = tuple(arm_name(f, a) for f in FAMILIES for a in ALPHAS)


def check_config(config):
    assert config['experiment'] == 'expv6'
    assert config['alphas'] == list(ALPHAS)
    v5.check_config(dict(config, experiment='expv5'))
    return config


def run_specs(config):
    return [dict(run_id=arm_name(f, a), method=FISHER_METHOD,
                 learning_rate=config['learning_rate'], adaptive_beta=f == 'adaptive_beta',
                 gamma=False, alpha=a) for f in FAMILIES for a in config['alphas']]


def interpolate_state(active, alpha):
    # Called once per refresh, after the inherited beta controller constructs H.
    return {name: dict(state, H=(1-alpha) + alpha*state['H'], alpha=alpha)
            for name, state in active.items()}


@torch.no_grad()
def decorate_metrics(rows, layers, optimizer, active, betas):
    # Reuse Adam measurements without the V5 gamma diagnostic.
    v5.decorate_metrics(rows, layers, optimizer, None, betas)
    clean, errors = [], []
    for row in layers:
        name = row['layer']
        params = optimizer.layers[name]
        s = torch.cat([p.summed_grad.reshape(p.shape[0], -1) for p in params], dim=1)
        filtered = apply_filter_to_copy(s, active[name], 'dp_fisher_wiener')
        error = (filtered-s).double()
        clean.append(s.double().flatten())
        errors.append(error.flatten())
        row.update(alpha=active[name]['alpha'], beta_train=betas[name],
                   filter_gradient_norm_ratio=row['filtered_gradient_norm']/row['noisy_gradient_norm'],
                   clean_gradient_distortion=float(error.norm()/s.double().norm()))
    rows.update(alpha=next(iter(active.values()))['alpha'],
                filter_gradient_norm_ratio=rows['filtered_gradient_norm']/rows['noisy_gradient_norm'],
                clean_gradient_distortion=float(torch.cat(errors).norm()/torch.cat(clean).norm()))


def train(config, seed, run_name, output, data_override=None):
    return train_v3(config, seed, run_name, output, data_override=data_override,
                    experiment=sys.modules[__name__])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run', choices=('all', *ARMS), default='all')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    config = check_config(json.loads(Path(args.config).read_text()))
    for seed in config['seeds'] if args.seed is None else [args.seed]:
        for arm in ARMS if args.run == 'all' else [args.run]:
            train(config, seed, arm, args.output)


if __name__ == '__main__':
    main()
