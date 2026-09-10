"""Merge existing ExpV5 endpoints with the six new partial-filter arms."""
import argparse
import json
from pathlib import Path
import pandas as pd
from expv3.common import save_json
from expv5.summarize_expv5 import UTILITY, MECHANISM as V5_MECHANISM
from expv6.train_expv6 import FAMILIES, arm_name

MECHANISM = [k for k in V5_MECHANISM if k != 'gamma_model_raw'] + ['clean_gradient_distortion']


def sweep_roots(config, runs, expv5_runs):
    for seed in config['seeds']:
        for family in FAMILIES:
            for alpha in (0., *config['alphas'], 1.):
                arm = ('dp_adam' if alpha == 0 else f'dp_fisher_wiener_{family}_adam') if alpha in (0, 1) else arm_name(family, alpha)
                source = Path(expv5_runs if alpha in (0, 1) else runs)
                yield dict(seed=seed, family=family, alpha=alpha, run_id=arm), source / f'seed{seed}' / arm


def with_mean(frame, groups, metrics):
    means = frame.groupby(groups, as_index=False)[metrics].mean()
    means['seed'] = 'mean'
    return pd.concat([frame, means], ignore_index=True)


def summarize(config, runs, expv5_runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    utility, mechanism = [], []
    for labels, root in sweep_roots(config, runs, expv5_runs):
        summary = json.loads((root / 'summary.json').read_text())
        assert summary['status'] == 'completed'
        utility.append(dict(labels, **{k: summary[k] for k in UTILITY}))
        layers = pd.read_csv(root / 'layer_metrics.csv')
        if labels['alpha'] == 0:
            layers['clean_gradient_distortion'] = 0.
            layers['beta_train'] = float('nan')
            layers['filter_gradient_norm_ratio'] = 1.
        if labels['alpha'] == 1:
            layers['clean_gradient_distortion'] = float('nan')
        for layer, values in layers.groupby('layer'):
            mechanism.append(dict(labels, layer=layer, **values[MECHANISM].mean().to_dict()))
    utility = pd.DataFrame(utility)
    utility[UTILITY] = utility[UTILITY].apply(pd.to_numeric)
    with_mean(utility, ['family', 'alpha'], UTILITY).to_csv(output / 'utility_alpha_sweep.csv', index=False)
    with_mean(pd.DataFrame(mechanism), ['family', 'alpha', 'layer'], MECHANISM).to_csv(output / 'mechanism_alpha_sweep.csv', index=False)
    for family in FAMILIES:
        values = utility[utility.family == family].set_index(['seed', 'alpha'])[UTILITY]
        baseline = utility[(utility.family == family) & (utility.alpha == 0)].set_index('seed')[UTILITY]
        paired = values.sub(baseline, level='seed').reset_index()
        label = 'adaptive' if family == 'adaptive_beta' else family
        with_mean(paired, ['alpha'], UTILITY).to_csv(output / f'paired_{label}_vs_dp_by_alpha.csv', index=False)
    left, right = [utility[utility.family == f].set_index(['seed', 'alpha'])[UTILITY] for f in FAMILIES]
    with_mean((right-left).reset_index(), ['alpha'], UTILITY).to_csv(output / 'paired_adaptive_minus_beta1_by_alpha.csv', index=False)
    means = utility.groupby(['family', 'alpha']).accuracy_auc.mean()
    best = {f: dict(alpha=float(means.loc[f].idxmax()), mean_accuracy_auc=float(means.loc[f].max())) for f in FAMILIES}
    result = dict(smoke_functional_only=config['smoke'], best_mean_auc=best)
    save_json(output / 'summary.json', result)
    report = json.dumps(result, indent=2) + '\nExpV5 alpha=1 clean_gradient_distortion was not recorded; left missing.\nThreshold differences remain missing when not reached.\n'
    if config['smoke']:
        report += 'Functional smoke only; no scientific conclusion.\n'
    (output / 'summary.txt').write_text(report)
    print(report)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for arg in ('config', 'runs', 'expv5-runs', 'output'):
        parser.add_argument('--' + arg, required=True)
    args = parser.parse_args()
    summarize(json.loads(Path(args.config).read_text()), args.runs, args.expv5_runs, args.output)


if __name__ == '__main__':
    main()
