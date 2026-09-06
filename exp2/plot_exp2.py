"""Plot all eight Exp2 diagnostic groups and validate paired result files."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import parse, run_root, save_json, LAYERS


def plot(c):
    root = run_root(c)
    out = root / 'figures'
    out.mkdir(exist_ok=True)
    tables = {k: pd.read_csv(root / 'results' / f'{k}_metrics.csv') for k in ('train', 'refresh', 'oracle')}
    summary = json.loads((root / 'results/summary.json').read_text())
    for method in ('dp_sgd', 'syn_diag'):
        assert summary[method]['config'] == c, 'Mismatched run configuration'
        train = tables['train'].query('method == @method')
        assert train.step.tolist() == list(range(1, summary[method]['total_steps']+1))
        required = ['train_loss', 'clip_rate', 'norm_q50', 'coefficient_q50', 'relative_distortion',
                    'coefficient_std', 'coefficient_cv', 'clipping_alpha_star', 'clipping_shape_error',
                    'actual_noise_norm', 'expected_noise_norm', 'update_norm', 'wall_seconds']
        assert np.isfinite(train[required].to_numpy()).all()
        assert (train.actual_noise_norm > 0).all() and (train.update_norm > 0).all()
        assert np.allclose(train[[f'contrib_{n}' for n in LAYERS]].sum(axis=1), 1, atol=1e-5)
        expected = list(range(0, summary[method]['total_steps'], c['K']))
        for key in ('refresh', 'oracle'):
            frame = tables[key].query('method == @method')
            for name in LAYERS:
                assert frame.query('layer == @name').step.tolist() == expected
        later = tables['refresh'].query('method == @method and step > 0')
        assert np.isfinite(later[['D_time', 'stale_ratio']].to_numpy()).all()
    a = json.loads((root / 'results/pairing_dp_sgd.json').read_text())
    b = json.loads((root / 'results/pairing_syn_diag.json').read_text())
    assert a == b, 'Dataloader order or DP noise stream differs'
    assert summary['dp_sgd']['initial_model_hash'] == summary['syn_diag']['initial_model_hash']
    assert summary['dp_sgd']['noise_multiplier'] == summary['syn_diag']['noise_multiplier']

    def figure(filename, table, columns, layers=False, syn_only=False):
        fig, axes = plt.subplots(len(columns), 1, figsize=(9, 3*len(columns)), squeeze=False)
        data = tables[table]
        if syn_only:
            data = data.query('method == "syn_diag"')
        keys = ['method', 'layer'] if layers else ['method']
        for ax, col in zip(axes[:, 0], columns):
            for label, group in data.groupby(keys):
                group = group.dropna(subset=[col])
                ax.plot(group.step, group[col], marker='o', markersize=3,
                        label='/'.join(label) if isinstance(label, tuple) else label)
            ax.set(xlabel='Private updates completed', ylabel=col)
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f'{filename}.png', dpi=150)
        plt.close(fig)

    figure('01_accuracy_loss', 'train', ['test_accuracy', 'test_loss', 'train_loss'])
    figure('02_clipping', 'train', ['clip_rate', 'norm_q50', 'coefficient_q50', 'coefficient_cv'])
    figure('03_distortion', 'train', ['aggregate_cosine', 'relative_distortion', 'clipping_shape_error'])
    figure('04_layer_contribution', 'train', [f'contrib_{n}' for n in LAYERS])
    figure('05_oracle_alignment', 'oracle', ['R', 'log_pearson'], layers=True)
    # One axis per layer puts sample variability and temporal drift on the same scale.
    fig, axes = plt.subplots(4, 1, figsize=(9, 12))
    for ax, name in zip(axes, LAYERS):
        frame = tables['refresh'].query('method == "syn_diag" and layer == @name')
        for col in ('D_sample', 'D_time'):
            ax.plot(frame.step, frame[col], 'o-', label=col)
        ax.set(xlabel='Private updates completed', ylabel=f'{name}: RMS centered log')
        ax.legend()
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / '06_ema_diagnostic.png', dpi=150)
    plt.close(fig)
    figure('07_staleness', 'refresh', ['stale_ratio'], layers=True, syn_only=True)
    figure('08_lambda', 'refresh', ['lambda_over_median', 'fraction_below_lambda'], layers=True, syn_only=True)
    figure('09_signal_noise', 'train', ['clipped_aggregate_norm', 'actual_noise_norm', 'expected_noise_norm', 'diagnostic_snr'])
    save_json(root / 'validation.json', {'passed': True, 'paired_initialization': True,
                                       'paired_batch_order_and_noise_rng': True,
                                       'all_four_layers': True, 'figures': 9,
                                       'rows': {k: len(v) for k, v in tables.items()}})
    print(f'Validated paired results and saved 9 figures: {out}', flush=True)


if __name__ == '__main__':
    plot(parse())
