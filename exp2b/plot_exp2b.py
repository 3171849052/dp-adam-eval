"""Plot Exp2-style diagnostic groups for a validated SynDiag run."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import parse, run_root, save_json, LAYERS
from summarize_exp2b import validate_run


def plot(c):
    root = run_root(c)
    out = root / 'figures'
    out.mkdir(exist_ok=True)
    tables, _ = validate_run(c)

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
    fig.savefig(out / '06_sample_time_diagnostic.png', dpi=150)
    plt.close(fig)
    figure('07_staleness', 'refresh', ['stale_ratio'], layers=True, syn_only=True)
    figure('08_lambda', 'refresh', ['lambda_over_median', 'fraction_below_lambda'], layers=True, syn_only=True)
    figure('09_signal_noise', 'train', ['clipped_aggregate_norm', 'actual_noise_norm', 'expected_noise_norm', 'diagnostic_snr'])
    figure('10_attenuation', 'refresh', ['attenuation_q05', 'attenuation_q50', 'attenuation_q95'], layers=True)
    figure('11_anisotropy', 'refresh', ['A_old', 'A_new'], layers=True)
    figure('12_P_quantiles', 'refresh', ['P_q05', 'P_q50', 'P_q95'], layers=True)
    save_json(root / 'validation.json', {'passed': True, 'all_four_layers': True,
                                       'figures': 12, 'rows': {k: len(v) for k, v in tables.items()}})
    print(f'Validated SynDiag results and saved 12 figures: {out}', flush=True)


if __name__ == '__main__':
    plot(parse())
