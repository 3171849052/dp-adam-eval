"""Produce the five requested figure groups, with Pink seed mean ± std."""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import parse, run_root, fingerprint, LAYERS

COLORS = {'private': 'black', 'public': 'tab:blue', 'pink': 'tab:red'}


def trend(ax, df, family, metric, private=False):
    sub = df[(df.family == family) & (df.metric == metric)]
    for group in (['private'] if private else []) + ['public', 'pink']:
        select = sub[sub.source.str.startswith(group)]
        table = select.groupby('checkpoint').value.agg(['mean', 'std']).sort_index()
        if table.empty:
            continue
        x, mean = table.index.to_numpy(), table['mean'].to_numpy()
        std = table['std'].fillna(0).to_numpy()
        ax.plot(x, mean, marker='.', label='Oracle' if group == 'private' else group.title(), color=COLORS[group])
        if group == 'pink':
            ax.fill_between(x, mean-std, mean+std, color=COLORS[group], alpha=.15)
    ax.set_xlabel('Training checkpoint (%)')
    ax.set_ylabel(metric)
    ax.grid(alpha=.2)


def plot(c):
    root = run_root(c)
    results, figures = root / 'results', root / 'figures'
    meta = json.loads((results / 'analysis.json').read_text())
    if meta['fingerprint'] != fingerprint(c):
        raise ValueError('Analysis configuration mismatch')
    figures.mkdir(exist_ok=True)
    df = pd.read_csv(results / 'metrics.csv')
    if len(df) != meta['rows'] or set(df.checkpoint) != set(range(0, 101, 10)):
        raise ValueError('Incomplete analysis table; rerun analyze_exp1.py')

    def finish(fig, name):
        for ax in fig.axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / f'{name}.png', dpi=160)
        fig.savefig(figures / f'{name}.pdf')
        plt.close(fig)

    for variant in ('raw', 'repo'):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        for col, layer in enumerate(LAYERS):
            sub = df[df.layer == layer]
            for row, factor in enumerate(('A', 'G')):
                trend(axes[row, col], sub, f'factor_{factor}_{variant}', 'cosine')
                axes[row, col].set_title(f'{layer}: cos({factor}) [{variant}]')
        finish(fig, f'figure1_kfac_{variant}')
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for col, layer in enumerate(LAYERS):
        for row, metric in enumerate(('cosine', 'shape_error')):
            trend(axes[row, col], df[df.layer == layer], 'empirical', metric)
            axes[row, col].set_title(layer)
            axes[row, col].set_xticks([0, 50, 100])
    finish(fig, 'figure2_empirical_alignment')

    # Full spectra stored as eigenvalue vectors only; each Pink seed is visible.
    for variant in ('raw', 'repo'):
        fig, axes = plt.subplots(3, 4, figsize=(16, 11))
        for row, percent in enumerate((0, 50, 100)):
            for path in sorted((results / f'checkpoint_{percent:03d}').glob('*.pt')):
                obj = torch.load(path, weights_only=True)
                source = obj['source']
                group = 'pink' if source.startswith('pink') else source
                for col, layer in enumerate(LAYERS):
                    entry = obj['layers'][layer]
                    for kind, style in [('empirical', '-'), (f'kfac_{variant}', '--')]:
                        eig = entry['spectra'][kind].numpy()
                        # Suppress numerical zeros in empirical nonzero spectrum.
                        tol = entry['empirical_rank_tolerance'] if kind == 'empirical' else 0
                        eig = eig[eig > tol]
                        label = f'{group} {kind}' if group != 'pink' or source == f"pink_{c['pink_seeds'][0]}" else None
                        axes[row, col].loglog(np.arange(1, len(eig)+1), eig, style, color=COLORS[group],
                                             alpha=.55 if group == 'pink' else .85, label=label)
                    axes[row, col].set_title(f'{layer} / {percent}%')
                    axes[row, col].set_xlabel('Eigenvalue rank')
                    axes[row, col].set_ylabel('Eigenvalue')
        finish(fig, f'figure3_spectrum_{variant}')

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for col, layer in enumerate(LAYERS):
        trend(axes[col], df[df.layer == layer], 'diagonal', 'log_ratio_spread')
        axes[col].set_title(layer)
    finish(fig, 'figure4_diagonal_spread')
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for row, percent in enumerate((0, 50, 100)):
        directory = results / f'checkpoint_{percent:03d}'
        priv = torch.load(directory / 'private.pt', weights_only=True)['layers']
        for path in sorted(directory.glob('*.pt')):
            if path.stem == 'private':
                continue
            obj = torch.load(path, weights_only=True)
            group = 'pink' if path.stem.startswith('pink') else 'public'
            for col, layer in enumerate(LAYERS):
                p = np.log10(priv[layer]['v_direct'].numpy() + c['eps_num'])
                s = np.log10(obj['layers'][layer]['v_direct'].numpy() + c['eps_num'])
                # Deterministic display subsampling only; all metrics use every coordinate.
                idx = np.linspace(0, len(p)-1, min(len(p), 2500), dtype=int)
                label = group if group != 'pink' or path.stem == f"pink_{c['pink_seeds'][0]}" else None
                axes[row, col].scatter(p[idx], s[idx], s=1, alpha=.15, color=COLORS[group], label=label)
                axes[row, col].set_title(f'{layer} / {percent}%')
                axes[row, col].set_xlabel('log10(v_private + eps)')
                axes[row, col].set_ylabel('log10(v_source + eps)')
        for col, layer in enumerate(LAYERS):
            p = np.log10(priv[layer]['v_direct'].numpy() + c['eps_num'])
            axes[row, col].plot([p.min(), p.max()], [p.min(), p.max()], 'k:', linewidth=.7)
    finish(fig, 'figure4_diagonal_scatter')
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for col, layer in enumerate(LAYERS):
        sub = df[df.layer == layer]
        trend(axes[0, col], sub, 'diagonal', 'A_residual', private=True)
        raw = sub[(sub.family == 'diagonal') & (sub.metric == 'A_raw') & (sub.source == 'private')]
        axes[0, col].plot(raw.checkpoint, raw.value, 'k--', label='Raw')
        trend(axes[1, col], sub, 'diagonal', 'R', private=True)
        axes[1, col].axhline(1, color='black', linestyle='--', label='Raw R=1')
        axes[0, col].set_title(layer)
    finish(fig, 'figure5_residual_anisotropy')
    print(f'Figures saved to {figures}')


if __name__ == '__main__':
    plot(parse(__doc__))
