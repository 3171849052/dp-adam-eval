"""Four mechanism figures and a separate reference performance figure."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exp4.common import ROOT, TRAJECTORIES, read_config
from exp4.analyze_exp4 import load_runs

COLORS = {'dp': '#d97919', 'syn': '#187e9e', 'adam': '#545454'}


def curve(ax, df, column, label, color, *, noisy=False, points=False):
    for _, seed in df.groupby('seed'):
        valid = seed.dropna(subset=[column])
        ax.plot(valid.step, valid[column], color=color, alpha=.10 if noisy else .17, lw=.5)
    mean = df.groupby('step')[column].mean().dropna()
    ax.plot(mean.index, mean, label=label, color=color, lw=.9 if noisy else 1.4,
            alpha=.35 if noisy else 1, linestyle='--' if noisy else '-',
            marker='o' if points else None, markersize=3)


def save(fig, out, name):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def decorate(ax, title, ylabel):
    ax.set(title=title, xlabel='Private step', ylabel=ylabel)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)


def plot(df, ev, out):
    out = Path(out)/'figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, t in zip(axes, TRAJECTORIES):
        frame = df[df.trajectory == t]
        for method, label in [('dp', 'DP-Adam'), ('syn', 'SynDiag')]:
            curve(ax, frame, f'cos_{method}_adam_current_noise_off', label+' current noise off', COLORS[method])
            curve(ax, frame, f'cos_{method}_adam_noisy', label+' noisy', COLORS[method], noisy=True)
        decorate(ax, t+' trajectory', 'Cosine with Adam direction')
    save(fig, out, 'figure1_adam_alignment')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, t in zip(axes, TRAJECTORIES):
        frame = df[df.trajectory == t]
        curve(ax, frame, 'alignment_advantage_current_noise_off', 'Current noise off', COLORS['syn'])
        curve(ax, frame, 'alignment_advantage_noisy', 'Noisy', COLORS['dp'], noisy=True)
        ax.axhline(0, color='black', lw=.8)
        decorate(ax, t+' trajectory', 'Alignment advantage: Syn − DP')
    save(fig, out, 'figure2_alignment_advantage')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, t in enumerate(TRAJECTORIES):
        frame = df[df.trajectory == t]
        for row, prefix, ylabel in [(0, 'noise_degradation', 'Current-step noise degradation'), (1, 'clip_cosine', 'Clipping cosine (own geometry)')]:
            for method, label in [('dp', 'DP-Adam'), ('syn', 'SynDiag')]:
                curve(axes[row, col], frame, f'{prefix}_{method}', label, COLORS[method])
            if row == 0:
                axes[row, col].axhline(0, color='black', lw=.8)
            decorate(axes[row, col], t+' trajectory', ylabel)
    save(fig, out, 'figure3_noise_and_clipping')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, t in enumerate(TRAJECTORIES):
        frame = df[df.trajectory == t]
        for method, label in [('dp', 'DP-Adam'), ('syn', 'SynDiag')]:
            curve(axes[0, col], frame, f'norm_ratio_{method}_adam', label, COLORS[method])
        axes[0, col].set_yscale('log')
        decorate(axes[0, col], t+' trajectory', 'Direction norm / Adam norm (log)')
        for method, label in [('adam', 'Adam'), ('dp', 'DP-Adam'), ('syn', 'SynDiag')]:
            curve(axes[1, col], frame, f'loss_progress_{method}', label, COLORS[method], points=True)
        axes[1, col].axhline(0, color='black', lw=.8)
        decorate(axes[1, col], t+' trajectory', 'Loss decrease / parameter movement')
    save(fig, out, 'figure4_magnitude_and_progress')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ['train_loss', 'test_loss', 'test_accuracy']):
        for t, color in [('adam', COLORS['adam']), ('dp_adam', COLORS['dp'])]:
            curve(ax, ev[ev.trajectory == t], metric, t+' reference', color)
        decorate(ax, 'Reference performance', metric.replace('_', ' '))
    save(fig, out, 'aux_reference_performance')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/full.json'))
    p.add_argument('--runs', required=True)
    a = p.parse_args()
    df, ev, _, _ = load_runs(read_config(a.config), a.runs)
    plot(df, ev, a.runs)
    print('Saved four mechanism figures and one auxiliary figure (PNG/PDF).')


if __name__ == '__main__':
    main()
