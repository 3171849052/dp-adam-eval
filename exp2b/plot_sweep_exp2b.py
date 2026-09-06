"""Generate matplotlib lambda x K heatmaps after validating all source runs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import LAYERS
from summarize_exp2b import arguments, summarize


def plot_sweep(frame, out):
    out.mkdir(parents=True, exist_ok=True)
    columns = ['final_test_accuracy', 'late_mean_accuracy',
               *(f'final_R_{n}' for n in LAYERS),
               *(f'final_attenuation_q50_{n}' for n in LAYERS),
               'final_coefficient_cv', 'final_clipping_shape_error']
    for col in columns:
        pivot = frame.pivot(index='lambda', columns='K', values=col).sort_index().sort_index(axis=1)
        values = pivot.to_numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(pivot.columns)), labels=[str(k) for k in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), labels=[f'{x:.0e}' for x in pivot.index])
        ax.set(xlabel='K (private steps)', ylabel='lambda', title=col)
        for i, j in np.ndindex(values.shape):
            ax.text(j, i, f'{values[i,j]:.4g}' if np.isfinite(values[i,j]) else 'N/A',
                    ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=.7, edgecolor='none'))
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / f'{col}.png', dpi=150)
        plt.close(fig)
    print(f'Saved {len(columns)} sweep heatmaps: {out}')


if __name__ == '__main__':
    args = arguments()
    frame = summarize(args.config_dir, args.output, args.smoke)
    plot_sweep(frame, args.output.parent / 'sweep_figures')
