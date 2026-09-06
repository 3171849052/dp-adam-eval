"""Validate every requested run before producing an objective factorial summary."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, LAYERS, fingerprint

LAMBDAS = {1e-4: '1e4', 3e-4: '3e4', 1e-3: '1e3'}
KS = (50, 100, 200)
FINAL_TRAIN = ('coefficient_cv', 'aggregate_cosine', 'clipping_shape_error', 'diagnostic_snr')
TRAIN_REQUIRED = ('train_loss', 'clip_rate', 'norm_q50', 'coefficient_q50',
                  'relative_distortion', 'coefficient_std', 'clipping_alpha_star',
                  'actual_noise_norm', 'expected_noise_norm', 'update_norm',
                  'wall_seconds', 'clipped_aggregate_norm', *FINAL_TRAIN)
REFRESH_REQUIRED = ('lambda_over_median', 'fraction_below_lambda', 'D_sample', 'A_new',
                    'attenuation_q05', 'attenuation_q50', 'attenuation_q95',
                    *(f'{prefix}_q{q:02d}' for prefix in ('v_syn', 'sqrt_v', 'P') for q in (1, 5, 50, 95, 99)))
ORACLE_REQUIRED = ('R', 'log_pearson', 'log_spearman', 'log_ratio_spread', 'A_raw', 'A_residual')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_refresh_steps(total, k):
    return list(range(0, total, k))


def load_grid(directory, smoke=False):
    paths = sorted(Path(directory).glob('*.json'))
    expected = {(1e-4, 2), (1e-3, 4)} if smoke else {(lam, k) for lam in LAMBDAS for k in KS}
    require(len(paths) == len(expected), f'Expected exactly {len(expected)} configs in {directory}')
    configs, outputs, combos = [], set(), set()
    fixed = dict(seed=42, learning_rate=.1, epsilon=1., delta=1e-5, max_grad_norm=1.,
                 batch_size=4 if smoke else 256, epochs=1 if smoke else 5,
                 M_syn=8 if smoke else 2560, M_oracle=8 if smoke else 512,
                 train_subset=16 if smoke else None, test_subset=32 if smoke else None,
                 oracle_enabled=True)
    for path in paths:
        c = json.loads(path.read_text())
        pair = (c['lambda'], c['K'])
        require(pair in expected and pair not in combos, f'Unexpected/duplicate lambda,K: {path}')
        require(path.stem == f"lambda{LAMBDAS[c['lambda']]}_K{c['K']}", f'Filename mismatch: {path}')
        require(all(c.get(key) == val for key, val in fixed.items()), f'Fixed settings mismatch: {path}')
        out = (ROOT / c['output']).resolve()
        require(out.is_relative_to(ROOT / 'runs') and out.name == path.stem, f'Invalid output: {path}')
        require(out not in outputs, f'Duplicate output: {out}')
        outputs.add(out)
        combos.add(pair)
        configs.append(c)
    require(combos == expected, 'Incomplete grid')
    return configs


def finite(frame, columns, context):
    require(set(columns) <= set(frame.columns), f'{context}: missing required columns')
    require(np.isfinite(frame[list(columns)].to_numpy(dtype=float)).all(), f'{context}: nonfinite metrics')



def validate_contributions(train, context='layer contributions'):
    columns = [f'contrib_{n}' for n in LAYERS]
    finite(train, columns, context)
    values = train[columns].to_numpy(dtype=float)
    # Sum over layers = mean_i(sq_i / max(sq_i, eps_num)).
    # Zero/tiny sample gradients legitimately reduce the sum below one.
    # CSVs do not retain individual sq_i, so exact equality is not recoverable.
    # Allow FP32 summation roundoff only at the upper bound.
    valid = (values >= 0).all(axis=1) & (values.sum(axis=1) <= 1 + 1e-5)
    require(valid.all(), f'{context}: invalid layer contributions at steps '
            f'{train.loc[~valid, "step"].head(10).tolist()}; expected nonnegative values with sum <= 1')


def validate_run(c):
    root = (ROOT / c['output']).resolve()
    require(root.is_relative_to(ROOT / 'runs'), 'Output outside exp2b/runs')
    try:
        summary = json.loads((root / 'results/summary.json').read_text())
        require(set(summary) == {'syn_diag'}, f'{root}: unexpected methods')
        meta = json.loads((root / 'metadata_syn_diag.json').read_text())
        saved = json.loads((root / 'config_syn_diag.json').read_text())
        require(summary['syn_diag'] == meta and saved == c and meta['config'] == c,
                f'{root}: mismatched configuration/summary')
        require(meta['fingerprint'] == fingerprint(c) and meta['method'] == 'syn_diag', f'{root}: provenance mismatch')
        total = (c['train_subset'] or 60000) // c['batch_size'] * c['epochs']
        require(meta['completed_steps'] == total == meta['total_steps'], f'{root}: incomplete run; expected {total} steps')
        tables = {}
        for kind in ('train', 'refresh', 'oracle'):
            frame = pd.read_csv(root / 'results' / f'{kind}_metrics_syn_diag.csv')
            require(not frame.empty and set(frame.method) == {'syn_diag'}, f'{kind}: invalid method')
            require(set(frame.config_fingerprint) == {fingerprint(c)}, f'{kind}: mixed config results')
            joint = pd.read_csv(root / 'results' / f'{kind}_metrics.csv')
            require(frame.equals(joint), f'{kind}: joint CSV mismatch')
            tables[kind] = frame
        train = tables['train']
        require(train.step.tolist() == list(range(1, total+1)), f'{root}: missing/duplicate train steps')
        finite(train, TRAIN_REQUIRED, 'train')
        eval_steps = [s for s in range(1, total+1) if s % c['eval_interval'] == 0 or s == total]
        require(train.loc[train.test_accuracy.notna(), 'step'].tolist() == eval_steps, 'Missing/unexpected evaluation')
        require(train.test_accuracy.isna().equals(train.test_loss.isna()), 'Unpaired evaluation metrics')
        finite(train[train.step.isin(eval_steps)], ('test_accuracy', 'test_loss'), 'eval')
        validate_contributions(train, str(root))
        for kind in ('refresh', 'oracle'):
            frame = tables[kind]
            require(set(frame.layer) == set(LAYERS), f'{kind}: expected all four layers')
            for layer in LAYERS:
                require(frame.loc[frame.layer == layer, 'step'].tolist() == expected_refresh_steps(total, c['K']),
                        f'{kind}/{layer}: invalid refresh schedule')
        refresh = tables['refresh']
        finite(refresh, REFRESH_REQUIRED, 'refresh')
        finite(refresh[refresh.step > 0], ('D_time', 'A_old', 'stale_ratio'), 'later refresh')
        require(refresh.loc[refresh.step == 0, ['D_time', 'A_old', 'stale_ratio']].isna().all().all(), 'Initial old geometry must be empty')
        a = refresh[['attenuation_q05', 'attenuation_q50', 'attenuation_q95']].to_numpy()
        require(((a >= 0) & (a <= 1)).all() and (np.diff(a, axis=1) >= 0).all(), 'Invalid attenuation quantiles')
        finite(tables['oracle'], ORACLE_REQUIRED, 'oracle')
        return tables, total
    except (OSError, KeyError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f'{root}: incomplete or malformed run: {exc}') from exc


def aggregate(c, tables, total):
    train = tables['train'].sort_values('step')
    evaluations = train.dropna(subset=['test_accuracy'])
    late = evaluations[evaluations.step > total / 2]
    require(not late.empty, 'No evaluation in second half of training')
    row = {'lambda': c['lambda'], 'K': c['K'], 'output': c['output'],
           'final_test_accuracy': float(train.iloc[-1].test_accuracy),
           'final_test_loss': float(train.iloc[-1].test_loss),
           'best_test_accuracy': float(evaluations.test_accuracy.max()),
           'late_mean_accuracy': float(late.test_accuracy.mean())}
    row.update({f'final_{key}': float(train.iloc[-1][key]) for key in FINAL_TRAIN})
    for layer in LAYERS:
        for kind, columns in [('oracle', ('R',)), ('refresh', ('attenuation_q50', 'A_old', 'A_new'))]:
            last = tables[kind].query('layer == @layer').sort_values('step').iloc[-1]
            row.update({f'final_{col}_{layer}': float(last[col]) for col in columns})
    return row


def summarize(config_dir, output, smoke=False):
    rows = []
    for c in load_grid(config_dir, smoke):
        tables, total = validate_run(c)
        rows.append(aggregate(c, tables, total))
    frame = pd.DataFrame(rows).sort_values(['lambda', 'K'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f'Validated {len(frame)} complete runs; summary: {output}')
    return frame


def arguments():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--smoke', action='store_true', help='Explicit two-config smoke grid, four steps per run')
    p.add_argument('--config-dir', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    args.config_dir = args.config_dir or ROOT / ('smoke_configs' if args.smoke else 'configs')
    args.output = args.output or ROOT / ('runs/smoke_validation_01/sweep_summary.csv' if args.smoke else 'sweep_summary.csv')
    return args


if __name__ == '__main__':
    args = arguments()
    summarize(args.config_dir, args.output, args.smoke)
