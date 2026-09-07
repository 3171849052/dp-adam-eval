"""Validate paired runs, write late summaries and answer the three mechanism questions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from exp4.common import ROOT, TRAJECTORIES, fingerprint, read_config, save_json, provenance

SUMMARY_METRICS = ['alignment_advantage_current_noise_off', 'alignment_advantage_noisy',
                   'noise_degradation_dp', 'noise_degradation_syn', 'clip_cosine_dp', 'clip_cosine_syn',
                   'cos_dp_adam_current_noise_off', 'cos_syn_adam_current_noise_off',
                   'cos_dp_adam_noisy', 'cos_syn_adam_noisy',
                   'norm_ratio_dp_adam', 'norm_ratio_syn_adam']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_runs(c, runs):
    frames, evaluations, metadata = [], [], []
    source = None
    for seed in c['seeds']:
        pair = []
        for trajectory in TRAJECTORIES:
            root = Path(runs)/f'seed{seed}'/trajectory
            m = json.loads((root/'metadata.json').read_text())
            require(m.get('complete') and m.get('diagnostics_enabled'), f'Incomplete diagnostic run: {root}')
            require(m['seed'] == seed and m['trajectory'] == trajectory, 'Run identity mismatch')
            require(m['config_fingerprint'] == fingerprint(c), 'Configuration mismatch')
            require(json.loads((root/'config.json').read_text()) == c, 'Saved config mismatch')
            if source is None:
                source = m['provenance']
            require(source == m['provenance'], 'Source provenance differs between runs')
            df = pd.read_csv(root/'shadow_metrics.csv')
            ev = pd.read_csv(root/'eval_metrics.csv')
            audit = json.loads((root/'pairing.json').read_text())
            total = m['total_steps']
            require(m['completed_steps'] == total, 'Partial training')
            if not c['smoke']:
                require(total == 1170, 'Formal runs must have 1170 steps')
            for table in (df, ev):
                require(table['step'].tolist() == list(range(1, total+1)), 'Missing/duplicate step')
                require((table['seed'] == seed).all() and (table['trajectory'] == trajectory).all(), 'CSV identity mismatch')
            require(len(audit['private']) == total, 'Incomplete private audit')
            require(np.isfinite(df[SUMMARY_METRICS].to_numpy()).all(), 'Undefined/nonfinite core metric')
            for suffix in ('noisy', 'current_noise_off'):
                require(np.allclose(df[f'alignment_advantage_{suffix}'], df[f'cos_syn_adam_{suffix}']-df[f'cos_dp_adam_{suffix}'], atol=1e-12), 'Alignment advantage formula mismatch')
            for method in ('dp', 'syn'):
                require(np.allclose(df[f'noise_degradation_{method}'], df[f'cos_{method}_adam_current_noise_off']-df[f'cos_{method}_adam_noisy'], atol=1e-12), 'Noise degradation formula mismatch')
                require(np.allclose(df[f'norm_ratio_{method}_adam'], df[f'{method}_direction_norm']/df.adam_direction_norm), 'Norm ratio formula mismatch')
            for column in df.columns:
                if column.startswith(('cos_', 'clip_cosine')):
                    require(df[column].between(-1, 1).all(), f'Invalid cosine: {column}')
            require(df.syn_refresh.tolist() == [s % c['K'] == 0 for s in range(total)], 'Refresh schedule mismatch')
            require([a['step'] for a in audit['synthetic']] == list(range(0, total, c['K'])), 'Synthetic refresh audit mismatch')
            for a in audit['synthetic']:
                require(a['count'] == c['M_syn'], 'Synthetic budget mismatch')
            for index, a in enumerate(audit['private']):
                require(a['step'] == index+1, 'Private audit step mismatch')
                require(a['noise_hash'] == a['syn_noise_hash'] == df.iloc[index].noise_hash, 'Shared noise mismatch')
                require(a['batch_hash'] == df.iloc[index].batch_hash, 'Batch audit mismatch')
            require(audit['private'][-1]['model_hash'] == m['final_model_hash'], 'Final model audit mismatch')
            diagnostic = (df.step == 1) | (df.step % c['diagnostic_interval'] == 0) | (df.step == total)
            for name in ('adam', 'dp', 'syn'):
                require(df[f'loss_progress_{name}'].notna().equals(diagnostic), 'Counterfactual schedule mismatch')
                require(np.isfinite(df.loc[diagnostic, f'loss_progress_{name}']).all(), 'Nonfinite progress')
            evaluated = (ev.step % c['eval_interval'] == 0) | (ev.step == total)
            require(ev.test_loss.notna().equals(evaluated) and ev.test_accuracy.notna().equals(evaluated), 'Evaluation schedule mismatch')
            require(np.isfinite(ev.train_loss).all(), 'Nonfinite train loss')
            frames.append(df)
            evaluations.append(ev)
            metadata.append(m)
            pair.append((m, audit))
        require(pair[0][0]['initial_model_hash'] == pair[1][0]['initial_model_hash'], 'Initial model pairing mismatch')
        require(pair[0][0]['noise_multiplier'] == pair[1][0]['noise_multiplier'], 'Noise calibration mismatch')
        for a, b in zip(pair[0][1]['private'], pair[1][1]['private']):
            for key in ('step', 'batch_hash', 'noise_hash', 'loader_rng_hash', 'noise_rng_before', 'noise_rng_after'):
                require(a[key] == b[key], f'Cross-trajectory pairing mismatch: {key}')
        for a, b in zip(pair[0][1]['synthetic'], pair[1][1]['synthetic']):
            require({k:v for k,v in a.items() if k != 'preconditioner_hash'} == {k:v for k,v in b.items() if k != 'preconditioner_hash'}, 'Synthetic sample/RNG pairing mismatch')
    # Archived results can be read after source edits, but the discrepancy is explicit.
    return pd.concat(frames, ignore_index=True), pd.concat(evaluations, ignore_index=True), metadata, source == provenance()


def summarize(df):
    per_seed = []
    for (trajectory, seed), frame in df.groupby(['trajectory', 'seed'], sort=False):
        late = frame[frame.step > frame.step.max()/2]
        for metric in SUMMARY_METRICS:
            per_seed.append(dict(trajectory=trajectory, seed=int(seed), metric=metric,
                                 late_median=float(late[metric].median()), late_steps=len(late)))
    per_seed = pd.DataFrame(per_seed)
    summary = per_seed.groupby(['trajectory', 'metric'], sort=False).late_median.agg(['mean', 'std', 'count']).reset_index()
    return per_seed, summary


def report(summary, metadata, c):
    def value(t, k):
        r = summary[(summary.trajectory == t) & (summary.metric == k)].iloc[0]
        return f'{r["mean"]:.4f} ± {r["std"]:.4f}' if r['count'] > 1 else f'{r["mean"]:.4f} (SD N/A)'
    lines = ['# Exp4：Shadow optimization dynamics', '',
             '各 seed 先取 step > T/2 的中位数，再报告跨 seed 均值 ± 样本标准差。方向均未乘学习率；不作显著性声明。', '',
             '| Reference trajectory | Syn–Adam (current noise off) | DP–Adam (current noise off) | Alignment advantage (current noise off) | Alignment advantage (noisy) |',
             '| --- | --- | --- | --- | --- |']
    if c['smoke']:
        lines.insert(2, '**Smoke validation only — 不作为正式机制结论。**\n')
    signs = []
    for t in TRAJECTORIES:
        keys = ['cos_syn_adam_current_noise_off', 'cos_dp_adam_current_noise_off', 'alignment_advantage_current_noise_off', 'alignment_advantage_noisy']
        lines.append('| '+t+' | '+' | '.join(value(t, k) for k in keys)+' |')
        vals = summary[(summary.trajectory == t) & (summary.metric == keys[2])]['mean'].iloc[0]
        signs.append(np.sign(vals))
    lines += ['', '**Q1 — 谁更像 Adam？**']
    for t, sign in zip(TRAJECTORIES, signs):
        lines.append(f'{t}：按后半程 current-noise-off alignment advantage 的跨 seed 均值，'+('SynDiag 更接近 Adam。' if sign > 0 else 'DP-Adam 更接近 Adam。' if sign < 0 else '两者相同。'))
    lines += ['', '**Q2 — 两条参考轨迹是否一致？** '+('主汇总指标的符号一致。' if signs[0] == signs[1] else '主汇总指标的符号不同，结论依赖参考轨迹。'), '',
              '**Q3 — 当前噪声和 clipping geometry 各有什么影响？**', '',
              '| Trajectory | Noise degradation DP | Noise degradation Syn | Clip cosine DP | Clip cosine Syn |',
              '| --- | --- | --- | --- | --- |']
    for t in TRAJECTORIES:
        lines.append('| '+t+' | '+' | '.join(value(t, k) for k in ['noise_degradation_dp', 'noise_degradation_syn', 'clip_cosine_dp', 'clip_cosine_syn'])+' |')
    for t in TRAJECTORIES:
        def mean(k):
            return float(summary[(summary.trajectory == t) & (summary.metric == k)]['mean'].iloc[0])
        off, noisy = mean('alignment_advantage_current_noise_off'), mean('alignment_advantage_noisy')
        affected = 'SynDiag' if mean('noise_degradation_syn') > mean('noise_degradation_dp') else 'DP-Adam'
        relation = '排名反转' if np.sign(off) != np.sign(noisy) else '排名保持'
        lines.append(f'\n{t}：恢复当前噪声后，alignment advantage 从 {off:.4f} 变为 {noisy:.4f}（{relation}）；当前噪声的对齐损失在 {affected} 上更大。')
    lines += ['', 'Noise degradation 为关闭当前噪声后的 alignment 减去 noisy alignment；正值表示当前噪声降低对齐，负值表示提高。Clipping cosine 越低，聚合方向改变越大，但两者位于各自裁剪前的几何中。', '',
              'current-noise-off 的 DP-Adam 仍包含历史 DP 噪声；SynDiag 还与 Adam 有预条件器、动量与历史状态差异。因此这两个诊断不是完整的因果分解，不能把关闭当前噪声后剩余的差异全部归因于 clipping。应结合四张时间曲线解释。', '',
              f'反事实 progress = (L(θ)−L(θ−ηu))/(实际参数移动范数+ε)，正值表示下降；使用各自实际学习率，仅在 step 1、每 {c["diagnostic_interval"]} 步及末步计算。', '',
              '参考性能仅属于两条真实轨迹，不报告 SynDiag test accuracy。', '',
              '| Trajectory | Seed | Final test loss | Final test accuracy |', '| --- | --- | --- | --- |']
    for m in metadata:
        lines.append(f'| {m["trajectory"]} | {m["seed"]} | {m["final_test_loss"]:.4f} | {m["final_test_accuracy"]:.4f} |')
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/full.json'))
    p.add_argument('--runs', required=True)
    a = p.parse_args()
    c = read_config(a.config)
    df, ev, metadata, current_source = load_runs(c, a.runs)
    out = Path(a.runs)
    per_seed, summary = summarize(df)
    per_seed.to_csv(out/'summary_per_seed.csv', index=False)
    summary.to_csv(out/'summary_across_seeds.csv', index=False)
    (out/'report.md').write_text(report(summary, metadata, c))
    save_json(out/'validation.json', dict(passed=True, runs=len(metadata), steps=len(df),
              seeds=c['seeds'], source_matches_current_checkout=current_source,
              pairing='batch, synthetic samples/labels/RNG, Gaussian z hashes all matched'))
    print(f'Validated {len(metadata)} runs, {len(df)} steps; wrote summaries and report.', flush=True)


if __name__ == '__main__':
    main()
