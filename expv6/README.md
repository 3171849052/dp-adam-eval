# ExpV6：Fisher-Wiener alpha sweep

研究问题：原始 Fisher-Wiener 是否滤波过强？训练使用
`H_alpha = (1-alpha) + alpha * H_beta`，其中
`H_beta = beta * lambda_F / (beta * lambda_F + r)`。
顺序保持为全局 clipping → batch average → DP noise → beta 构造 H → alpha 插值 → 过滤 y → Adam。
alpha=0 对应 DP-Adam，alpha=1 对应 ExpV5 原始滤波。

复用 ExpV5 的 Adam、V3 训练循环和 one_interval_lag_hold_last_positive beta controller。
V3 仅增加一次 refresh 后的插值调用；controller 仍从过滤前的 DP y 估计 beta。
controller 文件中的 H 是插值前的 H_beta，layer_metrics 中 H 是实际训练使用的 H_alpha。
无 gamma、cap、scheduler 或学习率 sweep。oracle clean gradient 仅用于诊断。
`clean_gradient_distortion = ||H_alpha s-s||/||s||` 同时记录每层和全模型值。

full 配置只运行两个 family × 三个 alpha (.25/.5/.75) × 两个 seed (123/456)，共 12 个新 run。
其余协议与 ExpV5 full 一致。每个 seed 沿用初始化、loader、evaluation、synthetic Fisher、DP noise RNG。
不重新训练 ExpV5 端点。默认参考目录为 `expv5/runs/formal_20260910_095056`（通过 CLI 显式传入）。

## Tiny smoke

使用已有 `expv5/runs/smoke` 的端点，六个新 arm 各运行 4 steps（跨两个 beta interval）。
输出目录须尚不存在；重复执行时换一个目录。

```bash
conda run -n curve python -m pytest expv6/tests -q -s
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m expv6.train_expv6 --config expv6/configs/smoke.json --output expv6/runs/smoke --run all
conda run -n curve python -m expv6.validate_expv6 \
  --config expv6/configs/smoke.json --runs expv6/runs/smoke \
  --expv5-runs expv5/runs/smoke --output expv6/runs/smoke
conda run -n curve python -m expv6.summarize_expv6 \
  --config expv6/configs/smoke.json --runs expv6/runs/smoke \
  --expv5-runs expv5/runs/smoke --output expv6/runs/smoke
```

validator 检查核心协议、完成状态、实际 Adam 输入范数、beta lag 和已有 batch/RNG/evaluation pairing。
ExpV6 certificate 显式记录 `alpha`、插值前的 `H_beta_hash` 和实际 active H 的 `H_alpha_hash`，
以及 `beta_train`、`r`、`lambda_A`、`lambda_G`；移除含义不明确的 `H_hash`，保留原有 `H_beta1_hash`。
验证时按训练设备上的 FP32 运算重算两个 H 并核对 hash，以及 layer_metrics 的 H 统计。
beta_controller_metrics 仍表示插值前的 H_beta。
已有训练循环的 artifact 写入保持原样。

## 汇总

输出 `utility_alpha_sweep.csv`、`mechanism_alpha_sweep.csv`、
`paired_beta1_vs_dp_by_alpha.csv`、`paired_adaptive_vs_dp_by_alpha.csv`、
`paired_adaptive_minus_beta1_by_alpha.csv`。
每个 alpha 保留各 seed 和 `seed=mean` 行；mechanism 按层先取每个 run 的时间均值，再对 seed 等权平均。
两个 family 共用同一个 ExpV5 DP-Adam baseline；alpha=1 分别读取 ExpV5 对应 family。
`summary.json`/`summary.txt` 给出各 family mean AUC 最大的 alpha（并列时选最小 alpha）。不做显著性检验。
ExpV5 未记录 alpha=1 的 clean distortion，保持缺失；alpha=0 distortion 为 0，beta 不适用。
未达到的 accuracy threshold 及其 paired difference 保持缺失。
smoke 仅验证功能，不能提供 hypothesis signal。

## Formal（仅提供命令，不自动启动）

```bash
EXPV5_REFERENCE=expv5/runs/formal_20260910_095056
EXPV6_FORMAL_RUNS=expv6/runs/formal_$(date +%Y%m%d_%H%M%S)

conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m expv6.train_expv6 \
  --config expv6/configs/full.json \
  --output "$EXPV6_FORMAL_RUNS" \
  --run all

conda run -n curve python -m expv6.validate_expv6 \
  --config expv6/configs/full.json \
  --runs "$EXPV6_FORMAL_RUNS" \
  --expv5-runs "$EXPV5_REFERENCE" \
  --output "$EXPV6_FORMAL_RUNS"

conda run -n curve python -m expv6.summarize_expv6 \
  --config expv6/configs/full.json \
  --runs "$EXPV6_FORMAL_RUNS" \
  --expv5-runs "$EXPV5_REFERENCE" \
  --output "$EXPV6_FORMAL_RUNS"
```
