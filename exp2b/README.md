# Exp2b: SynDiag lambda × K factorial sweep

目的：研究固定 damping lambda 与 block refresh period K 的 interaction。
这是单 seed 机制筛选，不用本轮直接声称统计显著性能优势；后续选出少量候选后再做 paired multi-seed。

从 `exp2/` 复制训练、synthetic preconditioner、metrics、diagnostics、common utilities 和测试。
只运行 `syn_diag`，固定 lr=0.1，其余 Exp2 当前算法不变。`exp2/` 及 `../DP-KFC/` 不需修改。

| 设置 | 固定值 |
| --- | --- |
| learning_rate | 0.1 |
| epsilon / delta | 1.0 / 1e-5 |
| max_grad_norm | 1.0 |
| batch_size / epochs | 256 / 5 |
| M_syn / M_oracle | 2560 / 512 |
| seed / optimizer | 42 / plain SGD |

Grid：lambda ∈ {1e-4, 3e-4, 1e-3}，K ∈ {50, 100, 200}，共 9 个配置。
文件命名为 `configs/lambda{1e4,3e4,1e3}_K{50,100,200}.json`，对应独立 `runs/<同名>/`。
文件名的 `3e4` 表示 `3e-4`，没有自适应选参。

算法严格保持：

```text
v_syn = mean_i(g_i^2)
P = 1 / (sqrt(v_syn) + lambda)
g_i -> P g_i -> global per-example L2 clip -> aggregate + Gaussian noise -> plain SGD
```

P 在 step 0、K、2K、…（小于总步数）构造，在接下来的 K 个 private steps 冻结，末块可以不足 K 步。
不是 `1/sqrt(v+lambda)`；不加 EMA、normalization、warm-up 或 ramp。
不修改 privacy accountant、Gaussian noise、C、M_syn、M_oracle；private oracle 只写诊断，不能反馈训练。
MNIST `drop_last=True`，每 epoch 234 步，完整 run 共 1170 步。
K=50/100/200 分别有 24/12/6 次 refresh，最后一次为 1150/1100/1000。

refresh_metrics CSV 在保留原指标的同时新增每层 `attenuation_q05/q50/q95`，使用当前 synthetic v：

```text
a_j(lambda) = sqrt(v_j) / (sqrt(v_j) + lambda)
```

attenuation_q50 用于判断 lambda 是否过度改变 bulk sqrt(v) geometry：越接近 1，影响越小；lower tail 仍由 lambda 限制。
A_old/A_new 比 stale_ratio 更应优先解读。step 0 没有旧 P，因此 A_old、D_time、stale_ratio 为空，不填造数值。

汇总每配置一行，不自动选择最佳配置。`late_mean_accuracy` 为 `step > total_steps / 2` 的所有非空 eval 点均值。
final train 指标取最后 private step，final 层指标取该层最后一次 refresh/oracle 记录，后者不等于训练终点重新估计。
`best_test_accuracy` 仅指该 run 已记录 eval 点的最大值。
校验会检查完整 grid、文件名和固定配置、独立 output、配置指纹、CSV 与 metadata/summary 一致、训练/refresh step 序列、四层及必要指标有限性。
每个 CSV 行携带配置指纹，合并 CSV 必须与 SynDiag CSV 一致。缺失或未完成 run 明确报错，所有 run 通过后才写汇总。

在项目根目录使用 conda `curve`。先测试：

```bash
conda run -n curve python -m unittest discover -s exp2b/tests -v
conda run -n curve python exp2b/train_exp2b.py --config exp2b/smoke_configs/lambda1e4_K2.json --method syn_diag
conda run -n curve python exp2b/train_exp2b.py --config exp2b/smoke_configs/lambda1e3_K4.json --method syn_diag
conda run -n curve python exp2b/plot_exp2b.py --config exp2b/smoke_configs/lambda1e4_K2.json
conda run -n curve python exp2b/plot_exp2b.py --config exp2b/smoke_configs/lambda1e3_K4.json
conda run -n curve python exp2b/summarize_exp2b.py --smoke
conda run -n curve python exp2b/plot_sweep_exp2b.py --smoke
```

Smoke 使用 train_subset=16、test_subset=32、batch_size=4、epochs=1、M_syn=M_oracle=8，每 run 4 步，仅验证代码闭环，不用于性能判断。
输出在全新 `runs/smoke_validation_01/`；训练拒绝覆盖已有 started marker。
重跑 smoke 请复制整个 smoke_configs 到新目录，统一将两个 config 的 output 父目录改为新的 `runs/smoke_validation_02/` 等；训练和单 run 图传入新 config 路径，汇总/绘图传入 `--smoke --config-dir <新配置目录> --output exp2b/runs/smoke_validation_02/sweep_summary.csv`。
K=4 只有首次 refresh，所以汇总中的 final_A_old 为空；两个 smoke 配置没有覆盖的 heatmap 格子显示 N/A。

完整实验命令（这里只记录，不自动启动）：

```bash
conda run -n curve python exp2b/train_exp2b.py --config exp2b/configs/lambda1e4_K50.json --method syn_diag
conda run -n curve python exp2b/train_exp2b.py --config exp2b/configs/lambda1e4_K100.json --method syn_diag
conda run -n curve python exp2b/train_exp2b.py --config exp2b/configs/lambda1e4_K200.json --method syn_diag
conda run -n curve python exp2b/train_exp2b.py --config exp2b/configs/lambda3e4_K50.json --method syn_diag
conda run -n curve python exp2b/train_exp2b.py --config exp2b/configs/lambda3e4_K100.json --method syn_diag
conda run -n curve python exp2b/train_exp2b.py --config exp2b/configs/lambda3e4_K200.json --method syn_diag
conda run -n curve python exp2b/train_exp2b.py --config exp2b/configs/lambda1e3_K50.json --method syn_diag
conda run -n curve python exp2b/train_exp2b.py --config exp2b/configs/lambda1e3_K100.json --method syn_diag
conda run -n curve python exp2b/train_exp2b.py --config exp2b/configs/lambda1e3_K200.json --method syn_diag
conda run -n curve python exp2b/summarize_exp2b.py
conda run -n curve python exp2b/plot_sweep_exp2b.py
```

正式 summary 在 `exp2b/sweep_summary.csv`，12 张 sweep heatmap 在 `exp2b/sweep_figures/`，均使用 matplotlib。
Sweep 绘图会重新验证源 run 并生成 summary，防止画出旧的或不完整的汇总。
单 run 的 Exp2 风格图仍可用 `plot_exp2b.py --config <config>` 生成，共 12 张（含 attenuation、A_old/A_new、P quantiles）。

层贡献校验：`contrib_layer = mean_i(||g_i,layer||² / max(||g_i||², eps_num))`。
四层之和等于 `mean_i(||g_i||² / max(||g_i||², eps_num))`，零梯度或极小梯度样本会使其小于 1。
因此检查有限、非负且总和 ≤ 1（上界允许 1e-5 浮点误差），不要求等于 1。
已有 CSV 未保存每个样本的平方范数，不能从这些汇总列恢复精确的预期总和。
此校验修复不改变训练、指标计算或已有 run 数据，无需重跑训练。
