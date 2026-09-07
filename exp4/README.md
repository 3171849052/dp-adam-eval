# Exp4: Adam / DP-Adam / SynDiag shadow dynamics

从仓库根目录、`curve` 环境运行。新建输出目录；已有 reference run 不覆盖。

```bash
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest exp4/tests -q
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m exp4.train_exp4 --config exp4/configs/smoke.json --output exp4/runs/smoke
conda run -n curve python -m exp4.analyze_exp4 --config exp4/configs/smoke.json --runs exp4/runs/smoke
conda run -n curve python -m exp4.plot_exp4 --config exp4/configs/smoke.json --runs exp4/runs/smoke

conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m exp4.train_exp4 --config exp4/configs/full.json --output exp4/runs/formal
conda run -n curve python -m exp4.analyze_exp4 --config exp4/configs/full.json --runs exp4/runs/formal
conda run -n curve python -m exp4.plot_exp4 --config exp4/configs/full.json --runs exp4/runs/formal
```

`--seed 42 --trajectory adam` 可选单个 run；默认 seeds 42/7/91 × adam/dp_adam。
六条真实轨迹各 1170 步；SynDiag 没有独立模型轨迹，不报告其 test accuracy。
正式配置固定 MNIST / SimpleCNN、B=256、5 epochs、C=1、ε=1、δ=1e-5、
Adam(lr=.001, betas=.9/.999, eps=1e-8, weight_decay=0)、SynDiag lr=.1、
M_syn=2560、K=50、λ=.001。只允许设备、线程和 synthetic 分析 microbatch 大小变化。
不针对 Exp4 指标调参。

每步在同一 reference 参数点、同一 batch 计算三种 candidate。普通 Adam 使用
SUM CE backward / B 的未裁剪梯度；DP-Adam 直接调用当前相邻 DP-KFC 仓库的
`clip_and_noise_gradients`，遵循 Exp1 的逐样本全局裁剪、Gaussian noise、
RDP `get_noise_multiplier(..., steps=T, accountant='rdp')` 和 torch Adam。
当前 upstream 已不再导出旧 Exp1 文件所引用的 `DPGradientAccumulator`，
所以调用其现有公开裁剪函数；系数仍为 `min(1,C/(norm+1e-6))`。
采样为 shuffle + drop_last，sample_rate=B/N，沿用原实验校准约定。
源文件哈希、upstream checkout 信息和噪声乘子写入 metadata。

SynDiag 直接复用 Exp3 `synthetic_samples` 和 `refresh(..., 'syn_diag')`：
Pink samples + uniform labels，FP64 累计 v=mean(g_i²)，P=1/(sqrt(v)+λ) 转为 FP32，
随后 P→per-sample transform→global clipping→noise→SGD direction。
refresh 发生在零基 step 0/50/...，读取该 private batch 之前，num_workers=0 无预取。
CSV 使用一步完成后的 1-based step，故 syn_refresh=True 在 1/51/...。
block 内 P 不变；两条轨迹的 P 数值可以不同，但构造样本完全相同。

AdamDirection 使用真正 torch Adam(lr=1) 的独立 scratch 参数读取 u。
每步 scratch 参数归零，m/v/step 始终保留；weight_decay=0 使状态与参数值无关。
真实 reference optimizer 使用实际学习率。当前噪声关闭的 DP 候选从 noisy
shadow 的上一时刻完整 state 的副本计算，随后才提交正式 noisy step。
它仍含历史噪声，不是完全无噪声 DP-Adam。SynDiag 当前噪声关闭候选为 mean clip(Pg)。

初始化／loader／test／synthetic／noise 分别为 seed 至 seed+4 的独立流。
每次调用上游噪声函数时按同一参数顺序重放 Gaussian 流来审计标准 z 哈希，
DP 与 Syn 使用同一流的副本；只有正式 noise stream 前进一次。
所有 reference 的 batch、synthetic、z、loader RNG 按 seed/step 配对。
只评估 reference train/test；train_loss 是更新前当前 batch loss，test 是更新后测试集。

输出：每个 `seed{seed}/{adam,dp_adam}/` 中保存 `shadow_metrics.csv`、
`eval_metrics.csv`、`pairing.json`、`metadata.json`、`config.json`、`final_state.pt`。
后者含 reference model/optimizer 和两套最终 shadow optimizer state。
主 CSV 覆盖需求中所有字段，并增加 `loss_progress_{adam,dp,syn}` 和 candidate loss。
反事实在 step 1、每 50 步和最终步计算，其余留空；正值表示下降：
E=(L(θ)−L(θ−ηu))/(实际参数移动范数+eps)。使用各算法实际 lr，诊断在模型副本上执行。
零向量 cosine/ratio 未定义时留空，正式汇总拒绝未定义核心指标。
这些私有机制诊断输出不是 DP 发布，Adam reference 本身也没有隐私保证。

`analyze_exp4` 在输出任何汇总前验证完整 steps、配置/身份/来源一致性、配对哈希、
refresh/诊断/eval 日程和主要指标公式。归档结果与当前源代码哈希不同时显式记录。
先取每个 seed 的后半程 step > T/2 中位数，再跨 seed 报均值和样本 SD (ddof=1)。
单 seed SD 留空；没有 p-value 或显著性判断。`report.md` 回答 Q1–Q3。
Noise degradation 和 clipping cosine 不是完整因果分解：剩余差异也包括历史噪声、
动量和预条件器差异，不能全部归因于 clipping。

`plot_exp4` 输出四张主图及一张辅助 reference performance 图，均为 PNG/PDF。
细淡线为各 seed 原始曲线，粗线为同一步的 seed 均值；不做时间平滑。
Figure 1/2 current-noise-off 为主，noisy 为浅虚线；Figure 2 包含零线。
Figure 4 的 norm ratio 使用 log y 轴，progress 使用原始固定诊断点。

测试覆盖 CPU/CUDA 的 Adam 与 DP-Adam 精确下一步/历史状态等价、current-noise-off
状态/RNG/模型隔离、SynDiag 与逐样本 v 定义和 upstream 单步等价、跨 epoch refresh
先于 private 读取且 block 内冻结、共同 z、跨轨迹配对，以及开关全部 shadow
诊断后的每一步 model/loader/reference/noise RNG hash 完全一致。

本次实验产物位于 `runs/formal_20260907/`：`report.md` 为结论报告，
`summary_per_seed.csv` 和 `summary_across_seeds.csv` 为可复核汇总，
`validation.json` 为完整性与配对审计，`figures/` 保存四张主图与辅助图。
最终默认 Adam 实现的 smoke 位于 `runs/smoke_default_adam_20260907/`。
CPU/CUDA 测试记录为 `test_results.log` 与 `verification.json`。
