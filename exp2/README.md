# Exp2：block-wise Synthetic Diagonal Preconditioning

从 `dp-adam-eval` 根目录运行，使用 conda `curve`。所有新增代码、缓存、结果位于 `exp2/`；`runs/` 已忽略。实现阶段仅运行单元测试与真实 MNIST smoke，未启动完整实验。

## 命令

```bash
conda run -n curve --no-capture-output python -m unittest discover -s exp2/tests -v
conda run -n curve --no-capture-output python exp2/train_exp2.py --config exp2/configs/smoke.json --method dp_sgd
conda run -n curve --no-capture-output python exp2/train_exp2.py --config exp2/configs/smoke.json --method syn_diag
conda run -n curve --no-capture-output python exp2/plot_exp2.py --config exp2/configs/smoke.json
```

完整实验命令（本阶段不执行）：

```bash
conda run -n curve --no-capture-output python exp2/train_exp2.py --config exp2/configs/default.json --method dp_sgd
conda run -n curve --no-capture-output python exp2/train_exp2.py --config exp2/configs/default.json --method syn_diag
conda run -n curve --no-capture-output python exp2/plot_exp2.py --config exp2/configs/default.json
```

训练拒绝覆盖已开始的方法目录记录；重复实验请复制配置并修改 `output` 为新的 `runs/<name>`。两个方法共用同一配置和 output，顺序执行。绘图可重复运行，并验证配置、配对、完整步数、四层、有限数值和所有预期 refresh。中断后保留已有 CSV，不将未完成运行标为完成；本版不支持恢复训练。

## 配置与训练路径

默认 MNIST / DP-KFC SimpleCNN，B=256、5 epochs、lr=0.001、epsilon=1、delta=1e-5、C=1、seed=42；K=100、M_syn=2560、M_oracle=512、lambda=1e-8，1170 private updates。学习率和 lambda 均为显式固定 scalar。

smoke 为真实 MNIST 训练集前 16 个样本、B=4、1 epoch，共 4 updates；在更新计数 0、2 处 refresh；M_syn=M_oracle=8；测试集前 32 个样本，仅检查闭环可运行，不能据此判断性能。训练指标的 step 表示该次更新完成后的计数；refresh/oracle 的 step 表示估计时已完成的更新数。

在 theta_t 上先用 Pink Noise 与 uniform random labels 估计 `v=mean(g_i²)`，构造 `P=1/(sqrt(v)+lambda)` 并冻结，然后获取下一个训练 batch。`syn_diag` 在 `.grad_sample` 上乘 P，再交给上游 `DPGradientAccumulator.accumulate/finalize`，执行全参数逐样本 global L2 clipping、Gaussian noise、除以实际 B，最后直接 `torch.optim.SGD.step()`。`dp_sgd` 的 active P 为 I。无 EMA、normalization、warm-up、ramp、Adam、任何自适应参数或提前停止。

复用上游裁剪实现的数值约定为 `min(1,C/(norm+1e-6))`，因此与理想的 `C/norm` 有固定的微小数值差异。clip_rate 定义为 `fraction(norm>C)`。噪声为各坐标独立 `N(0,sigma² C²)`；无 microbatch training accumulation，诊断梯度按 `analysis_batch_size` 分批，使用 float64 累加平方和。

与 Exp1 相同，sigma 由 Opacus `get_noise_multiplier(..., sample_rate=B/N, steps=epochs*len(loader), accountant='rdp')` 计算，逐步调用 `RDPAccountant.step`。保留上游 shuffle/drop_last 采样与 RDP 配置约定，未改成 Poisson sampling；报告的 epsilon 是这一上游约定的复现，不将它描述为对随机重排采样重新证明的界。

## 复用与顺序

- `exp1.common`：SimpleCNN、Pink generator、seed、device、JSON/checkpoint 保存、配置及上游源码 fingerprint 等适配器。
- `exp1.fisher_utils.layer_gradient`：conv1/conv2/fc1/fc2，各输出单元先 weight 再 bias 的交错向量。
- `exp1.metrics.diagonal_metrics/spread`：log Pearson/Spearman、log-ratio spread 与 Q95−Q5；residual 另外按实际 `P²*v_priv` 计算。
- DP-KFC 的 `data.get_mnist_loaders`：原始数据与 normalization 不变。仅调用时切换到 `exp2/runs/_cache/`，已有缓存只读复制，所有新下载留在 Exp2。
- DP-KFC `privacy.DPGradientAccumulator` 与其 norm/clip helpers：不复制 clipping、聚合、noise 实现。
- Opacus `GradSampleModule(loss_reduction='sum')` 计算逐样本 CE 梯度；RDP accountant 与 Exp1 同源。

P 的四层维数为 160、4640、200832、1290。应用时将 Exp1 向量 reshape 为输出单元行，拆出每行最后的 bias，其他坐标还原 weight shape，再对应真实 named parameters（逐层 weight/bias）。应用前断言维数、参数顺序和 grad_sample shape。统计与训练禁用 TF32。

## RNG 与因果隔离

初始化 seed=42；loader/test/synthetic/DP-noise/oracle-index 分别为 seed+1/+2/+3/+4/+5。独立 Torch RNGStream 在调用无 generator 参数的上游 Pink generator 和 Gaussian noise 时保存、切换、恢复 CPU/当前 CUDA 状态。Pink labels 使用上游 `torch.randint(0,10,(B,),device=dev)`；probes 按配置 B 生成，尾批使用实际剩余数量；split 也只消耗 synthetic stream。

DataLoader 传独立 generator；训练固定 num_workers=0，避免 worker 在 refresh 前预取 private batch。oracle 固定 subset indices，只有 P 已冻结后才读样本，使用独立模型副本计算未裁剪梯度；不改训练模型、梯度、P 或训练 RNG。dp_sgd 也估计 candidate synthetic P 以比较两条轨迹的 geometry，但训练始终用 I；其 oracle residual 使用实际 I。refresh CSV 中 dp_sgd 的 P/staleness 指 candidate synthetic P，而非 active P。

每步记录 batch indices、DP noise 调用前 RNG hash；metadata 保存初始化 hash、参数顺序、sigma 和 seeds。绘图验证两种方法对应记录完全相同。同一硬件/软件环境保证配对，不声称跨 CPU/CUDA 的逐位相等。

## 指标与隐私边界

所有 raw private clipping/oracle/train diagnostics 均为未加噪研究输出，不是实际 DP release，不包含在 accountant 的隐私预算内，不反馈给训练。checkpoint 模型只由 privatized updates 推进；不应把包含原始诊断的整个 runs 目录当作 DP 发布结果。

`train_metrics.csv` 每个 update 一行：train loss、间隔 test loss/accuracy、lr-scaled update norm、累计 wall time（包括 refresh/oracle/eval），以及：

- norm Q10/Q50/Q90/Q99、clip rate、上游 clipping coefficient mean/Q10/Q50/Q90。
- `aggregate_cosine` 与 `relative_distortion` 对比 transformed raw mean 与 clipped mean。
- `contrib_<layer>` 是每个样本该层 squared norm 占全参数 squared norm 的比例，再取 batch mean。
- `clipped_aggregate_norm` 是 clipped **mean** 的范数；`actual_noise_norm` 由上游 noisy mean 减保留的 clipped mean 重建（受 FP32 舍入影响）；`expected_noise_norm=(sigma*C/B)*sqrt(d)` 是噪声范数的 RMS 近似，二者单独记录。
- `noisy_update_norm` 是 noisy mean 的范数，`update_norm=lr*noisy_update_norm`；SNR 为 clipped mean norm / (expected noise norm + eps)。

新增四个字段直接写入既有 `train_metrics_<method>.csv` 和联合 `train_metrics.csv`，不新建 CSV：

- `coefficient_std = std(c_i, unbiased=False)`，使用 population std；`coefficient_cv = coefficient_std / (mean(c_i) + eps_num)`，衡量 sample-wise clipping heterogeneity。系数仍直接来自上游 `_compute_clip_factors`。
- `clipping_alpha_star = <mu_clip, mu_raw> / (||mu_raw||² + eps_num)`。
- `clipping_shape_error = ||mu_clip - clipping_alpha_star * mu_raw|| / (||mu_clip|| + eps_num)`，衡量去除最佳 global scalar 后的 aggregate distortion。跨四层完整 aggregate 拟合一个 scalar，inner product、norm 和直接 residual 平方和均以 float64 累加，eps_num 避免零或极小 aggregate 导致 inf/NaN。

100% clip rate 本身不代表 clipping direction bias：如果所有样本近似乘同一个 scalar，aggregate direction 可以保持，此时 aggregate_cosine 约为 1、clipping_shape_error 约为 0，而 relative_distortion 仍可能很大。固定 eps_num 在极小 aggregate 时会影响 scalar 拟合，这时指标应谨慎解释。判断 SynDiag 是否改善 clipping geometry，应结合 aggregate_cosine、clipping_shape_error、coefficient_cv 与 layer contribution，不能仅看 clip_rate 或 relative_distortion。

`02_clipping.png` 保留 clip_rate、norm_q50、coefficient_q50 并增加 coefficient_cv；`03_distortion.png` 保留原有两项并增加 clipping_shape_error。绘图 validation 要求四个新字段全部 finite。新增确定性测试覆盖纯 scalar clipping、异质/相同系数、跨层 global scalar 对照，以及零、极小和相互抵消的 aggregate。

诊断扩展使用全新的 smoke 目录，`smoke_clipshape.json` 相对原 smoke 配置只修改 output；不覆盖已有 `runs/smoke`：

```bash
conda run -n curve --no-capture-output python -m unittest discover -s exp2/tests -v
conda run -n curve --no-capture-output python exp2/train_exp2.py --config exp2/configs/smoke_clipshape.json --method dp_sgd
conda run -n curve --no-capture-output python exp2/train_exp2.py --config exp2/configs/smoke_clipshape.json --method syn_diag
conda run -n curve --no-capture-output python exp2/plot_exp2.py --config exp2/configs/smoke_clipshape.json
```

`refresh_metrics.csv` 每层每次 refresh 一行：v/sqrt(v)/P 的 Q1/Q5/Q50/Q95/Q99、lambda/median(sqrt(v))、fraction(sqrt(v)<lambda)。median 为零时比例为空，fraction 仍可读；不因此调整 lambda。

probes 随机均分两组（奇数时相差一个），分别按实际组大小平均。`z=log(v+eps)-mean(log(v+eps))`；D_sample 为两组 z 差的 RMS；D_time 为当前与上次 refresh z 差的 RMS，首次为空。`D_sample << D_time` 支持无须 EMA；相近时才为后续 EMA/增大 M 提供证据，本实验不改训练。

`A(x)=Q95(log10(x+eps))-Q5(log10(x+eps))`。在新 v 上计算 `A_old=A(old_P²*v)`、`A_new=A(new_P²*v)`、stale_ratio；首次 old/stale 为空。A_new 很小时 stale_ratio 可非常大，应结合 A_old/A_new 解读。

`oracle_metrics.csv` 每层每次 refresh 一行：synthetic/private log Pearson、Spearman、log-ratio spread、A_raw、A_residual=A(active_P²*v_priv)、R=A_residual/A_raw。常数向量相关性与零分母比例为空。所有 log 使用固定 eps_num=1e-12，不用于 P 的分母。

## 产物与验证

`runs/<name>/` 含各方法 config/metadata、started 防覆盖记录、`checkpoints/{method}_{initial,final}.pt`、`results/{train,refresh,oracle}_metrics.csv` 及各方法分表、pairing JSON、summary JSON、`validation.json` 和 `figures/`。checkpoint 保存模型和最后 active P、step、accountant；不保存 private oracle tensor。

绘图生成 accuracy/loss、clipping、distortion、四层 contribution、oracle R/Pearson、同轴 D_sample/D_time、staleness、lambda 和额外 signal/noise 共 9 张 PNG。

单元测试覆盖：逐样本 autograd 对照；P interleaving 逆映射；全局 clipping + 非零 Gaussian noise + SGD 对照；synthetic 重现及 global/loader/noise/Python/NumPy RNG 隔离；真实 MNIST 中 refresh 在下一 batch 读取前、block 内 P 不变。将 oracle 返回值反序并放大 1e12 与关闭 oracle 的训练进行比较，最终模型 hash 必须完全一致。

## 本次 smoke 验证结果

5 项 unit tests 全部通过；两个方法均完成 4 updates、2 refresh，epsilon≈0.99419，最终 32 张测试图的 accuracy 均为 0.1875。联合结果为 8 行 train、16 行 refresh、16 行 oracle，9 张图生成且配对验证通过。

观察到的异常/限制：两方法 clip rate 均为 100%；SynDiag 的各 batch norm 中位数均值约 1.10e8、coefficient 中位数均值约 9.45e-9，relative distortion 接近 1。首次 fc1 有约 34.3% synthetic 坐标的 sqrt(v)<lambda；8 个 probes 的零/稀疏统计导致这些坐标的 P 达到 1e8。conv1 第二次 refresh 的 stale_ratio 约 1.03e6，需结合接近零的 A_new 解读。这些是极小 smoke 设置下的诊断，不足以决定 EMA、K 或完整实验性能；未据此调整参数。

运行出现上游 RDP “optimal order is largest alpha” 和 PyTorch full backward hook 提示，未导致失败；逐样本梯度已通过单例 autograd 数值对照。完整 5-epoch 实验未启动。

Clipping shape 扩展验证：8 项 unit tests 全部通过；`runs/smoke_clipshape` 的两个方法各完成 4 updates、2 refresh，三个 train CSV 的四个新增字段全部 finite，9 张图及原有 validation 通过。与原 smoke 对照，两方法的最终模型 hash、RNG pairing 和上游源码 hash 完全一致。未覆盖原 smoke，也未启动完整实验。
