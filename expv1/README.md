# DP-Fisher-Wiener V1

这是完全隔离的研究实验。检验 data-free synthetic KFAC Fisher 是否能作为
clipped-clean gradient 的协方差先验，改善重建误差、方向、SNR 和完整训练质量。
scalar 对照用于判断各向异性是否比每层统一 shrinkage 更有价值；kappa 和
Fisher eigenmode diagnostics 用于观察 beta=1 的 absolute scale mismatch。
**beta 刻意固定为 1**，结果差、H 接近 0/1、kappa 极端均不是调参依据。
没有 beta calibration、私有 Fisher、额外缩放、学习率补偿或时间平滑。

## 已审计的上游协议

实现前阅读了 exp3 README/common/preconditioners/train/metrics/full/smoke，
以及实际 ../DP-KFC 的 covariance/recorder/privacy/optimizer/models。
模型、MNIST loader、pink noise、set_seed 直接导入相邻上游。没有运行时 exp3 import。
正式模式要求 clean checkout 且 commit 为
`eb31b9aeb2280642684f4cedfa65cc02b76c76cd`；不匹配立即失败。
smoke 允许 dirty，但必须保存完整 provenance。每个 run 保存 upstream commit、
dirty/status/remote、全部 upstream 包 Python 和 expv1 Python SHA256、config fingerprint。
测试证据另外绑定测试源文件摘要，验证器拒绝陈旧证据。

KFACRecorder 记录独立模型（当前 DP 参数副本）的 synthetic activations/backprops。
使用上游 generate_pink_noise 默认 alpha=1.0，标签为 torch.randint(0,10,...)。
正式每次严格 10×256 equal batches，每批默认 mean cross entropy。
直接 compute_covariances（默认 ridge eps=1e-5）后 accumulate_covariances 平均。
保留上游卷积空间平均和 bias augmentation；不复制 covariance 数学，不人为 rescale。
不使用 inverse-root、damping=1e-3 或 per-sample preconditioner。
独立普通模型的 recorder backprops 与 exp3 wrapper 的 mean-loss recorder convention 相同；
不需要为 synthetic batches 计算 Opacus per-sample gradients。

## 三个方法及矩阵约定

METHODS 严格为 dp_sgd、dp_scalar_wiener、dp_fisher_wiener。
全部 private 路径：sum CE backward 获取 raw per-sample gradients，调用上游
clip_and_noise_gradients(..., store_summed_grad=True) 完成 global L2 clipping、
Gaussian noise 和除 B，然后（可选）Wiener，最后 SGD(lr=.1,momentum=0)。
上游 clip coefficient 为 min(1,C/(norm+1e-6))。

令 s=g_clean=p.summed_grad，y=加噪后 filter 前 p.grad，n=y-s，hat=optimizer 实际梯度。
训练中的 active state 和 filter 不访问 summed_grad。

\[
r=(\sigma C/B)^2,\quad F_l=A_l\otimes G_l,\quad \beta=1.
\]

每层 Y 的 shape 是 [d_out,d_in_aug]。Linear weight 原矩阵、Conv weight
reshape(d_out,-1)，最后一列拼 bias；恢复时拆除最后一列并 reshape weight。
这里 F 对应 **column-major vec(Y)**，不是 PyTorch 默认 flatten 顺序。
仅数学单测在极小维度构造完整 F；训练和诊断均不构造 d×d Fisher。

DP-SGD 是 W=I。Scalar 方法使用
\[
\bar\lambda_l=\operatorname{tr}(A_l)\operatorname{tr}(G_l)/(d_A d_G),\quad
h_l=\bar\lambda_l/(\bar\lambda_l+r),\quad \hat Y=h_lY.
\]

Fisher 方法用 torch.linalg.eigh，特征值 clamp_min(0)，无额外 damping：
\[
H_{ji}=\frac{\lambda_{G,j}\lambda_{A,i}}{\lambda_{G,j}\lambda_{A,i}+r},\qquad
\hat Y=Q_G[H\odot(Q_G^TYQ_A)]Q_A^T.
\]

Fisher active state 为 FP32：Q_A/lambda_A/Q_G/lambda_G/H；scalar active state 仅包含每层
一个 FP32 scalar_h。trace 和 Fisher spectrum summary 属于独立 diagnostic state，不会被
filter、optimizer 或 refresh decision 使用。仅因零分母时定义 H=0 以避免 0/0；正式 r>0。
统计 power/sum 使用 float64，filter/basis 使用实际 FP32 state。

## Scalar baseline cost accounting

`dp_scalar_wiener` 的算法 active state 仅包含每层的 `scalar_h`。Scalar training 不需要
Fisher eigendecomposition；`scalar_h` 只由 `trace(A)`、`trace(G)` 和固定的 `r` 计算。

为保持研究诊断中的 `lambdaF_median`、`lambdaF_q10`、`lambdaF_q90`，在
`diagnostics=True` 时可以额外进行 diagnostic-only eigenspectrum computation。该计算：

- 不参与训练，不影响 `scalar_h` 或训练参数/RNG 轨迹；
- 不进入 `refresh_time` 或 `active_state_bytes`；
- 进入 `diagnostic_seconds`，因而从 `core_training_runtime` 中扣除；
- `diagnostics=False` 时完全不执行。

Fisher-Wiener 的 eigendecomposition 是算法本身所需，所以计入 Fisher 的
`refresh_time` 和 core cost。这样 scalar 与 Fisher 的 runtime/memory baseline 使用一致、
可审计的 cost accounting。

## 时序、RNG 和 privacy

所有 CSV 的 step 为 **zero-based private step**。refresh 位于 0,K,2K,...，
在 next(private_iterator) 前完成。评估判断使用 completed step=step+1。
初始化/loader/test/synthetic/noise 分别使用 seed、seed+1、seed+2、seed+3、seed+4。
synthetic 和 Gaussian 使用 fork_rng 隔离流，loader/test 使用独立 Generator。
scalar/fisher 按 refresh 配对 samples/labels hash；三方法初始化、batch indices 和
Gaussian before/after RNG hash 配对。模型分叉不会改变这些随机流。
诊断没有随机采样；测试比较 on/off 与 eigenmode layer budget 改变时每一步参数 hash。

继承 exp3 的 fixed shuffle/drop_last RDP convention，**不是 Poisson sampling**。
q=B/N，total_steps=floor(N/B)×epochs，get_noise_multiplier 使用 epsilon/delta/q/steps、
accountant='rdp'。每步 RDPAccountant.step(sigma,q)，最终记录 epsilon_spent。
这是沿用的实验 accounting convention，不声称 fixed shuffle 等同于 Poisson 定理条件。
Wiener 仅依赖已 DP 参数、data-free synthetic factors、sigma/C/B 和 noisy DP y，
因此是 DP output 的后处理，在该 accounting convention 下不增加 spending。

**研究诊断是非 DP 的统计量，不包含在 epsilon 保证中，不能作为可公开发布的 DP 输出。**
p.summed_grad、clip_rate、private train_loss、kappa、重建/SNR/eigenmode 统计仅供受控研究。
没有 raw private gradients 长期写盘；保存的 private batch indices/配对审计也属于受控研究产物。
这些数据不反馈 gain、optimizer、refresh、超参、checkpoint decision 或训练控制流。
诊断在 optimizer.step 后通过梯度副本计算，不更改任何参数或 active state。
没有 early stopping 或最佳 checkpoint 选择；best accuracy 只是训练完成后的汇总。

## 固定配置

full.json：MNIST/SimpleCNN，seeds=[42,7,91]，epochs=5，B=256，epsilon=1，
delta=1e-5，C=1，SGD lr=.1/momentum=0，beta=1，K=50，M_syn=2560，
eval_interval=100，threads=4，device=auto。N=60000 时每 run 1170 steps、24 refreshes。
正式仅 device 可指定；不提供超参搜索。eps_num=1e-12。

smoke.json：seeds=[42]，epochs=1，B=4，K=2，M_syn=8，train_subset=16，
test_subset=32，eval_interval=1，smoke=true；其它方法/隐私参数不变。
smoke 明确使用 2×4 equal synthetic batches，4 private steps，refresh=0,2。
它仅验证功能，accuracy 等不能支持研究结论。

## 指标定义

以下 e=1e-12，所有 global 指标每步记录，per-layer 分别覆盖 conv1/conv2/fc1/fc2。

| 指标 | 定义 |
|---|---|
| train_loss | private sum CE / B |
| clip_rate | mean(raw per-example global norm > C) |
| clean_clipped_norm / actual_noise_norm | ||s|| / ||n|| |
| noisy_gradient_norm / filtered_gradient_norm | ||y|| / ||hat|| |
| relmse_noisy | ||y-s||²/(||s||²+e) |
| relmse_filtered | ||hat-s||²/(||s||²+e) |
| cosine_noisy / cosine_filtered | dot(y,s)/(||y|| ||s||)，hat 同理；零 norm 定义 0 |
| signal_retention | ||Ws||²/(||s||²+e) |
| noise_retention | ||Wn||²/(||n||²+e) |
| snr_in / snr_out | ||s||²/(||n||²+e)，||Ws||²/(||Wn||²+e) |
| snr_gain_db | 10 log10((snr_out+e)/(snr_in+e)) |
| mse_reduction | 1-||hat-s||²/(||y-s||²+e)，允许负数 |
| kappa_l | ||s_l||²/(trace(A_l)trace(G_l)+e)，只诊断 |

DP-SGD 明确设 retention=1、snr_gain_db=0、mse_reduction=0，filtered/noisy metrics 相同。
DP-SGD 没有 Fisher，kappa/trace/lambdaF 为 N/A；H stats 为 identity。
Wiener 每层另外记录 H mean/population std/min/max/q10/q25/q50/q75/q90，
trace_A/G/F，lambdaF mean/median/q10/q90。scalar 的 H stats 来自 scalar_h（std=0）。

每次 Fisher refresh 对应首个 private batch 记录 eigenbins：
S'=Q_G^T S Q_A，N'=Q_G^T N Q_A，Y'=S'+N'，hat'=H*Y'。
按 lambdaF 升序稳定排序，分为 10 equal-count bins（无法整除时相差最多 1）。
每 bin 保存 count、lambda min/max/mean、log10(max(lambda_mean,e))；
signal_power=mean(S'²)，noise_power=mean(N'²)，empirical_snr=signal/(noise+e)，
H_mean；mse_before=mean(N'²)，mse_after=mean((H Y'-S')²)，
mse_reduction=1-after/(before+e)，signal_retention=mean((H S')²)/(signal+e)，
noise_retention=mean((H N')²)/(noise+e)。不保存 eigenmode raw gradients。

每 eval_interval 和最后一步评估 test loss/accuracy；final_accuracy、best_accuracy、
late_mean_accuracy（completed progress > 1/2）、final_test_loss。
accuracy_auc 对所有 evaluation points 的 normalized progress=(step+1)/total 做
trapezoid integral 再除 progress span；只有一个点则取其 accuracy，不补造 step 0 点。

wall_time 是训练 loop 含 evaluation/audit/diagnostics，不含 setup/data loading/最终序列化。
scalar `refresh_time` 包括 synthetic 生成、hash、covariance/trace/state 构建；Fisher
`refresh_time` 还包括算法所需的 covariance eigendecomposition/H 构建。Scalar 的
diagnostic-only spectrum reconstruction 单独记录在 `diagnostic_spectrum_time`，不污染
scalar core runtime。
filter_time 仅实际训练 filter（CUDA 同步计时），DP-SGD=0。
core_training_runtime=wall_time-diagnostic_seconds，包括共同 evaluation 和 pairing 审计。
记录 total/mean refresh/filter/diagnostic-spectrum time、refresh count、active tensor bytes 和 overall peak
CUDA allocated memory（包含诊断；CPU=0）。不将它标为排除诊断的 core peak。

## 输出与分析

所有指定 --output / --runs 必须位于 expv1/runs 下，防止写入其它实验。
默认 expv1/runs；cache 唯一例外为共享 expv1/runs/_cache/data/MNIST，允许从既有缓存复制。
新 run 不覆盖旧 run。每 seed<seed>/<method>/ 包含 config.json、metadata.json、summary.json、
train/layer/eigenbin/refresh_metrics.csv、pairing.json。无长期 raw private gradients。
测试临时目录/cache/log/evidence 也在 expv1/runs 下。

validate 严格检查完整 seed×method、step、finite、H/retention、DP-SGD identities、
refresh/eigenbin 结构、RNG 配对、当前 config/provenance 和诊断隔离测试证据。
summary_runs.csv 每 seed；summary_methods.csv 跨 seed mean/sample SD(ddof=1)；
summary_layers.csv 每 seed/layer 的后半段 median；summary_filter.csv 跨 seed/layer 汇总。
后半段定义 completed step>total/2；单 seed 的 sample SD 空白，不伪造 0。
paired_contrasts.csv 包含 scalar-SGD、Fisher-SGD、Fisher-scalar 的逐 seed 差和 mean/SD，
MSE/loss 负差改善，accuracy/cosine/SNR 正差改善，retention 联合解释。
不做 p-value 或显著性声称。

figures/ 全部 PNG+PDF：accuracy/test loss/train loss；重建 noisy/filtered；cosine/SNR；
first/middle/last refresh 每层 eigenvalue 对 signal/SNR/MSE reduction；每层 kappa 和 H quantiles。
seed 细线，mean 粗线；smoke 重复 refresh 选择只画一次。

## 运行

从项目根目录，在 curve 环境执行。实现阶段**不会自动启动 full experiment**。

```bash
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m pytest expv1/tests -q -s

EXPV1_SMOKE_RUNS=expv1/runs/smoke_$(date +%Y%m%d_%H%M%S)
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m expv1.train_expv1 --config expv1/configs/smoke.json --output "$EXPV1_SMOKE_RUNS" --method all
conda run -n curve python -m expv1.validate_expv1 --config expv1/configs/smoke.json --runs "$EXPV1_SMOKE_RUNS" --output "$EXPV1_SMOKE_RUNS"
conda run -n curve python -m expv1.summarize_expv1 --config expv1/configs/smoke.json --runs "$EXPV1_SMOKE_RUNS" --output "$EXPV1_SMOKE_RUNS"
conda run -n curve python -m expv1.plot_expv1 --config expv1/configs/smoke.json --runs "$EXPV1_SMOKE_RUNS" --output "$EXPV1_SMOKE_RUNS"
```

完整实验仅由用户显式启动：

```bash
EXPV1_FORMAL_RUNS=expv1/runs/formal_$(date +%Y%m%d_%H%M%S)
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m expv1.train_expv1 --config expv1/configs/full.json --output "$EXPV1_FORMAL_RUNS" --method all
conda run -n curve python -m expv1.validate_expv1 --config expv1/configs/full.json --runs "$EXPV1_FORMAL_RUNS" --output "$EXPV1_FORMAL_RUNS"
conda run -n curve python -m expv1.summarize_expv1 --config expv1/configs/full.json --runs "$EXPV1_FORMAL_RUNS" --output "$EXPV1_FORMAL_RUNS"
conda run -n curve python -m expv1.plot_expv1 --config expv1/configs/full.json --runs "$EXPV1_FORMAL_RUNS" --output "$EXPV1_FORMAL_RUNS"
```
