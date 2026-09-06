# Exp1：固定 MNIST DP 轨迹上的 Fisher / diagonal Fisher alignment

所有命令从 `dp-adam-eval` 项目根目录运行，使用已有 `curve` conda 环境。
本目录直接 import 相邻 `../DP-KFC/src/dp_kfac`，不修改 DP-KFC，不实现新的优化器或闭环预条件训练。

## 完整实验命令

```bash
conda run -n curve --no-capture-output python exp1/train_reference.py
conda run -n curve --no-capture-output python exp1/analyze_exp1.py
conda run -n curve --no-capture-output python exp1/plot_exp1.py
```

按顺序执行。训练 manifest 和 11 个 checkpoint 已存在且配置及上游源码 hash 相符时，第一条命令自动跳过训练；也可以直接从第二条开始。完整实验没有在实现阶段运行。

默认配置见 `configs/default.json`：seed=42，MNIST，SimpleCNN，Adam lr=1e-3，batch=256，epochs=5，target epsilon=1，delta=1e-5，clip norm=1。默认自动使用 CUDA（无 CUDA 则 CPU），分析 microbatch=32，CPU threads=4。
共 1170 次更新，在 round(total_steps × percent / 100) 保存 0/10/…/100% checkpoint。`M_diag=2560`，`M_full=512`，Pink seeds=0/1/2。

可复制配置到 `exp1/configs/` 并传 `--config exp1/configs/your_config.json` 给三个入口。更改配置须选择新的 `output` 子目录，以免将不同轨迹/探针混用。`analysis_batch_size` 控制瞬时梯度内存，`gram_chunk` 控制 CPU Gram 工作内存；二者不改变 Fisher 定义。`train_subset` 只用于测试，默认 null。

## 已核对并复用的 DP-KFC API

以当前本地源码和 `configs/exp1_cnn_mnist.yaml`、`scripts/paper/exp_cnn_mnist.py` 为准：

- `models.SimpleCNN`：Conv 1→16→32，FC 1568→128→10，不含 BN/dropout。
- `data.get_mnist_loaders` / `get_fashionmnist_loaders`：直接复用 `ToTensor` 和 `(0.1307, 0.3081)` normalization。训练 shuffle、drop_last=True；不采用 Poisson loader。
- `optimizer.generate_pink_noise`：不覆盖默认 `alpha=1.0`，直接保留 FFT 生成、归一化和 ×0.5。按上游 benchmark 的 batch 大小生成，随后按上游表达式 `torch.randint(0, 10, (batch_size,))` 生成 uniform random labels，无标签策略消融。
- `trainer.set_seed`：显式设定 Python、NumPy、Torch/CUDA seed 和 cuDNN deterministic 设置。
- `privacy.DPGradientAccumulator`：直接复用逐样本裁剪、求和、加 Gaussian noise、除以实际 batch size；基础 optimizer 为 `torch.optim.Adam`。使用 Opacus `GradSampleModule(batch_first=True, loss_reduction="sum")` 和 sum CE，与 `Trainer.train_dp_sgd` 的实现相同。
- Opacus `get_noise_multiplier(..., steps=total_steps, sample_rate=batch_size/N, accountant="rdp")` 与 `RDPAccountant`。这是上游固定 shuffle batching 下的 RDP 配置复现，未切换采样机制。
- `recorder.KFACRecorder` / `covariance.compute_covariances`：直接复用 hooks、bias augmentation、卷积 unfold 和空间平均，不复制近似 recorder。

上游 loader 硬编码 `./data`；适配层仅在调用 loader 时临时切换到 `exp1/`。若上游已有数据缓存，先复制到 `exp1/data/`；否则由原 loader 下载。所有新代码、数据和结果均写入 `exp1/`。manifest 保存相关上游文件 SHA256，防止源码变化后误用旧轨迹。

## Fisher 定义与尺度

诊断使用真实标签（Private MNIST/Public FashionMNIST）或固定随机标签（Pink），计算**未裁剪、未加噪**的 per-sample CE gradient。每层向量按输出单元排序，每行依次放该输出的 weight、bias。四层向量维度为 160、4640、200832、1290。

诊断阶段禁用 CUDA matmul/cuDNN TF32，采用 FP32 forward/backward 和 float64 统计累加，避免低精度舍入改变 ReLU/MaxPool 分支。训练保留上游默认精度设置。固定 seed 用于同一软件/硬件配置的重现，不承诺跨设备逐位相同。

`v_direct = mean_i(g_i**2)` 只从 Opacus grad_sample 计算，与 KFAC 独立。向量包含每个 weight 和 bias，不保存大型 d×d Fisher。

每个 source 的 `factors` 同时保存：

- `raw`：上游 `compute_covariances(eps=0)`，sum-loss backward，单样本 delta 尺度。对 diagnostic microbatch 按实际样本数加权。
- `repo`：模拟固定 benchmark batch B 的 mean-loss backward；`A_repo=A_raw+1e-5 I`，`G_repo=G_raw/B²+1e-5 I`。默认 B=256，等价于平均 10 个上游 batch 的 factors。分析 microbatch=32 不影响 B 或尺度。上游用于逆平方根的额外 damping=1e-3 不是 Fisher factor，不混入所存 Fisher，单独记在 metadata 中。

`F_KFAC=A⊗G` 的标准列向量顺序在本实现中作了共同置换（输出优先时为 G⊗A）。`v_kfac` 用 `G.diag()[:,None]*A.diag()[None,:]` 后 flatten，与 direct 向量严格对应；谱和 Frobenius 指标不受共同置换影响。

**卷积约定**：保留 DP-KFC 的空间平均 A/G。raw 只去除人为 damping 和 mean-loss 缩放，不添加上游没有的 spatial multiplier。卷积每样本梯度包含所有位置贡献之和，而上游 KFAC 因子是位置平均且忽略跨位置项，因此 raw KFAC 也不应被当作精确 direct Fisher。这种差异会在 trace/error 和 direct-vs-KFAC diagonal 指标中体现；不以 scalar alignment 掩盖绝对尺度误差。

数值稳定常量 `eps_num=1e-12` 用于所有 diagonal log/ratio 和虚拟预条件：
`residual=v_private/(v_source+eps_num)`。`A_raw` / `A_residual` 为 log10 的 Q95−Q5，`R=A_residual/A_raw`。Oracle 就是 Private 本身；有零坐标或 epsilon 影响时 residual 不严格等于 1。退化的相关系数/比例保存为空值，不伪造分数。

所有 source 同时计算以 Private 为 oracle 的 KFAC/empirical cosine、relative Frobenius error、trace ratio、alpha_star、scalar-aligned shape error（`||alpha*F_src-F_priv||/||F_priv||`）；diagonal 输出 log Pearson/Spearman、log-ratio spread、A_raw/A_residual/R。还输出每个 source 自身 direct diagonal 与 raw/repo KFAC diagonal 的比较。

## 固定探针和 full Fisher 内存

Private/Public 的样本索引分别由 seed+100/seed+101 决定；Pink 在 CPU 用各 probe seed 生成。索引、图像和标签保存在 `results/probes.pt`，跨 checkpoint 不变；full Fisher 使用各 source 的前 M_full 个样本。所有探针均来自训练 split，标签和模型参数不会在分析中更新。

仅 0/50/100% 保存 empirical spectrum。梯度暂存为 float32 `.npy` mmap，一次仅保留 Private 和当前 source，每层每次只复制 M×gram_chunk 到 float64 CPU 工作区。计算 `K=G G.T/M` 和 cross Gram；以 `||Gx Gy.T||²/(Mx My)` 求 Fisher 内积，从不构造 d×d 或 Kronecker Fisher。

默认 FC1 单个梯度文件约 392 MiB，两个 source 约 785 MiB（另有其它层）。chunk=4096、M=512 时，两块 float64 工作区约 32 MiB，小 Gram 约 2 MiB，另有模型、grad_sample 和 KFAC factors 内存。临时文件正常结束/异常退出时由 TemporaryDirectory 清理；强制 kill 后可手动删除 `results/gram_*`。只长期保存 diagonal、factors、特征值，不保留 full gradient tensor。KFAC 谱通过 A/G 各自特征值的两两乘积获得。

Private diagnostics 本身是未加噪的研究统计量，训练轨迹的 epsilon 不包含它们。

## 输出

默认目录 `exp1/runs/default/`（smoke 为 `exp1/runs/smoke/`）：

- `checkpoints/manifest.json` 和 11 个 `.pt`：参数、实际 step、checkpoint 百分比、epsilon spent、配置及源码指纹。
- `results/probes.pt`：固定样本/索引/标签。
- `results/checkpoint_NNN/{private,public,pink_0,pink_1,pink_2}.pt`：四层 v_direct、两套 factors、两套 v_kfac；0/50/100% 另含 empirical/raw KFAC/repo KFAC spectrum。
- `results/metrics.csv`：长表 `checkpoint,step,layer,source,family,metric,value`，保存每个 Pink seed 的独立结果。
- `results/analysis.json`：分析配置、统计约定、完成标记。
- `figures/figure1_kfac_{raw,repo}`：四层 cos(A)/cos(G) 随训练变化。
- `figures/figure2_empirical_alignment`：0/50/100% full empirical cosine/shape error。
- `figures/figure3_spectrum_{raw,repo}`：三 checkpoint、四层、三类来源的 empirical/KFAC 谱。
- `figures/figure4_diagonal_{spread,scatter}`：D_l 趋势和 log diagonal scatter。
- `figures/figure5_residual_anisotropy`：Raw/Public/Pink/Oracle 的各向异性和 R。

图均输出 PNG 和 PDF。Pink 趋势为三个 seed 的均值±样本标准差；谱和散点展示每个 seed。散点仅在绘图时确定性抽取最多 2500 坐标，数值指标使用所有坐标。各图展示的是 smoke 还是完整实验由其输出目录区分。

## 测试（不会运行完整实验）

```bash
conda run -n curve python -m unittest discover -s exp1/tests -v
conda run -n curve --no-capture-output python exp1/train_reference.py --config exp1/configs/smoke.json
conda run -n curve --no-capture-output python exp1/analyze_exp1.py --config exp1/configs/smoke.json
conda run -n curve --no-capture-output python exp1/plot_exp1.py --config exp1/configs/smoke.json
```

smoke 使用真实数据：20 张 MNIST 的 10 次 DP 更新；每 source 4 个诊断样本，full=3，仍覆盖 11 个 checkpoint、三类 source、三个 Pink seeds、四层和所有图。仅用于验证流程，不能用于实验结论。数值测试用独立逐样本 backward 验证 Opacus、用上游 covariance 验证两套 KFAC、用微型显式 Fisher 验证 Gram/Frobenius 恒等式、检查 bias 排序和 Oracle residual。

实现验证结果：3 项数值测试（包括 CUDA/CPU Pink 探针精度对照）通过；真实数据 smoke 训练、分析、绘图及跳过已存在训练均通过。已生成 11 个 checkpoint、55 个 source 统计文件、13060 条指标、8 张 PNG 和对应 PDF。`runs/smoke/validation.json` 记录产物检查结果；未启动默认完整实验。
