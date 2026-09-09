# DP-Wiener MNIST

将 ExpV1b 的 **DP-SGD / DP-Fisher-Wiener** 抽离为可独立安装、配置和运行的 MNIST 实验。只支持原 SimpleCNN、plain SGD（momentum=0、weight_decay=0）和 RDP accounting。算法迁移以源训练轨迹为优先。

当前工作区原有 `exp*` 文件保留用于历史实验；本工程训练、安装和测试只使用 `src/dp_wiener_mnist`、`config`、`scripts/train.py`、`tests`，不导入历史实验或 sibling checkout。打包仅包含 `dp_wiener_mnist`。可用下述导出命令得到不含历史实验的小仓库。

## 安装与运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
./run.sh config/mnist_dpsgd.yaml
./run.sh config/mnist_fisher_wiener.yaml
```

也支持 `./run.sh --config config/mnist_dpsgd.yaml`、`python scripts/train.py --config config/mnist_dpsgd.yaml` 和位置参数形式。需先安装本项目；入口不修改 `sys.path`。`run.sh` 使用当前 Python（可设置 `PYTHON`），直接前台运行，无 Conda 环境名和 tmux 依赖。

```bash
python scripts/train.py --config config/mnist_dpsgd_smoke.yaml
python scripts/train.py --config config/mnist_fisher_wiener_smoke.yaml
```

MNIST 自动下载到 `data.root`（默认 `data/`），使用 `ToTensor()` 和 `Normalize((0.1307,), (0.3081,))`。下载失败会保留失败运行目录；离线环境可预先将标准 `MNIST/raw/` 数据放入该目录。路径相对于启动时的工作目录。Smoke 使用真实 MNIST 前 16/32 个 train/test 样本，B=4、epochs=1、K=2、M=8，共 4 次更新；准确率不作为验收门槛。测试还使用固定 tiny tensor dataset 检查训练逻辑。

正式实验推荐 CUDA；`runtime.device: auto` 在无 CUDA 时使用 CPU。`runtime.gpu: 1` 在导入 PyTorch 前设置 `CUDA_VISIBLE_DEVICES=1`，内部使用 `cuda:0`；显式要求 CUDA 而不可用时失败。运行禁用 TF32、cuDNN benchmark、AMP，启用 deterministic algorithms，并设置线程数和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

## 算法数据流

```text
DP-SGD: per-example gradient → global clipping → Gaussian noise → SGD
Fisher: per-example gradient → global clipping → Gaussian noise → Fisher-Wiener → SGD
```

私有 batch 使用 `GradSampleModule(loss_reduction="sum")` 和 `cross_entropy(reduction="sum")`。每个样本的全模型 norm 先按源代码逐参数累加 `norm(2).pow(2)`，裁剪因子为 `min(1, C/(norm+1e-6))`。按参数顺序计算：

```python
summed = clipped.sum(dim=0)
noise = torch.randn_like(summed) * sigma * C
grad = (summed + noise) / B
```

Fisher 在上述 DP noisy gradient 之后处理，不再裁剪或加噪。两条训练路径共享同一 DP 函数；DP-SGD 不生成 synthetic probes、不构造 covariance、不分解特征值、不调用 filter。

SimpleCNN：Conv2d(1,16,3,padding=1) → ReLU → MaxPool(2) → Conv2d(16,32,3,padding=1) → ReLU → MaxPool(2) → Flatten → Linear(1568,128) → ReLU → Linear(128,10)。无归一化层、Dropout 或额外优化器状态。

## Wiener 数学与 synthetic Fisher

每层 $F\approx G\otimes A$，FP32 `eigh` 后将负特征值 clamp 至 0：

$$r=(\sigma C/B)^2,\qquad \lambda_F=\lambda_G\lambda_A,\qquad H=\frac{\lambda_F}{\lambda_F+r}.$$

分母为 0 的 mode 返回 0；对已加噪梯度矩阵 $M$：

$$\widetilde M=Q_G[H\odot(Q_G^\top M Q_A)]Q_A^\top.$$

按 `qg @ (h * (qg.T @ matrix @ qa)) @ qa.T` 的源运算顺序计算，不使用 inverse Fisher、natural gradient、对角近似或 EMA。固定层 `conv1, conv2, fc1, fc2`，weight 按输出维展平，bias 是最后一列。

在 step 0 及 `step % K == 0` 时、取下一 private batch 之前刷新。使用独立 SimpleCNN 加载当前私有模型参数。probes 为 `(1,28,28)` 的 complex FFT pink noise，逐样本标准化后乘 0.5，标签是随机 `randint(0,10)`。synthetic backward 保留源代码 **mean CE**；Linear 偏置增广，Conv 用 `F.unfold`，每个 batch 分别加 `1e-5` ridge，最终 `torch.stack(...).mean(0)`。formal 默认 10×256 probes，严格要求 M 能被 B 整除。

RNG 分流：init=seed、train loader=seed+1、test loader=seed+2、synthetic=seed+3、DP noise=seed+4。源 `RNGStream` 保存 CPU/CUDA 状态，通过 `fork_rng` 隔离；独立 Fisher 模型初始化也隔离。测试验证真实 refresh 不改变训练参数、梯度、grad_sample、全局 RNG 或下一次 DP noise。

## 隐私会计与研究输出

校准调用 `get_noise_multiplier(target_epsilon, target_delta, sample_rate=B/train_size, steps=total_steps, accountant="rdp")`，每次 private update 后调用 `RDPAccountant.step`。采样保留原固定 shuffle、drop_last=True，**不是 Poisson sampling**。README 与 resolved config 明确记录：

```yaml
sampling: fixed_shuffle_drop_last
accounting_convention: inherited_rdp_sample_rate_convention
poisson_sampling: false
```

这里继承源仓库的 sample-rate RDP 会计约定，不将它表述为对固定无放回采样的新增严格隐私证明。Fisher state 只读取当前模型参数和 synthetic probes；当前模型来自此前 DP 更新的后处理，probes 不读取原始 MNIST training examples。数据流是 `DP gradient → data-free Fisher-Wiener post-processing`。

`train_loss`、`clip_rate` 是未加隐私保护的研究统计；test metrics 假设测试集可公开。报告的 epsilon 描述继承的梯度机制会计，不能作为整个日志发布的隐私保证。

## 配置与输出

一个 YAML 是一次实验，`training.learning_rate` 直接控制优化器。正式默认 seed=42，epochs=5，B=256，lr=0.1，epsilon=1，delta=1e-5，C=1，K=50，M=2560，eval_interval=100。`config/sweeps/` 的六份配置仅 LR 不同（0.10/0.15/0.20/0.30/0.50/0.80）；trainer 不包含 sweep。未知字段、错误类型、非法范围和不支持的算法设置直接失败。

每次运行创建 `outputs/<timestamp>_simplecnn_mnist_<algorithm>_eps..._lr...[_K..._M...]/`：

- `config.yaml`：原始 YAML。
- `resolved_config.yaml`：默认值和实际 device/GPU、数据规模、steps、sigma、q、采样约定、RNG seeds、Wiener 开关。
- `metrics.csv`：每步 train loss、epsilon、clipping/noisy/filter norm、刷新标志、时间与 state bytes；评估步附 test loss/accuracy，最后一步总会评估。step 从 0 开始。
- `summary.json`：completed/diverged/failed、完成步数、最终/最佳评估、会计、耗时、模型 hash；Fisher 另有 refresh/filter 时间统计。
- `train.log`：逐步记录与异常堆栈。

默认不存 checkpoint。指标逐步 flush，非 finite 值使运行停止；已有指标保留，summary 记录 diverged_step。普通异常也写 failed summary，CLI 返回非零。

## 来源与轨迹验证

审计的 [dp-adam-eval](https://github.com/3171849052/dp-adam-eval) main 提交为 `36307b3d93f1a3a4db2e02189f5ba0efa0302605`，与本地一致：

- [expv1b/train_expv1b.py](https://github.com/3171849052/dp-adam-eval/blob/36307b3d93f1a3a4db2e02189f5ba0efa0302605/expv1b/train_expv1b.py)：更新和刷新顺序、加载器、会计。
- `expv1b/common.py`：formal/smoke 协议；`expv1/common.py`：RNGStream、初始化。
- [expv1/fisher_wiener.py](https://github.com/3171849052/dp-adam-eval/blob/36307b3d93f1a3a4db2e02189f5ba0efa0302605/expv1/fisher_wiener.py)：synthetic samples、covariance、FP32 state、pack/unpack、矩阵变换。
- [DP-KFC](https://github.com/molinamarcvdb/DP-KFC) 固定提交 `eb31b9aeb2280642684f4cedfa65cc02b76c76cd` 的 `src/dp_kfac/{models,privacy,recorder,covariance,data,optimizer,trainer}.py`：仅迁移所需 SimpleCNN、clipping/noise、KFAC、MNIST transform、pink noise、set_seed。
- [dp-adambc-ex](https://github.com/3171849052/dp-adambc-ex)：仅参考 config/src/scripts/tests、独立 run logging 的工程结构，未迁移其训练算法。

`tests/fixtures/expv1b_cpu_trajectory.json` 由**原 ExpV1b private_update 实际执行**生成。固定初始化、12 个 tiny 样本、batch order、sigma=2、seed=42、B=4、M=8、K=2，运行 3 步。逐参数 SHA256 检查初始化、每步 DP noisy gradient、filtered gradient、SGD 后参数、最终模型，以及 synthetic audit/noise RNG；Fisher 使用真实 covariance 和 eigendecomposition。参考生成脚本 `scripts/generate_reference.py` 是仅开发时运行的源仓库工具，需要显式 `PYTHONPATH=/path/to/dp-adam-eval`；普通训练和 pytest 只读取提交的 JSON，不需要任何上游 checkout。

回归锚点是在 CPU、4 threads、PyTorch 2.13.0+cu126、torchvision 0.28.0+cu126、Opacus 1.6.0 上生成。跨 PyTorch/BLAS/CUDA 版本或设备不承诺 bitwise 一致；测试不会跳过或自动重建不匹配锚点。若需比较另一数值环境，应在同一环境重新运行审计的原实现。此 tiny regression 不能替代五 epoch formal trajectory 的完整复现；本次不运行 formal 长实验。

工程差异：移除源 formal 固定参数锁以允许 YAML 实验（保留默认值和算法约束）；移除 LR 内部 sweep、诊断谱和旧输出布局；eval batch size 单独配置；增加 finite/shape 校验。源算术、synthetic mean CE 和随机数消费顺序保持不变。

## 导出独立小工程

```bash
python scripts/export_standalone.py /tmp/dp-wiener-mnist
cd /tmp/dp-wiener-mnist
pip install -r requirements.txt
pytest -q
```

导出只复制必需工程文件与固定参考，不复制 `exp*`、上游 checkout、数据缓存或历史运行；首次运行需下载 MNIST。验证记录见 `docs/verification.md`。
