# DP-Wiener MNIST

这是一个 standalone 的 MNIST 实验工程，保留 SimpleCNN、manual per-example
global clipping、manual Gaussian mechanism、RDP accountant，以及 DP-SGD 和
post-DP Fisher-Wiener 两条算法路径。

## 安装与运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
./run.sh config/mnist_dpsgd.yaml
./run.sh --config config/mnist_fisher_wiener.yaml
```

也可以直接运行：

```bash
python scripts/train.py --config config/mnist_dpsgd_smoke.yaml
python scripts/train.py --config config/mnist_fisher_wiener_smoke.yaml
```

MNIST 使用标准 `ToTensor()` 和 `(0.1307,)/(0.3081,)` normalization，默认下载到
`data/`。离线运行可预先放置标准 `MNIST/raw/` 数据。`runtime.device: auto` 会在
有 CUDA 时使用 CUDA，否则使用 CPU；`runtime.device: cpu` 不要求 GPU。

## Sampling / Privacy

训练使用真正的 Bernoulli Poisson subsampling。训练集大小为 `N`，配置中的逻辑
batch size 为 `B`，每个 planned step 对每个样本独立以

```text
q = B / N
```

的概率 inclusion；`steps_per_epoch = floor(N / B)`，总 mechanism 数为
`steps_per_epoch * epochs`。RDP accountant 使用同一个 `q`，因此 sampling protocol
与 accounting protocol 完全一致。采样器使用独立的 `seed+1` generator。

每一步的隐私流水线为：

```text
Poisson private batch
→ per-example global clipping
→ summed-space Gaussian noise
→ divide by expected batch size B
→ SGD
```

Fisher-Wiener 路径在 Gaussian mechanism 之后增加：

```text
→ Fisher-Wiener post-processing
```

仍然是 manual clipping、manual Gaussian noise 和 `RDPAccountant`，没有使用 GDP、
`PrivacyEngine`、Ghost Clipping、Adam 或 KFAC inverse preconditioning。

空 Poisson batch 是合法事件，不会重采样、跳过或合并。此时不做 private
forward/backward，直接执行零 summed gradient 加 Gaussian noise，再除以 `B`，计入
一次 accountant，并执行 Wiener（若启用）和 SGD。

## Metrics 与隐私计数

`privacy_steps` 表示已经执行 Gaussian mechanism 的次数；`completed_steps` 表示
成功完成 `optimizer.step()` 的次数。accountant 在 Gaussian noise 返回后、Fisher
和 optimizer 之前立即计数，所以 Fisher 发散也不会丢失已经发生的 privacy cost。

`metrics.csv` 每完成一个 epoch 写一行，并立即 `flush` 和 `fsync`。其中
`train_loss` 按实际 sampled examples 加权，`clip_rate` 为累计 clipped examples
除以累计 sampled examples；空 epoch 对这两个字段写空值。每个 epoch 结束后只评估
一次，不再进行按 step 的评估。

## RNG 与 Fisher 数学

RNG 分流为 `init=seed`、`train sampler=seed+1`、`test=seed+2`、
`synthetic=seed+3`、`DP noise=seed+4`。Poisson sampler 只消耗自己的 generator；
synthetic probes 和 DP noise 通过独立 `RNGStream` 隔离。

Fisher 使用 synthetic pink-noise probes 和独立模型构造 covariance，不读取真实
private batch。保留源实现的 FP32 eigendecomposition、`r=(sigma*C/B)^2` 和

```text
H = lambda_F / (lambda_F + r)
```

以及 `Q_G [H * (Q_G.T M Q_A)] Q_A.T` 的运算顺序。固定 batch、fixed RNG 下的
底层 DP/Wiener trajectory regression 仍应 bitwise 通过；完整 DataLoader trajectory
不再与旧 fixed-shuffle protocol 要求一致。

## Output naming 与文件

运行目录使用不含年份的秒级 timestamp，并只编码核心实验参数，
例如：

```text
0909-123456_simple_cnn_dp_fisher_wiener_s42_ep5_lr0.1_eps1_d1e-5_beta1_K50_M2560
```

碰撞时按秒递增 timestamp。`data.root` 和 `output.root` 不进入目录名；用户指定的
device/GPU 值进入目录名，实际解析设备和 GPU 信息写入 `resolved_config.yaml`。

每个 run 至少包含：

```text
config.yaml
resolved_config.yaml
metrics.csv
summary.json
train.log
```

## Launch / tmux

`run.sh` 先读取配置，执行 `--print-gpu` 和适用的 `--validate-gpu`，调用
`--prepare-run` 创建目录和 metadata，再生成基于 run directory 的安全 tmux session
名。存在 tmux 时后台启动，并打印：

```text
attach: tmux attach -t <session>
tail: tail -f <run-log>
kill: tmux kill-session -t <session>
```

没有 tmux 时会提示 `tmux is unavailable; running training in the foreground`，
然后使用同一个已准备的目录前台运行。Python 可通过 `PYTHON=/path/to/python`
覆盖，未硬编码 Conda 环境或机器路径。

## 历史实现与回归

`expv1/`、`expv1b/`、`expv1c/` 等历史实验不参与 standalone 训练。固定输入 batch
的 ExpV1b CPU trajectory fixture 保留在 `tests/fixtures/`，用于验证 per-example
gradient → clipping → noise → Fisher-Wiener → SGD 的底层数学和 RNG 顺序。
