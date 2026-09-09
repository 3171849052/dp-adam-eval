# 实际验证记录（2026-09-09）

审计：dp-adam-eval 本地 HEAD 与 fetch 后 origin/main 同为 `36307b3d93f1a3a4db2e02189f5ba0efa0302605`；DP-KFC 为 `eb31b9aeb2280642684f4cedfa65cc02b76c76cd`。已实际读取 dp-adambc-ex main 的 config、CLI、trainer、privacy、logging、utils、run.sh 和 packaging；只参考工程结构。

环境：Python 3.11、PyTorch 2.13.0+cu126、torchvision 0.28.0+cu126、Opacus 1.6.0。使用机器已有 Python 环境安装本包（环境路径没有写入运行脚本）。

执行 `python -m pytest -q`：**50 passed**。覆盖 DP 数学、Fisher gain/显式 Kronecker 变换、pack/unpack、Linear/Conv covariance、真实 refresh RNG 隔离、SGD bypass、刷新时序、真实 MNIST CPU 子进程、CLI、异常输出保留，以及原实现固定轨迹。

三步 CPU 源轨迹回归：两种算法的初始化、每步所有参数的 DP noisy gradient、filtered gradient、SGD 后参数、最终模型 hash、noise RNG 均 bitwise 一致。Fisher 在 step 0/2 使用真实 pink-noise covariance 和 eigendecomposition。锚点由 `scripts/generate_reference.py` 调用原 ExpV1b `private_update` 生成，测试不依赖源仓库。该结果只覆盖记录的环境和 tiny trajectory，不声称跨设备/依赖版本 bitwise 一致，也未执行五 epoch formal anchor 复现。

实际执行两个原始 smoke YAML（device=auto，RTX 3080 Ti，cuda:0）：

| 算法 | 状态 | steps | epsilon | sigma | final accuracy | final test loss | refreshes |
|---|---|---:|---:|---:|---:|---:|---:|
| dp_sgd | completed | 4 | 0.9941875234 | 2.734375 | 0.09375 | 9.7860746384 | 0 |
| dp_fisher_wiener | completed | 4 | 0.9941875234 | 2.734375 | 0.125 | 2.3027557731 | 2 |

完整 summary 和运行目录见 `smoke_results.json`。两个 run 都实际检查了五个必要输出、4 行 metrics、resolved config 采样约定及所有数字 finite。CPU smoke 也在测试子进程实际通过。

首次 MNIST 在线下载因服务器 TLS 错误失败，失败 summary/log 保留；随后把机器已有标准 MNIST 缓存复制到本工程 `data/` 后运行成功。运行时没有从旧实验目录读取数据。Opacus 提示最优 RDP alpha 位于搜索范围上界，PyTorch 提示 full backward hook 的触发位置；均为上游 warnings，没有屏蔽或改动会计/梯度计算。

工程调整包括独立 dataclass/YAML、逐步落盘、失败 summary、可配置 LR、独立输出目录和严格输入校验。相较源代码，允许 YAML 调整实验规模（formal 默认仍为原值），不包含诊断谱和内部 sweep。synthetic mean CE、每 batch ridge/stack mean、DP 算术顺序和 RNG 消费保持源行为。

独立安装验证：将所需文件导出到 `/tmp/dp-wiener-mnist-standalone`，在新的虚拟环境中构建并安装非 editable wheel，确认包来自该虚拟环境的 `site-packages`。从导出目录运行完整测试：**50 passed, 12 warnings (29.96s)**，包含两条真实 MNIST CPU CLI 运行。测试数据缓存为单独复制的标准 MNIST 文件；导出工程没有旧 `exp*` 或 sibling repository。虚拟环境复用已安装的第三方科学计算依赖，未重新下载 PyTorch。独立源码包位于工作区 `dist/dp-wiener-mnist.tar.gz`。
