# Exp4：Shadow optimization dynamics

各 seed 先取 step > T/2 的中位数，再报告跨 seed 均值 ± 样本标准差。方向均未乘学习率；不作显著性声明。

| Reference trajectory | Syn–Adam (current noise off) | DP–Adam (current noise off) | Alignment advantage (current noise off) | Alignment advantage (noisy) |
| --- | --- | --- | --- | --- |
| adam | 0.0804 ± 0.0187 | 0.0076 ± 0.0005 | 0.0723 ± 0.0188 | -0.0035 ± 0.0007 |
| dp_adam | 0.0925 ± 0.0283 | 0.0061 ± 0.0004 | 0.0872 ± 0.0282 | -0.0031 ± 0.0002 |

**Q1 — 谁更像 Adam？**
adam：按后半程 current-noise-off alignment advantage 的跨 seed 均值，SynDiag 更接近 Adam。
dp_adam：按后半程 current-noise-off alignment advantage 的跨 seed 均值，SynDiag 更接近 Adam。

**Q2 — 两条参考轨迹是否一致？** 主汇总指标的符号一致。

**Q3 — 当前噪声和 clipping geometry 各有什么影响？**

| Trajectory | Noise degradation DP | Noise degradation Syn | Clip cosine DP | Clip cosine Syn |
| --- | --- | --- | --- | --- |
| adam | 0.0009 ± 0.0000 | 0.0774 ± 0.0179 | 0.6345 ± 0.0073 | 0.3598 ± 0.0218 |
| dp_adam | 0.0006 ± 0.0000 | 0.0910 ± 0.0280 | 0.6826 ± 0.0080 | 0.4668 ± 0.0724 |

adam：恢复当前噪声后，alignment advantage 从 0.0723 变为 -0.0035（排名反转）；当前噪声的对齐损失在 SynDiag 上更大。

dp_adam：恢复当前噪声后，alignment advantage 从 0.0872 变为 -0.0031（排名反转）；当前噪声的对齐损失在 SynDiag 上更大。

Noise degradation 为关闭当前噪声后的 alignment 减去 noisy alignment；正值表示当前噪声降低对齐，负值表示提高。Clipping cosine 越低，聚合方向改变越大，但两者位于各自裁剪前的几何中。

current-noise-off 的 DP-Adam 仍包含历史 DP 噪声；SynDiag 还与 Adam 有预条件器、动量与历史状态差异。因此这两个诊断不是完整的因果分解，不能把关闭当前噪声后剩余的差异全部归因于 clipping。应结合四张时间曲线解释。

反事实 progress = (L(θ)−L(θ−ηu))/(实际参数移动范数+ε)，正值表示下降；使用各自实际学习率，仅在 step 1、每 50 步及末步计算。

参考性能仅属于两条真实轨迹，不报告 SynDiag test accuracy。

| Trajectory | Seed | Final test loss | Final test accuracy |
| --- | --- | --- | --- |
| adam | 42 | 0.0328 | 0.9896 |
| dp_adam | 42 | 0.4457 | 0.9171 |
| adam | 7 | 0.0368 | 0.9878 |
| dp_adam | 7 | 0.4268 | 0.9217 |
| adam | 91 | 0.0288 | 0.9897 |
| dp_adam | 91 | 0.4740 | 0.9134 |
