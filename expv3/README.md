# ExpV3 — Lagged Adaptive-Beta Fisher-Wiener vs DP-SGD

ExpV3 has exactly four runs per formal seed: `dp_sgd_lr0p50` and adaptive
Fisher-Wiener at learning rates `0.50`, `1.00`, and `5.00`.  The adaptive
controller starts at beta 1, pools one completed interval of pre-Wiener DP
observations, and applies that interval's positive finite ratio-of-sums
estimate only at the next refresh.  Negative or non-finite estimates use the
previous positive beta.  There is no beta floor, cap, smoothing, gamma,
scheduling, momentum, Adam, or layerwise learning rate.

The filter is applied after global clipping and Gaussian DP noise.  Its gain is
`H = beta * lambda_F / (beta * lambda_F + r)`, while `Q_A`, `Q_G`, and both
eigensystems are reused from ExpV1.  The beta estimator algebra is imported
from ExpV2.  DP-SGD still receives an identity gradient update; its synthetic
Fisher trace is measurement-only.

The training outputs include research-only clean/oracle diagnostics, so the
complete artifact is not DP-release-safe.  The deployable controller itself
uses only past DP `y`, known noise variance, and synthetic Fisher traces, and
therefore is DP post-processing with no additional accountant step.

Formal runs are intentionally never launched automatically.  Run the tests and
smoke pipeline first, then use the command printed by the agent.

