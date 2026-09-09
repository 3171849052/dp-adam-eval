# ExpV4a: Fisher-Wiener gamma diagnosis

ExpV4a reuses the ExpV3 adaptive-beta Fisher-Wiener trainer at the training
boundary. It reads the existing scalar layer and refresh artifacts after
training and computes the deployable model gamma, clean-signal oracle gamma,
and DP-energy gamma. Gamma is never passed to the beta controller, filter,
optimizer, learning rate, RNG, batch schedule, or stopping logic.

The smoke configuration intentionally runs only adaptive Fisher-Wiener at
learning rate 0.5. The formal configuration runs the three adaptive Fisher
learning rates and does not add a DP-SGD run.

The diagnostic outputs are gamma_refresh_metrics.csv and
gamma_interval_metrics.csv; summary_gamma.json contains the compact
layer/overall summary.
