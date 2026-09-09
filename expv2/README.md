# ExpV2 — DP-Safe Layerwise Beta Estimation for Fisher-Wiener

ExpV2 is a measurement experiment, not an adaptive training algorithm. It
tests whether a layerwise scalar

`E[s_l s_l^T] ~= beta_l F_l^syn`

can be estimated from an already privatized DP gradient `y_l`, the known
Gaussian variance `r`, and synthetic KFAC Fisher factors. It makes no prior
assumption that `beta > 1`; values much smaller than one are scientifically
valid and may imply a more aggressive statistically calibrated Wiener filter.
The experiment does not use accuracy to decide whether the estimator works.

## Fixed training invariant

Every formal and smoke trajectory trains with `beta_train=1`. The estimated
values `beta_oracle`, `beta_dp_raw`, positive, lagged, and hold-last-positive
are research diagnostics only. They never affect the Wiener filter, optimizer,
learning rate, clipping, noise, privacy accountant, Fisher refresh, stopping,
batch selection, or RNG. Fisher-Wiener remains

`H = lambda / (lambda + r)`

and is applied only after clipping and DP Gaussian noise. There is no KFC
preconditioning, private Fisher, layer-wise learning rate, momentum, scheduler,
Adam, warmup, EMA, or beta feedback.

The seven formal trajectories are DP-SGD lr=.1 for seeds 42/7/91, Fisher-Wiener
beta=1 lr=.1 for seeds 42/7/91, and Fisher-Wiener beta=1 lr=.8 for seed 42.
Smoke runs contain the three seed42 run IDs. DP-SGD also builds synthetic Fisher
as a measurement instrument at every refresh; that state cannot change its
private trajectory.

## Estimators

For packed layer dimension `d_l` (weight plus bias),

`r = (sigma * C / B)^2` and `E||n_l||^2 = d_l*r`.

The clean oracle records only the scalar `||s_l||^2`, where `s_l` is
`p.summed_grad`, the clean clipped batch mean. The deployable estimator records
only scalar `||y_l||^2` before Wiener filtering and computes

`beta_dp_step_raw = (||y_l||^2 - d_l*r) / trace(F_l^syn)`.

Raw estimates are allowed to be negative. `beta_dp_step_positive` is an
additional descriptive field; primary analysis uses the raw estimator. The
primary interval estimate is a ratio of sums, including the final partial
interval (20 steps in the 1170-step formal protocol), never a mean of per-step
ratios. The lagged and hold-last-positive files are offline evaluations and do
not feed training.

The synthetic Fisher is exactly the ExpV1 construction: upstream
`KFACRecorder`, `compute_covariances(..., eps=1e-5)`, and covariance averaging,
with the same pink-noise probes and random-uniform labels. ExpV2 reuses the
read-only ExpV1 Fisher-Wiener math and packing rather than reimplementing KFAC
eigendecomposition, `H`, or Conv/Linear bias conventions. `beta` is not the
DP-KFC paper's `alpha`: alpha scales a private Fisher approximation, while
ExpV2 measures the clipped-gradient signal second moment along the current
trajectory.

## Privacy and safety boundary

The deployable estimator uses only already privatized `y`, public/known `r`,
and synthetic Fisher factors, so it is post-processing and adds no privacy
accountant step. The oracle is explicitly a non-DP research diagnostic. Raw
`s`, `y`, Gaussian noise tensors, Fisher matrices, eigenvectors, private
per-example gradients, and actual noise realizations are never written to
artifact files. In particular, ExpV2 never records `n` and subtracts `y-n`;
it subtracts only the expected noise energy `d*r`.

## Artifacts

Each run preserves the usual `config.json`, `metadata.json`, `summary.json`,
`pairing.json`, `train_metrics.csv`, `layer_metrics.csv`,
`refresh_metrics.csv`, and empty-schema `eigenbin_metrics.csv`, and adds:

- `beta_step_metrics.csv`
- `beta_interval_metrics.csv`
- `beta_lagged_metrics.csv`
- `beta_hold_predictor.csv`
- `beta_window_sensitivity.csv`

The output root also receives beta layer, lagged, trajectory, and window
summary tables, `beta_trajectory_dependence.csv`, `summary.json`,
`validation.json`, and seven PNG/PDF figure pairs. Negative raw beta values are
not converted to absolute values for log plots; log plots show only positive-
valid observations and negative rates are reported separately.

## Reproducibility

The fixed ExpV1 reference is
`expv1/runs/formal_20260908_182603/`; the fixed ExpV1b reference is
`expv1b/runs/formal_20260909_003544/`. Formal validation compares private
pairing/model hashes and, where applicable, synthetic pairing to those roots.
Provenance records upstream commit/dirty/remote, all upstream Python hashes,
imported ExpV1/ExpV1b helper hashes, all ExpV2 Python hashes, and both fixed
reference roots.

Utility values (accuracy, loss, AUC) are retained only to establish trajectory
regression. Scientific endpoints are estimator quality: pooled beta error,
negative interval rate, positive coverage, correlation, lagged quality,
observability SNR, and window sensitivity. There is deliberately no
correlation or negative-rate pass/fail threshold.

## Smoke command

```bash
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m pytest expv2/tests -q -s

EXPV2_SMOKE_RUNS=expv2/runs/smoke_$(date +%Y%m%d_%H%M%S)
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m expv2.train_expv2 --config expv2/configs/smoke.json \
  --output "$EXPV2_SMOKE_RUNS" --run all
conda run -n curve python -m expv2.validate_expv2 \
  --config expv2/configs/smoke.json --runs "$EXPV2_SMOKE_RUNS" --output "$EXPV2_SMOKE_RUNS"
conda run -n curve python -m expv2.summarize_expv2 \
  --config expv2/configs/smoke.json --runs "$EXPV2_SMOKE_RUNS" --output "$EXPV2_SMOKE_RUNS"
conda run -n curve python -m expv2.plot_expv2 \
  --config expv2/configs/smoke.json --runs "$EXPV2_SMOKE_RUNS" --output "$EXPV2_SMOKE_RUNS"
```

## Formal command — intentionally not run by implementation

```bash
EXPV2_FORMAL_RUNS=expv2/runs/formal_$(date +%Y%m%d_%H%M%S)
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m expv2.train_expv2 --config expv2/configs/full.json \
  --output "$EXPV2_FORMAL_RUNS" --run all
conda run -n curve python -m expv2.validate_expv2 \
  --config expv2/configs/full.json --runs "$EXPV2_FORMAL_RUNS" --output "$EXPV2_FORMAL_RUNS"
conda run -n curve python -m expv2.summarize_expv2 \
  --config expv2/configs/full.json --runs "$EXPV2_FORMAL_RUNS" --output "$EXPV2_FORMAL_RUNS"
conda run -n curve python -m expv2.plot_expv2 \
  --config expv2/configs/full.json --runs "$EXPV2_FORMAL_RUNS" --output "$EXPV2_FORMAL_RUNS"
```

