# ExpV5: Fisher-Wiener with Adam

Research prototype comparing exactly three paired arms:

| Arm | Gradient into Adam |
| --- | --- |
| `dp_adam` (A) | noisy DP gradient y |
| `dp_fisher_wiener_beta1_adam` (B) | W(y), H = lambdaF / (lambdaF + r) |
| `dp_fisher_wiener_adaptive_beta_adam` (C) | W(y), H = beta * lambdaF / (beta * lambdaF + r) |

All use Adam: learning_rate=0.001, betas=(0.9, 0.999), eps=1e-8,
weight_decay=0, no scheduler. Adam m and v consume the actual post-filter
parameter gradients. No gamma, gamma cap, or gradient rescaling enters training.
`gamma_model_raw` is computed after the optimizer step as a read-only diagnostic.

The shared ExpV3 loop supplies per-example gradients -> global clipping -> batch
average -> DP Gaussian noise -> y -> optional Fisher-Wiener -> Adam.
Synthetic Fisher construction and eigensystems, the beta controller, DP accounting,
reconstruction diagnostics, and pairing audits are reused from V3/V4b.
The inherited internal method label `dp_sgd` identifies the unfiltered DP path;
ExpV5 run_id and optimizer metadata identify Adam.

Beta starts at 1. Interval j accumulates pre-Wiener DP noisy energy minus expected
noise energy, normalized by synthetic Fisher trace. Its estimate is finalized only
at the start of interval j+1; a finite positive estimate becomes active then,
otherwise the last positive beta is retained. B always holds beta=1.
Formal K=beta_window=50. All arms share initialization, batch ordering, DP noise RNG,
evaluation schedule, and synthetic RNG draws. B/C Fisher matrices can differ as
their parameters diverge; their synthetic samples and labels remain paired.

Formal protocol: seeds [123,456], 5 epochs, batch size 256, epsilon=1.0, delta=1e-5,
max_grad_norm=1.0, K=50, M_syn=2560, eval_interval=100. It inherits the V4b
fixed-shuffle/drop-last sampling and RDP accounting convention.

## Measurements

`layer_metrics.csv` records inherited reconstruction metrics and:

- `gradient_norm_into_adam`: norm of the gradient actually consumed by Adam.
- `adam_first_moment_norm`: norm of stored, uncorrected m.
- `adam_second_moment_sqrt_norm`: norm of sqrt(stored, uncorrected v).
- `adam_normalized_update_norm`: norm of bias-corrected m_hat / (sqrt(v_hat)+eps).
- `parameter_update_norm`: norm of actual theta_after - theta_before, including
  weight and bias. The difference tensor is measured in memory, not persisted.
- B/C: `filtered_gradient_norm`, `filter_gradient_norm_ratio` = norm(Wy)/norm(y),
  `beta_train`, and diagnostic-only `gamma_model_raw`.

Whole-model Adam norms are also recorded in train_metrics.csv. Inherited SGD
learning-rate interpretation columns are left empty for Adam.

Summary writes `paired_B_minus_A.csv` and `paired_C_minus_B.csv`, one row per seed,
plus utility summaries and B/C mechanism summaries by layer and by seed/layer.
Missing threshold times mean not reached and remain missing in paired differences.
AUC and late mean retain V3 definitions (normalized trapezoidal AUC over evaluation
points; mean evaluations in the latter half of training). Compare gradient norm
ratios against actual parameter update norms to assess Adam's scale response.
Research oracle diagnostics are not inputs to the beta controller or optimizer.

## Quick verification

Run from repository root, using the curve environment:

```bash
conda run -n curve python -m pytest expv5/tests -q -s
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m expv5.train_expv5 --config expv5/configs/smoke.json --output expv5/runs/smoke --run all
conda run -n curve python -m expv5.validate_expv5 --config expv5/configs/smoke.json --runs expv5/runs/smoke --output expv5/runs/smoke
conda run -n curve python -m expv5.summarize_expv5 --config expv5/configs/smoke.json --runs expv5/runs/smoke --output expv5/runs/smoke
```

Use a fresh output directory on subsequent runs. The single smoke configuration
has one seed, four steps per arm and two beta intervals. It only checks functionality;
it cannot establish a positive hypothesis signal. Unit tests cover actual optimizer
inputs, moments, parameter deltas, identical Adam options, and beta lag; CLI
validation checks completion, metrics, beta timing, and inherited pairing. No new
hash/checksum checks, golden regressions, or reproducibility infrastructure are added.
Formal execution is manual using full.json; it is never started by tests or smoke.
