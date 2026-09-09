# ExpV3 audit notes

This note records the read-only audit performed before implementation.

- `expv1/fisher_wiener.py`: `synthetic_samples` owns pink-noise probes and
  paired synthetic RNG; `build_covariances` uses the upstream KFAC recorder;
  `build_fisher_state` produces `Q_A`, `lambda_A`, `Q_G`, `lambda_G`, and the
  beta=1 gain; packed gradients include the bias column. The private update
  order is clipping, Gaussian noise, capture of `p.grad`, filter, then SGD.
- `expv1/metrics.py`: reconstruction and mechanism diagnostics operate on
  `summed_grad`, pre-filter noisy gradients, and filtered gradients. ExpV3
  passes the diagnostic method name `dp_fisher_wiener` while retaining its
  separate artifact method name.
- `expv1b`: divergence is algorithmic-only; noisy/filtered gradients and
  post-step parameters can stop a run, while research diagnostics do not.
  Diverged artifacts compare paired streams over their common completed
  prefix.
- `expv2/beta_estimation.py`: `noise_variance`, `single_step_beta`, and
  `aggregate_beta_window` provide the sole beta formulas. Window estimates
  are ratio-of-sums, not means of per-step ratios; `build_hold_predictor`
  confirms the one-interval hold-last-positive semantics.
- Fixed roots audited read-only: ExpV1 formal
  `expv1/runs/formal_20260908_182603`, ExpV1b formal
  `expv1b/runs/formal_20260909_003544`, and ExpV2 formal
  `expv2/runs/formal_20260909_124540`.

Implementation consequence: ExpV3 performs one ExpV1 eigendecomposition per
refresh, reuses that eigensystem for both adaptive and beta=1 counterfactual
gains, and lets only past DP-safe scalar observations update the controller.

Hardening consequence: `capture_deployable_beta_observations` reads only
pre-Wiener `p.grad`; `capture_oracle_beta_diagnostics` is a separate
research-only path reading the clean clipped gradient.  Controller rows now
retain exact previous-interval numerators, denominators, beta values, and
cumulative decision counts.  Runtime bookkeeping distinguishes adaptive beta
observation/controller time from research diagnostics and measurement-only
DP-SGD work.  The validator recomputes step, interval, controller, H, runtime,
and root-seed invariants rather than trusting derived artifact fields.
