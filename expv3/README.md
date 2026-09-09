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

The two beta data paths are physically separate.  The algorithmic path reads
only the pre-Wiener DP gradient `y`, computes noisy energy minus the known
expected noise energy, pools the ratio-of-sums beta estimate, and applies it at
the next interval's `H` refresh.  The research oracle path separately reads
the clipped clean gradient `s` (`summed_grad`) and computes oracle beta only.
Oracle fields are never passed to the controller.  Interval 0 has
`active_beta_source_interval=null`; every step in interval `m >= 1` has source
`m-1`.

Adaptive beta observation and controller work are algorithmic runtime.  For
DP-SGD, synthetic Fisher and beta measurement are research overhead because
the baseline does not use them for training.  Runtime summaries expose these
components separately.

The training outputs include research-only clean/oracle diagnostics, so the
complete artifact is not DP-release-safe.  The deployable controller itself
uses only past DP `y`, known noise variance, and synthetic Fisher traces, and
therefore is DP post-processing with no additional accountant step.

Formal runs are intentionally never launched automatically.  Run the tests and
smoke pipeline first, then use the command printed by the agent.

Privacy accounting occurs immediately after a Gaussian DP mechanism is
successfully executed, not after the Wiener filter or optimizer update.
`planned_privacy_steps` is the calibrated run length. `privacy_steps` counts
consumed Gaussian queries and may exceed `completed_steps` by one.
The validator independently recomputes RDP epsilon from consumed queries;
the formal full-run epsilon anchor applies only to completed runs. The fixed
formal sigma and sample rate apply to both completed and diverged runs.

A completed step has a successful optimizer update, finite parameters, and a
train artifact. A parameter-nonfinite update is excluded from completed steps
and ordinary evaluation, while its privacy query and valid beta observation
remain recorded. `divergence_stage` is null for completion, or one of `loss`,
`noisy_gradient`, `filtered_gradient`, `optimizer_exception`, `parameters`.

| Failure boundary | Extra privacy query | Extra beta observation | Completed update |
| --- | --- | --- | --- |
| Loss before DP | No | No | No |
| Nonfinite noisy gradient | Yes | No | No |
| Nonfinite filtered gradient | Yes | Yes | No |
| Optimizer exception / nonfinite parameters | Yes | Yes | No |

“Loss before DP -> No extra privacy query”. This only means that no additional Gaussian mechanism was consumed. Because the abort decision depends on the current pre-DP private loss, the accountant does not cover that observable stopping event. Therefore a loss-stage-diverged artifact retains the epsilon for mechanisms executed before the abort, but does not claim complete end-to-end DP accounting. Post-DP divergence and completed runs have complete end-to-end accounting when their privacy steps, epsilon, and pairing validate. This is separate from `release_safe_under_dp=false`, which remains false because the full research artifact contains a non-DP oracle.

`pairing.private` records successful finite training steps; `pairing.privacy`
records all executed DP mechanisms, including final post-noise failures;
`beta_step_metrics` records all successfully formed deployable observations.
These lengths can differ at divergence. Each beta row explicitly records
`dp_observation_recorded` and `optimizer_step_completed`. Partial intervals
include valid observations from failed updates. An interval is marked
`applied_to_training` / `was_used_by_next_interval` only when a successful
optimizer update used the next interval's H; constructing a refresh or
recording an observation does not suffice. No gradient or noise tensors are
saved. `actual_noise_saved=false` remains unchanged.

Oracle capture exceptions are recorded by exception type without exception
messages or sensitive payloads. They invalidate research diagnostics and
cannot change status, controller decisions, H, or training. Missing oracle
values remain missing in interval diagnostics while deployable algebra remains
available. Deployable capture, controller, clipping/noise, and H exceptions
are not swallowed as research errors.

The internal keyword-only `synthetic_measurement=False` control exists solely
for DP-SGD isolation tests and requires `beta_measurement=False`. It skips all
synthetic samples, covariances, traces, and measurement refreshes. Adaptive
Fisher rejects it. CLI and formal/smoke configs retain measurement enabled.

Raw beta median and negative rate use finite `beta_dp_raw` values from all
observed interval rows, including final and partial intervals. Empty finite
sets report null, including DP-SGD without measurements. Fallback and accepted
rates still use controller decision rows with `interval_index > 0`.

`failed_step_metrics.csv` is the authoritative record for partial runtime from
one failed post-DP step; it never adds a fake row to `train_metrics.csv`, whose
rows remain successful finite optimizer updates only. Loss-stage aborts occur
before `private_update` and therefore have no failed private timing row.

Multi-seed accuracy and test-loss curves are aggregated by step across seeds;
seed trajectories are never concatenated into one line. The `beta_train` curve
uses the median across seeds at each interval. `beta_raw/oracle` is the
contemporaneous `beta_dp_raw / beta_oracle` diagnostic, while
`beta_train/oracle` is the actual `beta_train` used in interval `m` divided by
the oracle beta from that same interval `m`.

`H_q50_adaptive_vs_beta1` uses the same-refresh Fisher spectrum to compare the
actual adaptive-beta `H_q50` with the beta=1 counterfactual `H_beta1_q50`, as
recorded in `beta_controller_metrics.csv`.

Mechanism and layer summary rows carry `seed`, `run_id`, `status`,
`completed_steps`, and `metric_scope` (`full` or `prefix`). Learning-rate full
utility aggregates use completed runs only and report `n_runs`, `n_completed`,
and `n_diverged`; paired matched-LR utility reports `n_total_pairs` and
`n_valid_pairs`.
