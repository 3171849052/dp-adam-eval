# ExpV1b — Global Learning-Rate Sweep for DP-Fisher-Wiener

ExpV1b is the isolated follow-up to ExpV1. ExpV1 showed that DP-Fisher-Wiener
improves gradient reconstruction, cosine, and SNR, but its utility at the
fixed 1170-step horizon remained below DP-SGD. Since Wiener filtering shrinks
the clean signal update, this experiment asks one question only:

> With beta, Fisher construction, privacy, clipping, noise, optimizer family,
> and the training horizon frozen, can a larger global SGD learning rate
> recover optimization speed and final utility?

The only scientific variable is the global learning rate. The formal grid is
`0.10, 0.15, 0.20, 0.30, 0.50, 0.80` for Fisher-Wiener, plus the fixed
DP-SGD anchor at `0.10`. There is no Scalar-Wiener run in ExpV1b.

## Fixed protocol

The formal experiment uses seed 42 only, MNIST, `SimpleCNN`, five epochs,
batch size 256, epsilon 1.0, delta `1e-5`, clipping norm 1.0, plain SGD with
momentum zero, beta 1, `K=50`, `M_syn=2560`, evaluation every 100 private
steps, four Torch threads, and the full train/test sets. This gives 1170
private steps and 24 Fisher refreshes. The smoke configuration inherits the
same protocol but uses 16/32 examples, batch size 4, one epoch, `K=2`, and
`M_syn=8`; it still runs all seven canonical run IDs.

Privacy uses the ExpV1 fixed-shuffle/drop-last convention, not Poisson
sampling. With `q=B/N`, the noise multiplier is computed once from epsilon,
delta, q, and the planned number of steps using the RDP accountant. Every
step accounts the same `(sigma, q)` pair. Learning rate does not affect the
accountant, clipping, Gaussian noise, refresh schedule, or any RNG stream.

## Unchanged Fisher-Wiener mechanism

ExpV1's read-only mathematical implementation is imported directly from
`expv1.fisher_wiener` and `expv1.metrics`; it is not copied here. For each
layer, `F=A otimes G`, beta is fixed at one, and

```
r = (sigma * C / B)^2
H[j,i] = lambda_G[j] * lambda_A[i]
         / (lambda_G[j] * lambda_A[i] + r)
Z = Q_G.T @ Y @ Q_A
Y_hat = Q_G @ (H * Z) @ Q_A.T
```

The private order is exactly:

```
loss.backward()
 -> global clipping
 -> DP Gaussian noise
 -> Fisher-Wiener post-noise filter
 -> SGD(lr=run learning rate, momentum=0)
```

There is no KFC preconditioning, private Fisher, beta calibration, layer-wise
learning rate, normalization, rescaling, update matching, scheduler, warmup,
momentum, Adam-family optimizer, mean prior, or extra private step.

## Pairing and diagnostics

All runs use init=`seed`, loader=`seed+1`, test=`seed+2`, synthetic=`seed+3`,
and noise=`seed+4`. Initial model hashes, private batch indices, noise RNG
before/after hashes, and Fisher synthetic sample/label hashes are paired. LR
does not enter any seed, branch, sampling decision, refresh schedule, or
synthetic generation. Different LR values do produce different parameter
trajectories; therefore later `A`, `G`, eigenvalues, and `H` values may differ
because the model state at refresh differs. The formula and configuration do
not change.

The following global and per-layer research diagnostics are retained from
ExpV1: reconstruction error, cosine, signal/noise retention, SNR gain, Fisher
trace/kappa, H statistics, and eigenvalue summaries. Eigenmode analysis is
disabled (`eigenmode_diagnostics=false`, `eigen_budget=0`) and
`eigenbin_metrics.csv` is intentionally empty with a valid schema.

ExpV1b additionally records:

```
signal_amplitude_retention = sqrt(max(signal_retention, 0))
effective_signal_lr       = learning_rate * signal_amplitude_retention
optimizer_update_norm     = learning_rate * filtered_gradient_norm
clean_reference_update_norm = 0.1 * clean_clipped_norm
update_to_clean_reference_ratio = optimizer_update_norm /
                                  (clean_reference_update_norm + 1e-12)
```

`effective_signal_lr` is the primary pure-signal step diagnostic. It uses the
non-DP `summed_grad` diagnostic and is never fed back into optimization. The
update-to-reference ratio includes residual noise and is therefore not a pure
signal-step estimator.

`steps_to_acc_50`, `steps_to_acc_70`, `steps_to_acc_80`, and `steps_to_acc_85`
are the first evaluation points meeting each threshold. They use completed
step `step+1`, do not interpolate, and remain null when unreached. Accuracy
AUC, final accuracy, best accuracy, late mean accuracy, and final test loss
are post-hoc summaries; there is no early stopping or best-checkpoint restore.

Runs that encounter non-finite loss, gradient, or update stop at the first
non-finite state as `status="diverged"`. They record the divergence step,
completed steps, epsilon at divergence, and last finite diagnostics. A
divergent run is not compared as a normal final-step utility result.

## Interpretation registered before the sweep

- If a middle LR improves accuracy/AUC and approaches or exceeds DP-SGD, that
  supports effective-step shrinkage as one important cause of ExpV1's gap.
- If accuracy keeps improving through 0.80, the upper boundary may still be
  low; a future experiment may extend it, but ExpV1b does not auto-expand.
- If middle LRs improve only partly and remain below DP-SGD, global LR explains
  only part of the gap; later work may study other mechanisms.
- If 0.10 is best or every larger LR worsens utility, the sweep does not
  support the hypothesis that the main issue was an overly small LR.

These are diagnostic rules, not automatic claims or tuning logic.

## Scope and provenance

This is a single-seed (`42`) hyperparameter/mechanism diagnostic, not a
confirmatory experiment. Even if one LR exceeds the DP-SGD anchor, the proper
statement is that it was best in the seed42 sweep. It is not evidence of
statistical superiority: no p-values, significance tests, confidence
intervals, or cross-seed averages are produced. A candidate LR must be fixed
and evaluated with new independent seeds in a later confirmatory experiment.

Every run stores upstream DP-KFC commit/dirty/remote information, upstream
Python hashes, ExpV1 dependency hashes (`common.py`, `fisher_wiener.py`, and
`metrics.py`), ExpV1b source hashes, config fingerprint, and pairing evidence.
Formal validation requires clean upstream commit
`eb31b9aeb2280642684f4cedfa65cc02b76c76cd`. The immutable ExpV1 seed42
reference anchors are:

```
DP-SGD lr=.10:           9e9c9e76e84680f6fcb1c34ded5047f73f36b54c8616fb14293a470270fce1b2
DP-Fisher-Wiener lr=.10: c04595c804fe9f04bd3421385e6e75e920bc288e9920c486fa0929c009ed6aa7
```

The full validator checks these hashes and, when the reference artifact is
available, all comparable private pairing/model-hash trajectories and Fisher
synthetic hashes. ExpV1 is read-only; all new code, caches, artifacts,
validation, summaries, evidence, and figures are confined to `expv1b/`.

## Commands

Implementation and smoke commands do not start the formal sweep. After tests,
the smoke pipeline is:

```bash
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m pytest expv1b/tests -q -s

EXPV1B_SMOKE_RUNS=expv1b/runs/smoke_$(date +%Y%m%d_%H%M%S)
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m expv1b.train_expv1b \
  --config expv1b/configs/smoke.json \
  --output "$EXPV1B_SMOKE_RUNS" \
  --run all

conda run -n curve python -m expv1b.validate_expv1b \
  --config expv1b/configs/smoke.json \
  --runs "$EXPV1B_SMOKE_RUNS" \
  --output "$EXPV1B_SMOKE_RUNS"

conda run -n curve python -m expv1b.summarize_expv1b \
  --config expv1b/configs/smoke.json \
  --runs "$EXPV1B_SMOKE_RUNS" \
  --output "$EXPV1B_SMOKE_RUNS"

conda run -n curve python -m expv1b.plot_expv1b \
  --config expv1b/configs/smoke.json \
  --runs "$EXPV1B_SMOKE_RUNS" \
  --output "$EXPV1B_SMOKE_RUNS"
```

The formal command is intentionally printed for the final handoff only and
must be explicitly launched by the user; this implementation does not run it.

```bash
EXPV1B_FORMAL_RUNS=expv1b/runs/formal_$(date +%Y%m%d_%H%M%S)
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m expv1b.train_expv1b \
  --config expv1b/configs/full.json \
  --output "$EXPV1B_FORMAL_RUNS" \
  --run all

conda run -n curve python -m expv1b.validate_expv1b \
  --config expv1b/configs/full.json \
  --runs "$EXPV1B_FORMAL_RUNS" \
  --output "$EXPV1B_FORMAL_RUNS"

conda run -n curve python -m expv1b.summarize_expv1b \
  --config expv1b/configs/full.json \
  --runs "$EXPV1B_FORMAL_RUNS" \
  --output "$EXPV1B_FORMAL_RUNS"

conda run -n curve python -m expv1b.plot_expv1b \
  --config expv1b/configs/full.json \
  --runs "$EXPV1B_FORMAL_RUNS" \
  --output "$EXPV1B_FORMAL_RUNS"
```
