# Exp3: paired geometry mechanism comparison

Run commands from the repository root. No formal runs are launched by tests.

```bash
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest exp3/tests -q
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m exp3.train_exp3 --config exp3/configs/full.json --output exp3/runs --method all
conda run -n curve python -m exp3.validate_exp3 --config exp3/configs/full.json --runs exp3/runs --output exp3
conda run -n curve python -m exp3.summarize_exp3 --config exp3/configs/full.json --runs exp3/runs --output exp3
conda run -n curve python -m exp3.plot_exp3 --config exp3/configs/full.json --runs exp3/runs --output exp3
```

The training command runs all nine combinations sequentially. For a single run,
add `--seed 42 --method dp_kfc` (also accepts `dp_kfc_pink_matched`). The three
output method directories are `dp_sgd`, `syn_diag`, and `dp_kfc`. Existing run
directories are never overwritten. Full configuration fixes all scientific
parameters; only device, thread count, analysis batch size, evaluation interval,
and Gram chunk size can change. A separate smoke config permits small budgets.

For smoke use the same commands with `--config exp3/configs/smoke.json`, and a
**new** directory as both training `--output` and later `--runs`/`--output`.
Smoke uses 4 private steps, refreshes at steps 0 and 2, and no oracle/refresh
point satisfies the strict late condition `step > total_steps/2`. These late
geometry summaries are N/A, never replaced by early or final-batch values.
Single-seed sample SD is also N/A. Curves still display both diagnostic points.

## Protocol and reuse

The adjacent `../DP-KFC/src` is imported read-only. SimpleCNN, MNIST transforms,
Pink noise, recorder, covariances, inverse roots, per-sample KFAC transformation,
DP clipping/noise accumulator, and RDP calibration retain upstream behavior.
The training metrics originate from exp2b, with additional population norm
mean/std/CV and quantiles. No upstream Trainer optimizer is used.

KFAC uses 10 equal 256-sample covariance batches per refresh, the upstream
covariance ridge (`eps=1e-5`) and inverse-root damping (`1e-3`). This preserves
upstream math without introducing EMA. SynDiag computes per-example squared
gradients in float64 and stores active `1/(sqrt(v)+lambda)` in FP32. All active
preconditioners remain frozen until the next refresh (0,50,...,1150).

DP-SGD uses identity. Every private update is raw per-example gradient,
precondition, global L2 clipping, average plus Gaussian noise, and explicit
SGD(lr=0.1,momentum=0). The upstream clipping stabilization is
`min(1,C/(norm+1e-6))`, unchanged from exp2/exp2b. RDP calibration uses the same
shuffle/drop_last sample-rate convention as exp2; it is not Poisson sampling.
No privacy-accountant or multiplier-calibration changes are made.

Model initialization, private loader, synthetic generation, and noise use
seed, seed+1, seed+3, and seed+4 respectively. Each RNG stream is isolated.
Synthetic sample/label hashes and before/after RNG hashes are saved per refresh;
private indices and DP-noise before/after hashes are saved at every step.
Private model hashes additionally support trajectory-isolation tests. The
oracle index seed is **314159 for every experimental seed**. Oracle diagnostics
run only after the preconditioner is frozen, on an independent model under
forked RNG; results are never inputs to training. These private diagnostic
outputs are not DP releases. Tests compare every training-step model hash with
oracle enabled/disabled for all three methods.

## Geometry and cost definitions

Layer vectors concatenate each output unit weight row with its bias. Both
private oracle and synthetic staleness use all requested samples, without
subsampling. Synthetic old/new transformations use the exact same current raw
per-example gradients. Step zero old/delta fields are empty; identity has no
refresh rows. Ratios of positive spreads use the requested definitions.
Degenerate spectra with fewer than two positive eigenvalues or zero spread
raise explicit errors instead of fabricating finite ratios.

Gradient matrices are temporary disk-backed FP32 arrays under each run.
Parameter-chunked Gram multiplication and eigensolvers use FP64 on the selected
device. Eigenvalues at or below `max(M,d)*float64_eps*largest_eigenvalue` are
excluded; significant negative eigenvalues fail. No d-by-d Fisher is formed.
Temporary arrays are removed even after errors. Full staleness needs several
GB of scratch disk per sequential run; it includes all 2560 synthetic examples.

Refresh time includes sample generation, factor/diagonal estimation, and inverse
roots, excluding staleness/oracle diagnostics. Diagnostic time is separate.
Wall time includes the training loop, refresh, diagnostics, evaluation, and
per-step audits; initialization/data loading and final file serialization are
excluded. CUDA peak allocated/reserved includes diagnostics, resets per run,
and is zero on CPU. Active state bytes count actual FP32 diagonal P or KFAC
inverse roots; covariance tensors are discarded after construction. DP-SGD
has zero refresh count/time and zero active state bytes.

Train metrics use population standard deviation. Each seed gets strictly late
medians and Q25/Q75/IQR. Oracle/refresh layer metrics first get a late median
per seed. G_diag/G_full are the geometric mean of the four late layer R
medians; plotted G values instead combine four layers at each diagnostic point.
Accuracy is evaluated every 100 steps and at the final step; late accuracy is
the mean over late evaluation points. Cross-seed tables use sample SD (ddof=1),
retain raw seed values, and report within-seed paired differences without
p-values or significance claims.

Each run stores config, metadata/source hashes, three metric CSVs, pairing.json,
and summary.json. Top-level summary tables and 12 matplotlib PNG figures are
written only by the explicit summary/plot commands. Validation requires all
expected runs, 1170 steps for full mode, complete finite layer diagnostics,
matched config/source provenance, refresh schedules, and paired RNG audits.
