# ExpV4b: adaptive beta after Wiener signal-scale recovery

Research prototype. Primary question: does adaptive beta improve accuracy AUC
once both Fisher arms actually use their own raw model gamma? No full run is
started automatically. Smoke is functional only.

| Arm | lr | Wiener beta | gamma |
| --- | --- | --- | --- |
| `dp_sgd_lr0p50` | 0.5 | no filter/controller | none |
| `dp_fisher_wiener_beta1_gamma_lr0p50` | 0.5 | fixed 1 | raw model gamma |
| `dp_fisher_wiener_adaptive_beta_gamma_lr0p50` | 0.5 | ExpV3 adaptive controller | raw model gamma |

For each layer, a = beta * lambda_F, H = a/(a+r), and
`gamma = sqrt(sum(a) / sum(a * H**2))`. ExpV4a's estimator reads the actual
active state once per refresh. There is no gamma cap, clipping, or fallback.
The gradient path is unchanged through global per-example clipping, averaging,
and DP Gaussian noise. ExpV3 applies Wiener, then ExpV4b multiplies each layer's
weight and bias `.grad` by gamma, then the real SGD optimizer steps.

The fixed arm finalizes measurements with `apply=False`, keeping beta=1.
The adaptive arm uses the unchanged `one_interval_lag_hold_last_positive`
controller: interval 0 uses beta=1; each refresh finalizes the previous interval,
rebuilds H with the resulting active beta, and computes gamma from that H.
H/gamma are held together until the next refresh. Oracle diagnostics run
downstream and cannot feed gamma, the filter, or the controller.

The training loop, synthetic Fisher construction, clipping/noise, and RNG
streams are reused from V3. Its existing pairing/hashes are retained, including
measurement-only synthetic Fisher for the DP-SGD baseline. Synthetic **randomness**
is paired; Fisher matrices can differ once model trajectories diverge.
V4b uses its own config/output location and does not require V3's old pinned
commit; no new provenance or reproducibility infrastructure is added.

## Metrics and minimal validation

`layer_metrics.csv` keeps V3's `*_filtered`, `signal_retention`, and
`noise_retention` as **pre-gamma Wy** diagnostics. Additional `*_after_gamma`
metrics describe the compensated update. `optimizer_update_norm` and effective
signal LR use the compensated gradient and signal retention. Beta/model/oracle
gamma are per-layer quantities, not fictitious whole-model scalars; the global
`gamma_update_ratio` is the ratio of concatenated gradient norms.
`gamma_refresh_metrics.csv` records the model gamma actually used at refresh.
Nonfinite compensated gradients use divergence stage `compensated_gradient`.

The validator checks completion in smoke, beta lag/fixed beta, gamma alignment,
actual scaling ratios, retention identities, and the existing pairing audit.
The three focused unit cases check gamma=1, uncapped gamma=1e6 through actual
SGD, and controller/H/gamma math. No golden regression or hardening suite is added.

Summary outputs:

- `utility_by_seed.csv`, `utility_summary.csv`: A/B/C utility and available counts.
- `paired_C_minus_B.csv`, `primary_endpoint.json`: each seed's C-B differences,
  mean/median AUC difference, positive and completed pair counts.
- `mechanism_summary.csv`: B/C overall and per-layer median/q10/q90, pooling
  layer-step observations; includes update magnitude and pre/post gamma quality.
- `gamma_ranges.csv`, `summary.txt`: observed raw gamma ranges and divergence.

Unreached thresholds remain missing. Diverged trajectories do not contribute a
completed AUC, and missing pairs are reported explicitly. Formal conclusions
should compare post-gamma signal scale and update norms alongside cosine/SNR/
relMSE, rather than treating a larger update as evidence for better filtering.

## Run

All Python commands use `curve`:

```bash
conda run -n curve python -m pytest expv4b/tests -q -s
EXPV4B_SMOKE_RUNS=expv4b/runs/smoke_$(date +%Y%m%d_%H%M%S)
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m expv4b.train_expv4b --config expv4b/configs/smoke.json --output "$EXPV4B_SMOKE_RUNS" --run all
conda run -n curve python -m expv4b.validate_expv4b --config expv4b/configs/smoke.json --runs "$EXPV4B_SMOKE_RUNS" --output "$EXPV4B_SMOKE_RUNS"
conda run -n curve python -m expv4b.summarize_expv4b --config expv4b/configs/smoke.json --runs "$EXPV4B_SMOKE_RUNS" --output "$EXPV4B_SMOKE_RUNS"
```

For a manually launched formal run, use `configs/full.json` and a fresh output
directory. It retains V3's five-epoch formal protocol at lr=0.5, K=beta_window=50,
M_syn=2560, batch_size=256, epsilon=1, delta=1e-5, C=1, eval_interval=100,
and uses seeds 123/456/789/1024/2027. Formal has not been run as part of setup.
