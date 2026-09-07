# Exp5: Adam-style EMA SynDiag

Run from the repository root in the `curve` conda environment. Exp5 is
isolated from `exp1`--`exp4`; it imports their pinned upstream adapters but
does not rewrite their implementations or result directories.

The six methods are `syn_diag`, `syn_diag_beta1`, `syn_diag_beta2`,
`syn_diag_beta12`, `dp_adam`, and `dp_kfc_adam`. SynDiag refreshes happen at
zero-based private steps `0, K, 2K, ...`, before the next private batch is
read. The beta2 state uses the actual refresh step gap in
`beta2 ** delta_t`; beta1 is an explicit privatized-direction EMA with Adam
bias correction. DP-Adam and DP-KFC+Adam use `torch.optim.Adam` with the
fixed Exp5 parameters.

Each run records model, batch, Gaussian-noise, synthetic sample/label, RNG,
and upstream provenance hashes. The five synthetic methods share the same
synthetic stream for a seed. `dp_adam` does not generate synthetic samples.
Diagnostics are research-only and run on disposable state/model copies.

Smoke and analysis loop:

```bash
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest exp5/tests -q
EXP5_SMOKE_RUNS=exp5/runs/smoke_$(date +%Y%m%d_%H%M%S)
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m exp5.train_exp5 \
  --config exp5/configs/smoke.json --output "$EXP5_SMOKE_RUNS" --method all
conda run -n curve python -m exp5.validate_exp5 \
  --config exp5/configs/smoke.json --runs "$EXP5_SMOKE_RUNS" --output "$EXP5_SMOKE_RUNS"
conda run -n curve python -m exp5.analyze_exp5 \
  --config exp5/configs/smoke.json --runs "$EXP5_SMOKE_RUNS" --output "$EXP5_SMOKE_RUNS"
conda run -n curve python -m exp5.plot_exp5 \
  --config exp5/configs/smoke.json --runs "$EXP5_SMOKE_RUNS" --output "$EXP5_SMOKE_RUNS"
```

The full experiment is intentionally not run during implementation. To start
it manually in a fresh output directory:

```bash
EXP5_FORMAL_RUNS=exp5/runs/formal_$(date +%Y%m%d_%H%M%S)
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m exp5.train_exp5 \
  --config exp5/configs/full.json --output "$EXP5_FORMAL_RUNS" --method all
conda run -n curve python -m exp5.validate_exp5 \
  --config exp5/configs/full.json --runs "$EXP5_FORMAL_RUNS" --output "$EXP5_FORMAL_RUNS"
conda run -n curve python -m exp5.analyze_exp5 \
  --config exp5/configs/full.json --runs "$EXP5_FORMAL_RUNS" --output "$EXP5_FORMAL_RUNS"
conda run -n curve python -m exp5.plot_exp5 \
  --config exp5/configs/full.json --runs "$EXP5_FORMAL_RUNS" --output "$EXP5_FORMAL_RUNS"
```
