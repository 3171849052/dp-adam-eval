# Exp5b: matched SGD-scale temporal optimization

Exp5b is a follow-up to Exp5. Exp5 used the same nominal `lr=.001` for
direct/EMA SynDiag and Adam optimizers, although their actual parameter-update
scales were not matched. This is an experimental-design optimizer-scale
confound, not an Exp5 code bug.

Exp5b isolates the matched SGD-style regime by removing Adam training
baselines, setting `learning_rate=.1` for all six methods, and using matched
post-DP first-moment EMA baselines. The six methods are:

`syn_diag`, `syn_diag_beta1`, `syn_diag_beta2`, `syn_diag_beta12`,
`dp_sgd_momentum`, and `dp_kfc_momentum`.

`syn_diag_beta12` is Full. All four momentum-enabled methods use exactly the
same explicit `FirstMomentState`:

```text
m_t = 0.9 m_(t-1) + 0.1 tilde_g_t
m_hat_t = m_t / (1 - 0.9^t)
theta_t = theta_(t-1) - 0.1 m_hat_t
```

Here `tilde_g_t` is produced only after optional preconditioning, global
per-example clipping, Gaussian noise, and batch averaging. No official method
uses `torch.optim.SGD(momentum=.9)` or Adam. The clean Adam direction retained
in the logs is a mechanistic diagnostic reference only, not an Exp5b competing
optimizer.

Synthetic refresh uses pink-noise images and labels from
`torch.randint(0, 10, ...)`. Refresh occurs before reading the current private
batch. β2 uses `v_0=q_0` and `beta2 ** delta_t`, where `delta_t` is the actual
number of private steps since the preceding refresh. The KFAC path uses the
pinned upstream DP-KFC commit `eb31b9aeb2280642684f4cedfa65cc02b76c76cd`,
covariance ridge `1e-5`, and inverse-root damping `1e-3`.

The preregistered paired contrasts are Full minus SynDiag, SynDiag+β1,
SynDiag+β2, DP-SGD-Momentum, and DP-KFC-Momentum. Results report paired
per-seed deltas followed by mean and sample SD (`ddof=1`), without p-values or
significance claims.

## Research questions

1. Does post-DP first-moment EMA improve SynDiag utility under an SGD-scale learning rate?
2. Does true-private-time synthetic β2 smoothing improve SynDiag utility?
3. Do β1 and β2 interact constructively?
4. How does Full compare with matched DP-SGD-Momentum and DP-KFC-Momentum in utility, geometry, and efficiency?

## Smoke run

```bash
EXP5B_SMOKE_RUNS=exp5b/runs/smoke_$(date +%Y%m%d_%H%M%S)
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m exp5b.train_exp5b \
  --config exp5b/configs/smoke.json --output "$EXP5B_SMOKE_RUNS" --method all
conda run -n curve python -m exp5b.validate_exp5b \
  --config exp5b/configs/smoke.json --runs "$EXP5B_SMOKE_RUNS" --output "$EXP5B_SMOKE_RUNS"
conda run -n curve python -m exp5b.analyze_exp5b \
  --config exp5b/configs/smoke.json --runs "$EXP5B_SMOKE_RUNS" --output "$EXP5B_SMOKE_RUNS"
conda run -n curve python -m exp5b.plot_exp5b \
  --config exp5b/configs/smoke.json --runs "$EXP5B_SMOKE_RUNS" --output "$EXP5B_SMOKE_RUNS"
```

The full experiment is intentionally not run during implementation. Run it
manually only after reviewing the code and smoke validation.

## Formal run command

```bash
EXP5B_FORMAL_RUNS=exp5b/runs/formal_$(date +%Y%m%d_%H%M%S)
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m exp5b.train_exp5b \
  --config exp5b/configs/full.json --output "$EXP5B_FORMAL_RUNS" --method all
conda run -n curve python -m exp5b.validate_exp5b \
  --config exp5b/configs/full.json --runs "$EXP5B_FORMAL_RUNS" --output "$EXP5B_FORMAL_RUNS"
conda run -n curve python -m exp5b.analyze_exp5b \
  --config exp5b/configs/full.json --runs "$EXP5B_FORMAL_RUNS" --output "$EXP5B_FORMAL_RUNS"
conda run -n curve python -m exp5b.plot_exp5b \
  --config exp5b/configs/full.json --runs "$EXP5B_FORMAL_RUNS" --output "$EXP5B_FORMAL_RUNS"
```
