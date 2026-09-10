# Exp6: post-DP synthetic second moments

Exp6 asks:

> Can synthetic second-moment estimates improve DP optimization when used purely as Adam-style post-DP normalization of the privatized first moment, and does beta2 EMA improve those estimates?

The three paired methods are `syn_adam`, `syn_adam_ema`, and `dp_adam`. All
three use the same private pipeline:

```
per-sample gradients -> global clip -> aggregate -> Gaussian noise -> tilde g
```

They then use the explicit first-moment EMA
`m_t = beta1*m_(t-1) + (1-beta1)*tilde_g_t`. `syn_adam` uses the latest
synthetic `q`; `syn_adam_ema` updates a beta2 EMA of that q on every private
step; and `dp_adam` updates beta2 from `tilde_g_t**2`.

Synthetic inputs are pink noise with random labels. They are refreshed every
`K` private steps using the current model, before the current private batch is
read. The resulting q is optimizer state/post-processing only: it never
modifies the private `.grad_sample` and does not participate in clipping or
noise addition.

Smoke run from the repository root:

```bash
EXP6_SMOKE_RUNS=exp6/runs/smoke_$(date +%Y%m%d_%H%M%S)
conda run -n curve python -m exp6.train_exp6 --config exp6/configs/smoke.json --output "$EXP6_SMOKE_RUNS" --method all
conda run -n curve python -m exp6.validate_exp6 --config exp6/configs/smoke.json --runs "$EXP6_SMOKE_RUNS" --output "$EXP6_SMOKE_RUNS"
conda run -n curve python -m exp6.analyze_exp6 --config exp6/configs/smoke.json --runs "$EXP6_SMOKE_RUNS" --output "$EXP6_SMOKE_RUNS"
```

Start the complete three-seed experiment with:

```bash
conda run -n curve python -m exp6.train_exp6 \
  --config exp6/configs/full.json \
  --output exp6/runs/formal \
  --method all
```
