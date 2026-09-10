# Exp6b: post-DP synthetic second moments with a noise floor

Exp6b tests whether the catastrophic post-DP SynAdam failure observed in Exp6
is primarily caused by the absence of the DP Gaussian noise variance in the
synthetic second-moment denominator. The two synthetic methods therefore use
an additive floor `r=(sigma*C/B)^2`, while DP-Adam remains unchanged.

The methods are `syn_adam_floor`, `syn_adam_ema_floor`, and `dp_adam`. The
synthetic q state is post-processing only: it does not modify private
`.grad_sample` tensors or participate in clipping or noise addition. The EMA
method applies the floor after bias correction and does not put it into its
synthetic state. This is not a complete noisy-gradient second-moment
estimator; it only tests whether the DP noise variance floor is enough to
restore stable training.

Smoke run from the repository root:

```bash
EXP6B_SMOKE_RUNS=exp6b/runs/smoke_$(date +%Y%m%d_%H%M%S)
conda run -n curve python -m exp6b.train_exp6b --config exp6b/configs/smoke.json --output "$EXP6B_SMOKE_RUNS" --method all
conda run -n curve python -m exp6b.validate_exp6b --config exp6b/configs/smoke.json --runs "$EXP6B_SMOKE_RUNS" --output "$EXP6B_SMOKE_RUNS"
conda run -n curve python -m exp6b.analyze_exp6b --config exp6b/configs/smoke.json --runs "$EXP6B_SMOKE_RUNS" --output "$EXP6B_SMOKE_RUNS"
```

Complete experiment:

```bash
conda run -n curve python -m exp6b.train_exp6b \
  --config exp6b/configs/full.json \
  --output exp6b/runs/formal \
  --method all
```
