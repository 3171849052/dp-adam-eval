# Exp3b: offline mechanism analysis

Exp3b reads the immutable formal Exp3 result directory
`exp3/runs/formal_20260906_200824/`. It performs no training and never edits
`exp3/`, `exp2/`, `exp2b/`, `../DP-KFC/`, or the source result files.

Run from the repository root in `curve`:

```bash
conda run -n curve python -m pytest exp3b/tests -q -s
conda run -n curve python -m exp3b.analyze_exp3b \
  --source exp3/runs/formal_20260906_200824 \
  --output exp3b/results
conda run -n curve python -m exp3b.validate_exp3b \
  --source exp3/runs/formal_20260906_200824 \
  --results exp3b/results
conda run -n curve python -m exp3b.plot_exp3b --results exp3b/results
```

The four windows are train steps 1–200, 201–585, 586–1170, and 1–1170.
For oracle geometry, the early window includes step 0 and the all window
includes steps 0–1150. Geometry AUC is the mean of discrete oracle-point log
ratios, not a continuous-time integral.

`contrib_*` is retained as an epsilon-floored per-example contribution
average. Its sum is `contrib_mass`, which can be below one because Exp3 uses
`max(||h_i||², eps_num)` in the sample denominator. `nearzero_mass_deficit` is
`1 - mean_i[||h_i||² / max(||h_i||², eps_num)]`; it is not a near-zero sample
fraction.

Exp3b divides raw `contrib_*` by `contrib_mass` to form `share_*`. All layer
composition metrics (`fc_share`, `fc_to_conv`, HHI, and entropy) use these
epsilon-weighted normalized shares, whose sum is one. They are not exactly
`mean_i[||h_il||² / ||h_i||²]`: near-zero samples are implicitly down-weighted
by `min(1, ||h_i||²/eps_num)`. Existing CSVs cannot recover the exact
unfloored per-example fractions.

Exp3 did not save per-layer clipped aggregate vectors, so Exp3b does not infer
layer-wise clipped norms, layer-wise DP SNR, or layer-wise noisy update norms
from `contrib_*` or `share_*`. A larger `fc_share` is therefore described only
as redistribution of epsilon-weighted pre-clipping transformed-gradient mass,
not as proof of more useful signal.

The clipping outputs are kept as separate descriptive diagnostics: scalar
attenuation (`clipping_alpha_star` and `alpha_over_mean_coeff`), directional
distortion (`aggregate_cosine` and `clipping_shape_error`), sample-level
heterogeneity (`coefficient_cv` and `norm_cv`), and usable aggregate magnitude
(`clipped_aggregate_norm` and `diagnostic_snr`). No composite score or
significance test is constructed; temporal associations are Spearman
descriptive associations per seed.

`main_mechanism_table.csv` stores numeric mean, sample standard deviation, and
n columns for each method. `paired_deltas.csv` stores seed-level and mean ±
sample-SD paired contrasts without p-values. `deep_geometry_vs_energy.png` is
the focused SynDiag/DP-KFC comparison of diagonal FC-vs-convolution geometry
and `fc_share`.

`clipping_alpha_star` is the signed, unconstrained least-squares projection
`dot(mu_clip, mu_raw) / (||mu_raw||^2 + eps)`, so negative values are legal.
`alpha_over_mean_coeff` preserves that sign and is never log-transformed in the
primary analysis. A negative value means a signed projection reversal for that
aggregate; it does not prove that the entire training update reversed direction.
The sign is checked against `aggregate_cosine` only for rows away from zero.
Negative-alpha conditional summaries report `N/A` when no such rows exist.

`clipping_alpha_star` is the signed, unconstrained least-squares projection
`dot(mu_clip, mu_raw) / (||mu_raw||^2 + eps)`, so negative values are legal.
`alpha_over_mean_coeff` preserves that sign and is never log-transformed in the
primary analysis. A negative value means a signed projection reversal for that
aggregate; it does not prove that the entire training update reversed direction.
The sign is checked against `aggregate_cosine` only for rows away from zero.
Negative-alpha conditional summaries report `N/A` when no such rows exist.

Exp3b additionally tests whether lower `contrib_mass` / higher
`nearzero_mass_deficit` is associated with norm CV, coefficient CV, clipping,
and aggregate diagnostics. These are descriptive temporal associations only;
they are not causal claims and have no p-values.
