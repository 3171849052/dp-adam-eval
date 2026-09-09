# ExpV1c completion report

Implementation, 27 passing tests (0 failures), MNIST smoke and the full formal sweep are complete. The latest user instruction authorized formal execution after testing.

## Implementation

Shared ExpV1 Fisher math and ExpV1b private-update/evaluation/optimizer helpers; independent upper-LR protocol, T90, full .80 and DP-SGD regressions, finite-safe validation/summaries/figures, and combined history.

New source/configuration/documentation files:

- `expv1c/README.md`
- `expv1c/__init__.py`
- `expv1c/common.py`
- `expv1c/configs/full.json`
- `expv1c/configs/smoke.json`
- `expv1c/plot_expv1c.py`
- `expv1c/run_pipeline.sh`
- `expv1c/summarize_expv1c.py`
- `expv1c/test_requirements.py`
- `expv1c/tests/__init__.py`
- `expv1c/tests/conftest.py`
- `expv1c/tests/test_config.py`
- `expv1c/tests/test_diagnostic_isolation.py`
- `expv1c/tests/test_lr_sweep.py`
- `expv1c/tests/test_pairing.py`
- `expv1c/tests/test_pipeline.py`
- `expv1c/tests/test_regression.py`
- `expv1c/tests/test_validation.py`
- `expv1c/train_expv1c.py`
- `expv1c/validate_expv1c.py`

## Verification

- Tests: 27 passed, 0 failed; explicit required nodeids and current provenance in `expv1c/runs/test_evidence.json`.
- Exact grid, LR-only optimizer scaling, paired RNG, diagnostic isolation, nonfinite diagnostic validator, real NaN divergence, pre/post-update divergence artifacts, .80 tiny trajectory regression, and complete pipeline passed.
- Smoke: `expv1c/runs/smoke_20260909_095210`; all five runs completed, validation passed, summary generated, 7 PNG and 7 PDF.
- Formal: `expv1c/runs/formal_20260909_095301`; all five runs completed 1170 steps, each Fisher run refreshed 24 times, validation passed, 7 PNG and 7 PDF.
- Both full anchors exactly match every private step, model hash, batch and noise RNG; Fisher .80 also exactly matches all synthetic pairing. Final hashes match the pinned requirements.
- All runs: sigma=1.068115234375, epsilon=0.995693195331761, zero nonfinite diagnostics, zero divergence.
- Final isolation check: 2,842 existing files unchanged, no new files in protected directories; git status shows only new expv1c/.

## Formal results

| Method | LR | Final accuracy | Accuracy AUC | Test CE loss | T90 |
|---|---:|---:|---:|---:|---:|
| dp_sgd | 0.10 | 89.7500% | 0.8436415888 | 0.4802647718 | unreached |
| dp_fisher_wiener | 0.80 | 90.0600% | 0.8557799065 | 0.6363517887 | 1100 |
| dp_fisher_wiener | 1.00 | 90.5600% | 0.8619467290 | 0.6460649539 | 1100 |
| dp_fisher_wiener | 1.20 | 90.7000% | 0.8649845794 | 0.6854450998 | 1170 |
| dp_fisher_wiener | 1.50 | 90.7600% | 0.8684242991 | 0.7327912292 | 1100 |

## Interpretation and invariants

Best tested LR by AUC and final accuracy: 1.5. boundary_best=true; peak_bracketed=false; stability_boundary_observed=false. No automatic grid extension. Accuracy/AUC improved while CE loss worsened: this does not establish overall or statistical superiority. Single seed42 only.

Beta1; filter after DP noise; no KFC preconditioning, private calibration, layer-wise LR, momentum, scheduler, adaptive scaling, or extra private steps. ExpV1, ExpV1b and DP-KFC unchanged.

Reproduction commands are in `expv1c/README.md`; the exact tested smoke-to-formal orchestration is `expv1c/run_pipeline.sh`. Complete execution logs are in `expv1c/runs/pipeline.log` and per-run-root stage logs.
