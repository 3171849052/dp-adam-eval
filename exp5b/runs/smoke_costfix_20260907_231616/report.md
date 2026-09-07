# Exp5b: Matched SGD-scale temporal optimization

Exp5b removes Adam training baselines and sets learning_rate=0.1 for all six methods.
All four momentum-enabled methods use the same post-DP, bias-corrected FirstMomentState.
The clean Adam direction is a mechanistic diagnostic reference only, not an Exp5b competing optimizer.
All contrasts are paired within seed and report mean +/- sample SD (ddof=1); no p-values or significance claims are reported.

Smoke validation only; this output is not a formal experimental conclusion.

## Main results

| Method | Final Acc ↑ | Final Loss ↓ | Late Acc ↑ | Core Time ↓ | Refresh Time ↓ | Algorithm State Bytes ↓ | Peak CUDA Memory Core ↓ | Best Acc | Mean Refresh Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| syn_diag | 0.0625 | 10.88 | 0.0625 | 0.5953 | 0.528 | 827688 | 91642368 | 0.0625 | 0.264 |
| syn_diag_beta1 | 0.125 | 7.943 | 0.125 | 0.113 | 0.04672 | 1655376 | 94130688 | 0.125 | 0.02336 |
| syn_diag_beta2 | 0.0625 | 10.78 | 0.0625 | 0.1139 | 0.04567 | 2483064 | 96563712 | 0.0625 | 0.02283 |
| syn_diag_beta12 | 0.0625 | 7.978 | 0.0625 | 0.1203 | 0.05256 | 3310752 | 99052032 | 0.0625 | 0.02628 |
| dp_sgd_momentum | 0.125 | 7.612 | 0.125 | 0.05999 | 0 | 827688 | 84245504 | 0.125 | 0 |
| dp_kfc_momentum | 0.125 | 7.833 | 0.125 | 0.2031 | 0.1299 | 10896852 | 169951232 | 0.125 | 0.06497 |

## Paired contrasts

| Contrast | Metric | Mean | Sample SD | N |
| --- | --- | ---: | ---: | ---: |
| Full - SynDiag | final_accuracy | 0 | N/A | 1 |
| Full - SynDiag | final_test_loss | -2.89976 | N/A | 1 |
| Full - SynDiag | late_mean_accuracy | 0 | N/A | 1 |
| Full - syn_diag_beta1 | final_accuracy | -0.0625 | N/A | 1 |
| Full - syn_diag_beta1 | final_test_loss | 0.0350184 | N/A | 1 |
| Full - syn_diag_beta1 | late_mean_accuracy | -0.0625 | N/A | 1 |
| Full - syn_diag_beta2 | final_accuracy | 0 | N/A | 1 |
| Full - syn_diag_beta2 | final_test_loss | -2.80424 | N/A | 1 |
| Full - syn_diag_beta2 | late_mean_accuracy | 0 | N/A | 1 |
| Full - dp_sgd_momentum | final_accuracy | -0.0625 | N/A | 1 |
| Full - dp_sgd_momentum | final_test_loss | 0.36601 | N/A | 1 |
| Full - dp_sgd_momentum | late_mean_accuracy | -0.0625 | N/A | 1 |
| Full - dp_kfc_momentum | final_accuracy | -0.0625 | N/A | 1 |
| Full - dp_kfc_momentum | final_test_loss | 0.144783 | N/A | 1 |
| Full - dp_kfc_momentum | late_mean_accuracy | -0.0625 | N/A | 1 |

## SynDiag factorial effects

Y00=SynDiag, Y10=SynDiag+β1, Y01=SynDiag+β2, Y11=Full. Values are paired within seed; no significance claims.

| Metric | β1 effect mean +/- SD | β2 effect mean +/- SD | Interaction mean +/- SD | N |
| --- | ---: | ---: | ---: | ---: |
| final_accuracy | 0.0625 +/- N/A | 0 +/- N/A | -0.0625 +/- N/A | 1 |
| final_test_loss | -2.93478 +/- N/A | -0.0955234 +/- N/A | 0.130542 +/- N/A | 1 |
| late_mean_accuracy | 0.0625 +/- N/A | 0 +/- N/A | -0.0625 +/- N/A | 1 |

## Research questions

- RQ1: Does post-DP first-moment EMA improve SynDiag utility under an SGD-scale learning rate?
- RQ2: Does true-private-time synthetic β2 smoothing improve SynDiag utility?
- RQ3: Do β1 and β2 interact constructively?
- RQ4: How does Full compare with matched DP-SGD-Momentum and DP-KFC-Momentum in utility, geometry, and efficiency?

Results are reported as observed; no outcome is hardcoded or treated as statistically significant.
