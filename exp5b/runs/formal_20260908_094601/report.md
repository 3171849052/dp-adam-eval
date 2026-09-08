# Exp5b: Matched SGD-scale temporal optimization

Exp5b removes Adam training baselines and sets learning_rate=0.1 for all six methods.
All four momentum-enabled methods use the same post-DP, bias-corrected FirstMomentState.
The clean Adam direction is a mechanistic diagnostic reference only, not an Exp5b competing optimizer.
All contrasts are paired within seed and report mean +/- sample SD (ddof=1); no p-values or significance claims are reported.

## Main results

| Method | Final Acc ↑ | Final Loss ↓ | Late Acc ↑ | Core Time ↓ | Refresh Time ↓ | Algorithm State Bytes ↓ | Peak CUDA Memory Core ↓ | Best Acc | Mean Refresh Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| syn_diag | 0.903 +/- 0.004771 | 0.465 +/- 0.02735 | 0.8858 +/- 0.007039 | 62.42 +/- 0.9806 | 12.25 +/- 0.3247 | 827688 | 519717376 | 0.903 +/- 0.004771 | 0.5103 +/- 0.01353 |
| syn_diag_beta1 | 0.9029 +/- 0.006475 | 0.4669 +/- 0.02483 | 0.8858 +/- 0.008768 | 62.82 +/- 0.7138 | 12.12 +/- 0.1771 | 1655376 | 521376256 | 0.9029 +/- 0.006475 | 0.5049 +/- 0.007379 |
| syn_diag_beta2 | 0.9029 +/- 0.005285 | 0.4931 +/- 0.02592 | 0.8845 +/- 0.008174 | 61.62 +/- 0.4593 | 12.11 +/- 0.1705 | 2483064 | 521466880 | 0.9029 +/- 0.005285 | 0.5047 +/- 0.007105 |
| syn_diag_beta12 | 0.9023 +/- 0.007063 | 0.4937 +/- 0.02688 | 0.8852 +/- 0.009168 | 62.17 +/- 0.5968 | 12.1 +/- 0.176 | 3310752 | 523125760 | 0.9023 +/- 0.007063 | 0.5042 +/- 0.007332 |
| dp_sgd_momentum | 0.8966 +/- 0.00311 | 0.48 +/- 0.01464 | 0.8863 +/- 0.003979 | 48.94 +/- 0.2706 | 0 +/- 0 | 827688 | 520547328 | 0.8966 +/- 0.00311 | 0 +/- 0 |
| dp_kfc_momentum | 0.9456 +/- 0.002157 | 0.3116 +/- 0.006643 | 0.9347 +/- 0.003667 | 67.89 +/- 0.5695 | 3.484 +/- 0.05132 | 10896852 | 1116627968 | 0.9456 +/- 0.002157 | 0.1452 +/- 0.002138 |

## Paired contrasts

| Contrast | Metric | Mean | Sample SD | N |
| --- | --- | ---: | ---: | ---: |
| Full - SynDiag | final_accuracy | -0.000733333 | 0.00248261 | 3 |
| Full - SynDiag | final_test_loss | 0.0287765 | 0.00953712 | 3 |
| Full - SynDiag | late_mean_accuracy | -0.000519048 | 0.00213534 | 3 |
| Full - syn_diag_beta1 | final_accuracy | -0.000633333 | 0.00085049 | 3 |
| Full - syn_diag_beta1 | final_test_loss | 0.0268428 | 0.00289127 | 3 |
| Full - syn_diag_beta1 | late_mean_accuracy | -0.00052381 | 0.000968939 | 3 |
| Full - syn_diag_beta2 | final_accuracy | -0.0006 | 0.00217025 | 3 |
| Full - syn_diag_beta2 | final_test_loss | 0.000678286 | 0.00477138 | 3 |
| Full - syn_diag_beta2 | late_mean_accuracy | 0.000747619 | 0.00107725 | 3 |
| Full - dp_sgd_momentum | final_accuracy | 0.0057 | 0.0041328 | 3 |
| Full - dp_sgd_momentum | final_test_loss | 0.0137308 | 0.012261 | 3 |
| Full - dp_sgd_momentum | late_mean_accuracy | -0.00104762 | 0.0052001 | 3 |
| Full - dp_kfc_momentum | final_accuracy | -0.0433333 | 0.00542433 | 3 |
| Full - dp_kfc_momentum | final_test_loss | 0.182126 | 0.0332141 | 3 |
| Full - dp_kfc_momentum | late_mean_accuracy | -0.0495 | 0.00627221 | 3 |

## SynDiag factorial effects

Y00=SynDiag, Y10=SynDiag+β1, Y01=SynDiag+β2, Y11=Full. Values are paired within seed; no significance claims.

| Metric | β1 effect mean +/- SD | β2 effect mean +/- SD | Interaction mean +/- SD | N |
| --- | ---: | ---: | ---: | ---: |
| final_accuracy | -0.0001 +/- 0.00175214 | -0.000133333 +/- 0.000550757 | -0.0005 +/- 0.000888819 | 3 |
| final_test_loss | 0.0019337 +/- 0.00756241 | 0.0280982 +/- 0.00488856 | -0.00125541 +/- 0.00371447 | 3 |
| late_mean_accuracy | 4.7619e-06 +/- 0.00183627 | -0.00126667 +/- 0.00124616 | 0.000742857 +/- 0.00076037 | 3 |

## Research questions

- RQ1: Does post-DP first-moment EMA improve SynDiag utility under an SGD-scale learning rate?
- RQ2: Does true-private-time synthetic β2 smoothing improve SynDiag utility?
- RQ3: Do β1 and β2 interact constructively?
- RQ4: How does Full compare with matched DP-SGD-Momentum and DP-KFC-Momentum in utility, geometry, and efficiency?

Results are reported as observed; no outcome is hardcoded or treated as statistically significant.
