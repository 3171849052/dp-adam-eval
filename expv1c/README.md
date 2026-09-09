# ExpV1c — Upper-Bound Learning-Rate Sweep for DP-Fisher-Wiener

ExpV1b's seed42 sweep improved final accuracy and accuracy AUC throughout
LR 0.10–0.80. Its pinned `formal_20260909_003544` Fisher .80 result was
90.06% final accuracy and 0.855779906542056 AUC. This experiment asks whether
utility keeps improving above .80, or whether an optimization/stability
boundary appears. The only scientific variable is global SGD learning rate.

The five canonical runs are `dp_sgd_lr0p10`, `dp_fisher_wiener_lr0p80`,
`dp_fisher_wiener_lr1p00`, `dp_fisher_wiener_lr1p20`, and
`dp_fisher_wiener_lr1p50`. The Fisher grid is exactly [.8, 1., 1.2, 1.5]; it
never expands automatically, including when 1.5 is best.

## Frozen protocol and shared implementation

Seed42, MNIST, SimpleCNN, five epochs, batch256, epsilon1, delta1e-5,
clip norm1, beta1, SGD momentum0, K50, M_syn2560, eval interval100, four
threads, device auto, full datasets: 1170 private steps and 24 refreshes.
Noise multiplier is 1.068115234375; completed epsilon is
0.995693195331761. Accounting inherits fixed shuffle/drop_last and RDP
q=B/N from ExpV1b. LR never enters the accountant.

Fisher math and research diagnostics remain read-only imports from
`expv1.fisher_wiener` and `expv1.metrics`. The private update, optimizer,
evaluation, divergence exception, summary calculations, and CSV schemas are
shared with `expv1b.train_expv1b`. Only experiment-loop glue is adapted from
that module so config, output routing, provenance, and T90 belong to ExpV1c.
ExpV1b's generic validation primitives and I/O helpers are also shared; its
fixed run_specs and check_config are not used for ExpV1c runs.

The unchanged operation order is backward → global per-sample clipping →
DP Gaussian noise → Fisher-Wiener → SGD. With r=(sigma*C/B)^2 and
F=A⊗G, H[j,i]=lambda_G[j]*lambda_A[i]/(lambda_G[j]*lambda_A[i]+r), and
Y_hat=Q_G@(H*(Q_G.T@Y@Q_A))@Q_A.T. No KFC preconditioning, private
calibration/Fisher, layer-wise LR, momentum, scheduler, adaptive scaling,
normalization, update matching, or extra private steps are introduced.

RNG streams are init=42, loader=43, test=44, synthetic=45, noise=46.
Private batches/noise and synthetic samples/labels must match across runs.
Fisher values can differ after model trajectories diverge with LR.

## Diagnostics, divergence, and validation

Research diagnostics never control optimization, RNG, privacy, LR, refresh,
or termination. `diagnostics_finite` reflects global and layer diagnostics;
summary fields record all_finite, nonfinite_count, and nonfinite_steps.
Correctly flagged research NaN/Inf is valid in a completed artifact. Only
finite-flagged steps undergo research formula/nonnegativity assertions.
Summaries exclude diagnostic-invalid steps and explicitly mask nonfinite
values; diagnostic masks never remove utility observations.

Algorithmic nonfinite loss, noisy gradient, filtered gradient, or parameters
still causes divergence, with no rescue or LR changes. Completed steps must
be diverged_step or diverged_step+1. Refresh occurs before the attempted
private step; divergent refresh schedules include diverged_step when it is
a refresh. Pairing compares strict common executed prefixes for divergent
runs. The remaining canonical runs continue.

Formal validation requires the fixed read-only reference
`expv1b/runs/formal_20260909_003544`. Both anchors compare every private step,
batch index, noise RNG state, and model hash; Fisher .80 additionally compares
every synthetic sample/label and RNG hash. Final anchors:

- DP-SGD .10: `9e9c9e76e84680f6fcb1c34ded5047f73f36b54c8616fb14293a470270fce1b2`
- Fisher .80: `38c1dfe4ca6ad65b49598e8a327edaf81f72873ff5a1df28f5884d34028a1b33`

New LRs have no expected model hash. All artifacts, caches, test evidence,
figures and code stay in expv1c. Provenance includes all upstream Python,
ExpV1 dependencies, ExpV1b Python dependencies (including validator and
summary helpers), and ExpV1c Python. Formal upstream must be clean commit
`eb31b9aeb2280642684f4cedfa65cc02b76c76cd`. Explicit required pytest node IDs
and matching source provenance are required by artifact validation.

## Metrics and interpretation

Primary metrics are final accuracy, accuracy AUC, late mean accuracy and
T90. T50/T70/T80/T85/T90 are the first evaluation meeting the threshold,
reported as completed step=CSV step+1, without interpolation; unreached is
null. CE test loss remains a separate outcome: ExpV1b .80 exceeded baseline
accuracy while its CE was worse (0.63635 versus 0.48026). Higher accuracy
alone does not establish comprehensive superiority.

Rankings include only completed runs with a finite metric; CE loss ranks
ascending. Diverged runs appear separately in `stability.csv`. Summary flags
use AUC: `boundary_best` means the best completed tested AUC is at 1.5;
`peak_bracketed` means a higher completed LR has strictly worse AUC than the
best. Ties resolve to the smaller LR and do not alone bracket a peak.
Divergence sets `stability_boundary_observed`; it does not establish a true
optimum. If 1.5 remains best, peak_bracketed=false and no new LR is run.

The AUC winner is a post-hoc descriptive candidate, reported together with
accuracy and CE loss. This is a single-seed boundary search, not a
confirmatory experiment: no significance tests, p-values, confidence
intervals, or statistical superiority claims. A later independent multi-seed
experiment would need a fixed candidate LR.

Formal `combined_lr_history.csv` reads the pinned ExpV1b summary at
.10/.15/.20/.30/.50, verifies the .80 anchor, and uses ExpV1c at
.80/1.00/1.20/1.50, with source labels. Smoke never mixes tiny utility with
formal history. Seven PNG/PDF figure families show upper sweep, history
(smoke placeholder), accuracy trajectory, train/test losses, threshold
steps, late effective signal LR by layer, and late mechanisms. Runtime,
diagnostic seconds, core training runtime, refresh/filter time, state bytes,
and peak CUDA memory retain ExpV1b semantics.

## Execution

The user's latest instruction authorizes the complete formal sweep after
successful tests and smoke. Smoke has 16/32 examples, batch4, one epoch,
K2, M_syn8 and four steps, still with all five runs. It cannot tune the grid.

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m pytest expv1c/tests -q -s

# Use smoke.json and a fresh smoke_<timestamp> directory first.
EXPV1C_FORMAL_RUNS=expv1c/runs/formal_$(date +%Y%m%d_%H%M%S)
PYTHONDONTWRITEBYTECODE=1 conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m expv1c.train_expv1c --config expv1c/configs/full.json \
  --output "$EXPV1C_FORMAL_RUNS" --run all
for stage in validate summarize plot; do
  PYTHONDONTWRITEBYTECODE=1 conda run -n curve python -m expv1c.${stage}_expv1c \
    --config expv1c/configs/full.json --runs "$EXPV1C_FORMAL_RUNS" --output "$EXPV1C_FORMAL_RUNS"
done
```
