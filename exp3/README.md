# Exp3: paired DP-SGD / SynDiag / DP-KFC Pink matched

All commands run from the repository root, in `curve`. No tests launch formal
experiments. Existing run directories are never overwritten.

```bash
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest exp3/tests -q -s
EXP3_FORMAL_RUNS=exp3/runs/formal_YYYYMMDD
conda run -n curve env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m exp3.train_exp3 --config exp3/configs/full.json --output "$EXP3_FORMAL_RUNS" --method all
conda run -n curve python -m exp3.validate_exp3 --config exp3/configs/full.json --runs "$EXP3_FORMAL_RUNS" --output "$EXP3_FORMAL_RUNS"
conda run -n curve python -m exp3.summarize_exp3 --config exp3/configs/full.json --runs "$EXP3_FORMAL_RUNS" --output "$EXP3_FORMAL_RUNS"
conda run -n curve python -m exp3.plot_exp3 --config exp3/configs/full.json --runs "$EXP3_FORMAL_RUNS" --output "$EXP3_FORMAL_RUNS"
```

Formal mode is pinned to public DP-KFC commit
`eb31b9aeb2280642684f4cedfa65cc02b76c76cd`.
Formal training, validation, summary, and plotting reject both dirty upstream
provenance and any other commit. The current `../DP-KFC` checkout is clean and
at that exact pin. Smoke retains its documented ability to audit a dirty
checkout and records the provenance.

The full command covers seeds 42/7/91 × all three methods sequentially, each
1170 steps. For one run add `--seed 42 --method dp_kfc` (alias
`dp_kfc_pink_matched`). Use `configs/smoke.json` and a **fresh** output directory
for smoke, passing that directory as `--runs` and `--output` to postprocessing.
Smoke has 4 steps and refreshes at 0/2, so no oracle/refresh point is strictly
late (`step > 2`). Late geometry summaries and single-seed sample SD are N/A.
Plots still display the diagnostic points. Never substitute early points.

## Upstream audit and exact KFAC conventions

`upstream_audit.json` records the read-only audit; rerun with
`conda run -n curve python -m exp3.audit_upstream` when intentionally re-auditing.
The audited local HEAD and public remote HEAD are both
`eb31b9aeb2280642684f4cedfa65cc02b76c76cd`, remote
`https://github.com/molinamarcvdb/DP-KFC.git`. The worktree is clean with no
untracked files, and all required dependency files match that commit.

AST comparison against public HEAD finds the Pink generator, KFAC recorder,
covariance definitions/aggregation/inverse roots, and per-sample transform
unchanged. Covariance adds an unused EMA helper. Optimizer/privacy add logical
batch accumulation; trainer has batching/training changes. Exp3 calls the
single-batch upstream `clip_and_noise_gradients` API, available in both public
HEAD and this checkout, rather than depending on the local-only accumulator
class. Its current implementation delegates to that accumulator. No upstream
Trainer optimizer creation is used; only its unchanged `set_seed` is imported.

Every run stores upstream repo/commit/dirty/remote/status at metadata top level
and in provenance, plus SHA256 for **all** upstream package Python files and
Exp3 Python files, covering all direct/transitive local dependencies including
models, data, optimizer, privacy, covariance, recorder, precondition, trainer,
and types. Different config, seed labels, or provenance cannot be aggregated.

KFAC construction is **10 × 256** equal synthetic microbatches. Every batch
uses default **MEAN cross entropy** before directly calling upstream
`compute_covariances`, with its separate default ridge **eps=1e-5**. Upstream
`accumulate_covariances` averages the 10 CovariancePairs, then upstream
`compute_inverse_sqrt(..., damping=1e-3)` constructs the active inverse roots.
No manual scaling, normalization, EMA, or replacement mathematics is used.
The old SUM-loss construction incorrectly multiplied the data-dependent G
factor by B² (excluding its ridge). Tests explicitly reject that implementation.
Opacus remains configured with loss_reduction="sum" like upstream; construction
uses recorder backprops, not the synthetic grad_sample tensors produced by
that wrapper. Private training and raw-gradient diagnostics still use SUM loss
to obtain raw per-example gradients.

SynDiag is unchanged: `v=mean_i(g_i²)` in float64, and active FP32
`P=1/(sqrt(v)+1e-3)`. It uses exactly the same 2560 construction samples/labels
as KFAC, with its own raw per-example definition. K=50 and M_syn=2560 are fixed.
All three methods use SGD(lr=.1,momentum=0). DP-SGD uses identity. The sequence is
raw per-example gradients, upstream KFAC transformation (or diagonal/identity),
global clipping, aggregation plus Gaussian noise, then SGD. Conv/Linear reshape
and bias augmentation remain upstream. The upstream stabilized coefficient
`min(1,C/(norm+1e-6))`, RDP accountant/calibration, and shuffle/drop_last
sample-rate convention are unchanged; sampling is not Poisson.

The deliberate Exp3 experimental controls remain matched refresh/budget,
paired RNG streams, fixed SGD, and diagnostics. This is not a reproduction of
the upstream default Trainer schedule or optimizer settings. KFAC construction
and transformation directly follow the pinned public checkout; its relevant
KFAC functions match public HEAD byte-for-byte at the pinned commit.

## Independent probes and old/new oracle

Initialization/loader/test/construction/noise streams use seed, seed+1,
seed+2, seed+3, seed+4. Independent staleness probes use **seed+100003**;
oracle indices always use **314159**, shared across all experimental seeds.
Construction and staleness save before/after RNG states as hashes, sample
hashes, label hashes, counts and step. Validation requires both synthetic audit
sequences to match between SynDiag and KFAC while remaining independent of
each other. Private batch/noise audits and per-step model hashes are retained.

At each refresh, new P is constructed from M_syn=2560 and frozen before any
private batch access. Staleness uses **fresh independent M_stale=512** Pink
samples/uniform labels, never the fitting samples. Old and new P act on exactly
the same current probe gradients; deltas are old spread minus new spread.
Private M_oracle=512 uses a separate diagnostic model to compare raw/old/new
on fixed private samples. Oracle outputs never enter training or decisions.
Tests compare all private-step model/RNG audit hashes with oracle on and off
for all three methods. Diagnostic-only probe-budget changes are also isolated.
These private diagnostic files are not DP releases.

Oracle CSV records R_diag_new/old, R_full_new/old, A_diag_raw/new/old,
S_full_raw/new/old, ranks, tolerances and private_delta_stale_diag/full.
Legacy R_diag/R_full and pre columns alias new. At step zero **all old/delta
fields are empty**, including DP-SGD; later DP-SGD old/new are identity with R=1.
DP-SGD has no refresh/stale probe rows. Summary and validation use this schema.

Disk-backed FP32 gradient matrices, parameter chunking, FP64 Gram accumulation
and eigensolver, PSD checking and numerical-zero filtering are preserved.
Only 512 samples enter oracle or full-spectrum staleness, not 2560. Eigenvalues
must exceed `max(M,d)*float64_eps*max_eigenvalue`. Degenerate undefined spreads
fail explicitly instead of fabricating ratios. No d×d Fisher is constructed;
temporary arrays are removed on success/failure.

## Cost and summaries

`total_refresh_time` includes construction samples/audit, factor estimation and
inverse roots; it excludes stale probes/oracle. `wall_time` covers the loop,
evaluation, audits and diagnostics, excluding setup/data loading/final file
serialization. `diagnostic_seconds` includes probe generation, full geometry,
oracle and diagnostic cleanup. **core_wall_time = wall_time - diagnostic_seconds**
is the primary algorithm runtime comparison; it still includes common
evaluation, training audits, and peak-tracker boundary overhead.

**peak_cuda_memory_core** is allocated CUDA bytes: maxima accumulated only over
training/evaluation/audit and preconditioner-construction segments. Before a
diagnostic segment, the core peak is captured and CUDA peak stats reset. After
diagnostic tensors and model/hook cycles are released, the overall diagnostic
peak is captured and stats reset again before returning to core. Thus the Gram
and eigensolver peaks never enter core memory. **peak_cuda_memory_overall** is
the maximum across all segments. Legacy allocated/reserved fields report
overall peaks; reserved is not presented as core memory because allocator
caches can survive diagnostics. All CUDA metrics are zero on CPU.

State bytes count actual active diagonal P or KFAC **inv_A+inv_G**, excluding
discarded covariance temporaries. DP-SGD refresh count/time/state bytes are zero.

Train metrics get late median/Q25/Q75/IQR per seed. Oracle new/old metrics first
get a late median per layer per seed, then four-layer geometric means:
G_diag_new, G_full_new, G_diag_old, G_full_old. Legacy G aliases new.
Synthetic delta_stale_diag/full get late per-layer medians. Cross-seed tables
retain raw values and report mean ± sample SD (ddof=1), plus three paired
contrasts; no p-values or significance claims. Final test loss is in run JSON
and all three summary tables. Accuracy remains secondary, evaluated every 100
steps plus final; late accuracy averages late evaluation points.

Matplotlib writes 16 figures: accuracy; four fresh/stale G curves; five late
training metrics; two synthetic staleness plots; refresh time; core wall time;
active state bytes; core peak CUDA memory. Seed traces are faint with means.
