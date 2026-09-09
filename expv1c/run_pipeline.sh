#!/usr/bin/env bash
# Execute only after pytest succeeds; every stage stops on unexpected failure.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for phase in smoke formal; do
  config=expv1c/configs/smoke.json
  if [[ "$phase" == formal ]]; then config=expv1c/configs/full.json; fi
  run_path="expv1c/runs/${phase}_$(date +%Y%m%d_%H%M%S)"
  mkdir "$run_path"
  printf '%s\n' "$run_path" > "expv1c/runs/active_${phase}_path.txt"
  conda run --no-capture-output -n curve python -m expv1c.train_expv1c \
    --config "$config" --output "$run_path" --run all 2>&1 | tee "$run_path/train.log"
  for stage in validate summarize plot; do
    conda run --no-capture-output -n curve python -m "expv1c.${stage}_expv1c" \
      --config "$config" --runs "$run_path" --output "$run_path" 2>&1 | tee "$run_path/${stage}.log"
  done
  printf '%s\n' "$run_path" > "expv1c/runs/latest_${phase}_path.txt"
  printf 'Completed %s pipeline: %s\n' "$phase" "$run_path"
done
