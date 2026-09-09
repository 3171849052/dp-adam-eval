#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -eq 1 && ( $1 == "--help" || $1 == "-h" ) ]]; then
  echo "usage: $0 [--config CONFIG]"
  echo "       $0 [CONFIG]"
  exit 0
elif [[ $# -eq 2 && $1 == "--config" ]]; then
  CONFIG="$2"
elif [[ $# -eq 1 && $1 != "--config" ]]; then
  CONFIG="$1"
else
  echo "usage: $0 [--config CONFIG]" >&2
  echo "       $0 [CONFIG]" >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "config file does not exist: $CONFIG" >&2
  exit 1
fi
CONFIG="$(realpath "$CONFIG")"

if command -v tmux >/dev/null 2>&1; then
  HAS_TMUX=1
else
  HAS_TMUX=0
  echo "tmux is unavailable; running training in the foreground" >&2
fi

GPU="$(cd "$ROOT" && env -u CUDA_VISIBLE_DEVICES "$PYTHON" -u scripts/train.py --config "$CONFIG" --print-gpu)"
if ! (cd "$ROOT" && env -u CUDA_VISIBLE_DEVICES "$PYTHON" -u scripts/train.py --config "$CONFIG" --validate-gpu >/dev/null); then
  echo "GPU validation failed; refusing to start training" >&2
  exit 1
fi

RUN_DIR="$(cd "$ROOT" && CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u scripts/train.py --config "$CONFIG" --prepare-run)"
RUN_DIR="$(realpath "$RUN_DIR")"
TRAIN_LOG="$RUN_DIR/train.log"

if [[ "$HAS_TMUX" -eq 0 ]]; then
  echo "physical GPU: $GPU"
  echo "run directory: $RUN_DIR"
  echo "log: $TRAIN_LOG"
  set -o pipefail
  (cd "$ROOT" && CUDA_VISIBLE_DEVICES="$GPU" PYTHONFAULTHANDLER=1 "$PYTHON" -u scripts/train.py \
    --config "$CONFIG" --run-dir "$RUN_DIR" 2>&1 | tee -a "$TRAIN_LOG")
  exit $?
fi

SESSION="$(cd "$ROOT" && "$PYTHON" -u scripts/train.py --tmux-session-name "$RUN_DIR")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

printf -v COMMAND \
  'cd %q && export CUDA_VISIBLE_DEVICES=%q && export PYTHONFAULTHANDLER=1 && set -o pipefail && %q -u %q --config %q --run-dir %q 2>&1 | tee -a %q' \
  "$ROOT" "$GPU" "$PYTHON" "$ROOT/scripts/train.py" "$CONFIG" "$RUN_DIR" "$TRAIN_LOG"
tmux new-session -d -s "$SESSION" "$COMMAND"

echo "physical GPU: $GPU"
echo "tmux session: $SESSION"
echo "run directory: $RUN_DIR"
echo "log: $TRAIN_LOG"
echo "attach: tmux attach -t $SESSION"
echo "tail: tail -f $TRAIN_LOG"
echo "kill: tmux kill-session -t $SESSION"
