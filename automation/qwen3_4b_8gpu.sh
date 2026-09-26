#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -lt 1 ]]; then
  echo "Usage: bash automation/qwen3_4b_8gpu.sh /path/to/job.env [--dry-run|--prepare-only]" >&2
  exit 2
fi
CONFIG=$1
shift
source "$CONFIG"
: "${PYTHON:?Set PYTHON in job.env}"
: "${MODEL:?Set MODEL in job.env}"
: "${TRAIN_DATA:?Set TRAIN_DATA in job.env}"
: "${TEST_DATA:?Set TEST_DATA in job.env}"
: "${WORK_ROOT:?Set a persistent WORK_ROOT in job.env}"
MODE=${1:-run}
if [[ $# -gt 1 || ( "$MODE" != run && "$MODE" != --dry-run && "$MODE" != --prepare-only ) ]]; then
  echo "Unknown arguments" >&2
  exit 2
fi
mkdir -p -- "$WORK_ROOT"
WORK_ROOT=$(cd -- "$WORK_ROOT" && pwd)
SUITE=${SUITE:-"$WORK_ROOT/suite"}
DEPENDENCY_DIR=${DEPENDENCY_DIR:-"$WORK_ROOT/incoming"}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
LOCAL_EXPERIMENT=${LOCAL_EXPERIMENT:-no_joint/seed_42}
export HF_HOME=${HF_HOME:-"$WORK_ROOT/cache/huggingface"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$WORK_ROOT/cache/datasets"}
export TMPDIR=${TMPDIR:-"$WORK_ROOT/cache/tmp"}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"$WORK_ROOT/cache/torchinductor"}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"$WORK_ROOT/cache/triton"}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export DISABLE_VERSION_CHECK=1 PYTHONUNBUFFERED=1
mkdir -p -- "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$DEPENDENCY_DIR"
if [[ -n ${DEPENDENCY_SOURCE:-} ]]; then
  command -v ssh >/dev/null
  command -v scp >/dev/null
fi
# Prevent concurrent jobs preparing/running the same remote suite.
exec 9>"$WORK_ROOT/automation.lock"
flock -n 9 || { echo "Another automation task is using WORK_ROOT" >&2; exit 1; }
"$PYTHON" "$ROOT/scripts/verify_package.py"
if [[ ! -f "$SUITE/plan.json" ]]; then
  "$PYTHON" "$ROOT/local_setup/prepare_qwen3_4b_matrix.py" \
    --model "$MODEL" --train-data "$TRAIN_DATA" --test-data "$TEST_DATA" --output "$SUITE"
fi
ARGS=(--suite "$SUITE" --gpus "$GPUS" --exclude "$LOCAL_EXPERIMENT" --dependency-dir "$DEPENDENCY_DIR")
if [[ -n ${DEPENDENCY_SOURCE:-} ]]; then
  ARGS+=(--dependency-source "$DEPENDENCY_SOURCE")
fi
if [[ "$MODE" == --dry-run || "$MODE" == --prepare-only ]]; then
  ARGS+=(--dry-run)
fi
# Foreground execution lets SSH/nohup or the automation platform own job lifetime.
exec "$PYTHON" -u "$SUITE/source_snapshot/local_setup/run_qwen3_4b_workers.py" "${ARGS[@]}"
