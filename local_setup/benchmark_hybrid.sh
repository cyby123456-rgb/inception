#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
: "${MODEL:?Set MODEL to a local Qwen3-8B directory}"
: "${BEFORE:?Set BEFORE to the initial or comparison checkpoint}"
: "${AFTER:?Set AFTER to the jointly trained checkpoint}"
: "${DATA:?Set DATA to the GSM8K question/answer JSONL file}"
GPU=${GPU:-0}
SAMPLES=${SAMPLES:-16}
REPEATS=${REPEATS:-3}
START=${START:-80}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
OUT=${OUT:-$ROOT/runs/hybrid_$(date +%Y%m%d_%H%M%S)_$$}
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u "$ROOT/local_setup/benchmark_inference_matrix.py" \
  --model "$MODEL" --before "$BEFORE" --after "$AFTER" --data "$DATA" \
  --output "$OUT" --start "$START" --samples "$SAMPLES" --repeats "$REPEATS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --merge-target --lower-right --fused-norms --compact --t-graph --gpu-greedy --lookup \
  --variants greedy_lower_fused,before_b3_lower_fused_compact_tgraph_lookup,after_b3_lower_fused_compact_tgraph,after_b3_lower_fused_compact_tgraph_lookup,after_b3_lower_fused_compact_tgraph_lookuponly \
  "$@"
