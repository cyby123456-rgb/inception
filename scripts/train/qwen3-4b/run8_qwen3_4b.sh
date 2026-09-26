#!/bin/bash
# 自动化任务直接运行此文件；需要 8 卡，所有路径使用当前共享盘。
# 本机保留 no_joint/seed_42；本脚本只运行另外 8 个实验。
set -euo pipefail

export PATH="/root/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/llmshared-ssd-hd/wangruitao/conda-envs/inception

BASE_DIR="/mnt/llmshared-ssd-hd/wangruitao/inception-qwen3-4b"
PYTHON="/mnt/llmshared-ssd-hd/wangruitao/conda-envs/inception/bin/python"
MODEL="/mnt/llmshared-ssd-hd/wangruitao/models/Qwen--Qwen3-4B/snapshots/master"
TRAIN_DATA="/mnt/llmshared-ssd-hd/wangruitao/datasets/MetaMathQA/MetaMathQA-valid.json"
TEST_DATA="/mnt/llmshared-ssd-hd/wangruitao/datasets/GSM8K/test_official.jsonl"
SUITE="$BASE_DIR/runs/qwen3_4b_remote8_20260918"
DEPENDENCY_DIR="$BASE_DIR/runs/qwen3_4b_matrix_20260918_v2_audited/dependency_exports"
MODE=${1:-run}
if [[ $# -gt 1 || ( "$MODE" != run && "$MODE" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi

cd "$BASE_DIR"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HOME="$BASE_DIR/runs/remote8_cache/huggingface"
export HF_DATASETS_CACHE="$BASE_DIR/runs/remote8_cache/datasets"
export TMPDIR="$BASE_DIR/runs/remote8_cache/tmp"
export TORCHINDUCTOR_CACHE_DIR="$BASE_DIR/runs/remote8_cache/torchinductor"
export TRITON_CACHE_DIR="$BASE_DIR/runs/remote8_cache/triton"
mkdir -p "$BASE_DIR/logs" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" \
    "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$DEPENDENCY_DIR"
LOG_DIR="$BASE_DIR/logs/qwen3_4b_remote8_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOG_DIR"

# 不重复启动同一批实验；日志、checkpoint 保留在共享盘。
exec 9>"$BASE_DIR/runs/qwen3_4b_remote8_20260918.launch.lock"
flock -n 9 || { echo "This eight-experiment job is already running" >&2; exit 1; }
"$PYTHON" "$BASE_DIR/scripts/verify_package.py"
if [[ ! -f "$SUITE/plan.json" ]]; then
    "$PYTHON" "$BASE_DIR/local_setup/prepare_qwen3_4b_matrix.py" \
        --model "$MODEL" --train-data "$TRAIN_DATA" --test-data "$TEST_DATA" --output "$SUITE"
fi
RUNNER="$SUITE/source_snapshot/local_setup/run_qwen3_4b_workers.py"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ ${#GPU_IDS[@]} -ne 8 ]]; then
    echo "This job needs exactly eight allocated GPUs" >&2
    exit 1
fi
"$PYTHON" "$RUNNER" --suite "$SUITE" --exclude no_joint/seed_42 \
    --gpus "$(IFS=,; echo "${GPU_IDS[*]}")" --dependency-dir "$DEPENDENCY_DIR" --dry-run
if [[ "$MODE" != --dry-run ]]; then
    "$PYTHON" -c 'import torch; assert torch.cuda.device_count() >= 8, "Submit this script to an eight-GPU task"'
fi

PIDS=()
NAMES=()
cleanup() {
    trap - INT TERM
    for pid in "${PIDS[@]}"; do
        # 每个 Python 协调器会停止它自己的训练/测速子进程。
        if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

launch() {
    local gpu="$1" experiment="$2"
    local log="$LOG_DIR/${experiment//\//_}.log"
    echo "Launching $experiment on GPU $gpu -> $log"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        args=(--suite "$SUITE" --only "$experiment" --gpus "$gpu")
        if [[ "$experiment" == post_joint/seed_42 ]]; then
            args+=(--external-dependency no_joint/seed_42 --dependency-dir "$DEPENDENCY_DIR")
        fi
        if [[ "$MODE" == --dry-run ]]; then args+=(--dry-run); fi
        exec "$PYTHON" -u "$RUNNER" "${args[@]}"
    ) >"$log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$experiment")
}

launch "${GPU_IDS[0]}" post_joint/seed_42  # 等本机 no_joint seed 42 的 checkpoint
launch "${GPU_IDS[1]}" full_joint/seed_42
launch "${GPU_IDS[2]}" no_joint/seed_43
launch "${GPU_IDS[3]}" post_joint/seed_43  # 等本任务 no_joint seed 43
launch "${GPU_IDS[4]}" full_joint/seed_43
launch "${GPU_IDS[5]}" no_joint/seed_44
launch "${GPU_IDS[6]}" post_joint/seed_44  # 等本任务 no_joint seed 44
launch "${GPU_IDS[7]}" full_joint/seed_44

RESULT=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "Finished: ${NAMES[$index]}"
    else
        code=$?
        echo "Failed: ${NAMES[$index]} (exit $code); inspect $LOG_DIR" >&2
        RESULT=1
    fi
done
if [[ "$MODE" == --dry-run ]]; then
    echo "Eight experiment dry-run finished; no GPU training was started."
elif [[ "$RESULT" == 0 ]]; then
    echo "All eight experiments finished. Logs: $LOG_DIR"
fi
exit "$RESULT"
