# Joint training and hybrid inference

These modules extend the existing RecurFT launchers with frozen-target joint
training, exact training-state resume, paired inference measurements, and
prompt/history lookup with a neural T fallback. Model weights and datasets are
supplied separately. Run commands from the repository root with the dependencies
in `requirements-linux-cu121.lock.txt` and `requirements-test.txt` installed.

## Hybrid inference

The hybrid decoder matches the last 2–5 known tokens against the current prompt
and generated history. A match proposes up to eight continuation tokens;
otherwise T and its boundary head propose up to two. The full target verifies
the continuous matching prefix, rolls back rejected KV entries, and supplies the
correction. Deferred T synchronization incorporates the accepted target hidden
states before neural drafting resumes. Reference answers are used only for scoring.

The benchmark includes optimized greedy, T-only, lookup-only, and hybrid
controls. Target LoRA merging, fused RMSNorm and attention optimizations apply
to the corresponding greedy baseline too. T uses a static cache and CUDA Graph.

```bash
MODEL=/path/to/Qwen3-8B \
BEFORE=/path/to/initial_step0 \
AFTER=/path/to/milestones/joint_22000 \
DATA=/path/to/GSM8K/test_official.jsonl \
GPU=0 START=48 SAMPLES=32 REPEATS=2 \
OUT="$PWD/runs/hybrid_check" \
bash local_setup/benchmark_hybrid.sh

python local_setup/analyze_acceleration_goal.py runs/hybrid_check \
  --out runs/hybrid_check/analysis
```

`BEFORE` and `AFTER` must have identical target adapters. To evaluate one
checkpoint without a training comparison, pass that checkpoint to both.
The result directory must be new. `results.jsonl` contains the token IDs and
per-block draft source; `manifest.json` records source/asset hashes;
`gpu_telemetry.jsonl` records concurrent workloads. `summary.json` updates live.

Measurements use BF16, batch size one, thinking disabled, and at most 1,024
generated tokens by default. Timings include prefill and per-question T/cache
initialization; loading, compilation and warmup are excluded. Report both the
total-time ratio and token/s ratio because output lengths can differ.

**Strict block verification does not guarantee token-identical serial greedy
output in BF16.** Block and single-token numerical paths can diverge. Report
full-output equality and answer quality with speed; this implementation has
observed mismatches and is not established as lossless. Repeats of the same
question do not increase the number of independent evaluation questions.

`ACCELERATION_GOAL.md` and `INFERENCE_OPTIMIZATION.md` preserve the earlier
experiment notes. Named local run directories in those notes are historical
references; use the explicit paths above on another machine. Optional branch,
shortlist and whole-draft-graph ablations are included but are not enabled by
the hybrid wrapper. `finalize_hybrid_report.py`, `report_joint_from_base_retest.py`,
`summarize_inference_optimizations.py` and `benchmark_target_forward.py` retain
experiment-specific assumptions; the general report entry point is
`analyze_acceleration_goal.py`.

## Joint T + boundary training from the base model

```bash
python local_setup/prepare_joint_from_base.py \
  --model /path/to/Qwen3-8B \
  --train-data /path/to/MetaMathQA-valid.json \
  --test-data /path/to/GSM8K/test_official.jsonl \
  --output runs/joint_from_base --gpus 0

python runs/joint_from_base/run_pipeline.py
```

Preparation creates an isolated source snapshot, dataset mapping and run plan.
The recipe trains T LoRA and the boundary head from update one while freezing
the target and its adapters. T is initialized from pretrained layers 33–34.
The fixed 49,375-update budget corresponds to one pass through the original
394,996-example dataset at effective batch eight. A different dataset size
does not automatically change that budget. Data use `query`/`response` fields;
evaluation JSONL uses `question`/`answer`.

Training uses BF16, learning rate 3e-6, sequence cap 1,024, and a two-step
rollout-loss warmup over the first 200 updates. Both trainable components are
enabled throughout. The pipeline first tests a two-update checkpoint and
two-update resume, then measures periodically. Its scheduled neural-only
evaluations use FP32 and require equality with greedy; hybrid BF16 inference
is a separate invocation of the command above.

The scheduler waits for an empty selected GPU with at least 45,000 MiB free;
it leaves other processes running. It saves the full optimizer/scheduler budget
across measurement pauses. Optimizer, scheduler and Python/NumPy/Torch RNG state
use safetensors plus JSON with checksums, without pickle loading. Training-state
files are required for resume; model-only milestones are for inference.

The orchestrator handles scheduled pauses, not arbitrary crash recovery.
It refuses to restart over an existing status file. Diagnose failed stages
before resuming them with the source snapshot runner, for example:

```bash
python runs/joint_from_base/source_snapshot/local_setup/run_qwen3_stage.py \
  --config runs/joint_from_base/joint/train.yaml --gpu 0 \
  --resume-from runs/joint_from_base/joint/checkpoints/checkpoint-1000 \
  --stop-after-step 2000
```

`--initialize-from` instead starts a new joint stage from model weights with a
fresh optimizer. It is distinct from `--resume-from` and cannot be combined
with it. A new stage must use a new output directory.

## Checks

```bash
PYTHONPATH="$PWD/LLaMA-Factory/src:$PWD/LLaMA-Factory/experiments/recurft_math" \
DISABLE_VERSION_CHECK=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
python -m pytest -q tests
```

Tests exercise rejection/EOS/cache rollback, lookup-only and mixed execution,
frozen PEFT scope, and an actual tiny Trainer resumed versus uninterrupted run
with identical parameters and data order. CUDA numerical validators run when
the associated optimization is enabled by the inference benchmark.

## Audited 4B training modes

For one local staged experiment plus eight remote GPU workers, see [the Linux/SSH automation guide](../docs/qwen3_4b_eight_gpu.md) and `automation/qwen3_4b_8gpu.sh`. Continuations wait for the exact same-seed checkpoint; the local checkpoint can be published and fetched automatically over SSH.

The legacy preparer above is specific to Qwen3-8B and now rejects 4B configurations. Use [the 4B guide](../docs/qwen3_4b_matrix.md) and [training/code audit](../docs/code_and_training_audit.md) for the three-seed staged, frozen-target continuation, and trainable-target joint experiments. These are distinct training recipes. Audited v2 snapshots also fix the greedy token-cap termination accounting; capped historical timings must not be pooled with corrected timings.
