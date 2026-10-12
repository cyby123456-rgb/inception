# OneiroDraft: Reading the Latent Dreams for Speculative Decoding

OneiroDraft is a recurrent latent drafter that adapts a small contiguous interval of a pretrained target Transformer's layers into a shared transition across future token positions. A lightweight boundary readout turns the predicted hidden states into draft tokens, and the complete target model verifies the proposals in parallel.

The Hybrid decoder combines recurrent latent drafting with contextual candidate reuse: it uses a continuation found in the committed prompt/history when available, and otherwise falls back to neural drafting.

This README follows the anonymous ICLR 2027 submission *OneiroDraft: Reading the Latent Dreams for Speculative Decoding*. The implementation retains the historical names **LayerLoop** and **RecurFT** in module names, configuration fields, and checkpoint files.

## Highlights

- **Target-derived recurrence:** independently copy a short target-layer interval and adapt it from transformation along model depth to prediction across token positions.
- **Rollout-aware training:** establish the transition and readout separately, then jointly calibrate them on non-detached recurrent trajectories.
- **Verified proposals:** the target model decides which draft tokens to commit; temporary latent predictions are replaced by target-computed states before subsequent neural drafting.
- **Complementary proposal sources:** contextual lookup supplies inexpensive drafts on a hit, while the recurrent transition handles lookup misses.
- **Measured acceleration:** on Qwen3-8B, latent drafting alone achieves **1.20x-1.27x** wall-clock speedup; Hybrid achieves **1.59x-1.70x** wall-clock speedup and **1.60x-1.70x** throughput speedup over paired greedy decoding.

## Method

### Recurrent latent drafting

Let `h_t` be the target-computed hidden state at an intermediate boundary. The copied transition `T` is conditioned on the committed hidden-state history and recursively predicts future boundary states. Readout `B` maps each predicted state to vocabulary logits.

```text
Committed prefix -> complete target model -> target anchor + boundary states
                                                       |
                                                       v
                                  copied transition T -> predicted state
                                                       |
                                                  readout B
                                                       |
                                                   draft token
                                                       |
                         predicted state feeds the next application of T
                                                       |
                          target verifies anchor + candidate block
                                                       |
                           commit accepted prefix and roll back suffix
                                                       |
                               re-anchor with target-computed states
```

The recurrent object is the hidden representation. Neural draft tokens are produced for verification; they are not additional token-conditioning inputs to the transition in the reported configuration.

For Qwen3-4B and Qwen3-8B, the recurrent boundary is `hidden_states[33]`: the output of zero-based decoder layer 32 and the input to layer 33. The transition independently copies layers `33:35`, i.e. layers 33 and 34. Verification still executes all 36 target layers.

The boundary readout applies a low-rank residual adapter before reusing the target's final normalization and vocabulary projection. It avoids executing the remaining target Transformer layers for every draft token.

### Staged initialization and joint calibration

Training uses MetaMathQA. Backbone base weights remain frozen; trainable target and transition adaptation uses LoRA.

1. **One-step transition training:** predict the next target-computed boundary state with an MSE objective; target LoRA and transition LoRA are trainable.
2. **Detached multistep training:** reuse predicted states over short rollouts while detaching reused predictions, exposing `T` to its inference-time input distribution.
3. **Readout initialization:** freeze the adapted target and transition; train `B` on target-computed states using next-token cross-entropy and target-distribution distillation.
4. **Joint recurrent calibration:** keep the adapted target fixed and jointly train transition LoRA and `B` on recurrent rollouts without detaching the predicted states.

The compact joint objective is

```text
L_joint = alpha * L_step + w(s) * (L_state + gamma * L_dist)
```

`L_step` supervises the first predicted state; `L_state` uses smooth-L1 supervision on later states; `L_dist` matches predicted-state readouts to detached target token distributions using `KL(target || draft)`. In the reported frozen-target Qwen3-8B configuration, `alpha = 0.1`, `gamma = 10`, rollout depth is 2, and `w(s)` increases to 0.1 over the first 200 joint updates. After warmup, the effective objective is `0.1 * L_step + 0.1 * L_state + 1.0 * L_dist`.

| Component | Qwen3-4B | Qwen3-8B |
| --- | ---: | ---: |
| Hidden dimension | 2,560 | 4,096 |
| Target LoRA rank | 8 | 8 |
| Transition LoRA rank | 80 | 128 |
| Boundary adapter rank | 160 | 256 |
| Recurrent boundary | `hidden_states[33]` | `hidden_states[33]` |
| Copied layers, zero-based | 33-34 | 33-34 |

The replicated staged-to-joint schedule in the paper is:

| Stage | Updates in this stage | Learning rate | Trainable components |
| --- | ---: | ---: | --- |
| Core transition | 49,199 | `1e-5` | Target LoRA + transition LoRA |
| Detached multistep | 9,840 | `3e-6` | Target LoRA + transition LoRA |
| Readout warmup | 222 | `1e-4` | Boundary readout |
| Readout continuation | 762 | `5e-5` | Boundary readout |
| Joint calibration | 4,000 additional | `3e-6` | Transition LoRA + boundary readout |
| Extended calibration, Qwen3-8B | 20,000 additional in total | `3e-6` | Transition LoRA + boundary readout |

Staged initialization totals **60,023 updates**. Joint calibration reaches **64,023 cumulative updates**, and Extended calibration reaches **80,023**. The 20,000-update branch is the total joint-calibration budget, not an additional 20,000 updates after the 4,000-update branch. Qwen3-4B auxiliary losses and the trainable-target Full-joint ablation are separate variants; see the paper's Appendix B.1 and the [training audit](docs/code_and_training_audit.md).

### Hybrid decoding and verification

At each speculative round, the target provides an anchor token. The Hybrid route searches exact token-ID continuations in the current example's retained prompt and committed generation, trying match lengths 5, 4, 3, and 2. A lookup hit supplies up to eight draft tokens. On a miss, `T` and `B` supply up to two neural draft tokens.

Lookup and neural drafting are alternative sources for a round: a short lookup continuation is not extended with neural proposals. Only the longest consecutively accepted candidate prefix is committed. Rejected tokens are removed from speculative caches and are never added to the committed lookup history. Target boundary states are synchronized before neural drafting resumes.

Strict verification (`lambda = 1.00`) requires each accepted token ID to equal the target argmax from the block-verification forward pass. The paper also studies relaxed target support (`lambda < 1`), defined as `p(candidate) / max_v p(v) >= lambda`.

**Strict verification does not guarantee token-identical serial greedy output in BF16.** Block and single-token forwards can use different floating-point execution paths. Report answer quality and output equality alongside speed. The submission's Appendix D provides the numerical diagnostic; this implementation should not be described as empirically established lossless decoding.

## Results Reported in the Paper

The following results are from Tables 2 and 3 of the submitted manuscript, using Qwen3-8B after **20,000 joint-calibration updates**. Values are means over training seeds **42, 43, and 44**; uncertainties are sample standard deviations.

| Benchmark | Examples | Greedy accuracy (%) | Hybrid accuracy (%) | Hybrid wall-clock speedup | Hybrid throughput speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| MATH-500 | 500 | 66.60 +/- 0.92 | 67.27 +/- 0.12 | 1.63x +/- 0.01 | 1.63x +/- 0.01 |
| AIME 2024 | 30 | 4.44 +/- 1.92 | 3.33 +/- 0.00 | 1.70x +/- 0.05 | 1.70x +/- 0.02 |
| AIME 2025 | 30 | 4.44 +/- 1.92 | 4.44 +/- 1.92 | 1.65x +/- 0.01 | 1.65x +/- 0.01 |
| AIME 2026 | 30 | 1.11 +/- 1.92 | 1.11 +/- 1.92 | 1.64x +/- 0.05 | 1.64x +/- 0.04 |
| AMC 2023 | 40 | 27.50 +/- 4.33 | 29.17 +/- 1.44 | 1.67x +/- 0.04 | 1.66x +/- 0.03 |
| Minerva/OCW | 272 | 25.61 +/- 0.42 | 26.72 +/- 1.39 | 1.59x +/- 0.02 | 1.60x +/- 0.01 |
| OlympiadBench | 675 | 23.51 +/- 0.75 | 23.11 +/- 0.90 | 1.66x +/- 0.02 | 1.66x +/- 0.02 |

Proposal-source ablation, wall-clock speedup over the same paired greedy target:

| Benchmark | Latent-only | Lookup-only | Hybrid |
| --- | ---: | ---: | ---: |
| MATH-500 | 1.27x +/- 0.01 | 1.51x +/- 0.01 | 1.63x +/- 0.01 |
| AIME 2024 | 1.21x +/- 0.04 | 1.64x +/- 0.01 | 1.70x +/- 0.05 |
| AIME 2025 | 1.20x +/- 0.01 | 1.59x +/- 0.02 | 1.65x +/- 0.01 |
| AIME 2026 | 1.22x +/- 0.01 | 1.56x +/- 0.05 | 1.64x +/- 0.05 |
| AMC 2023 | 1.27x +/- 0.01 | 1.56x +/- 0.04 | 1.67x +/- 0.04 |
| Minerva/OCW | 1.20x +/- 0.01 | 1.50x +/- 0.01 | 1.59x +/- 0.02 |
| OlympiadBench | 1.23x +/- 0.01 | 1.56x +/- 0.01 | 1.66x +/- 0.02 |

These controls separate the recurrent drafter's contribution from contextual reuse. Joint calibration improves neural acceptance and Latent-only speed across all seven benchmarks. Hybrid achieves the largest mean speedup in every row, while accuracy varies modestly in both directions rather than improving uniformly.

With a **4,096-token response budget**, Hybrid reaches **2.17x-2.53x** wall-clock speedup on the four benchmarks with complete three-seed coverage: AIME 2024, AIME 2025, AIME 2026, and AMC 2023. This is a response-length sensitivity result, not a complete seven-benchmark long-generation result.

### Main evaluation protocol

- BF16 precision, SDPA attention, batch size 1, one GPU per worker.
- Maximum prompt length 4,096 tokens and maximum response length 1,024 tokens.
- Speculation begins after 32 generated tokens; neural horizon 2, lookup horizon up to 8, excluding the target anchor.
- Strict verification with `lambda = 1.00`.
- Each speculative run is paired with greedy decoding of the **same adapted target checkpoint**, including corresponding target-side inference optimizations.
- Wall-clock speedup is `T_greedy / T_method`; throughput speedup is `(tokens_method / T_method) / (tokens_greedy / T_greedy)`.
- Ratios are computed within each seed before reporting the three-seed mean and sample standard deviation. Repeated timings of one checkpoint are not independent training seeds.

The paper reports a mixed remote-worker setting; accelerator identity was not recorded consistently for every run. Results should not be attributed to a single GPU model.

## Repository Guide

```text
LLaMA-Factory/src/llamafactory/
  model/model_utils/recurft.py       Recurrent transition and boundary modules
  model/adapter.py                   Adapter loading and trainable scope
  train/sft/recurft.py               Transition, readout, and rollout objectives
  hparams/finetuning_args.py          RecurFT configuration fields
LLaMA-Factory/experiments/recurft_math/
  recurft_speculative_generate.py    Speculative decoder and cache control
  recurft_rollout_eval.py            Rollout evaluation
  recurft_block_numerics_audit.py    Block-versus-serial numerical audit
local_setup/                        Joint-training and Hybrid benchmark tools
scripts/                            Installation, staged training, and launchers
configs/reference/                  Historical staged recipes
configs/experimental/               Joint and Qwen3-4B experiment recipes
docs/                               Training audit and experiment guides
reports/                            Historical run and validation reports
provenance/                         Source, environment, asset, and route audits
```

Key guides: [joint training and Hybrid inference](local_setup/README.md), [Qwen3-4B experiment matrix](docs/qwen3_4b_matrix.md), [training/code audit](docs/code_and_training_audit.md), and [eight-GPU automation](docs/qwen3_4b_eight_gpu.md).

## Setup and Assets

Run the following commands from the repository root on **Linux x86_64, Python 3.11, and NVIDIA CUDA**. The recorded environment uses PyTorch **2.4.1+cu121**, Transformers **4.57.1**, and PEFT **0.17.1**.

```bash
bash scripts/install.sh
source .venv/bin/activate
```

The installer creates an isolated virtual environment, installs the pinned environment and bundled LLaMA-Factory, and runs offline CPU checks. Existing environments can use [the dependency lock](requirements-linux-cu121.lock.txt) and [test requirements](requirements-test.txt). Launchers disable upstream dependency-range checks to use the recorded environment; this is not a general compatibility guarantee.

Provide the following local assets:

- A target model directory, such as Qwen3-4B or Qwen3-8B, matching the checkpoint architecture.
- Training data in MetaMathQA-style JSON with `query` and `response` fields.
- Evaluation JSONL with `question` and `answer` fields, using the benchmark's official split and task-appropriate scoring protocol.
- A compatible checkpoint containing target adapters, recurrent weights, boundary readout, tokenizer metadata, and `recurft_config.json`.

Model weights, datasets, and trained checkpoints are supplied separately. Model-only checkpoints support inference or stage initialization; exact training resume additionally requires optimizer, scheduler, and RNG state.

## Run Paired Decoding

The GitHub snapshot provides an executable Hybrid benchmark with paired greedy, Latent-only, Lookup-only, and Hybrid routes:

```bash
MODEL=/path/to/Qwen3-8B \
BEFORE=/path/to/staged_checkpoint \
AFTER=/path/to/joint_checkpoint \
DATA=/path/to/benchmark_test.jsonl \
GPU=0 START=0 SAMPLES=32 REPEATS=2 MAX_NEW_TOKENS=1024 \
OUT="$PWD/runs/hybrid_comparison" \
bash local_setup/benchmark_hybrid.sh

python local_setup/analyze_acceleration_goal.py runs/hybrid_comparison \
  --out runs/hybrid_comparison/analysis
```

`BEFORE` and `AFTER` must contain identical target adapters for a frozen-target continuation comparison. To evaluate one checkpoint, pass the same path to both. Use a new output directory for each experiment.

Outputs include `results.jsonl` with token IDs and per-block proposal sources, `manifest.json` with asset/source hashes, `load_audit.json`, `gpu_telemetry.jsonl`, and a live `summary.json`. Timings include prefill and per-question initialization, while model loading, compilation, and warmup are excluded.

**Snapshot protocol:** this wrapper is an implementation benchmark, not the exact paper sweep. Its underlying matrix fixes the prompt cap at **1,024** and neural speculation start at **0**, whereas the paper uses a prompt cap of **4,096** and speculation start **32**. `START` selects the first dataset row; it does not set the speculation start position. The snapshot also retains historical task-specific answer extraction. Exact paper-table reproduction requires the corresponding final protocol, trained checkpoints, and per-benchmark scorers; the command above does not automatically reproduce Tables 2-5.

For a strict neural-only preflight using the historical decoder wrapper:

```bash
python scripts/run_decode.py \
  --model /path/to/Qwen3-8B \
  --checkpoint /path/to/boundary_checkpoint \
  --data /path/to/benchmark_test.jsonl \
  --output "$PWD/runs/latent_smoke" --gpu 0 \
  --route boundary --samples 2 --max-new-tokens 64 --dry-run
```

Remove `--dry-run` after checking the resolved assets and command. This historical wrapper disables lookup for the neural routes. `boundary` uses the lightweight readout; `tail` uses the remaining target layers; `ngram` is a separate lookup control. These route names should remain explicit in reports.

## Train the Drafter

### Audited Qwen3-4B staged and joint experiments

```bash
python local_setup/prepare_qwen3_4b_matrix.py \
  --model /path/to/Qwen3-4B \
  --train-data /path/to/MetaMathQA-valid.json \
  --test-data /path/to/benchmark_test.jsonl \
  --output /path/to/new_suite

bash /path/to/new_suite/run_all.sh --gpus 0,1,2 --dry-run
bash /path/to/new_suite/run_all.sh --gpus 0,1,2

python /path/to/new_suite/source_snapshot/local_setup/summarize_qwen3_4b_matrix.py \
  --suite /path/to/new_suite
```

The matrix uses seeds 42, 43, and 44 and distinguishes Staged (`no_joint`), frozen-target continuation (`post_joint`), and trainable-target joint (`full_joint`). It prepares a fixed held-out split, stage configurations, source snapshots, and protocol manifests. Verify the generated update budget against the paper schedule; changing the data changes the staged budget. The trainable-target comparison evaluates different complete recipes, not a single-loss ablation. See the [4B guide](docs/qwen3_4b_matrix.md) for gradient scope, checkpoint dependencies, and resume behavior.

### Frozen-target continuation from an existing boundary checkpoint

```bash
python scripts/train.py --stage joint \
  --model /path/to/Qwen3-8B \
  --data /path/to/MetaMathQA-valid.json \
  --checkpoint /path/to/model_only_boundary_checkpoint \
  --source-config /path/to/source_train.yaml \
  --output /path/to/new_joint_stage \
  --template qwen3_nothink --gpu 0 --steps 4000 --dry-run
```

Remove `--dry-run` to launch the new stage. The source configuration must match the checkpoint's layers, adapter ranks, and boundary layout. This launcher changes optimizer groups and therefore requires a model-only source checkpoint; do not pass an old optimizer state as a new stage's resume state. `--steps` is the number of additional updates.

The [from-base 8B preparer](local_setup/README.md) is another experimental recipe: it freezes the original target and trains `T` and `B` from update one. It is distinct from the paper's staged initialization followed by frozen-target calibration. The `configs/reference/` Llama recipes also retain historical layer layouts and update budgets; do not apply them unchanged to Qwen3.

## Checks and Reproducibility

```bash
CUDA_VISIBLE_DEVICES="" python scripts/self_test.py

PYTHONPATH="$PWD/LLaMA-Factory/src:$PWD/LLaMA-Factory/experiments/recurft_math" \
DISABLE_VERSION_CHECK=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
python -m pytest -q tests
```

These checks exercise model loading, rejection/EOS handling, cache rollback, lookup routing, frozen parameter scope, and resume behavior. CPU/tiny-model checks validate implementation behavior; they do not reproduce GPU speed or benchmark accuracy.

The supplied GitHub snapshot contains earlier local recipes and reports rather than the complete final paper sweep. Historical GSM8K/Llama runs, worker timing repeats, and later seven-benchmark paper results use different protocols and must be labeled separately. In particular, the [execution-route audit](provenance/execution_route_audit.json) identifies historical lookup-only results that cannot substantiate neural-drafter acceleration.

For comparable measurements, preserve the target checkpoint, tokenizer, task prompt, output budget, runtime settings, and sample identities. Record generated-token counts, final-answer accuracy, draft-source acceptance, output equality, source/checkpoint hashes, and GPU workload telemetry alongside time and throughput.

## Scope and Limitations

The primary evaluation covers Qwen3 mathematical reasoning. Broader model families, generation domains, adaptive boundaries, and longer neural horizons remain directions for evaluation. Lookup speed depends on reusable continuations, and increasing recurrent capacity or horizon can raise drafting cost or reduce later-step acceptance. The method periodically re-anchors predicted states to limit within-block error accumulation, while final-answer accuracy still requires independent measurement.

The project bundles a modified LLaMA-Factory implementation. Its upstream license and citation are available in [LICENSE](LLaMA-Factory/LICENSE) and [CITATION.cff](LLaMA-Factory/CITATION.cff).
