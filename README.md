# OneiroDraft: Reading the Latent Dreams for Speculative Decoding

OneiroDraft is a recurrent latent drafter that adapts a small contiguous interval of a pretrained target Transformer's layers into a shared transition across future token positions. A lightweight boundary readout turns the predicted hidden states into draft tokens, and the complete target model verifies the proposals in parallel.

The Hybrid decoder combines recurrent latent drafting with contextual candidate reuse: it uses a continuation found in the committed prompt/history when available, and otherwise falls back to neural drafting.

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

For the Qwen3-8B configuration, the recurrent boundary is `hidden_states[33]`: the output of zero-based decoder layer 32 and the input to layer 33. The transition independently copies layers `33:35`, i.e. layers 33 and 34. Verification still executes all 36 target layers.

The boundary readout applies a low-rank residual adapter before reusing the target's final normalization and vocabulary projection. It avoids executing the remaining target Transformer layers for every draft token.

### Staged initialization and joint calibration

Training uses MetaMathQA. Backbone base weights remain frozen; trainable target and transition adaptation uses LoRA.

1. **One-step transition training:** predict the next target-computed boundary state with an MSE objective; target LoRA and transition LoRA are trainable.
2. **Detached multistep training:** reuse predicted states over short rollouts while detaching reused predictions, exposing `T` to its inference-time input distribution.
3. **Readout initialization:** freeze the adapted target and transition; train `B` on target-computed states using next-token cross-entropy and target-distribution distillation.
4. **Joint recurrent calibration:** keep the adapted target fixed and jointly train transition LoRA and `B` on recurrent rollouts without detaching the predicted states.

The joint objective is

```text
L_joint = alpha * L_step + w(s) * (L_state + gamma * L_dist)
```

`L_step` supervises the first predicted state; `L_state` uses smooth-L1 supervision on later states; `L_dist` matches predicted-state readouts to detached target token distributions using `KL(target || draft)`. In the reported frozen-target Qwen3-8B configuration, `alpha = 0.1`, `gamma = 10`, rollout depth is 2, and `w(s)` increases to 0.1 over the first 200 joint updates. After warmup, the effective objective is `0.1 * L_step + 0.1 * L_state + 1.0 * L_dist`.

| Component | Qwen3-8B |
| --- | ---: |
| Hidden dimension | 4,096 |
| Target LoRA rank | 8 |
| Transition LoRA rank | 128 |
| Boundary adapter rank | 256 |
| Recurrent boundary | `hidden_states[33]` |
| Copied layers, zero-based | 33-34 |

The final training schedule uses staged initialization followed by **20,000 frozen-target joint-calibration updates**:

| Stage | Updates in this stage | Learning rate | Trainable components |
| --- | ---: | ---: | --- |
| Core transition | 49,199 | `1e-5` | Target LoRA + transition LoRA |
| Detached multistep | 9,840 | `3e-6` | Target LoRA + transition LoRA |
| Readout warmup | 222 | `1e-4` | Boundary readout |
| Readout continuation | 762 | `5e-5` | Boundary readout |
| Joint calibration | 20,000 | `3e-6` | Transition LoRA + boundary readout |

Staged initialization totals **60,023 updates**; the final checkpoint has **80,023 cumulative updates**. The adapted target remains fixed throughout joint calibration, and all weights are fixed during inference.

### Hybrid decoding and verification

At each speculative round, the target provides an anchor token. The Hybrid route searches exact token-ID continuations in the current example's retained prompt and committed generation, trying match lengths 5, 4, 3, and 2. A lookup hit supplies up to eight draft tokens. On a miss, `T` and `B` supply up to two neural draft tokens.

Lookup and neural drafting are alternative sources for a round: a short lookup continuation is not extended with neural proposals. Only the longest consecutively accepted candidate prefix is committed. Rejected tokens are removed from speculative caches and are never added to the committed lookup history. Target boundary states are synchronized before neural drafting resumes.

The main configuration uses strict verification (`lambda = 1.00`): each accepted token ID must equal the target argmax from the block-verification forward pass. The acceptance rule also supports a relaxed threshold (`lambda < 1`), defined as `p(candidate) / max_v p(v) >= lambda`.

Block and single-token forwards in BF16 can use different floating-point execution paths. Strict matching to block logits therefore requires answer-quality and output-equality measurements alongside speed; token-identical serial greedy output is not guaranteed.

## Results

The final Qwen3-8B configuration uses **20,000 joint-calibration updates** and strict Hybrid decoding. Values are means over training seeds **42, 43, and 44**; uncertainties are sample standard deviations.

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

The proposal-source controls measure latent drafting, contextual reuse, and their combination under the same paired greedy target. Hybrid achieves the largest mean speedup on all seven benchmarks. Accuracy changes in both directions across tasks.

With a **4,096-token response budget**, Hybrid reaches **2.17x-2.53x** wall-clock speedup on AIME 2024, AIME 2025, AIME 2026, and AMC 2023, each evaluated with complete three-seed coverage.

### Main evaluation protocol

- BF16 precision, SDPA attention, batch size 1, one GPU per worker.
- Maximum prompt length 4,096 tokens and maximum response length 1,024 tokens.
- Speculation begins after 32 generated tokens; neural horizon 2, lookup horizon up to 8, excluding the target anchor.
- Strict verification with `lambda = 1.00`.
- Each speculative run is paired with greedy decoding of the **same adapted target checkpoint**, including corresponding target-side inference optimizations.
- Wall-clock speedup is `T_greedy / T_method`; throughput speedup is `(tokens_method / T_method) / (tokens_greedy / T_greedy)`.
- Ratios are computed within each seed before reporting the three-seed mean and sample standard deviation. Repeated timings of one checkpoint are not independent training seeds.

Evaluation uses remote GPU workers. Accelerator identity was not recorded consistently across all runs, so the reported results are not tied to a single GPU model.

## Implementation

| File | Role |
| --- | --- |
| [Recurrent modules](LLaMA-Factory/src/llamafactory/model/model_utils/recurft.py) | Copied transition, LoRA adaptation, boundary readout, and recurrent cache |
| [Adapter configuration](LLaMA-Factory/src/llamafactory/model/adapter.py) | Adapter loading and trainable parameter scope |
| [Training objectives](LLaMA-Factory/src/llamafactory/train/sft/recurft.py) | One-step, multistep, readout, and joint-calibration losses |
| [Configuration fields](LLaMA-Factory/src/llamafactory/hparams/finetuning_args.py) | Transition, readout, and rollout settings |
| [Decoder](LLaMA-Factory/experiments/recurft_math/recurft_speculative_generate.py) | Candidate generation, target verification, and cache control |
| [Numerical audit](LLaMA-Factory/experiments/recurft_math/recurft_block_numerics_audit.py) | Block-versus-serial forward comparison |
| [Joint recipe](configs/experimental/recurft_joint_boundary.yaml) | Frozen-target joint-calibration settings |

Efficient execution retains the target KV for committed candidate prefixes, defers corrective-token processing to the next block, and synchronizes recurrent state before neural drafting is needed. Batched readout and merged adapters reduce proposal overhead. Corresponding target-side optimizations are also applied to the paired greedy baseline.

## Setup and Assets

Run the following commands from the repository root on **Linux x86_64, Python 3.11, and NVIDIA CUDA**. The pinned environment uses PyTorch **2.4.1+cu121**, Transformers **4.57.1**, and PEFT **0.17.1**.

```bash
bash scripts/install.sh
source .venv/bin/activate
```

The installer creates an isolated virtual environment, installs the pinned dependencies and bundled LLaMA-Factory, and runs offline CPU checks. Dependency versions are specified in [the environment lock](requirements-linux-cu121.lock.txt), with checks listed in [test requirements](requirements-test.txt).

Provide the following local assets:

- A Qwen3-8B target model directory matching the checkpoint architecture.
- Training data in MetaMathQA-style JSON with `query` and `response` fields.
- Evaluation JSONL with `question` and `answer` fields, using the benchmark's official split and task-appropriate scoring protocol.
- A compatible checkpoint containing target adapters, recurrent weights, boundary readout, tokenizer metadata, and `recurft_config.json`.

Model weights, datasets, and trained checkpoints are supplied separately. Model-only checkpoints support inference or stage initialization; exact training resume additionally requires optimizer, scheduler, and RNG state.

## Joint Calibration

Initialize from the staged boundary checkpoint and run **20,000 additional updates** with the target frozen:

```bash
python scripts/train.py --stage joint \
  --model /path/to/Qwen3-8B \
  --data /path/to/MetaMathQA-valid.json \
  --checkpoint /path/to/model_only_boundary_checkpoint \
  --source-config /path/to/source_train.yaml \
  --output /path/to/new_joint_stage \
  --template qwen3_nothink --gpu 0 --steps 20000 --dry-run
```

Remove `--dry-run` to launch the new stage. The source configuration must match the checkpoint's layers, adapter ranks, and boundary layout. Joint calibration uses a new optimizer parameter group and requires a model-only source checkpoint. `--steps` specifies the number of additional optimizer updates.

## Checks and Reproducibility

```bash
CUDA_VISIBLE_DEVICES="" python scripts/self_test.py

PYTHONPATH="$PWD/LLaMA-Factory/src:$PWD/LLaMA-Factory/experiments/recurft_math" \
DISABLE_VERSION_CHECK=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
python -m pytest -q tests
```

These checks exercise model loading, rejection/EOS handling, cache rollback, lookup routing, frozen parameter scope, and resume behavior. CPU/tiny-model checks validate implementation behavior; they do not reproduce GPU speed or benchmark accuracy.

For comparable measurements, preserve the target checkpoint, tokenizer, task prompt, output budget, runtime settings, and sample identities. Record generated-token counts, final-answer accuracy, draft-source acceptance, output equality, source/checkpoint hashes, and GPU workload telemetry alongside time and throughput.

## Scope and Limitations

The primary evaluation covers Qwen3 mathematical reasoning. Broader model families, generation domains, adaptive boundaries, and longer neural horizons remain directions for evaluation. Lookup speed depends on reusable continuations, and increasing recurrent capacity or horizon can raise drafting cost or reduce later-step acceptance. The method periodically re-anchors predicted states to limit within-block error accumulation, while final-answer accuracy still requires independent measurement.

The project bundles a modified LLaMA-Factory implementation. Its upstream license and citation are available in [LICENSE](LLaMA-Factory/LICENSE) and [CITATION.cff](LLaMA-Factory/CITATION.cff).
