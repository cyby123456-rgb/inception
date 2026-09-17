#!/usr/bin/env python3
"""Greedy or quality-tolerant decoding with RecurFT latent drafts.

T proposes future hidden states and tokens. The target model verifies every
draft block in one forward pass. Drafts can either be corrected at the first
target mismatch or committed as a whole when final-task quality matters more
than reproducing token-wise greedy decoding.
"""

from __future__ import annotations

# Isolated runtime: original ongoing evaluators keep their original source.
from adaptive_two_stage_policy import recent_budget
from collections import deque

import argparse
import inspect
import json
import math
import re
import time
from bisect import bisect_right
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from llamafactory.model.model_utils.recurft import RECURFT_CONFIG_NAME, find_decoder_layers
from recurft_ngram_tree import (
    DraftTree,
    build_draft_tree,
    target_guided_leaf_path,
    tree_attention_mask,
    tree_branch_budget,
    tree_position_ids,
)
from recurft_verification_policy import agreement_gate_decision
from recurft_rollout_eval import (
    build_recurrent_module,
    get_base_causal_lm,
    get_decoder,
    maybe_position_embeddings,
    torch_dtype,
)


_SYNC_COMPONENT_TIMING = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--answer-metric",
        choices=["math", "rouge_l"],
        default="math",
        help="Task-quality metric. math preserves the existing extracted-answer exact match.",
    )
    parser.add_argument(
        "--target-lora-removal",
        choices=["none", "pre", "post"],
        default="none",
        help=(
            "Inference-time diagnostic that sets the selected target-path LoRA scaling to zero. "
            "`pre` removes layers before the recurrent interface; `post` removes the interface "
            "and downstream projection layers. T is unchanged. This is not a retrained ablation."
        ),
    )
    parser.add_argument(
        "--merge-target-lora",
        action="store_true",
        help=(
            "Merge the target-path PEFT adapter into the base weights before decoding. "
            "The separately loaded recurrent T weights are unchanged."
        ),
    )
    parser.add_argument(
        "--merge-recurrent-lora",
        action="store_true",
        help=(
            "Merge the custom recurrent T LoRA into T's private base weights after all "
            "recurrent/verifier weights are loaded. Verifier LoRA remains separate."
        ),
    )
    parser.add_argument(
        "--target-lora-dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help=(
            "Optional runtime dtype for target-path LoRA A/B weights. `auto` preserves "
            "the checkpoint dtype; this does not affect recurrent T weights."
        ),
    )
    parser.add_argument("--mode", choices=["heuristic", "verifier", "fixed", "schedule"], default="heuristic")
    parser.add_argument(
        "--draft-commit-policy",
        choices=[
            "target_match",
            "target_match_expand",
            "draft_margin",
            "whole_block",
            "category_block",
            "refine_block",
        ],
        default="target_match",
        help=(
            "How verified draft blocks are committed. category_block ignores target-token match "
            "and accepts only the contiguous prefix allowed by --draft-token-category-gate."
        ),
    )
    parser.add_argument(
        "--latent-draft-commit-policy",
        choices=[
            "target_match",
            "target_match_expand",
            "draft_margin",
            "whole_block",
            "category_block",
            "refine_block",
        ],
        help=(
            "Optional commit policy used only for recurrent latent drafts. This keeps prompt-lookup "
            "drafts on --draft-commit-policy, allowing strict PLD with a no-match latent policy."
        ),
    )
    parser.add_argument(
        "--target-match-lambda",
        type=float,
        default=1.0,
        help=(
            "Minimum relative target support for accepting a drafted token under target_match. "
            "0 accepts the full verifier-approved block; 1 requires an exact target top-1 match."
        ),
    )
    parser.add_argument("--target-match-lambda-early", type=float)
    parser.add_argument("--target-match-lambda-mid", type=float)
    parser.add_argument("--target-match-lambda-late", type=float)
    parser.add_argument(
        "--mismatch-min-target-entropy",
        type=float,
        help=(
            "Only accept a target-supported mismatch when the target distribution entropy is at least "
            "this value. Exact draft matches are unaffected."
        ),
    )
    parser.add_argument("--mismatch-min-target-entropy-early", type=float)
    parser.add_argument("--mismatch-min-target-entropy-mid", type=float)
    parser.add_argument("--mismatch-min-target-entropy-late", type=float)
    parser.add_argument(
        "--max-accepted-mismatches-per-sequence",
        type=int,
        help=(
            "Cap accepted draft mismatches per generated sequence for target_match or "
            "target_match_expand; exact matches are unaffected."
        ),
    )
    parser.add_argument(
        "--max-accepted-unchecked-per-sequence",
        type=int,
        help=(
            "Cap category_block or draft_margin tokens committed without target-match "
            "diagnostics per generated sequence. Disabled by default."
        ),
    )
    parser.add_argument(
        "--max-accepted-unchecked-per-window",
        type=int,
        help=(
            "Cap unchecked category_block or draft_margin tokens in each fixed generated-token "
            "window. Requires --unchecked-budget-window-tokens. Disabled by default."
        ),
    )
    parser.add_argument(
        "--unchecked-budget-window-tokens",
        type=int,
        help="Generated-token window size for --max-accepted-unchecked-per-window.",
    )
    parser.add_argument(
        "--precommit-unchecked-drafts",
        action="store_true",
        help=(
            "For latent whole_block or draft_margin policies, decide the accepted prefix before "
            "the target tail forward, trim rejected drafts, and request only the final-position "
            "logits. This removes target-match diagnostics and avoids processing drafts that the "
            "unchecked policy would reject. Disabled by default."
        ),
    )
    parser.add_argument(
        "--latent-budget-short-circuit",
        action="store_true",
        help=(
            "Skip latent T rollout after the active unchecked-token budget is exhausted. "
            "Strict prompt-lookup drafts remain enabled. This is opt-in for controlled timing."
        ),
    )
    parser.add_argument(
        "--split-unchecked-target-forward",
        action="store_true",
        help=(
            "For precommitted latent blocks, update target prefix-layer KV from token IDs and "
            "projection-layer KV/logits from T-predicted anchors instead of running the complete "
            "target stack serially. This experimental hybrid-cache path is disabled by default."
        ),
    )
    parser.add_argument(
        "--split-unchecked-parallel",
        action="store_true",
        help=(
            "Run the disjoint prefix-KV and latent-projection branches on separate CUDA streams. "
            "Requires --split-unchecked-target-forward."
        ),
    )
    parser.add_argument(
        "--split-unchecked-min-position",
        type=int,
        default=0,
        help="Use hybrid split execution only at or after this generated-token position.",
    )
    parser.add_argument(
        "--split-unchecked-min-draft-margin",
        type=float,
        help="Require every precommitted latent draft to meet this margin before split execution.",
    )
    parser.add_argument(
        "--target-match-category-lambdas",
        default="",
        help=(
            "Comma-separated token-category lambda overrides for target_match, e.g. "
            "number:0.03,symbol:0.3,newline:1,word:1. Categories are space, symbol, "
            "number, word, newline, other. Unspecified categories use --target-match-lambda."
        ),
    )
    parser.add_argument(
        "--expand-after-position",
        type=int,
        default=128,
        help="For target_match_expand, only enlarge strict accepted drafts at or after this generated position.",
    )
    parser.add_argument(
        "--expand-min-safe-drafts",
        type=int,
        default=2,
        help="For target_match_expand, require this many strict matching draft tokens before expansion.",
    )
    parser.add_argument(
        "--expand-multiplier",
        type=float,
        default=2.0,
        help="For target_match_expand, accept up to ceil(strict_safe_drafts * multiplier) draft tokens.",
    )
    parser.add_argument(
        "--expand-max-accepted-drafts",
        type=int,
        default=16,
        help="For target_match_expand, cap the number of accepted draft tokens after expansion.",
    )
    parser.add_argument("--verifier-weights")
    parser.add_argument("--verifier-threshold", type=float, default=0.5)
    parser.add_argument("--verifier-threshold-early", type=float)
    parser.add_argument("--verifier-threshold-mid", type=float)
    parser.add_argument("--verifier-threshold-late", type=float)
    parser.add_argument(
        "--verifier-cumulative-survival",
        action="store_true",
        help=(
            "Treat each probe output as a conditional survival probability and compare the "
            "running product against the active verifier threshold."
        ),
    )
    parser.add_argument(
        "--verifier-score-temperature",
        type=float,
        default=1.0,
        help="Temperature-scale verifier logits before converting them to probabilities.",
    )
    parser.add_argument(
        "--verifier-skip-threshold",
        type=float,
        help=(
            "If the active verifier threshold t at a position is at least this value, "
            "skip T/probe for the cycle and emit only the target token."
        ),
    )
    parser.add_argument("--verifier-interval", type=int, default=1)
    parser.add_argument("--allow-zero-draft", action="store_true")
    parser.add_argument(
        "--cheap-verifier-policy",
        choices=["none", "entropy", "logit_margin", "anchor_category"],
        default="none",
        help=(
            "Optional low-cost rollout trigger. The entropy policy uses the current target "
            "next-token distribution to choose a dynamic verifier threshold in verifier mode, "
            "and can skip T or reduce the block in fixed/schedule/heuristic modes."
        ),
    )
    parser.add_argument("--cheap-verifier-entropy-low", type=float, default=0.05)
    parser.add_argument("--cheap-verifier-entropy-mid", type=float, default=0.20)
    parser.add_argument("--cheap-verifier-entropy-high", type=float, default=0.70)
    parser.add_argument("--cheap-verifier-threshold-low", type=float, default=0.25)
    parser.add_argument("--cheap-verifier-threshold-mid", type=float, default=0.40)
    parser.add_argument("--cheap-verifier-threshold-high", type=float, default=0.55)
    parser.add_argument("--cheap-verifier-threshold-very-high", type=float, default=0.80)
    parser.add_argument(
        "--cheap-verifier-skip-threshold",
        type=float,
        default=0.75,
        help="Skip T/verifier for the cycle when the dynamic threshold t is above this value.",
    )
    parser.add_argument(
        "--cheap-verifier-top1-skip-min",
        type=float,
        default=0.0,
        help="If >0, skip T/verifier when current target top-1 probability is below this value.",
    )
    parser.add_argument(
        "--cheap-verifier-margin-skip-below",
        type=float,
        default=0.0,
        help="For logit_margin policy, skip T below this target top-1/top-2 raw-logit margin.",
    )
    parser.add_argument(
        "--cheap-verifier-margin-block2-below",
        type=float,
        default=0.0,
        help="For logit_margin policy, cap rollout at two total tokens below this target logit margin.",
    )
    parser.add_argument(
        "--latent-min-target-margin",
        type=float,
        default=0.0,
        help=(
            "Only invoke latent T when the current target top-1/top-2 raw-logit margin "
            "meets this threshold. N-gram drafting remains enabled below the threshold; "
            "zero disables the latent-only gate."
        ),
    )
    parser.add_argument(
        "--latent-margin-gate-close-after",
        type=int,
        default=0,
        help=(
            "Permanently disable latent T for a request after this many consecutive "
            "latent candidates fail --latent-min-target-margin. N-gram drafting remains "
            "active; zero disables the request-level circuit breaker."
        ),
    )
    parser.add_argument(
        "--cheap-verifier-dynamic-block",
        action="store_true",
        help="Map lower dynamic thresholds to larger blocks and higher thresholds to smaller blocks.",
    )
    parser.add_argument("--cheap-verifier-low-t-block", type=int, default=16)
    parser.add_argument("--cheap-verifier-mid-t-block", type=int, default=8)
    parser.add_argument("--cheap-verifier-high-t-block", type=int, default=4)
    parser.add_argument("--cheap-verifier-very-high-t-block", type=int, default=2)
    parser.add_argument(
        "--cheap-verifier-rollout-anchor-categories",
        default="",
        help=(
            "Comma-separated target anchor token categories that may enable T/verifier rollout. "
            "Categories are space, symbol, number, word, newline, other. When set, cycles outside "
            "the position window or with an anchor category outside the set skip T/verifier."
        ),
    )
    parser.add_argument("--cheap-verifier-rollout-min-position", type=int, default=0)
    parser.add_argument("--cheap-verifier-rollout-max-position", type=int)
    parser.add_argument(
        "--draft-token-category-gate",
        default="",
        help=(
            "Comma-separated categories allowed for accepting drafted tokens. "
            "Categories are space, symbol, number, word, newline, other. "
            "When set, acceptance stops at the first drafted token outside the set."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-block-tokens", type=int, default=6)
    parser.add_argument("--min-block-tokens", type=int, default=1)
    parser.add_argument(
        "--verification-frequency-policy",
        choices=["fixed", "risk_budget", "agreement_gated"],
        default="fixed",
        help=(
            "Choose tentative verification blocks by the configured fixed cap or stop early "
            "when cumulative verifier-head rejection risk would exceed a budget. Agreement-gated "
            "frequency first observes exact one-token shadow probes, then enables the configured "
            "block cap only while exact high-margin agreement continues."
        ),
    )
    parser.add_argument(
        "--verification-risk-budget",
        type=float,
        default=1.0,
        help=(
            "Maximum cumulative rejection risk for --verification-frequency-policy risk_budget. "
            "The cap remains --max-block-tokens and --min-block-tokens is always honored."
        ),
    )
    parser.add_argument(
        "--verification-agreement-warmup",
        type=int,
        default=4,
        help="Consecutive qualified shadow agreements required to enable multi-token verification.",
    )
    parser.add_argument(
        "--verification-agreement-min-draft-margin",
        type=float,
        default=2.0,
        help="Minimum draft top-1 logit margin for an agreement observation to qualify.",
    )
    parser.add_argument(
        "--verification-agreement-min-target-margin",
        type=float,
        default=3.0,
        help="Minimum target top-1 logit margin for an agreement observation to qualify.",
    )
    parser.add_argument(
        "--omit-target-attention-mask",
        action="store_true",
        help=(
            "Do not allocate an all-ones attention mask for batch-1 target forwards. "
            "The model derives causal masking from cache_position; use this production "
            "fast path only for unpadded single-sequence decoding."
        ),
    )
    parser.add_argument(
        "--compact-runtime-stats",
        action="store_true",
        help=(
            "Use the production inference path: compute target softmax/entropy and token text only "
            "when an active acceptance policy needs them, and omit bulky per-token diagnostics."
        ),
    )
    parser.add_argument(
        "--anchor-hook-hidden-states",
        action="store_true",
        help=(
            "Capture only the recurrent boundary hidden state with a decoder-layer hook "
            "instead of retaining every layer hidden state from target forwards."
        ),
    )
    parser.add_argument(
        "--production-async-timing",
        action="store_true",
        help=(
            "Do not synchronize the whole CUDA device around every component timer. "
            "The outer baseline/adaptive wall timers remain synchronized, while component "
            "times become host enqueue times and must not be used for decomposition."
        ),
    )
    parser.add_argument(
        "--sparse-target-entropy",
        action="store_true",
        help=(
            "Compute target entropy only for mismatch/support candidate rows. This can help "
            "large verification blocks, but dense softmax is faster for the usual block-3 path."
        ),
    )
    parser.add_argument(
        "--single-gpu-parallel-draft",
        action="store_true",
        help=(
            "Run the independent draft-logit projection and recurrent T step on two CUDA "
            "streams. This applies to both boundary-head and tail-logit drafting when the "
            "verifier has not already advanced T."
        ),
    )
    parser.add_argument(
        "--single-gpu-parallel-correction-sync",
        action="store_true",
        help=(
            "Overlap the target corrective-token forward with recurrent-cache synchronization "
            "for the accepted prefix on a second CUDA stream. This applies only to the "
            "--reuse-verify-cache-for-correction path."
        ),
    )
    parser.add_argument(
        "--preemptive-lookahead",
        action="store_true",
        help=(
            "After the target reaches the recurrent boundary, overlap its remaining tail "
            "layers with T-cache synchronization and next-block drafting. The prefetched "
            "block is reused only when the current block is fully accepted and its first "
            "token agrees with the target next token."
        ),
    )
    parser.add_argument(
        "--preemptive-min-draft-margin",
        type=float,
        default=0.0,
        help=(
            "Launch preemptive lookahead only when every current draft has at least this "
            "boundary-logit margin. Zero disables the gate."
        ),
    )
    parser.add_argument("--inplace-draft-cache", action="store_true")
    parser.add_argument(
        "--full-draft-cache-clone",
        action="store_true",
        help=(
            "Clone the full target KV cache for draft-tail projection instead of only tail layers. "
            "This is slower and uses more memory, but is useful for exactness diagnostics."
        ),
    )
    parser.add_argument(
        "--replay-accepted-target-cache",
        action="store_true",
        help=(
            "After verifying a fully accepted block, replay the committed tokens from the live target "
            "cache instead of adopting the verification cache. This is a conservative exactness "
            "diagnostic and increases target calls."
        ),
    )
    parser.add_argument(
        "--reuse-verify-cache-for-correction",
        action="store_true",
        help=(
            "On a partially accepted block, crop and adopt the verification KV cache, then run only "
            "the corrective token. This avoids recomputing the accepted prefix and is intended for "
            "task-quality decoding where token-level numerical equivalence is not required."
        ),
    )
    parser.add_argument(
        "--defer-correction-to-next-verify",
        action="store_true",
        help=(
            "On a partially accepted block, leave the target corrective token pending and include it "
            "as the first token of the next verification block. This one-token-lag pipeline removes "
            "the standalone corrective forward while preserving target verification of every block."
        ),
    )
    parser.add_argument(
        "--defer-correction-min-accepted-drafts",
        type=int,
        default=0,
        help=(
            "Only defer correction after accepting at least this many drafted tokens in the "
            "current block. Zero preserves unconditional deferred correction."
        ),
    )
    parser.add_argument(
        "--defer-correction-after-position",
        type=int,
        default=0,
        help="Only defer correction at or after this generated-token position.",
    )
    parser.add_argument(
        "--defer-correction-before-position",
        type=int,
        help="Only defer correction before this generated-token position.",
    )
    parser.add_argument(
        "--serial-target-cache-commit",
        action="store_true",
        help=(
            "Commit target-cache tokens one by one instead of feeding multi-token commit tensors. "
            "This is slower, but useful for checking whether batched correction commits introduce "
            "small exactness differences."
        ),
    )
    parser.add_argument(
        "--serial-fallback-margin",
        type=float,
        help=(
            "Selectively replay the accepted prefix token by token when the target logit margin at "
            "the commit boundary is at or below this value. The corrective token is then recomputed "
            "from the serial cache. Disabled by default."
        ),
    )
    parser.add_argument(
        "--serial-fallback-full-replay-margin",
        type=float,
        help=(
            "Rebuild the live target cache from the prompt and replay the complete generated prefix when "
            "the target margin is at or below this stricter threshold. The check is applied before each "
            "cycle's first target token and at a corrective-token boundary. This repairs numerical drift "
            "accumulated by earlier block commits, but is more expensive than local-prefix replay."
        ),
    )
    parser.add_argument(
        "--batched-draft-tail",
        action="store_true",
        help="Project all token-independent recurrent anchors through the draft tail in one call.",
    )
    parser.add_argument(
        "--batched-draft-boundary",
        action="store_true",
        help=(
            "Project all token-independent recurrent anchors through the boundary head and "
            "shared LM head in one call."
        ),
    )
    parser.add_argument(
        "--draft-logit-source",
        choices=["tail", "boundary"],
        default="tail",
        help="Use the original projection tail or the optional lightweight RecurFT boundary head for drafts.",
    )
    parser.add_argument("--min-draft-margin", type=float, default=0.0)
    parser.add_argument(
        "--min-draft-margin-early",
        type=float,
        help="Override the latent draft-margin threshold before --medium-position.",
    )
    parser.add_argument(
        "--min-draft-margin-mid",
        type=float,
        help="Override the latent draft-margin threshold from --medium-position to --long-position.",
    )
    parser.add_argument(
        "--min-draft-margin-late",
        type=float,
        help="Override the latent draft-margin threshold at or after --long-position.",
    )
    parser.add_argument(
        "--ngram-draft-mode",
        choices=["off", "prefer", "only"],
        default="off",
        help=(
            "Use a prompt/generated suffix match as a zero-model-cost drafter. Prefer mode "
            "falls back to latent T when no suffix match exists; only mode falls back to one target token."
        ),
    )
    parser.add_argument("--ngram-min-size", type=int, default=2)
    parser.add_argument("--ngram-max-size", type=int, default=8)
    parser.add_argument(
        "--indexed-ngram-lookup",
        action="store_true",
        help="Cache n-gram occurrence positions instead of rescanning the full prefix each cycle.",
    )
    parser.add_argument(
        "--ngram-tree-verify",
        action="store_true",
        help=(
            "Verify several distinct indexed n-gram continuations in one ancestor-masked "
            "target tree forward. This first implementation is restricted to strict "
            "n-gram-only decoding and is disabled by default."
        ),
    )
    parser.add_argument("--ngram-tree-early-branches", type=int, default=1)
    parser.add_argument("--ngram-tree-mid-branches", type=int, default=2)
    parser.add_argument("--ngram-tree-late-branches", type=int, default=4)
    parser.add_argument(
        "--ngram-tree-max-nodes",
        type=int,
        default=96,
        help="Maximum root-plus-draft nodes in one target tree forward.",
    )
    parser.add_argument(
        "--ngram-occurrence-policy",
        choices=[
            "latest",
            "prompt_latest",
            "consensus",
            "prompt_consensus",
            "weighted",
            "prompt_weighted",
        ],
        default="latest",
        help=(
            "Choose among repeated suffix matches. The default preserves the latest-match "
            "behavior; prompt variants exclude generated-history occurrences, while consensus "
            "builds a token-wise majority continuation with recency tie-breaking. Weighted "
            "policies exponentially favor recent occurrences."
        ),
    )
    parser.add_argument(
        "--ngram-occurrence-recency-decay",
        type=float,
        default=0.9,
        help=(
            "Per-occurrence recency decay for weighted n-gram occurrence policies. The latest "
            "occurrence has weight 1 and each older occurrence is multiplied by this value."
        ),
    )
    parser.add_argument(
        "--ngram-strict-target-match",
        action="store_true",
        help=(
            "Require exact target-token agreement for prompt-lookup drafts while retaining "
            "the configured quality-tolerant target-match policy for latent T drafts."
        ),
    )
    parser.add_argument(
        "--fast-strict-ngram-diagnostics",
        action="store_true",
        help=(
            "For compact strict prompt-lookup verification, transfer only target argmax ids "
            "instead of materializing float32 full-vocabulary support diagnostics."
        ),
    )
    parser.add_argument(
        "--fast-strict-diagnostics", action="store_true",
        help="For compact exact target-match decoding, compare token IDs without full-vocabulary support statistics.",
    )
    parser.add_argument(
        "--ngram-max-draft-tokens",
        type=int,
        default=0,
        help=(
            "Optional n-gram-specific proposal width. Zero reuses the normal block limit, "
            "allowing latent T and prompt lookup to use different widths when set."
        ),
    )
    parser.add_argument(
        "--ngram-draft-tokens-by-match-size",
        default="",
        help=(
            "Optional comma-separated prompt-lookup proposal widths keyed by matched n-gram "
            "size, for example 2:3,3:4,4:5,5:5,6:7,7:7,8:7. Strict target matching is unchanged."
        ),
    )
    parser.add_argument(
        "--ngram-max-draft-tokens-by-position",
        default="",
        help=(
            "Optional comma-separated maximum prompt-lookup proposal widths keyed by the "
            "first generated position where each width applies, for example 0:7,512:31. "
            "This cap is applied before --ngram-draft-tokens-by-match-size."
        ),
    )
    parser.add_argument(
        "--ngram-wide-auto-route-after-hits",
        type=int,
        default=0,
        help=(
            "After this many prompt-lookup proposals wider than the fallback width, "
            "keep or disable wide proposals using their observed extra accepted tokens. "
            "Zero disables this request-local controller."
        ),
    )
    parser.add_argument(
        "--ngram-wide-auto-route-min-extra-per-hit",
        type=float,
        default=0.0,
        help="Disable wide prompt-lookup proposals when extra accepted drafts per wide hit fall below this value.",
    )
    parser.add_argument(
        "--ngram-wide-auto-route-fallback-width",
        type=int,
        default=7,
        help="Prompt-lookup draft width used after the request-local wide controller disables widening.",
    )
    parser.add_argument(
        "--ngram-predecode-cost-route",
        action="store_true",
        help=(
            "Choose a request-level prompt-lookup width before decoding. Explicit position "
            "schedules are preserved. Otherwise, long prompts with short outputs use the wide "
            "width, medium prompts with long outputs use the medium width, and all remaining "
            "requests use the narrow width."
        ),
    )
    parser.add_argument("--ngram-predecode-medium-prompt-tokens", type=int, default=512)
    parser.add_argument("--ngram-predecode-long-prompt-tokens", type=int, default=2048)
    parser.add_argument("--ngram-predecode-short-output-tokens", type=int, default=128)
    parser.add_argument("--ngram-predecode-narrow-width", type=int, default=7)
    parser.add_argument("--ngram-predecode-medium-width", type=int, default=15)
    parser.add_argument("--ngram-predecode-wide-width", type=int, default=63)
    parser.add_argument(
        "--ngram-predecode-occurrence-route",
        action="store_true",
        help=(
            "Choose the repeated-suffix occurrence policy before decoding from prompt and "
            "requested-output lengths. Explicit position schedules retain the configured "
            "occurrence policy."
        ),
    )
    parser.add_argument(
        "--ngram-unchecked-min-match-size",
        type=int,
        default=0,
        help=(
            "Precommit prompt-lookup drafts only when the matched suffix size is at least "
            "this value. Zero disables unchecked n-gram commits; all other n-gram cycles "
            "retain the configured target verification policy."
        ),
    )
    parser.add_argument(
        "--ngram-unchecked-min-position",
        type=int,
        default=0,
        help="Do not precommit eligible n-gram drafts before this generated-token position.",
    )
    parser.add_argument(
        "--ngram-unchecked-token-categories",
        default="",
        help=(
            "Optional comma-separated token categories allowed in an unchecked n-gram prefix. "
            "The prefix stops before the first disallowed token; categories are word, number, "
            "space, newline, symbol, and other. Empty disables this additional gate."
        ),
    )
    parser.add_argument(
        "--ngram-unchecked-category-reject-closes-sequence",
        action="store_true",
        help=(
            "After a token-category gate rejects or truncates an eligible n-gram prefix, "
            "disable later unchecked n-gram opportunities for that sequence instead of "
            "reallocating the remaining risk budget."
        ),
    )
    parser.add_argument(
        "--latent-unchecked-token-categories",
        default="",
        help=(
            "Optional comma-separated token categories allowed for unchecked latent drafts. "
            "The precommit prefix stops before the first disallowed token; categories are "
            "word, number, space, newline, symbol, and other."
        ),
    )
    parser.add_argument(
        "--latent-unchecked-category-reject-closes-sequence",
        action="store_true",
        help=(
            "After the latent token-category gate rejects a precommit candidate, disable "
            "later unchecked latent opportunities for that sequence."
        ),
    )
    parser.add_argument(
        "--lazy-t-sync",
        action="store_true",
        help=(
            "Delay recurrent-cache synchronization across target-only or n-gram cycles, "
            "then batch the pending boundary states immediately before T is needed again."
        ),
    )
    parser.add_argument(
        "--defer-t-init-until-latent",
        action="store_true",
        help=(
            "Defer the prompt-wide recurrent T prefill until the first latent draft. "
            "Requests served entirely by the target model or prompt lookup then avoid T compute. "
            "Requires --lazy-t-sync."
        ),
    )
    parser.add_argument(
        "--latent-draft-min-position",
        type=int,
        default=0,
        help=(
            "Do not invoke latent T before this generated-token position. Prompt lookup "
            "remains available, and --lazy-t-sync can batch the skipped recurrent-cache updates."
        ),
    )
    parser.add_argument(
        "--draft-cooldown-after-failures",
        type=int,
        default=0,
        help=(
            "After this many consecutive latent draft cycles accept zero draft tokens, "
            "temporarily switch to target-only decoding. Zero disables the controller."
        ),
    )
    parser.add_argument(
        "--draft-cooldown-cycles",
        type=int,
        default=0,
        help="Number of target-only cycles entered by the online draft cooldown controller.",
    )
    parser.add_argument(
        "--auto-route-after-latent-drafts",
        type=int,
        default=0,
        help=(
            "After observing this many latent draft tokens, permanently disable T for the "
            "request if its cumulative acceptance is below the configured threshold."
        ),
    )
    parser.add_argument(
        "--auto-route-min-latent-acceptance",
        type=float,
        default=0.0,
        help="Minimum warmup latent-draft acceptance required to keep T active.",
    )
    parser.add_argument(
        "--auto-route-reevaluate-every",
        type=int,
        default=0,
        help=(
            "After a latent route passes its initial warmup, re-evaluate cumulative "
            "acceptance after this many additional route observations. Zero preserves "
            "the existing one-shot decision."
        ),
    )
    parser.add_argument(
        "--auto-route-early-zero-after",
        type=int,
        default=0,
        help=(
            "Optional first-stage router: disable T immediately when none of the first N "
            "latent drafts are accepted. A nonzero acceptance defers the decision to the "
            "regular cumulative-acceptance gate."
        ),
    )
    parser.add_argument("--short-block", type=int, default=2)
    parser.add_argument("--medium-block", type=int, default=4)
    parser.add_argument("--long-block", type=int, default=6)
    parser.add_argument("--medium-position", type=int, default=96)
    parser.add_argument("--long-position", type=int, default=192)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "eager", "sdpa"],
        default="auto",
        help=(
            "Attention kernel used by the target model. Eager mode is useful for checking whether "
            "BF16 block-vs-token exactness differences come from fused attention kernels."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--question-prefix", default="Solve the following math problem step by step.")
    parser.add_argument("--question-suffix", default="Put the final answer on its own line after 'Answer:'.")
    parser.add_argument(
        "--synthetic-prefix-tokens",
        type=int,
        default=0,
        help="Insert exactly this many neutral tokens after BOS for controlled long-context timing.",
    )
    parser.add_argument(
        "--synthetic-prefix-text",
        default="Background context: this sentence is unrelated to the math problem.\n",
        help="Text whose token IDs are repeated by --synthetic-prefix-tokens.",
    )
    parser.add_argument("--disable-thinking", action="store_true", default=True)
    parser.add_argument("--enable-thinking", action="store_true", help="Allow Qwen-style thinking in chat templates.")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument(
        "--decode-order",
        choices=["baseline_first", "adaptive_first", "alternate"],
        default="baseline_first",
        help=(
            "Order the baseline and adaptive decoders within each sample. Alternate balances "
            "warm-up, thermal, and transient-load effects in wall-clock comparisons."
        ),
    )
    return parser.parse_args()


def load_rows(path: str, start: int, limit: int) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def apply_target_lora_removal(model: torch.nn.Module, metadata: dict[str, Any], group: str) -> list[str]:
    """Disable one target-path LoRA group by zeroing its runtime scaling."""
    if group == "none":
        return []

    boundary = int(metadata["recurrent_hidden_state_index"])
    removed = []
    for name, module in model.named_modules():
        match = re.search(r"\.layers\.(\d+)\.", name)
        scaling = getattr(module, "scaling", None)
        if match is None or not isinstance(scaling, dict):
            continue

        layer_index = int(match.group(1))
        selected = layer_index < boundary if group == "pre" else layer_index >= boundary
        if not selected:
            continue
        for adapter_name in list(scaling):
            scaling[adapter_name] = 0.0
        removed.append(name)

    if not removed:
        raise ValueError(f"No target-path LoRA modules matched --target-lora-removal {group!r}.")
    return removed


def cast_target_lora_parameters(model: torch.nn.Module, dtype: torch.dtype) -> list[str]:
    """Cast only target PEFT low-rank matrices, leaving base and recurrent weights untouched."""
    cast_names = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_A." not in name and ".lora_B." not in name:
                continue
            parameter.data = parameter.data.to(dtype=dtype)
            cast_names.append(name)
    if not cast_names:
        raise ValueError("No target LoRA A/B parameters were found to cast.")
    return cast_names


def make_prompt(tokenizer, question: str, prefix: str, suffix: str, disable_thinking: bool) -> torch.Tensor:
    content = f"{prefix}\n\n{question.strip()}\n\n{suffix}".strip()
    kwargs = {
        "conversation": [{"role": "user", "content": content}],
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    if disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(**kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(**kwargs)


def add_synthetic_prefix(
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    token_count: int,
    prefix_text: str,
) -> torch.Tensor:
    if token_count <= 0:
        return prompt_ids
    unit_ids = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if unit_ids.numel() == 0:
        raise ValueError("--synthetic-prefix-text must tokenize to at least one token.")
    repeats = math.ceil(token_count / unit_ids.numel())
    prefix_ids = unit_ids.repeat(repeats)[:token_count].unsqueeze(0)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    insert_at = int(bool(bos_id is not None and int(prompt_ids[0, 0]) == int(bos_id)))
    return torch.cat(
        [prompt_ids[:, :insert_at], prefix_ids, prompt_ids[:, insert_at:]],
        dim=1,
    )


def normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().replace(",", "").replace("$", "")
    value = value.replace("−", "-")
    value = re.sub(r"\s+", "", value)
    value = value.rstrip(".")
    return value.lower()


def extract_answer(text: str) -> str | None:
    patterns = [
        r"answer\s*:\s*([^\n]+)",
        r"####\s*([^\n]+)",
        r"\\boxed\{([^{}]+)\}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            candidate = matches[-1]
            numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", candidate.replace(",", ""))
            return numbers[-1] if numbers else candidate.strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


def rouge_l_f1(prediction: str, reference: str) -> float:
    prediction_tokens = re.findall(r"\w+|[^\w\s]", prediction.lower(), flags=re.UNICODE)
    reference_tokens = re.findall(r"\w+|[^\w\s]", reference.lower(), flags=re.UNICODE)
    if not prediction_tokens or not reference_tokens:
        return 0.0

    previous = [0] * (len(reference_tokens) + 1)
    for prediction_token in prediction_tokens:
        current = [0]
        for index, reference_token in enumerate(reference_tokens, start=1):
            if prediction_token == reference_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current

    lcs_length = previous[-1]
    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(reference_tokens)
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def score_response(
    text: str,
    row: dict[str, Any],
    metric: str,
) -> tuple[str | None, str | None, float]:
    raw_references = row.get("answers")
    if isinstance(raw_references, list) and raw_references:
        references = [str(value) for value in raw_references]
    else:
        references = [str(row.get("answer", ""))]

    if metric == "math":
        prediction = normalize_answer(extract_answer(text))
        # Official GSM8K answers contain a rationale followed by "#### <answer>".
        reference = normalize_answer(references[0].rsplit("####", 1)[-1])
        return prediction, reference, float(prediction == reference)

    scores = [rouge_l_f1(text, reference) for reference in references]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    return text, references[best_index], scores[best_index]


def find_suffix_ngram_draft(
    prefix_tokens: list[int],
    max_draft_tokens: int,
    min_ngram_size: int,
    max_ngram_size: int,
) -> tuple[list[int], int | None]:
    if max_draft_tokens <= 0 or len(prefix_tokens) <= min_ngram_size:
        return [], None

    largest_ngram = min(max_ngram_size, len(prefix_tokens) - 1)
    for ngram_size in range(largest_ngram, min_ngram_size - 1, -1):
        suffix = prefix_tokens[-ngram_size:]
        latest_start = len(prefix_tokens) - ngram_size - 1
        for match_start in range(latest_start, -1, -1):
            if prefix_tokens[match_start : match_start + ngram_size] != suffix:
                continue
            continuation_start = match_start + ngram_size
            continuation = prefix_tokens[
                continuation_start : continuation_start + max_draft_tokens
            ]
            if continuation:
                return continuation, ngram_size
    return [], None


class SuffixNgramIndex:
    """Incremental occurrence index equivalent to ``find_suffix_ngram_draft``."""

    def __init__(self, tokens: list[int], min_ngram_size: int, max_ngram_size: int) -> None:
        self.tokens = list(tokens)
        self.prompt_length = len(tokens)
        self.min_ngram_size = min_ngram_size
        self.max_ngram_size = max_ngram_size
        self.positions: dict[int, dict[tuple[int, ...], list[int]]] = {
            size: {} for size in range(min_ngram_size, max_ngram_size + 1)
        }
        for size in self.positions:
            for start in range(0, len(self.tokens) - size + 1):
                key = tuple(self.tokens[start : start + size])
                self.positions[size].setdefault(key, []).append(start)

    def append(self, tokens: list[int]) -> None:
        for token in tokens:
            self.tokens.append(int(token))
            for size in self.positions:
                if len(self.tokens) < size:
                    continue
                start = len(self.tokens) - size
                key = tuple(self.tokens[start:])
                self.positions[size].setdefault(key, []).append(start)

    def _continuation(
        self,
        start: int,
        ngram_size: int,
        first_token: int,
        max_draft_tokens: int,
        *,
        prompt_only: bool,
    ) -> list[int]:
        continuation_start = start + ngram_size
        prefix_length = len(self.tokens) + 1
        continuation_limit = self.prompt_length if prompt_only else prefix_length
        continuation_end = min(
            continuation_start + max_draft_tokens,
            continuation_limit,
        )
        history_end = min(continuation_end, len(self.tokens))
        continuation = list(self.tokens[continuation_start:history_end])
        if continuation_end > len(self.tokens):
            continuation.append(int(first_token))
        return continuation

    def _consensus_continuation(
        self,
        starts: list[int],
        ngram_size: int,
        first_token: int,
        max_draft_tokens: int,
        *,
        prompt_only: bool,
        recency_decay: float | None = None,
    ) -> list[int]:
        continuations = [
            self._continuation(
                start,
                ngram_size,
                first_token,
                max_draft_tokens,
                prompt_only=prompt_only,
            )
            for start in starts
        ]
        active = [
            (start, continuation)
            for start, continuation in zip(starts, continuations)
            if continuation
        ]
        weights = {
            start: (
                recency_decay ** (len(starts) - rank - 1)
                if recency_decay is not None
                else 1.0
            )
            for rank, start in enumerate(starts)
        }
        consensus: list[int] = []
        for offset in range(max_draft_tokens):
            candidates = [
                (start, continuation[offset])
                for start, continuation in active
                if offset < len(continuation)
            ]
            if not candidates:
                break
            counts: dict[int, float] = {}
            latest_starts: dict[int, int] = {}
            for start, token in candidates:
                counts[token] = counts.get(token, 0.0) + weights[start]
                latest_starts[token] = max(latest_starts.get(token, -1), start)
            selected = max(
                counts,
                key=lambda token: (counts[token], latest_starts[token]),
            )
            consensus.append(selected)
            active = [
                (start, continuation)
                for start, continuation in active
                if offset < len(continuation) and continuation[offset] == selected
            ]
        return consensus

    def find(
        self,
        first_token: int,
        max_draft_tokens: int,
        occurrence_policy: str = "latest",
        occurrence_recency_decay: float = 0.9,
    ) -> tuple[list[int], int | None]:
        prefix_length = len(self.tokens) + 1
        if max_draft_tokens <= 0 or prefix_length <= self.min_ngram_size:
            return [], None

        prompt_only = occurrence_policy in {
            "prompt_latest",
            "prompt_consensus",
            "prompt_weighted",
        }
        use_consensus = occurrence_policy in {
            "consensus",
            "prompt_consensus",
            "weighted",
            "prompt_weighted",
        }
        recency_decay = (
            occurrence_recency_decay
            if occurrence_policy in {"weighted", "prompt_weighted"}
            else None
        )

        largest_ngram = min(self.max_ngram_size, prefix_length - 1)
        for ngram_size in range(largest_ngram, self.min_ngram_size - 1, -1):
            if ngram_size == 1:
                suffix = (int(first_token),)
            else:
                suffix = tuple(self.tokens[-(ngram_size - 1) :]) + (int(first_token),)
            starts = self.positions[ngram_size].get(suffix)
            if not starts:
                continue
            latest_start = prefix_length - ngram_size - 1
            match_index = bisect_right(starts, latest_start) - 1
            if match_index < 0:
                continue
            valid_starts = starts[: match_index + 1]
            if prompt_only:
                valid_starts = [
                    start
                    for start in valid_starts
                    if start + ngram_size < self.prompt_length
                ]
            if not valid_starts:
                continue
            if use_consensus:
                continuation = self._consensus_continuation(
                    valid_starts,
                    ngram_size,
                    first_token,
                    max_draft_tokens,
                    prompt_only=prompt_only,
                    recency_decay=recency_decay,
                )
            else:
                continuation = self._continuation(
                    valid_starts[-1],
                    ngram_size,
                    first_token,
                    max_draft_tokens,
                    prompt_only=prompt_only,
                )
            if continuation:
                return continuation, ngram_size
        return [], None

    def find_branches(
        self,
        first_token: int,
        max_draft_tokens: int,
        max_branches: int,
    ) -> tuple[list[list[int]], int | None]:
        """Return distinct recent continuations for the largest matching suffix."""

        prefix_length = len(self.tokens) + 1
        if (
            max_draft_tokens <= 0
            or max_branches <= 0
            or prefix_length <= self.min_ngram_size
        ):
            return [], None

        largest_ngram = min(self.max_ngram_size, prefix_length - 1)
        for ngram_size in range(largest_ngram, self.min_ngram_size - 1, -1):
            if ngram_size == 1:
                suffix = (int(first_token),)
            else:
                suffix = tuple(self.tokens[-(ngram_size - 1) :]) + (int(first_token),)
            starts = self.positions[ngram_size].get(suffix)
            if not starts:
                continue
            latest_start = prefix_length - ngram_size - 1
            match_index = bisect_right(starts, latest_start) - 1
            if match_index < 0:
                continue

            branches = []
            seen = set()
            for start in reversed(starts[: match_index + 1]):
                continuation = self._continuation(
                    start,
                    ngram_size,
                    first_token,
                    max_draft_tokens,
                    prompt_only=False,
                )
                key = tuple(continuation)
                if not continuation or key in seen:
                    continue
                seen.add(key)
                branches.append(continuation)
                if len(branches) >= max_branches:
                    break
            if branches:
                return branches, ngram_size
        return [], None


def clone_cache(cache: Any, layer_ids: list[int] | None = None) -> DynamicCache:
    cloned = DynamicCache()
    selected = set(layer_ids) if layer_ids is not None else None
    for layer_idx, layer in enumerate(cache.layers):
        if selected is not None and layer_idx not in selected:
            continue
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if torch.is_tensor(keys) and torch.is_tensor(values):
            cloned.update(keys.detach().clone(), values.detach().clone(), layer_idx)
    return cloned


def cache_length(cache: Any, layer_idx: int | None = None) -> int:
    if layer_idx is not None:
        try:
            return int(cache.get_seq_length(layer_idx))
        except (IndexError, TypeError):
            pass
    try:
        return int(cache.get_seq_length())
    except (IndexError, TypeError):
        return 0


def timed(fn, device: torch.device):
    if device.type == "cuda" and _SYNC_COMPONENT_TIMING:
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda" and _SYNC_COMPONENT_TIMING:
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def wall_timed(fn, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def target_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    past_key_values: DynamicCache | None = None,
    start_position: int | None = None,
    output_hidden_states: bool = True,
    return_dict: bool = True,
    omit_attention_mask: bool = False,
    logits_to_keep: int | None = None,
):
    kwargs: dict[str, Any] = {
        "use_cache": True,
        "output_hidden_states": output_hidden_states,
        "return_dict": return_dict,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = logits_to_keep
    if start_position is not None:
        seq_len = input_ids.size(1)
        position_ids = torch.arange(
            start_position,
            start_position + seq_len,
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0)
        kwargs["position_ids"] = position_ids
        kwargs["cache_position"] = position_ids[0]
        if not omit_attention_mask:
            kwargs["attention_mask"] = torch.ones(
                (input_ids.size(0), start_position + seq_len),
                dtype=torch.long,
                device=input_ids.device,
            )
    while True:
        try:
            return model(input_ids, **kwargs)
        except TypeError as exc:
            message = str(exc)
            if "cache_position" in message and "cache_position" in kwargs:
                kwargs.pop("cache_position")
                continue
            if "logits_to_keep" in message and "logits_to_keep" in kwargs:
                kwargs.pop("logits_to_keep")
                continue
            raise


def target_tree_forward(
    model: torch.nn.Module,
    tree: DraftTree,
    *,
    past_key_values: DynamicCache,
    start_position: int,
    output_hidden_states: bool,
):
    device = past_key_values.layers[0].keys.device
    dtype = past_key_values.layers[0].keys.dtype
    input_ids = torch.tensor(
        [[node.token_id for node in tree.nodes]],
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.tensor(
        [tree_position_ids(start_position, tree)],
        dtype=torch.long,
        device=device,
    )
    cache_position = torch.arange(
        start_position,
        start_position + len(tree.nodes),
        dtype=torch.long,
        device=device,
    )
    allowed = torch.tensor(
        tree_attention_mask(start_position, tree),
        dtype=torch.bool,
        device=device,
    )
    attention_mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    attention_mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return model(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask.unsqueeze(0).unsqueeze(0),
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=True,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )


def gather_tree_path_cache(
    cache: Any,
    prefix_length: int,
    path_indices: tuple[int, ...],
) -> DynamicCache:
    physical_indices = torch.tensor(
        list(range(prefix_length)) + [prefix_length + index for index in path_indices],
        dtype=torch.long,
        device=cache.layers[0].keys.device,
    )
    gathered = DynamicCache()
    for layer_idx, layer in enumerate(cache.layers):
        gathered.update(
            layer.keys.index_select(-2, physical_indices),
            layer.values.index_select(-2, physical_indices),
            layer_idx,
        )
    return gathered


def select_tree_path_outputs(
    outputs: Any,
    tree: DraftTree,
    prefix_length: int,
) -> tuple[Any, list[int], int, tuple[int, ...]]:
    predictions = outputs.logits[0].argmax(dim=-1).tolist()
    path_indices, matched_drafts = target_guided_leaf_path(tree, predictions)
    index_tensor = torch.tensor(path_indices, dtype=torch.long, device=outputs.logits.device)
    outputs.logits = outputs.logits.index_select(1, index_tensor)
    if outputs.hidden_states is not None:
        if isinstance(outputs.hidden_states, dict):
            outputs.hidden_states = {
                layer_idx: hidden.index_select(1, index_tensor)
                for layer_idx, hidden in outputs.hidden_states.items()
            }
        else:
            outputs.hidden_states = tuple(
                hidden.index_select(1, index_tensor) for hidden in outputs.hidden_states
            )
    outputs.past_key_values = gather_tree_path_cache(
        outputs.past_key_values,
        prefix_length,
        path_indices,
    )
    tokens = [tree.nodes[index].token_id for index in path_indices]
    return outputs, tokens, matched_drafts, path_indices


def parallel_draft_recurrent_step(
    recurrent: torch.nn.Module,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    next_anchor: torch.Tensor,
    t_cache: DynamicCache,
    tail_cache: DynamicCache,
    previous_token: int,
    rollout_step: int,
    position: int,
    draft_logit_source: str,
    device: torch.device,
    streams: tuple[torch.cuda.Stream, torch.cuda.Stream],
) -> tuple[torch.Tensor, torch.Tensor, DynamicCache]:
    r"""Overlap draft projection and T, two independent consumers of the latent anchor."""
    projection_stream, recurrent_stream = streams
    caller_stream = torch.cuda.current_stream(device)
    token_ids = torch.tensor([[previous_token]], dtype=torch.long, device=device)
    projection_stream.wait_stream(caller_stream)
    recurrent_stream.wait_stream(caller_stream)
    next_anchor.record_stream(projection_stream)
    next_anchor.record_stream(recurrent_stream)
    token_ids.record_stream(recurrent_stream)
    record_cache_stream(t_cache, recurrent_stream)
    if draft_logit_source == "tail":
        record_cache_stream(tail_cache, projection_stream)

    with torch.cuda.stream(projection_stream):
        if draft_logit_source == "boundary":
            draft_logits = recurrent.boundary_logits(next_anchor, get_base_causal_lm(model))
        elif draft_logit_source == "tail":
            draft_logits = project_anchor_step(model, metadata, next_anchor, tail_cache, position)
        else:
            raise ValueError(f"Unsupported parallel draft logit source: {draft_logit_source}")

    with torch.cuda.stream(recurrent_stream):
        following_anchor, updated_cache = recurrent.forward_with_cache(
            next_anchor,
            model=get_base_causal_lm(model),
            past_key_value=t_cache,
            token_ids=token_ids,
            rollout_step=rollout_step,
        )

    caller_stream.wait_stream(projection_stream)
    caller_stream.wait_stream(recurrent_stream)
    draft_logits.record_stream(caller_stream)
    following_anchor.record_stream(caller_stream)
    record_cache_stream(tail_cache, caller_stream)
    record_cache_stream(updated_cache, caller_stream)
    return draft_logits, following_anchor, updated_cache


def record_cache_stream(cache: Any, stream: torch.cuda.Stream) -> None:
    for layer in getattr(cache, "layers", []):
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if torch.is_tensor(tensor):
                tensor.record_stream(stream)


def parallel_corrective_target_and_t_sync(
    model: torch.nn.Module,
    recurrent: torch.nn.Module,
    accepted_anchor: torch.Tensor,
    accepted_tokens: list[int],
    target_cache: DynamicCache,
    true_t_cache: DynamicCache,
    corrective: int,
    start_position: int,
    omit_attention_mask: bool,
    device: torch.device,
    recurrent_stream: torch.cuda.Stream,
) -> tuple[Any, torch.Tensor, DynamicCache]:
    r"""Overlap target correction with the independent accepted-prefix T sync."""
    caller_stream = torch.cuda.current_stream(device)
    accepted_token_tensor = torch.tensor([accepted_tokens], dtype=torch.long, device=device)
    corrective_tensor = torch.tensor([[corrective]], dtype=torch.long, device=device)
    recurrent_stream.wait_stream(caller_stream)

    with torch.cuda.stream(recurrent_stream):
        accepted_t_output, updated_t_cache = recurrent.forward_with_cache(
            accepted_anchor,
            model=get_base_causal_lm(model),
            past_key_value=true_t_cache,
            token_ids=accepted_token_tensor,
        )

    corrective_outputs = target_forward(
        model,
        corrective_tensor,
        past_key_values=target_cache,
        start_position=start_position,
        omit_attention_mask=omit_attention_mask,
    )
    caller_stream.wait_stream(recurrent_stream)
    accepted_t_output.record_stream(caller_stream)
    record_cache_stream(updated_t_cache, caller_stream)
    return corrective_outputs, accepted_t_output, updated_t_cache


def target_forward_with_preemptive_lookahead(
    model: torch.nn.Module,
    recurrent: torch.nn.Module,
    metadata: dict[str, Any],
    input_ids: torch.Tensor,
    *,
    past_key_values: DynamicCache,
    start_position: int,
    true_t_cache: DynamicCache,
    block_tokens: list[int],
    future_block_limit: int,
    omit_attention_mask: bool,
    device: torch.device,
    lookahead_stream: torch.cuda.Stream,
) -> tuple[Any, dict[str, Any]]:
    r"""Launch the all-accepted next-block branch while the target tail is running."""
    base_model = get_base_causal_lm(model)
    layers, _ = find_decoder_layers(base_model)
    anchor_idx = int(metadata.get("recurrent_hidden_state_index", metadata["anchor_layer"] + 1))
    hook_layer_idx = anchor_idx - 1
    if not 0 <= hook_layer_idx < len(layers):
        raise ValueError(f"Invalid recurrent boundary hook layer {hook_layer_idx}.")

    state: dict[str, Any] = {}
    block_token_tensor = torch.tensor([block_tokens], dtype=torch.long, device=device)

    def launch_lookahead(_module, _inputs, output):
        if state:
            return output
        anchor_block = output[0] if isinstance(output, tuple) else output
        producer_stream = torch.cuda.current_stream(device)
        lookahead_stream.wait_stream(producer_stream)
        anchor_block.record_stream(lookahead_stream)
        block_token_tensor.record_stream(lookahead_stream)
        record_cache_stream(true_t_cache, lookahead_stream)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(lookahead_stream):
            start_event.record(lookahead_stream)
            branch_cache = clone_cache(true_t_cache)
            synced_output, branch_cache = recurrent.forward_with_cache(
                anchor_block,
                model=base_model,
                past_key_value=branch_cache,
                token_ids=block_token_tensor,
            )
            synced_cache = clone_cache(branch_cache)
            synced_next_anchor = synced_output[:, -1:, :]

            first_logits = recurrent.boundary_logits(anchor_block[:, -1:, :], base_model)
            previous_token = first_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            lookahead_tokens = [previous_token]
            next_anchor = synced_next_anchor
            for rollout_step in range(2, future_block_limit + 1):
                draft_logits = recurrent.boundary_logits(next_anchor, base_model)
                draft_token = draft_logits[:, -1, :].argmax(dim=-1, keepdim=True)
                following_anchor, branch_cache = recurrent.forward_with_cache(
                    next_anchor,
                    model=base_model,
                    past_key_value=branch_cache,
                    token_ids=previous_token,
                    rollout_step=rollout_step,
                )
                lookahead_tokens.append(draft_token)
                previous_token = draft_token
                next_anchor = following_anchor[:, -1:, :]

            token_tensor = torch.cat(lookahead_tokens, dim=1)
            end_event.record(lookahead_stream)

        token_tensor.record_stream(lookahead_stream)
        synced_next_anchor.record_stream(lookahead_stream)
        record_cache_stream(synced_cache, lookahead_stream)
        state.update(
            {
                "tokens": token_tensor,
                "synced_t_cache": synced_cache,
                "synced_next_anchor": synced_next_anchor,
                "start_event": start_event,
                "end_event": end_event,
            }
        )
        return output

    hook = layers[hook_layer_idx].register_forward_hook(launch_lookahead)
    try:
        outputs = target_forward(
            model,
            input_ids,
            past_key_values=past_key_values,
            start_position=start_position,
            omit_attention_mask=omit_attention_mask,
        )
    finally:
        hook.remove()

    if not state:
        raise RuntimeError("The preemptive lookahead hook did not observe the recurrent boundary.")
    return outputs, state


def target_prefix_anchor_block(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: DynamicCache,
    start_position: int,
    anchor_idx: int,
) -> torch.Tensor:
    """Advance only layers below the recurrent boundary and return true anchors."""
    base = get_base_causal_lm(model)
    decoder = get_decoder(base)
    layers, _ = find_decoder_layers(base)
    embed_tokens = getattr(decoder, "embed_tokens", None) or getattr(decoder, "wte", None)
    if embed_tokens is None:
        raise ValueError("Decoder does not expose token embeddings for split target execution.")
    if not 0 < anchor_idx <= len(layers):
        raise ValueError(f"Invalid split recurrent boundary {anchor_idx} for {len(layers)} layers.")

    hidden_states = embed_tokens(input_ids)
    seq_len = hidden_states.size(1)
    position_ids = torch.arange(
        start_position,
        start_position + seq_len,
        dtype=torch.long,
        device=hidden_states.device,
    ).unsqueeze(0)
    cache_position = position_ids[0]
    causal_mask = make_projection_causal_mask(hidden_states, start_position)
    for layer_idx in range(anchor_idx):
        layer = layers[layer_idx]
        signature = inspect.signature(layer.forward)
        kwargs: dict[str, Any] = {
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "use_cache": True,
            "cache_position": cache_position,
        }
        if "past_key_values" in signature.parameters:
            kwargs["past_key_values"] = cache
        elif "past_key_value" in signature.parameters:
            kwargs["past_key_value"] = cache
        position_embeddings = maybe_position_embeddings(base, hidden_states, position_ids)
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        output = layer(hidden_states, **kwargs)
        hidden_states = output[0] if isinstance(output, tuple) else output
    return hidden_states


def rollout_recurrent_anchor_block(
    recurrent: torch.nn.Module,
    model: torch.nn.Module,
    initial_anchor: torch.Tensor,
    cache: DynamicCache,
    block_tokens: list[int],
) -> torch.Tensor:
    anchors = [initial_anchor]
    next_anchor = initial_anchor
    device = initial_anchor.device
    for token_index in range(1, len(block_tokens)):
        following_anchor, cache = recurrent.forward_with_cache(
            next_anchor,
            model=get_base_causal_lm(model),
            past_key_value=cache,
            token_ids=torch.tensor(
                [[block_tokens[token_index - 1]]], dtype=torch.long, device=device
            ),
            rollout_step=token_index + 1,
        )
        next_anchor = following_anchor[:, -1:, :]
        anchors.append(next_anchor)
    return torch.cat(anchors, dim=1)


def split_unchecked_target_forward(
    model: torch.nn.Module,
    metadata: dict[str, Any],
    input_ids: torch.Tensor,
    predicted_anchors: torch.Tensor,
    cache: DynamicCache,
    start_position: int,
    device: torch.device,
    projection_stream: torch.cuda.Stream | None = None,
) -> Any:
    """Build a hybrid cache from true prefix KV and T-predicted projection KV."""
    base = get_base_causal_lm(model)
    layers, _ = find_decoder_layers(base)
    anchor_idx = int(metadata.get("recurrent_hidden_state_index", metadata["anchor_layer"] + 1))
    projection_ids = list(metadata.get("projection_layer_ids") or metadata.get("tail_layer_ids") or [])
    if projection_ids != list(range(anchor_idx, len(layers))):
        raise ValueError(
            "Split target execution requires projection layers to cover every layer from "
            f"the recurrent boundary {anchor_idx} through {len(layers) - 1}; got {projection_ids}."
        )
    if predicted_anchors.size(1) != input_ids.size(1):
        raise ValueError("Split target execution requires one predicted anchor per committed token.")

    if projection_stream is None:
        true_anchors = target_prefix_anchor_block(
            model, input_ids, cache, start_position, anchor_idx
        )
        projected_logits = project_anchor_block(
            model, metadata, predicted_anchors, cache, start_position
        )
    else:
        caller_stream = torch.cuda.current_stream(device)
        projection_stream.wait_stream(caller_stream)
        predicted_anchors.record_stream(projection_stream)
        record_cache_stream(cache, projection_stream)
        with torch.cuda.stream(projection_stream):
            projected_logits = project_anchor_block(
                model, metadata, predicted_anchors, cache, start_position
            )
        true_anchors = target_prefix_anchor_block(
            model, input_ids, cache, start_position, anchor_idx
        )
        caller_stream.wait_stream(projection_stream)
        projected_logits.record_stream(caller_stream)
        record_cache_stream(cache, caller_stream)

    return SimpleNamespace(
        logits=projected_logits,
        hidden_states={anchor_idx: true_anchors},
        past_key_values=cache,
    )


def project_anchor_step(
    model: torch.nn.Module,
    metadata: dict[str, Any],
    anchor_hidden: torch.Tensor,
    cache: DynamicCache,
    position: int,
) -> torch.Tensor:
    base = get_base_causal_lm(model)
    layers, _ = find_decoder_layers(base)
    hidden_states = anchor_hidden
    position_ids = torch.tensor([[position]], dtype=torch.long, device=hidden_states.device)
    cache_position = torch.tensor([position], dtype=torch.long, device=hidden_states.device)
    projection_ids = metadata.get("projection_layer_ids") or metadata.get("tail_layer_ids")
    for layer_idx in projection_ids:
        layer = layers[layer_idx]
        signature = inspect.signature(layer.forward)
        kwargs: dict[str, Any] = {
            "position_ids": position_ids,
            "use_cache": True,
            "cache_position": cache_position,
        }
        if "past_key_values" in signature.parameters:
            kwargs["past_key_values"] = cache
        elif "past_key_value" in signature.parameters:
            kwargs["past_key_value"] = cache
        position_embeddings = maybe_position_embeddings(base, hidden_states, position_ids)
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        output = layer(hidden_states, **kwargs)
        hidden_states = output[0] if isinstance(output, tuple) else output

    decoder = get_decoder(base)
    norm = getattr(decoder, "norm", None) or getattr(decoder, "ln_f", None)
    if norm is not None:
        hidden_states = norm(hidden_states)
    return base.get_output_embeddings()(hidden_states[:, -1, :])


def make_projection_causal_mask(hidden_states: torch.Tensor, past_length: int) -> torch.Tensor:
    batch_size, query_length = hidden_states.shape[:2]
    key_length = past_length + query_length
    query_positions = torch.arange(
        past_length, past_length + query_length, device=hidden_states.device
    )[:, None]
    key_positions = torch.arange(key_length, device=hidden_states.device)[None, :]
    allowed = key_positions <= query_positions
    min_dtype = torch.finfo(hidden_states.dtype).min
    mask = torch.zeros((query_length, key_length), dtype=hidden_states.dtype, device=hidden_states.device)
    mask = mask.masked_fill(~allowed, min_dtype)
    return mask[None, None, :, :].expand(batch_size, 1, query_length, key_length).clone()


def project_anchor_block(
    model: torch.nn.Module,
    metadata: dict[str, Any],
    anchor_hidden: torch.Tensor,
    cache: DynamicCache,
    start_position: int,
) -> torch.Tensor:
    base = get_base_causal_lm(model)
    layers, _ = find_decoder_layers(base)
    hidden_states = anchor_hidden
    seq_len = hidden_states.size(1)
    position_ids = torch.arange(
        start_position, start_position + seq_len, dtype=torch.long, device=hidden_states.device
    ).unsqueeze(0)
    cache_position = torch.arange(
        start_position, start_position + seq_len, dtype=torch.long, device=hidden_states.device
    )
    causal_mask = make_projection_causal_mask(hidden_states, start_position)
    projection_ids = metadata.get("projection_layer_ids") or metadata.get("tail_layer_ids")
    for layer_idx in projection_ids:
        layer = layers[layer_idx]
        signature = inspect.signature(layer.forward)
        kwargs: dict[str, Any] = {
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "use_cache": True,
            "cache_position": cache_position,
        }
        if "past_key_values" in signature.parameters:
            kwargs["past_key_values"] = cache
        elif "past_key_value" in signature.parameters:
            kwargs["past_key_value"] = cache
        position_embeddings = maybe_position_embeddings(base, hidden_states, position_ids)
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        output = layer(hidden_states, **kwargs)
        hidden_states = output[0] if isinstance(output, tuple) else output

    decoder = get_decoder(base)
    norm = getattr(decoder, "norm", None) or getattr(decoder, "ln_f", None)
    if norm is not None:
        hidden_states = norm(hidden_states)
    return base.get_output_embeddings()(hidden_states)


def choose_block_limit(args: argparse.Namespace, generated_length: int) -> int:
    if args.mode == "fixed":
        return args.max_block_tokens
    if generated_length < args.medium_position:
        return min(args.short_block, args.max_block_tokens)
    if generated_length < args.long_position:
        return min(args.medium_block, args.max_block_tokens)
    return min(args.long_block, args.max_block_tokens)


def verifier_threshold_for_position(args: argparse.Namespace, generated_position: int) -> float:
    if generated_position >= args.long_position and args.verifier_threshold_late is not None:
        return args.verifier_threshold_late
    if generated_position >= args.medium_position and args.verifier_threshold_mid is not None:
        return args.verifier_threshold_mid
    if args.verifier_threshold_early is not None:
        return args.verifier_threshold_early
    return args.verifier_threshold


def verifier_previous_score_rejects(
    scores: list[float | None],
    thresholds: list[float | None],
    default_threshold: float,
) -> bool:
    """Return whether an already-scored draft should stop the rollout."""
    if not scores or not thresholds or scores[-1] is None:
        return False
    threshold = thresholds[-1] if thresholds[-1] is not None else default_threshold
    return scores[-1] < threshold


def verification_risk_budget_decision(
    survival_probability: float,
    conditional_acceptance_probability: float,
    risk_budget: float,
    current_block_tokens: int,
    min_block_tokens: int,
) -> tuple[bool, float, float]:
    """Decide whether adding one more tentative draft would exceed the risk budget."""
    if not 0.0 <= survival_probability <= 1.0:
        raise ValueError("survival_probability must be between 0 and 1.")
    if not 0.0 <= conditional_acceptance_probability <= 1.0:
        raise ValueError("conditional_acceptance_probability must be between 0 and 1.")
    if not 0.0 <= risk_budget <= 1.0:
        raise ValueError("risk_budget must be between 0 and 1.")
    next_survival = survival_probability * conditional_acceptance_probability
    next_risk = 1.0 - next_survival
    should_stop = current_block_tokens >= min_block_tokens and next_risk > risk_budget
    return should_stop, next_survival, next_risk


def draft_margin_threshold_for_position(
    base_threshold: float,
    generated_position: int,
    medium_position: int,
    long_position: int,
    *,
    early_threshold: float | None = None,
    mid_threshold: float | None = None,
    late_threshold: float | None = None,
) -> float:
    if generated_position >= long_position and late_threshold is not None:
        return late_threshold
    if generated_position >= medium_position and mid_threshold is not None:
        return mid_threshold
    if early_threshold is not None:
        return early_threshold
    return base_threshold


def min_draft_margin_for_position(args: argparse.Namespace, generated_position: int) -> float:
    return draft_margin_threshold_for_position(
        args.min_draft_margin,
        generated_position,
        args.medium_position,
        args.long_position,
        early_threshold=args.min_draft_margin_early,
        mid_threshold=args.min_draft_margin_mid,
        late_threshold=args.min_draft_margin_late,
    )


def target_match_lambda_for_position(args: argparse.Namespace, generated_position: int) -> float:
    if generated_position >= args.long_position and args.target_match_lambda_late is not None:
        return args.target_match_lambda_late
    if generated_position >= args.medium_position and args.target_match_lambda_mid is not None:
        return args.target_match_lambda_mid
    if args.target_match_lambda_early is not None:
        return args.target_match_lambda_early
    return args.target_match_lambda


def target_match_accepts_candidate(
    match: bool,
    target_relative_support: float | None,
    lambda_for_token: float,
) -> bool:
    """Apply the target-match schedule, with lambda=1 as the exact anchor."""
    if lambda_for_token <= 0.0:
        return True
    if lambda_for_token >= 1.0:
        return bool(match)
    if target_relative_support is None:
        raise ValueError("Target relative support is required when 0 < lambda < 1.")
    return target_relative_support >= lambda_for_token


def mismatch_entropy_for_position(
    args: argparse.Namespace,
    generated_position: int,
) -> float | None:
    if (
        generated_position >= args.long_position
        and args.mismatch_min_target_entropy_late is not None
    ):
        return args.mismatch_min_target_entropy_late
    if (
        generated_position >= args.medium_position
        and args.mismatch_min_target_entropy_mid is not None
    ):
        return args.mismatch_min_target_entropy_mid
    if args.mismatch_min_target_entropy_early is not None:
        return args.mismatch_min_target_entropy_early
    return args.mismatch_min_target_entropy


def token_category_from_text(token_text: str) -> str:
    if "\n" in token_text:
        return "newline"
    stripped = token_text.strip()
    if not stripped:
        return "space"
    if any(ch.isdigit() for ch in stripped):
        return "number"
    if all(ch in "+-*/=<>^%.,:;()[]{}$\\|_#" for ch in stripped):
        return "symbol"
    if any(ch.isalpha() for ch in stripped):
        return "word"
    return "other"


def allowed_category_prefix_length(
    categories: list[str],
    allowed_categories: set[str],
) -> int:
    """Return the number of leading categories allowed by a precommit gate."""
    for index, category in enumerate(categories):
        if category not in allowed_categories:
            return index
    return len(categories)


def parse_category_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def count_accepted_draft_margin_records(
    records: list[dict[str, Any]],
    min_margin: float,
    max_unchecked_per_sequence: int | None,
    unchecked_budget_used: int,
    *,
    max_unchecked_per_window: int | None = None,
    unchecked_budget_window_tokens: int | None = None,
    accepted_unchecked_positions: list[int] | None = None,
    margin_threshold_for_position: Callable[[int], float] | None = None,
) -> int:
    accepted = 0
    accepted_positions = accepted_unchecked_positions or []
    pending_positions: list[int] = []
    for record in records:
        if record.get("draft_category_allowed") is False:
            break
        if (
            max_unchecked_per_sequence is not None
            and unchecked_budget_used + accepted >= max_unchecked_per_sequence
        ):
            break
        if max_unchecked_per_window is not None:
            if unchecked_budget_window_tokens is None:
                raise ValueError("unchecked_budget_window_tokens is required for a window budget")
            generated_position = int(record["generated_position"])
            window = generated_position // unchecked_budget_window_tokens
            used_in_window = sum(
                int(position) // unchecked_budget_window_tokens == window
                for position in accepted_positions + pending_positions
            )
            if used_in_window >= max_unchecked_per_window:
                break
        required_margin = min_margin
        if margin_threshold_for_position is not None:
            required_margin = margin_threshold_for_position(int(record["generated_position"]))
        if record.get("margin") is None or record["margin"] < required_margin:
            break
        accepted += 1
        if max_unchecked_per_window is not None:
            pending_positions.append(int(record["generated_position"]))
    return accepted


def limit_target_match_expand_records(
    records: list[dict[str, Any]],
    proposed_drafts: int,
    accepted_mismatches: int,
    max_accepted_mismatches: int | None,
) -> int:
    """Trim an expanded prefix at the first mismatch that exceeds its sequence budget."""
    accepted = 0
    mismatch_budget_used = accepted_mismatches
    for record in records[:proposed_drafts]:
        if record.get("draft_category_allowed") is False:
            break
        if not record["match"]:
            if (
                max_accepted_mismatches is not None
                and mismatch_budget_used >= max_accepted_mismatches
            ):
                break
            mismatch_budget_used += 1
        accepted += 1
    return accepted


def precommit_unchecked_prefix_length(
    policy: str,
    block_margins: list[float | None],
    generated_start: int,
    min_margin: float,
    max_unchecked_per_sequence: int | None,
    unchecked_budget_used: int,
    *,
    max_unchecked_per_window: int | None = None,
    unchecked_budget_window_tokens: int | None = None,
    accepted_unchecked_positions: list[int] | None = None,
    margin_threshold_for_position: Callable[[int], float] | None = None,
    ngram_draft_allowed: list[bool] | None = None,
) -> int | None:
    """Return the target-input prefix length for an unchecked latent policy."""
    if policy == "whole_block":
        return 1 + len(block_margins)
    if policy == "ngram_prefix":
        accepted = 0
        pending_positions: list[int] = []
        for draft_index in range(len(block_margins)):
            if ngram_draft_allowed is not None and (
                draft_index >= len(ngram_draft_allowed)
                or not ngram_draft_allowed[draft_index]
            ):
                break
            if (
                max_unchecked_per_sequence is not None
                and unchecked_budget_used + accepted >= max_unchecked_per_sequence
            ):
                break
            generated_position = generated_start + draft_index + 1
            if max_unchecked_per_window is not None:
                if unchecked_budget_window_tokens is None:
                    raise ValueError("unchecked_budget_window_tokens is required for a window budget")
                window = generated_position // unchecked_budget_window_tokens
                used_in_window = sum(
                    int(position) // unchecked_budget_window_tokens == window
                    for position in (accepted_unchecked_positions or []) + pending_positions
                )
                if used_in_window >= max_unchecked_per_window:
                    break
                pending_positions.append(generated_position)
            accepted += 1
        return 1 + accepted
    if policy != "draft_margin":
        return None
    records = [
        {
            "margin": margin,
            "generated_position": generated_start + draft_index + 1,
        }
        for draft_index, margin in enumerate(block_margins)
    ]
    return 1 + count_accepted_draft_margin_records(
        records,
        min_margin,
        max_unchecked_per_sequence,
        unchecked_budget_used,
        max_unchecked_per_window=max_unchecked_per_window,
        unchecked_budget_window_tokens=unchecked_budget_window_tokens,
        accepted_unchecked_positions=accepted_unchecked_positions,
        margin_threshold_for_position=margin_threshold_for_position,
    )


def unchecked_budget_exhausted_for_position(
    generated_position: int,
    max_unchecked_per_sequence: int | None,
    unchecked_budget_used: int,
    *,
    max_unchecked_per_window: int | None = None,
    unchecked_budget_window_tokens: int | None = None,
    accepted_unchecked_positions: list[int] | None = None,
) -> bool:
    """Return whether an unchecked latent draft cannot be accepted here."""
    if (
        max_unchecked_per_sequence is not None
        and unchecked_budget_used >= max_unchecked_per_sequence
    ):
        return True
    if max_unchecked_per_window is None:
        return False
    if unchecked_budget_window_tokens is None:
        raise ValueError("unchecked_budget_window_tokens is required for a window budget")
    window = generated_position // unchecked_budget_window_tokens
    used_in_window = sum(
        int(position) // unchecked_budget_window_tokens == window
        for position in (accepted_unchecked_positions or [])
    )
    return used_in_window >= max_unchecked_per_window


VALID_TOKEN_CATEGORIES = {"space", "symbol", "number", "word", "newline", "other"}


def parse_category_float_map(value: str, *, flag_name: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{flag_name} entries must be category:value, got {item!r}.")
        category, raw_value = item.split(":", 1)
        category = category.strip()
        if category not in VALID_TOKEN_CATEGORIES:
            raise ValueError(
                f"Unknown token category {category!r} in {flag_name}; "
                f"valid categories are {sorted(VALID_TOKEN_CATEGORIES)}."
            )
        parsed[category] = float(raw_value)
    return parsed


def parse_nonnegative_int_map(value: str, *, flag_name: str) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{flag_name} entries must be integer:integer, got {item!r}.")
        raw_key, raw_value = item.split(":", 1)
        key = int(raw_key)
        mapped_value = int(raw_value)
        if key <= 0 or mapped_value < 0:
            raise ValueError(f"{flag_name} keys must be positive and values non-negative.")
        parsed[key] = mapped_value
    return parsed


def parse_position_width_schedule(value: str, *, flag_name: str) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{flag_name} entries must be position:width, got {item!r}.")
        raw_position, raw_width = item.split(":", 1)
        position = int(raw_position)
        width = int(raw_width)
        if position < 0 or width <= 0:
            raise ValueError(f"{flag_name} positions must be non-negative and widths positive.")
        parsed[position] = width
    return parsed


def ngram_width_for_position(
    default_width: int,
    generated_position: int,
    width_schedule: dict[int, int],
) -> int:
    width = default_width
    for position in sorted(width_schedule):
        if generated_position < position:
            break
        width = width_schedule[position]
    return width


def ngram_wide_route_decision(
    observations: int,
    extra_accepted_tokens: int,
    after_hits: int,
    min_extra_per_hit: float,
) -> tuple[bool, float | None]:
    if after_hits <= 0 or observations < after_hits:
        return False, None
    score = extra_accepted_tokens / max(1, observations)
    return score < min_extra_per_hit, score


def ngram_predecode_cost_route(
    *,
    enabled: bool,
    prompt_tokens: int,
    requested_output_tokens: int,
    has_position_schedule: bool,
    medium_prompt_tokens: int,
    long_prompt_tokens: int,
    short_output_tokens: int,
    narrow_width: int,
    medium_width: int,
    wide_width: int,
) -> tuple[int | None, str]:
    if not enabled:
        return None, "disabled"
    if has_position_schedule:
        return None, "explicit_position_schedule"
    if requested_output_tokens > short_output_tokens:
        if medium_prompt_tokens <= prompt_tokens < long_prompt_tokens:
            return medium_width, "medium_prompt_long_output"
        return narrow_width, "short_or_long_prompt_long_output"
    if prompt_tokens >= long_prompt_tokens:
        return wide_width, "long_prompt_short_output"
    return narrow_width, "short_prompt_narrow"


def ngram_predecode_occurrence_route(
    *,
    enabled: bool,
    prompt_tokens: int,
    requested_output_tokens: int,
    has_position_schedule: bool,
    medium_prompt_tokens: int,
    long_prompt_tokens: int,
    short_output_tokens: int,
) -> tuple[str | None, str]:
    if not enabled:
        return None, "disabled"
    if has_position_schedule and requested_output_tokens <= short_output_tokens:
        return None, "explicit_position_schedule"
    if requested_output_tokens > short_output_tokens:
        if medium_prompt_tokens <= prompt_tokens < long_prompt_tokens:
            return "prompt_consensus", "medium_prompt_long_output"
        if prompt_tokens >= long_prompt_tokens:
            return "consensus", "long_prompt_long_output"
        reason = (
            "explicit_position_schedule_long_output"
            if has_position_schedule
            else "short_prompt_long_output"
        )
        return "consensus", reason
    if medium_prompt_tokens <= prompt_tokens < long_prompt_tokens:
        return "consensus", "medium_prompt_short_output"
    return "latest", "short_or_long_prompt_short_output"


def limit_ngram_draft_tokens(
    tokens: list[int],
    ngram_size: int | None,
    draft_tokens_by_match_size: dict[int, int],
) -> list[int]:
    if ngram_size is None or ngram_size not in draft_tokens_by_match_size:
        return tokens
    return tokens[: draft_tokens_by_match_size[ngram_size]]


def ngram_unchecked_gate_allows(
    match_size: int | None,
    generated_position: int,
    min_match_size: int,
    min_position: int,
) -> bool:
    """Return whether a prompt-lookup cycle may use its bounded unchecked path."""
    return bool(
        min_match_size > 0
        and match_size is not None
        and match_size >= min_match_size
        and generated_position >= min_position
    )


def infer_verifier_tag(verifier_weights: str | None) -> str | None:
    if not verifier_weights:
        return None
    text = verifier_weights.lower()
    for tag in ("cont15x", "cont10x", "cont5x"):
        if tag in text:
            return tag
    return Path(verifier_weights).parent.name or None


def next_token_distribution_stats(logits: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits.float(), dim=-1)
    top_values = torch.topk(probs, k=2, dim=-1).values[0]
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
    top1 = float(top_values[0].item())
    top2 = float(top_values[1].item())
    return {
        "entropy": entropy,
        "top1_prob": top1,
        "top1_margin": top1 - top2,
    }


def top_token_and_logit_margin(logits: torch.Tensor) -> tuple[int, float]:
    """Read the target top token and top-2 margin with one device-to-host transfer."""
    top_values, top_indices = torch.topk(logits[0].float(), k=2)
    # topk tie order is not argmax tie order. Target token selection must match greedy.
    packed = torch.cat((top_values, logits[0].argmax().reshape(1).to(top_values.dtype))).cpu().tolist()
    return int(packed[2]), float(packed[0] - packed[1])


def exact_id_diagnostics_allowed(args: argparse.Namespace, policy: str, need_margin: bool) -> bool:
    if not getattr(args, "fast_strict_diagnostics", False) or not args.compact_runtime_stats:
        return False
    if policy != "target_match" or need_margin or args.target_match_lambda != 1.0:
        return False
    if getattr(args, "target_match_category_lambdas", ""):
        return False
    for phase in ("early", "mid", "late"):
        if getattr(args, "target_match_lambda_" + phase, None) not in (None, 1.0):
            return False
    return all(getattr(args, name, None) is None for name in (
        "mismatch_min_target_entropy", "mismatch_min_target_entropy_early",
        "mismatch_min_target_entropy_mid", "mismatch_min_target_entropy_late"))


def latent_target_margin_allows(threshold: float, margin: float | None) -> bool:
    """Return whether a latent T attempt passes the target-confidence gate."""
    return threshold <= 0.0 or (margin is not None and margin >= threshold)


def latent_margin_gate_route_update(
    gated: bool,
    consecutive_skips: int,
    close_after: int,
) -> tuple[int, bool]:
    """Update the request-level margin-gate streak and decide whether to close T."""
    if not gated:
        return 0, False
    next_skips = consecutive_skips + 1
    return next_skips, close_after > 0 and next_skips >= close_after


def auto_route_monitor_decision(
    observations: int,
    successes: int,
    next_evaluation: int,
    min_acceptance: float,
    reevaluate_every: int,
) -> tuple[bool, bool, int, float | None]:
    """Evaluate a one-shot or periodically monitored request-level latent route."""
    if next_evaluation <= 0 or observations < next_evaluation:
        return False, False, next_evaluation, None
    score = successes / max(1, observations)
    disable = score < min_acceptance
    following_evaluation = (
        observations + reevaluate_every if not disable and reevaluate_every > 0 else 0
    )
    return True, disable, following_evaluation, score


def cheap_verifier_decision(
    args: argparse.Namespace,
    generated_position: int,
    next_logits: torch.Tensor,
    base_block_limit: int,
    anchor_token: int | None = None,
    anchor_text: str | None = None,
    target_top1_margin: float | None = None,
) -> dict[str, Any]:
    anchor_category = token_category_from_text(anchor_text or "") if anchor_text is not None else None
    decision: dict[str, Any] = {
        "policy": args.cheap_verifier_policy,
        "generated_position": generated_position,
        "anchor_token": anchor_token,
        "anchor_text": anchor_text,
        "anchor_category": anchor_category,
        "target_entropy": None,
        "target_top1_prob": None,
        "target_top1_margin": None,
        "dynamic_threshold": None,
        "skip": False,
        "skip_reason": None,
        "block_limit": base_block_limit,
    }
    if args.cheap_verifier_policy == "none":
        return decision

    rollout_categories = parse_category_set(args.cheap_verifier_rollout_anchor_categories)
    if rollout_categories:
        in_window = generated_position >= args.cheap_verifier_rollout_min_position
        if args.cheap_verifier_rollout_max_position is not None:
            in_window = in_window and generated_position < args.cheap_verifier_rollout_max_position
        if not in_window:
            decision["skip"] = True
            decision["skip_reason"] = "anchor_position"
        elif anchor_category not in rollout_categories:
            decision["skip"] = True
            decision["skip_reason"] = "anchor_category"

    if args.cheap_verifier_policy == "anchor_category":
        if decision["skip"]:
            decision["block_limit"] = 1
        return decision

    if args.cheap_verifier_policy == "logit_margin":
        if target_top1_margin is None:
            _, target_top1_margin = top_token_and_logit_margin(next_logits)
        margin = float(target_top1_margin)
        decision["target_top1_margin"] = margin
        if args.cheap_verifier_margin_skip_below > 0.0 and margin < args.cheap_verifier_margin_skip_below:
            decision["skip"] = True
            decision["skip_reason"] = "logit_margin"
            decision["block_limit"] = 1
        elif (
            args.cheap_verifier_margin_block2_below > 0.0
            and margin < args.cheap_verifier_margin_block2_below
        ):
            decision["block_limit"] = min(base_block_limit, 2)
        return decision

    stats = next_token_distribution_stats(next_logits)
    decision["target_entropy"] = stats["entropy"]
    decision["target_top1_prob"] = stats["top1_prob"]
    decision["target_top1_margin"] = stats["top1_margin"]

    entropy = stats["entropy"]
    if entropy <= args.cheap_verifier_entropy_low:
        threshold = args.cheap_verifier_threshold_low
    elif entropy <= args.cheap_verifier_entropy_mid:
        threshold = args.cheap_verifier_threshold_mid
    elif entropy <= args.cheap_verifier_entropy_high:
        threshold = args.cheap_verifier_threshold_high
    else:
        threshold = args.cheap_verifier_threshold_very_high

    decision["dynamic_threshold"] = threshold
    if args.cheap_verifier_top1_skip_min > 0.0 and stats["top1_prob"] < args.cheap_verifier_top1_skip_min:
        decision["skip"] = True
        decision["skip_reason"] = "top1_prob"
    if threshold > args.cheap_verifier_skip_threshold:
        decision["skip"] = True
        decision["skip_reason"] = "threshold"

    if decision["skip"]:
        decision["block_limit"] = 1
    elif args.cheap_verifier_dynamic_block:
        if threshold <= args.cheap_verifier_threshold_low:
            block = args.cheap_verifier_low_t_block
        elif threshold <= args.cheap_verifier_threshold_mid:
            block = args.cheap_verifier_mid_t_block
        elif threshold <= args.cheap_verifier_threshold_high:
            block = args.cheap_verifier_high_t_block
        else:
            block = args.cheap_verifier_very_high_t_block
        decision["block_limit"] = max(1, min(base_block_limit, block, args.max_block_tokens))

    return decision


@torch.no_grad()
def greedy_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | list[int] | None,
    anchor_idx: int,
    device: torch.device,
    omit_target_attention_mask: bool = False,
) -> dict[str, Any]:
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
    generated: list[int] = []
    timings = {"prefill_s": 0.0, "decode_s": 0.0}

    outputs, timings["prefill_s"] = timed(
        lambda: target_forward(
            model,
            input_ids,
            start_position=0,
            output_hidden_states=False,
            omit_attention_mask=omit_target_attention_mask,
        ),
        device,
    )
    cache = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :]

    while len(generated) < max_new_tokens:
        token = int(next_logits.argmax(dim=-1).item())
        generated.append(token)
        if token in eos_ids or len(generated) >= max_new_tokens:
            break
        token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
        outputs, elapsed = timed(
            lambda: target_forward(
                model,
                token_tensor,
                past_key_values=cache,
                start_position=input_ids.size(1) + len(generated) - 1,
                output_hidden_states=False,
                omit_attention_mask=omit_target_attention_mask,
            ),
            device,
        )
        timings["decode_s"] += elapsed
        cache = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

    return {
        "token_ids": generated,
        "timings": timings,
        "target_calls": max(1, len(generated)),
    }


@torch.no_grad()
def speculative_decode(
    model: torch.nn.Module,
    recurrent: torch.nn.Module,
    metadata: dict[str, Any],
    input_ids: torch.Tensor,
    args: argparse.Namespace,
    eos_token_id: int | list[int] | None,
    device: torch.device,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
    anchor_idx = metadata.get("recurrent_hidden_state_index", metadata["anchor_layer"] + 1)
    projection_ids = metadata.get("projection_layer_ids") or metadata.get("tail_layer_ids")
    prompt_token_ids = [int(token) for token in input_ids[0].tolist()]
    recurrent_active = args.ngram_draft_mode != "only"
    generated: list[int] = []
    ngram_index = (
        SuffixNgramIndex(prompt_token_ids, args.ngram_min_size, args.ngram_max_size)
        if args.indexed_ngram_lookup and args.ngram_draft_mode != "off"
        else None
    )
    ngram_index_generated_tokens = 0
    draft_category_gate = parse_category_set(args.draft_token_category_gate)
    latent_unchecked_categories = parse_category_set(args.latent_unchecked_token_categories)
    target_match_category_lambdas = parse_category_float_map(
        args.target_match_category_lambdas,
        flag_name="--target-match-category-lambdas",
    )
    ngram_draft_tokens_by_match_size = parse_nonnegative_int_map(
        args.ngram_draft_tokens_by_match_size,
        flag_name="--ngram-draft-tokens-by-match-size",
    )
    ngram_max_draft_tokens_by_position = parse_position_width_schedule(
        args.ngram_max_draft_tokens_by_position,
        flag_name="--ngram-max-draft-tokens-by-position",
    )
    ngram_predecode_width, ngram_predecode_reason = ngram_predecode_cost_route(
        enabled=args.ngram_predecode_cost_route,
        prompt_tokens=len(prompt_token_ids),
        requested_output_tokens=args.max_new_tokens,
        has_position_schedule=bool(ngram_max_draft_tokens_by_position),
        medium_prompt_tokens=args.ngram_predecode_medium_prompt_tokens,
        long_prompt_tokens=args.ngram_predecode_long_prompt_tokens,
        short_output_tokens=args.ngram_predecode_short_output_tokens,
        narrow_width=args.ngram_predecode_narrow_width,
        medium_width=args.ngram_predecode_medium_width,
        wide_width=args.ngram_predecode_wide_width,
    )
    routed_occurrence_policy, ngram_predecode_occurrence_reason = (
        ngram_predecode_occurrence_route(
            enabled=args.ngram_predecode_occurrence_route,
            prompt_tokens=len(prompt_token_ids),
            requested_output_tokens=args.max_new_tokens,
            has_position_schedule=bool(ngram_max_draft_tokens_by_position),
            medium_prompt_tokens=args.ngram_predecode_medium_prompt_tokens,
            long_prompt_tokens=args.ngram_predecode_long_prompt_tokens,
            short_output_tokens=args.ngram_predecode_short_output_tokens,
        )
    )
    active_ngram_occurrence_policy = (
        routed_occurrence_policy or args.ngram_occurrence_policy
    )
    ngram_unchecked_categories = parse_category_set(args.ngram_unchecked_token_categories)
    timings = {
        "prefill_s": 0.0,
        "t_init_s": 0.0,
        "t_sync_s": 0.0,
        "t_step_s": 0.0,
        "tail_s": 0.0,
        "boundary_s": 0.0,
        "draft_parallel_s": 0.0,
        "correction_tsync_parallel_s": 0.0,
        "preemptive_compute_s": 0.0,
        "probe_s": 0.0,
        "ngram_lookup_s": 0.0,
        "ngram_tree_gather_s": 0.0,
        "verify_s": 0.0,
        "acceptance_diagnostics_s": 0.0,
        "target_match_compare_s": 0.0,
        "acceptance_record_s": 0.0,
        "acceptance_policy_s": 0.0,
        "split_anchor_s": 0.0,
        "split_forward_s": 0.0,
        "refine_s": 0.0,
        "correction_s": 0.0,
    }
    stats = {
        "cycles": 0,
        "target_calls": 1,
        "draft_tokens": 0,
        "accepted_draft_tokens": 0,
        "matched_draft_tokens": 0,
        "target_match_observations": 0,
        "accepted_mismatch_tokens": 0,
        "accepted_mismatch_positions": [],
        "accepted_mismatch_events": [],
        "accepted_ngram_mismatch_tokens": 0,
        "accepted_latent_mismatch_tokens": 0,
        "accepted_unchecked_tokens": 0,
        "accepted_unchecked_positions": [],
        "accepted_unchecked_events": [],
        "precommit_unchecked_blocks": 0,
        "precommit_trimmed_draft_tokens": 0,
        "last_token_logits_only_blocks": 0,
        "split_unchecked_blocks": 0,
        "split_unchecked_tokens": 0,
        "split_parallel_blocks": 0,
        "split_unchecked_gate_skips": 0,
        "target_match_expand_events": 0,
        "target_match_expand_extra_tokens": 0,
        "target_cache_replays": 0,
        "fast_correction_cache_reuses": 0,
        "deferred_corrective_tokens": 0,
        "serial_target_commits": 0,
        "serial_fallback_events": 0,
        "serial_fallback_token_changes": 0,
        "full_prefix_replays": 0,
        "cycle_start_fallback_events": 0,
        "cycle_start_fallback_token_changes": 0,
        "refinement_passes": 0,
        "corrective_tokens": 0,
        "block_hist": {},
        "accepted_hist": {},
        "block_hist_by_phase": {"early": {}, "mid": {}, "late": {}},
        "accepted_hist_by_phase": {"early": {}, "mid": {}, "late": {}},
        "verifier_scores": [],
        "verifier_conditional_scores": [],
        "verifier_survival_scores": [],
        "target_relative_supports": [],
        "rejected_draft_tokens": 0,
        "rejected_draft_positions": [],
        "draft_margins": [],
        "draft_records": [],
        "accelerated_positions": [],
        "target_generated_positions": [],
        "block_records": [],
        "cheap_policy_records": [],
        "cheap_policy_evaluations": 0,
        "cheap_policy_skips": 0,
        "cheap_policy_dynamic_thresholds": [],
        "cheap_policy_target_margins": [],
        "verifier_threshold_skips": 0,
        "preemptive_attempts": 0,
        "preemptive_sync_reuses": 0,
        "preemptive_token_hits": 0,
        "preemptive_token_misses": 0,
        "preemptive_branch_discards": 0,
        "preemptive_gate_skips": 0,
        "preemptive_reused_draft_tokens": 0,
        "ngram_attempts": 0,
        "ngram_hits": 0,
        "fast_strict_ngram_diagnostic_blocks": 0,
        "fast_strict_target_diagnostic_blocks": 0,
        "ngram_draft_tokens": 0,
        "accepted_ngram_tokens": 0,
        "ngram_tree_cycles": 0,
        "ngram_tree_branches": 0,
        "ngram_tree_nodes": 0,
        "ngram_tree_selected_nodes": 0,
        "ngram_tree_target_matched_drafts": 0,
        "ngram_wide_route_evaluations": 0,
        "ngram_wide_route_disables": 0,
        "ngram_wide_route_observations": 0,
        "ngram_wide_route_extra_accepted_tokens": 0,
        "ngram_wide_route_score": None,
        "ngram_wide_route_disable_position": None,
        "ngram_wide_route_fallback_cycles": 0,
        "ngram_predecode_route_width": ngram_predecode_width,
        "ngram_predecode_route_reason": ngram_predecode_reason,
        "ngram_predecode_occurrence_policy": active_ngram_occurrence_policy,
        "ngram_predecode_occurrence_reason": ngram_predecode_occurrence_reason,
        "ngram_unchecked_gate_evaluations": 0,
        "ngram_unchecked_gate_eligible": 0,
        "ngram_unchecked_budget_skips": 0,
        "ngram_unchecked_category_gate_evaluations": 0,
        "ngram_unchecked_category_gate_rejects": 0,
        "ngram_unchecked_category_trimmed_tokens": 0,
        "ngram_unchecked_category_budget_closes": 0,
        "ngram_unchecked_category_closed_skips": 0,
        "lazy_t_sync_queued_tokens": 0,
        "lazy_t_sync_flushes": 0,
        "lazy_t_sync_flushed_tokens": 0,
        "deferred_t_initializations": 0,
        "deferred_t_init_tokens": 0,
        "latent_position_gate_skips": 0,
        "latent_target_margin_gate_evaluations": 0,
        "latent_target_margin_gate_skips": 0,
        "latent_target_margin_gate_closes": 0,
        "latent_target_margin_gate_close_position": None,
        "latent_target_margins": [],
        "latent_unchecked_budget_skips": 0,
        "latent_unchecked_category_gate_evaluations": 0,
        "latent_unchecked_category_gate_rejects": 0,
        "latent_unchecked_category_trimmed_tokens": 0,
        "latent_unchecked_category_budget_closes": 0,
        "latent_unchecked_category_closed_skips": 0,
        "draft_cooldown_skips": 0,
        "draft_cooldown_activations": 0,
        "auto_route_evaluations": 0,
        "auto_route_t_disables": 0,
        "auto_route_early_evaluations": 0,
        "auto_route_early_t_disables": 0,
        "auto_route_disable_position": None,
        "auto_route_observed_acceptance": None,
        "auto_route_latent_drafts": 0,
        "auto_route_accepted_drafts": 0,
        "auto_route_latent_candidate_cycles": 0,
        "auto_route_confident_candidate_cycles": 0,
        "auto_route_score_kind": None,
        "missing_t_anchor_fallbacks": 0,
        "skipped_terminal_t_steps": 0,
        "verification_intervals": [],
        "verification_risk_budget_stops": 0,
        "verification_risk_at_stops": [],
        "verification_rollbacks": 0,
        "verification_wasted_drafts": 0,
        "verification_first_divergence_events": [],
        "verification_agreement_shadow_probes": 0,
        "verification_agreement_shadow_matches": 0,
        "verification_agreement_gate_openings": 0,
        "verification_agreement_gate_closures": 0,
        "verification_agreement_active_cycles": 0,
        "verification_agreement_max_streak": 0,
        "verification_agreement_events": [],
    }
    draft_streams = (
        (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
        if args.single_gpu_parallel_draft and device.type == "cuda"
        else None
    )
    correction_sync_stream = (
        torch.cuda.Stream(device=device)
        if args.single_gpu_parallel_correction_sync and device.type == "cuda"
        else None
    )
    lookahead_stream = (
        torch.cuda.Stream(device=device)
        if args.preemptive_lookahead and device.type == "cuda"
        else None
    )
    split_projection_stream = (
        torch.cuda.Stream(device=device)
        if args.split_unchecked_parallel and device.type == "cuda"
        else None
    )
    pending_lookahead_tokens: list[int] | None = None
    pending_t_sync_anchors: list[torch.Tensor] = []
    pending_t_sync_tokens: list[int] = []
    deferred_t_prompt_anchors: torch.Tensor | None = None
    recent_drafts = deque(maxlen=getattr(args, "acceptance_window", 8))
    consecutive_failed_draft_cycles = 0
    consecutive_latent_margin_gate_skips = 0
    cooldown_remaining = 0
    auto_route_decided = False
    auto_route_early_decided = False
    auto_route_next_evaluation = args.auto_route_after_latent_drafts
    ngram_wide_route_decided = False
    ngram_wide_route_disabled = False
    ngram_unchecked_category_closed = False
    latent_unchecked_category_closed = False
    verification_agreement_streak = 0
    verification_agreement_gate_open = False
    token_text_cache: dict[int, str] = {}
    need_draft_margin = bool(
        not args.compact_runtime_stats
        or args.mode == "heuristic"
        or args.draft_commit_policy == "draft_margin"
        or args.latent_draft_commit_policy == "draft_margin"
        or args.preemptive_lookahead
        or args.verification_frequency_policy == "agreement_gated"
    )
    need_target_margin = bool(
        not args.compact_runtime_stats
        or args.serial_fallback_margin is not None
        or args.verification_frequency_policy == "agreement_gated"
    )
    anchor_hook = None
    anchor_capture: dict[str, Any] = {"enabled": False, "value": None}
    if args.anchor_hook_hidden_states and recurrent_active:
        target_layers, _ = find_decoder_layers(get_base_causal_lm(model))
        hook_layer_idx = anchor_idx - 1
        if not 0 <= hook_layer_idx < len(target_layers):
            raise ValueError(f"Invalid recurrent boundary hook layer {hook_layer_idx}.")

        def capture_anchor(_module, _inputs, output):
            if anchor_capture["enabled"]:
                anchor_capture["value"] = output[0] if isinstance(output, tuple) else output
            return output

        anchor_hook = target_layers[hook_layer_idx].register_forward_hook(capture_anchor)

    def tracked_target_forward(*forward_args, **forward_kwargs):
        if anchor_hook is None:
            return target_forward(*forward_args, **forward_kwargs)
        anchor_capture["enabled"] = recurrent_active
        anchor_capture["value"] = None
        forward_kwargs["output_hidden_states"] = False
        outputs_local = target_forward(*forward_args, **forward_kwargs)
        anchor_capture["enabled"] = False
        if recurrent_active:
            captured = anchor_capture["value"]
            if captured is None:
                raise RuntimeError("The target boundary hook did not capture a hidden state.")
            outputs_local.hidden_states = {anchor_idx: captured}
        anchor_capture["value"] = None
        return outputs_local

    def tracked_target_tree_forward(
        tree: DraftTree,
        *,
        past_key_values: DynamicCache,
        start_position: int,
    ):
        if anchor_hook is None:
            return target_tree_forward(
                model,
                tree,
                past_key_values=past_key_values,
                start_position=start_position,
                output_hidden_states=recurrent_active,
            )
        anchor_capture["enabled"] = recurrent_active
        anchor_capture["value"] = None
        outputs_local = target_tree_forward(
            model,
            tree,
            past_key_values=past_key_values,
            start_position=start_position,
            output_hidden_states=False,
        )
        anchor_capture["enabled"] = False
        if recurrent_active:
            captured = anchor_capture["value"]
            if captured is None:
                raise RuntimeError("The target boundary hook did not capture a tree hidden state.")
            outputs_local.hidden_states = {anchor_idx: captured}
        anchor_capture["value"] = None
        return outputs_local

    def decode_token(token: int) -> str | None:
        if tokenizer is None:
            return None
        if token not in token_text_cache:
            token_text_cache[token] = tokenizer.decode([token], skip_special_tokens=False)
        return token_text_cache[token]

    outputs, timings["prefill_s"] = timed(
        lambda: tracked_target_forward(
            model,
            input_ids,
            start_position=0,
            output_hidden_states=recurrent_active,
            omit_attention_mask=args.omit_target_attention_mask,
        ),
        device,
    )
    target_cache = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :]
    true_t_cache = DynamicCache()
    if recurrent_active:
        anchor_history = outputs.hidden_states[anchor_idx]
        if args.defer_t_init_until_latent:
            deferred_t_prompt_anchors = anchor_history
            true_next_anchor = None
        else:
            (true_t_output, true_t_cache), timings["t_init_s"] = timed(
                lambda: recurrent.forward_with_cache(
                    anchor_history,
                    model=get_base_causal_lm(model),
                    past_key_value=true_t_cache,
                    token_ids=input_ids,
                ),
                device,
            )
            true_next_anchor = true_t_output[:, -1:, :]
    else:
        true_next_anchor = None

    def queue_t_sync(anchor: torch.Tensor, tokens: list[int]) -> None:
        if anchor.size(1) != len(tokens):
            raise ValueError("Lazy T sync requires one boundary state per committed token.")
        pending_t_sync_anchors.append(anchor)
        pending_t_sync_tokens.extend(tokens)
        stats["lazy_t_sync_queued_tokens"] += len(tokens)

    def flush_t_sync() -> None:
        nonlocal deferred_t_prompt_anchors, true_t_cache, true_next_anchor
        if deferred_t_prompt_anchors is not None:
            anchor_chunks = [deferred_t_prompt_anchors, *pending_t_sync_anchors]
            token_ids = prompt_token_ids + list(pending_t_sync_tokens)
            anchors = torch.cat(anchor_chunks, dim=1)
            (true_t_output, true_t_cache), elapsed = timed(
                lambda: recurrent.forward_with_cache(
                    anchors,
                    model=get_base_causal_lm(model),
                    past_key_value=true_t_cache,
                    token_ids=torch.tensor([token_ids], dtype=torch.long, device=device),
                ),
                device,
            )
            timings["t_init_s"] += elapsed
            true_next_anchor = true_t_output[:, -1:, :]
            stats["deferred_t_initializations"] += 1
            stats["deferred_t_init_tokens"] += len(token_ids)
            deferred_t_prompt_anchors = None
            pending_t_sync_anchors.clear()
            pending_t_sync_tokens.clear()
            return
        if not pending_t_sync_tokens:
            return
        anchors = torch.cat(pending_t_sync_anchors, dim=1)
        tokens = list(pending_t_sync_tokens)
        (true_t_output, true_t_cache), elapsed = timed(
            lambda: recurrent.forward_with_cache(
                anchors,
                model=get_base_causal_lm(model),
                past_key_value=true_t_cache,
                token_ids=torch.tensor([tokens], dtype=torch.long, device=device),
            ),
            device,
        )
        timings["t_sync_s"] += elapsed
        true_next_anchor = true_t_output[:, -1:, :]
        stats["lazy_t_sync_flushes"] += 1
        stats["lazy_t_sync_flushed_tokens"] += len(tokens)
        pending_t_sync_anchors.clear()
        pending_t_sync_tokens.clear()

    def commit_target_tokens(
        tokens: list[int],
        start_position: int,
        *,
        count_as_replay: bool = False,
        force_serial: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal target_cache
        if args.serial_target_cache_commit or force_serial:
            committed_anchors = []
            next_logits_local = None
            for offset, token in enumerate(tokens):
                token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
                commit_outputs, elapsed = timed(
                    lambda: tracked_target_forward(
                        model,
                        token_tensor,
                        past_key_values=target_cache,
                        start_position=start_position + offset,
                        omit_attention_mask=args.omit_target_attention_mask,
                    ),
                    device,
                )
                timings["correction_s"] += elapsed
                stats["target_calls"] += 1
                stats["serial_target_commits"] += 1
                if count_as_replay:
                    stats["target_cache_replays"] += 1
                target_cache = commit_outputs.past_key_values
                committed_anchors.append(commit_outputs.hidden_states[anchor_idx])
                next_logits_local = commit_outputs.logits[:, -1, :]
            return torch.cat(committed_anchors, dim=1), next_logits_local

        commit_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        commit_outputs, elapsed = timed(
            lambda: tracked_target_forward(
                model,
                commit_tensor,
                past_key_values=target_cache,
                start_position=start_position,
                omit_attention_mask=args.omit_target_attention_mask,
            ),
            device,
        )
        timings["correction_s"] += elapsed
        stats["target_calls"] += 1
        if count_as_replay:
            stats["target_cache_replays"] += 1
        target_cache = commit_outputs.past_key_values
        return commit_outputs.hidden_states[anchor_idx], commit_outputs.logits[:, -1, :]

    def rebuild_target_cache_serial(
        generated_prefix: list[int],
        return_tail_anchors: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal target_cache
        prompt_outputs, elapsed = timed(
            lambda: tracked_target_forward(
                model,
                input_ids,
                start_position=0,
                omit_attention_mask=args.omit_target_attention_mask,
            ),
            device,
        )
        timings["correction_s"] += elapsed
        stats["target_calls"] += 1
        stats["target_cache_replays"] += 1
        stats["full_prefix_replays"] += 1
        target_cache = prompt_outputs.past_key_values
        next_logits_local = prompt_outputs.logits[:, -1, :]
        tail_anchors = []
        tail_start = max(0, len(generated_prefix) - return_tail_anchors)
        for offset, token in enumerate(generated_prefix):
            token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
            replay_outputs, elapsed = timed(
                lambda token_tensor=token_tensor, offset=offset: tracked_target_forward(
                    model,
                    token_tensor,
                    past_key_values=target_cache,
                    start_position=input_ids.size(1) + offset,
                    omit_attention_mask=args.omit_target_attention_mask,
                ),
                device,
            )
            timings["correction_s"] += elapsed
            stats["target_calls"] += 1
            stats["serial_target_commits"] += 1
            stats["target_cache_replays"] += 1
            target_cache = replay_outputs.past_key_values
            next_logits_local = replay_outputs.logits[:, -1, :]
            if offset >= tail_start:
                tail_anchors.append(replay_outputs.hidden_states[anchor_idx])

        if not tail_anchors:
            raise ValueError("Full-prefix fallback must return at least one committed anchor.")
        return torch.cat(tail_anchors, dim=1), next_logits_local

    while len(generated) < args.max_new_tokens:
        stats["cycles"] += 1
        cycle = stats["cycles"]
        generated_start = len(generated)
        old_length = input_ids.size(1) + len(generated)
        agreement_gate_active_cycle = bool(
            args.verification_frequency_policy == "agreement_gated"
            and verification_agreement_gate_open
        )
        agreement_shadow_requested = False
        agreement_shadow_probe: dict[str, Any] | None = None
        if args.serial_fallback_full_replay_margin is not None and generated:
            cycle_top2 = torch.topk(next_logits[0].float(), k=2).values
            cycle_margin = float((cycle_top2[0] - cycle_top2[1]).item())
            if cycle_margin <= args.serial_fallback_full_replay_margin:
                block_first_token = int(next_logits.argmax(dim=-1).item())
                _, next_logits = rebuild_target_cache_serial(generated, 1)
                serial_first_token = int(next_logits.argmax(dim=-1).item())
                stats["serial_fallback_events"] += 1
                stats["cycle_start_fallback_events"] += 1
                stats["serial_fallback_token_changes"] += int(serial_first_token != block_first_token)
                stats["cycle_start_fallback_token_changes"] += int(
                    serial_first_token != block_first_token
                )
        target_top1_margin = None
        if args.cheap_verifier_policy == "logit_margin" or args.latent_min_target_margin > 0.0:
            first_token, target_top1_margin = top_token_and_logit_margin(next_logits)
        else:
            first_token = int(next_logits.argmax(dim=-1).item())
        if first_token in eos_ids:
            stats["target_generated_positions"].append(
                {
                    "cycle": cycle,
                    "generated_position": generated_start,
                    "sequence_position": old_length,
                    "source": "target_next",
                }
            )
            generated.append(first_token)
            break

        remaining = args.max_new_tokens - len(generated)
        block_limit = max(1, min(choose_block_limit(args, len(generated)), remaining))
        verifier_survival_probability = 1.0
        verification_risk_survival = 1.0
        if args.mode == "verifier" and stats["cycles"] % max(1, args.verifier_interval) != 0:
            block_limit = min(block_limit, args.min_block_tokens)
        needs_cheap_anchor_text = bool(
            args.cheap_verifier_policy == "anchor_category"
            or args.cheap_verifier_rollout_anchor_categories
        )
        first_token_text = decode_token(first_token) if needs_cheap_anchor_text else None
        cheap_decision = cheap_verifier_decision(
            args,
            generated_start,
            next_logits,
            block_limit,
            anchor_token=first_token,
            anchor_text=first_token_text,
            target_top1_margin=target_top1_margin,
        )
        if args.cheap_verifier_policy != "none":
            block_limit = max(1, min(block_limit, int(cheap_decision["block_limit"]), remaining))
            stats["cheap_policy_evaluations"] += 1
            if not args.compact_runtime_stats:
                stats["cheap_policy_records"].append(cheap_decision)
            if cheap_decision["dynamic_threshold"] is not None:
                stats["cheap_policy_dynamic_thresholds"].append(float(cheap_decision["dynamic_threshold"]))
            if cheap_decision["target_top1_margin"] is not None:
                stats["cheap_policy_target_margins"].append(float(cheap_decision["target_top1_margin"]))
            if cheap_decision["skip"]:
                stats["cheap_policy_skips"] += 1

        recent_rate = None
        history_budget = None
        if getattr(args, "two_stage_adaptive", False):
            history_budget, recent_rate = recent_budget(
                recent_drafts, args.max_drafts, args.explore_drafts,
                args.good_acceptance, args.strong_acceptance,
            )
            block_limit = min(block_limit, 1 + history_budget)
        budget_before_cooldown = block_limit
        draft_confidence_stopped = False

        if (
            args.mode == "verifier"
            and args.verifier_skip_threshold is not None
            and block_limit > 1
        ):
            first_draft_position = generated_start + 1
            active_threshold = (
                float(cheap_decision["dynamic_threshold"])
                if cheap_decision["dynamic_threshold"] is not None
                else verifier_threshold_for_position(args, first_draft_position)
            )
            if active_threshold >= args.verifier_skip_threshold:
                block_limit = 1
                stats["verifier_threshold_skips"] += 1
        cooldown_cycle = False
        if cooldown_remaining > 0:
            block_limit = 1
            cooldown_remaining -= 1
            cooldown_cycle = True
            stats["draft_cooldown_skips"] += 1
        if args.verification_frequency_policy == "agreement_gated":
            if agreement_gate_active_cycle:
                stats["verification_agreement_active_cycles"] += 1
            elif block_limit > 1:
                # A closed gate may compute one latent draft, but the target still advances
                # serially. The discarded draft becomes an exact shadow observation.
                block_limit = min(block_limit, 2)
                agreement_shadow_requested = True
        prefetched_cycle = False
        ngram_cycle = False
        ngram_wide_cycle = False
        ngram_size = None
        ngram_tree: DraftTree | None = None
        ngram_tree_branches: list[list[int]] = []
        ngram_tree_branch_limit = 1
        ngram_scheduled_width = ngram_width_for_position(
            args.ngram_max_draft_tokens,
            generated_start,
            ngram_max_draft_tokens_by_position,
        )
        if ngram_predecode_width is not None:
            ngram_scheduled_width = (
                min(ngram_scheduled_width, ngram_predecode_width)
                if ngram_scheduled_width > 0
                else ngram_predecode_width
            )
        active_ngram_max_draft_tokens = ngram_scheduled_width
        latent_position_gated = generated_start < args.latent_draft_min_position
        if pending_lookahead_tokens is not None and pending_lookahead_tokens[0] == first_token:
            block_tokens = pending_lookahead_tokens[:block_limit]
            prefetched_cycle = True
            reused_drafts = max(0, len(block_tokens) - 1)
            stats["draft_tokens"] += reused_drafts
            stats["preemptive_reused_draft_tokens"] += reused_drafts
        else:
            block_tokens = [first_token]
            if args.ngram_draft_mode != "off" and block_limit > 1:
                stats["ngram_attempts"] += 1
                if ngram_wide_route_disabled:
                    active_ngram_max_draft_tokens = min(
                        active_ngram_max_draft_tokens,
                        args.ngram_wide_auto_route_fallback_width,
                    )
                ngram_block_limit = (
                    min(remaining, 1 + active_ngram_max_draft_tokens)
                    if active_ngram_max_draft_tokens > 0
                    else block_limit
                )
                ngram_lookup_start = time.perf_counter()
                if ngram_index is not None:
                    if ngram_index_generated_tokens < len(generated):
                        ngram_index.append(generated[ngram_index_generated_tokens:])
                        ngram_index_generated_tokens = len(generated)
                    if args.ngram_tree_verify:
                        ngram_tree_branch_limit = tree_branch_budget(
                            generated_start,
                            args.medium_position,
                            args.long_position,
                            args.ngram_tree_early_branches,
                            args.ngram_tree_mid_branches,
                            args.ngram_tree_late_branches,
                        )
                        ngram_tree_branches, ngram_size = ngram_index.find_branches(
                            first_token,
                            ngram_block_limit - 1,
                            ngram_tree_branch_limit,
                        )
                        limited_branches = []
                        for branch in ngram_tree_branches:
                            branch = limit_ngram_draft_tokens(
                                branch,
                                ngram_size,
                                ngram_draft_tokens_by_match_size,
                            )
                            if branch:
                                for token_index, token in enumerate(branch):
                                    if token in eos_ids:
                                        branch = branch[: token_index + 1]
                                        break
                                if branch not in limited_branches:
                                    limited_branches.append(branch)
                        ngram_tree_branches = limited_branches
                        ngram_tokens = ngram_tree_branches[0] if ngram_tree_branches else []
                        if len(ngram_tree_branches) > 1:
                            ngram_tree = build_draft_tree(
                                first_token,
                                ngram_tree_branches,
                                max_nodes=args.ngram_tree_max_nodes,
                            )
                    else:
                        ngram_tokens, ngram_size = ngram_index.find(
                            first_token,
                            ngram_block_limit - 1,
                            active_ngram_occurrence_policy,
                            args.ngram_occurrence_recency_decay,
                        )
                else:
                    ngram_tokens, ngram_size = find_suffix_ngram_draft(
                        prompt_token_ids + generated + [first_token],
                        ngram_block_limit - 1,
                        args.ngram_min_size,
                        args.ngram_max_size,
                    )
                if ngram_tree is None:
                    ngram_tokens = limit_ngram_draft_tokens(
                        ngram_tokens,
                        ngram_size,
                        ngram_draft_tokens_by_match_size,
                    )
                ngram_wide_cycle = bool(
                    len(ngram_tokens) > args.ngram_wide_auto_route_fallback_width
                )
                timings["ngram_lookup_s"] += time.perf_counter() - ngram_lookup_start
                if ngram_tokens:
                    for token in ngram_tokens:
                        block_tokens.append(token)
                        if token in eos_ids:
                            break
                    block_limit = len(block_tokens)
                    ngram_cycle = True
                    stats["ngram_hits"] += 1
                    if ngram_wide_route_disabled:
                        stats["ngram_wide_route_fallback_cycles"] += 1
                    stats["ngram_draft_tokens"] += len(block_tokens) - 1
                    stats["draft_tokens"] += len(block_tokens) - 1
                elif (
                    args.ngram_draft_mode == "only"
                    or not recurrent_active
                    or latent_position_gated
                ):
                    block_limit = 1
                    if latent_position_gated and recurrent_active:
                        stats["latent_position_gate_skips"] += 1
            elif latent_position_gated and recurrent_active and block_limit > 1:
                block_limit = 1
                stats["latent_position_gate_skips"] += 1
        if not recurrent_active and not ngram_cycle:
            # Auto-routing can disable T with n-gram drafting off.  In that case
            # no drafter can fill the remaining block, so commit target tokens only.
            block_limit = len(block_tokens)
        pending_lookahead_tokens = None
        block_margins: list[float | None] = [None] * max(0, len(block_tokens) - 1)
        block_scores: list[float | None] = [None] * max(0, len(block_tokens) - 1)
        block_thresholds: list[float | None] = [None] * max(0, len(block_tokens) - 1)
        block_draft_records: list[dict[str, Any]] = []
        block_rejected_records: list[dict[str, Any]] = []

        latent_drafter_candidate = bool(
            recurrent_active
            and not prefetched_cycle
            and not ngram_cycle
            and len(block_tokens) < block_limit
        )
        if latent_drafter_candidate and latent_unchecked_category_closed:
            block_limit = len(block_tokens)
            latent_drafter_candidate = False
            stats["latent_unchecked_category_closed_skips"] += 1
        latent_target_margin_gated = False
        if latent_drafter_candidate and args.latent_min_target_margin > 0.0:
            stats["latent_target_margin_gate_evaluations"] += 1
            stats["latent_target_margins"].append(float(target_top1_margin))
            latent_target_margin_gated = not latent_target_margin_allows(
                args.latent_min_target_margin,
                target_top1_margin,
            )
            consecutive_latent_margin_gate_skips, close_latent_route = (
                latent_margin_gate_route_update(
                    latent_target_margin_gated,
                    consecutive_latent_margin_gate_skips,
                    args.latent_margin_gate_close_after,
                )
            )
            if latent_target_margin_gated:
                block_limit = len(block_tokens)
                latent_drafter_candidate = False
                stats["latent_target_margin_gate_skips"] += 1
                if close_latent_route:
                    recurrent_active = False
                    true_t_cache = DynamicCache()
                    true_next_anchor = None
                    deferred_t_prompt_anchors = None
                    pending_t_sync_anchors.clear()
                    pending_t_sync_tokens.clear()
                    stats["latent_target_margin_gate_closes"] += 1
                    stats["latent_target_margin_gate_close_position"] = generated_start
        latent_policy = args.latent_draft_commit_policy or args.draft_commit_policy
        latent_budget_exhausted = bool(
            latent_drafter_candidate
            and args.latent_budget_short_circuit
            and latent_policy in {"draft_margin", "category_block"}
            and unchecked_budget_exhausted_for_position(
                generated_start + 1,
                args.max_accepted_unchecked_per_sequence,
                stats["accepted_unchecked_tokens"],
                max_unchecked_per_window=args.max_accepted_unchecked_per_window,
                unchecked_budget_window_tokens=args.unchecked_budget_window_tokens,
                accepted_unchecked_positions=stats["accepted_unchecked_positions"],
            )
        )
        uses_latent_drafter = latent_drafter_candidate and not latent_budget_exhausted
        if latent_budget_exhausted:
            block_limit = len(block_tokens)
            stats["latent_unchecked_budget_skips"] += 1
        if args.lazy_t_sync and uses_latent_drafter:
            flush_t_sync()

        if uses_latent_drafter and true_next_anchor is None:
            # A correction/acceptance path can temporarily invalidate the recurrent anchor.
            # Keep this cycle target-only; the normal commit path rebuilds T state afterwards.
            uses_latent_drafter = False
            block_limit = len(block_tokens)
            stats["missing_t_anchor_fallbacks"] += 1

        ngram_unchecked_eligible = False
        ngram_unchecked_draft_allowed: list[bool] | None = None
        if ngram_cycle and args.ngram_unchecked_min_match_size > 0:
            stats["ngram_unchecked_gate_evaluations"] += 1
            if (
                args.ngram_unchecked_category_reject_closes_sequence
                and ngram_unchecked_category_closed
            ):
                stats["ngram_unchecked_category_closed_skips"] += 1
            elif ngram_unchecked_gate_allows(
                ngram_size,
                generated_start,
                args.ngram_unchecked_min_match_size,
                args.ngram_unchecked_min_position,
            ):
                category_gate_allows = True
                if ngram_unchecked_categories:
                    stats["ngram_unchecked_category_gate_evaluations"] += 1
                    ngram_unchecked_draft_allowed = [
                        token_category_from_text(decode_token(token) or "")
                        in ngram_unchecked_categories
                        for token in block_tokens[1:]
                    ]
                    category_gate_allows = bool(
                        ngram_unchecked_draft_allowed
                        and ngram_unchecked_draft_allowed[0]
                    )
                    if not category_gate_allows:
                        stats["ngram_unchecked_category_gate_rejects"] += 1
                        if args.ngram_unchecked_category_reject_closes_sequence:
                            ngram_unchecked_category_closed = True
                            stats["ngram_unchecked_category_budget_closes"] += 1
                if category_gate_allows and unchecked_budget_exhausted_for_position(
                    generated_start + 1,
                    args.max_accepted_unchecked_per_sequence,
                    stats["accepted_unchecked_tokens"],
                    max_unchecked_per_window=args.max_accepted_unchecked_per_window,
                    unchecked_budget_window_tokens=args.unchecked_budget_window_tokens,
                    accepted_unchecked_positions=stats["accepted_unchecked_positions"],
                ):
                    stats["ngram_unchecked_budget_skips"] += 1
                elif category_gate_allows:
                    ngram_unchecked_eligible = True
                    stats["ngram_unchecked_gate_eligible"] += 1

        if ngram_unchecked_eligible:
            effective_commit_policy = "ngram_prefix"
        elif uses_latent_drafter and args.latent_draft_commit_policy is not None:
            effective_commit_policy = args.latent_draft_commit_policy
        else:
            effective_commit_policy = args.draft_commit_policy

        t_cache_length = cache_length(true_t_cache, metadata["loop_start_layer"])
        # No draft forward can mutate T's cache when this cycle skips T.
        t_cache = true_t_cache if (args.inplace_draft_cache or not uses_latent_drafter) else clone_cache(true_t_cache)
        next_anchor = true_next_anchor
        cycle_initial_anchor = true_next_anchor
        draft_cache_layer_ids = None if args.full_draft_cache_clone else projection_ids
        tail_cache = None
        if args.draft_logit_source == "tail":
            tail_cache = (
                target_cache
                if args.inplace_draft_cache
                else clone_cache(target_cache, draft_cache_layer_ids)
            )

        batched_draft_projection = bool(
            uses_latent_drafter
            and (
                (args.draft_logit_source == "tail" and args.batched_draft_tail)
                or (
                    args.draft_logit_source == "boundary"
                    and args.batched_draft_boundary
                )
            )
            and args.mode in {"fixed", "schedule", "verifier", "heuristic"}
            and not recurrent.has_token_conditioning()
        )
        if batched_draft_projection:
            draft_count = max(0, block_limit - 1)
            draft_anchors = []
            verifier_probs: list[float | None] = []
            verifier_thresholds: list[float | None] = []
            if args.mode == "verifier":
                previous_probability: float | None = None
                previous_threshold: float | None = None
                for _ in range(draft_count):
                    candidate_draft_index = len(block_tokens) + len(draft_anchors)
                    candidate_generated_position = generated_start + candidate_draft_index
                    candidate_sequence_position = old_length + candidate_draft_index
                    if (
                        len(block_tokens) + len(draft_anchors) >= args.min_block_tokens
                        and previous_probability is not None
                        and previous_threshold is not None
                        and previous_probability < previous_threshold
                    ):
                        break

                    candidate_anchor = next_anchor
                    (following_anchor, t_cache), elapsed = timed(
                        lambda: recurrent.forward_with_cache(
                            candidate_anchor,
                            model=get_base_causal_lm(model),
                            past_key_value=t_cache,
                            token_ids=torch.tensor(
                                [[block_tokens[-1]]], dtype=torch.long, device=device
                            ),
                            rollout_step=candidate_draft_index + 1,
                        ),
                        device,
                    )
                    timings["t_step_s"] += elapsed
                    next_anchor = following_anchor[:, -1:, :]
                    verifier_cache = clone_cache(t_cache)
                    score, elapsed = timed(
                        lambda: recurrent.probe(
                            batch_size=1,
                            model=get_base_causal_lm(model),
                            past_key_value=verifier_cache,
                        ),
                        device,
                    )
                    timings["probe_s"] += elapsed
                    conditional_probability = float(
                        torch.sigmoid(score.float() / args.verifier_score_temperature).item()
                    )
                    verifier_survival_probability *= conditional_probability
                    probability = (
                        verifier_survival_probability
                        if args.verifier_cumulative_survival
                        else conditional_probability
                    )
                    stats["verifier_conditional_scores"].append(conditional_probability)
                    stats["verifier_survival_scores"].append(verifier_survival_probability)
                    stats["verifier_scores"].append(probability)
                    if args.verification_frequency_policy == "risk_budget":
                        should_stop, next_survival, next_risk = verification_risk_budget_decision(
                            verification_risk_survival,
                            conditional_probability,
                            args.verification_risk_budget,
                            len(block_tokens) + len(draft_anchors),
                            args.min_block_tokens,
                        )
                        if should_stop:
                            stats["verification_risk_budget_stops"] += 1
                            stats["verification_risk_at_stops"].append(next_risk)
                            break
                        verification_risk_survival = next_survival
                    threshold = (
                        float(cheap_decision["dynamic_threshold"])
                        if cheap_decision["dynamic_threshold"] is not None
                        else verifier_threshold_for_position(args, candidate_generated_position)
                    )
                    if args.allow_zero_draft and probability < threshold:
                        rejected_record = {
                            "cycle": cycle,
                            "draft_index": candidate_draft_index,
                            "generated_position": candidate_generated_position,
                            "sequence_position": candidate_sequence_position,
                            "verifier_score": probability,
                            "verifier_threshold": threshold,
                            "reason": "verifier_threshold",
                        }
                        block_rejected_records.append(rejected_record)
                        stats["rejected_draft_positions"].append(rejected_record)
                        stats["rejected_draft_tokens"] += 1
                        break

                    draft_anchors.append(candidate_anchor)
                    previous_probability = probability
                    previous_threshold = threshold
                    verifier_probs.append(probability)
                    verifier_thresholds.append(threshold)
            else:
                if draft_count > 0:
                    draft_anchors.append(next_anchor)
                for rollout_step in range(2, draft_count + 1):
                    (following_anchor, t_cache), elapsed = timed(
                        lambda: recurrent.forward_with_cache(
                            next_anchor,
                            model=get_base_causal_lm(model),
                            past_key_value=t_cache,
                            token_ids=torch.tensor(
                                [[block_tokens[-1]]], dtype=torch.long, device=device
                            ),
                            rollout_step=rollout_step,
                        ),
                        device,
                    )
                    timings["t_step_s"] += elapsed
                    next_anchor = following_anchor[:, -1:, :]
                    draft_anchors.append(next_anchor)
                verifier_probs = [None] * len(draft_anchors)
                verifier_thresholds = [None] * len(draft_anchors)

            if draft_anchors:
                anchor_block = torch.cat(draft_anchors, dim=1)
                if args.draft_logit_source == "boundary":
                    draft_logits, elapsed = timed(
                        lambda: recurrent.boundary_logits(
                            anchor_block, get_base_causal_lm(model)
                        ),
                        device,
                    )
                    timings["boundary_s"] += elapsed
                else:
                    draft_logits, elapsed = timed(
                        lambda: project_anchor_block(
                            model, metadata, anchor_block, tail_cache, old_length
                        ),
                        device,
                    )
                    timings["tail_s"] += elapsed
                top_values, top_indices = torch.topk(draft_logits.float(), k=2, dim=-1)
                for draft_idx in range(draft_logits.size(1)):
                    draft_token = int(top_indices[0, draft_idx, 0].item())
                    margin = (
                        float((top_values[0, draft_idx, 0] - top_values[0, draft_idx, 1]).item())
                        if need_draft_margin
                        else None
                    )
                    if (
                        effective_commit_policy == "draft_margin"
                        and margin is not None
                        and margin
                        < min_draft_margin_for_position(
                            args, generated_start + len(block_tokens)
                        )
                    ):
                        rejected_record = {
                            "cycle": cycle,
                            "draft_index": len(block_tokens),
                            "generated_position": generated_start + len(block_tokens),
                            "sequence_position": old_length + len(block_tokens),
                            "margin": margin,
                            "reason": "draft_margin",
                        }
                        block_rejected_records.append(rejected_record)
                        if not args.compact_runtime_stats:
                            stats["rejected_draft_positions"].append(rejected_record)
                        stats["rejected_draft_tokens"] += 1
                        break
                    block_tokens.append(draft_token)
                    block_margins.append(margin)
                    block_scores.append(verifier_probs[draft_idx] if draft_idx < len(verifier_probs) else None)
                    block_thresholds.append(
                        verifier_thresholds[draft_idx] if draft_idx < len(verifier_thresholds) else None
                    )
                    if margin is not None:
                        stats["draft_margins"].append(margin)
                    stats["draft_tokens"] += 1
                    if draft_token in eos_ids:
                        break
        else:
            while len(block_tokens) < block_limit:
                if len(block_tokens) >= args.min_block_tokens:
                    if (
                        args.mode == "heuristic"
                        and block_margins
                        and block_margins[-1]
                        < min_draft_margin_for_position(
                            args,
                            generated_start + len(block_tokens) - 1,
                        )
                    ):
                        draft_confidence_stopped = True
                        break
                    if (
                        args.mode == "verifier"
                        and verifier_previous_score_rejects(
                            block_scores,
                            block_thresholds,
                            args.verifier_threshold,
                        )
                    ):
                        break

                probability = None
                threshold = None
                following_anchor = None
                if args.mode == "verifier" and args.allow_zero_draft:
                    candidate_draft_index = len(block_tokens)
                    candidate_generated_position = generated_start + candidate_draft_index
                    candidate_sequence_position = old_length + candidate_draft_index
                    threshold = (
                        float(cheap_decision["dynamic_threshold"])
                        if cheap_decision["dynamic_threshold"] is not None
                        else verifier_threshold_for_position(args, candidate_generated_position)
                    )
                    (following_anchor, t_cache), elapsed = timed(
                        lambda: recurrent.forward_with_cache(
                            next_anchor,
                            model=get_base_causal_lm(model),
                            past_key_value=t_cache,
                            token_ids=torch.tensor(
                                [[block_tokens[-1]]], dtype=torch.long, device=device
                            ),
                            rollout_step=candidate_draft_index + 1,
                        ),
                        device,
                    )
                    timings["t_step_s"] += elapsed
                    verifier_cache = clone_cache(t_cache)
                    score, elapsed = timed(
                        lambda: recurrent.probe(
                            batch_size=1,
                            model=get_base_causal_lm(model),
                            past_key_value=verifier_cache,
                        ),
                        device,
                    )
                    timings["probe_s"] += elapsed
                    conditional_probability = float(
                        torch.sigmoid(score.float() / args.verifier_score_temperature).item()
                    )
                    verifier_survival_probability *= conditional_probability
                    probability = (
                        verifier_survival_probability
                        if args.verifier_cumulative_survival
                        else conditional_probability
                    )
                    stats["verifier_conditional_scores"].append(conditional_probability)
                    stats["verifier_survival_scores"].append(verifier_survival_probability)
                    stats["verifier_scores"].append(probability)
                    if args.verification_frequency_policy == "risk_budget":
                        should_stop, next_survival, next_risk = verification_risk_budget_decision(
                            verification_risk_survival,
                            conditional_probability,
                            args.verification_risk_budget,
                            len(block_tokens),
                            args.min_block_tokens,
                        )
                        if should_stop:
                            stats["verification_risk_budget_stops"] += 1
                            stats["verification_risk_at_stops"].append(next_risk)
                            break
                        verification_risk_survival = next_survival
                    if probability < threshold:
                        rejected_record = {
                            "cycle": cycle,
                            "draft_index": candidate_draft_index,
                            "generated_position": candidate_generated_position,
                            "sequence_position": candidate_sequence_position,
                            "verifier_score": probability,
                            "verifier_threshold": threshold,
                            "reason": "verifier_threshold",
                        }
                        block_rejected_records.append(rejected_record)
                        stats["rejected_draft_positions"].append(rejected_record)
                        stats["rejected_draft_tokens"] += 1
                        break

                needs_following_anchor = len(block_tokens) + 1 < block_limit
                if (
                    draft_streams is not None
                    and following_anchor is None
                    and needs_following_anchor
                ):
                    (draft_logits, following_anchor, t_cache), elapsed = timed(
                        lambda: parallel_draft_recurrent_step(
                            recurrent,
                            model,
                            metadata,
                            next_anchor,
                            t_cache,
                            tail_cache,
                            block_tokens[-1],
                            len(block_tokens) + 1,
                            old_length + len(block_tokens) - 1,
                            args.draft_logit_source,
                            device,
                            draft_streams,
                        ),
                        device,
                    )
                    timings["draft_parallel_s"] += elapsed
                elif args.draft_logit_source == "boundary":
                    draft_logits, elapsed = timed(
                        lambda: recurrent.boundary_logits(next_anchor, get_base_causal_lm(model)),
                        device,
                    )
                    timings["boundary_s"] += elapsed
                else:
                    draft_logits, elapsed = timed(
                        lambda: project_anchor_step(
                            model, metadata, next_anchor, tail_cache, old_length + len(block_tokens) - 1
                        ),
                        device,
                    )
                    timings["tail_s"] += elapsed
                top_values, top_indices = torch.topk(draft_logits.float(), k=2, dim=-1)
                if top_indices.dim() == 3:
                    draft_token = int(top_indices[0, -1, 0].item())
                    margin = (
                        float((top_values[0, -1, 0] - top_values[0, -1, 1]).item())
                        if need_draft_margin
                        else None
                    )
                else:
                    draft_token = int(top_indices[0, 0].item())
                    margin = (
                        float((top_values[0, 0] - top_values[0, 1]).item())
                        if need_draft_margin
                        else None
                    )
                if margin is not None:
                    stats["draft_margins"].append(margin)

                if (
                    effective_commit_policy == "draft_margin"
                    and margin is not None
                    and margin
                    < min_draft_margin_for_position(
                        args,
                        generated_start + len(block_tokens),
                    )
                ):
                    candidate_draft_index = len(block_tokens)
                    rejected_record = {
                        "cycle": cycle,
                        "draft_index": candidate_draft_index,
                        "generated_position": generated_start + candidate_draft_index,
                        "sequence_position": old_length + candidate_draft_index,
                        "margin": margin,
                        "reason": "draft_margin",
                    }
                    block_rejected_records.append(rejected_record)
                    if not args.compact_runtime_stats:
                        stats["rejected_draft_positions"].append(rejected_record)
                    stats["rejected_draft_tokens"] += 1
                    break

                if following_anchor is None and needs_following_anchor:
                    (following_anchor, t_cache), elapsed = timed(
                        lambda: recurrent.forward_with_cache(
                            next_anchor,
                            model=get_base_causal_lm(model),
                            past_key_value=t_cache,
                            token_ids=torch.tensor(
                                [[block_tokens[-1]]], dtype=torch.long, device=device
                            ),
                            rollout_step=len(block_tokens) + 1,
                        ),
                        device,
                    )
                    timings["t_step_s"] += elapsed
                elif following_anchor is None:
                    stats["skipped_terminal_t_steps"] += 1

                if args.mode == "verifier" and not args.allow_zero_draft and len(block_tokens) + 1 < block_limit:
                    verifier_cache = clone_cache(t_cache)
                    score, elapsed = timed(
                        lambda: recurrent.probe(
                            batch_size=1,
                            model=get_base_causal_lm(model),
                            past_key_value=verifier_cache,
                        ),
                        device,
                    )
                    timings["probe_s"] += elapsed
                    conditional_probability = float(
                        torch.sigmoid(score.float() / args.verifier_score_temperature).item()
                    )
                    verifier_survival_probability *= conditional_probability
                    probability = (
                        verifier_survival_probability
                        if args.verifier_cumulative_survival
                        else conditional_probability
                    )
                    stats["verifier_conditional_scores"].append(conditional_probability)
                    stats["verifier_survival_scores"].append(verifier_survival_probability)
                    stats["verifier_scores"].append(probability)
                    if args.verification_frequency_policy == "risk_budget":
                        should_stop, next_survival, next_risk = verification_risk_budget_decision(
                            verification_risk_survival,
                            conditional_probability,
                            args.verification_risk_budget,
                            len(block_tokens),
                            args.min_block_tokens,
                        )
                        if should_stop:
                            stats["verification_risk_budget_stops"] += 1
                            stats["verification_risk_at_stops"].append(next_risk)
                            break
                        verification_risk_survival = next_survival
                    threshold = (
                        float(cheap_decision["dynamic_threshold"])
                        if cheap_decision["dynamic_threshold"] is not None
                        else verifier_threshold_for_position(args, generated_start + len(block_tokens))
                    )

                block_tokens.append(draft_token)
                block_margins.append(margin)
                block_scores.append(probability)
                block_thresholds.append(threshold)
                stats["draft_tokens"] += 1
                if following_anchor is not None:
                    next_anchor = following_anchor[:, -1:, :]
                if draft_token in eos_ids:
                    break

        original_proposed_len = len(block_tokens)
        if agreement_shadow_requested and len(block_tokens) > 1:
            agreement_shadow_probe = {
                "token": block_tokens[1],
                "draft_margin": block_margins[0],
                "verifier_score": block_scores[0],
                "generated_position": generated_start + 1,
                "sequence_position": old_length + 1,
            }
            stats["verification_agreement_shadow_probes"] += 1
            block_tokens = block_tokens[:1]
            block_margins = []
            block_scores = []
            block_thresholds = []

        if args.inplace_draft_cache:
            target_cache.crop(old_length)
            true_t_cache.crop(t_cache_length)

        precommit_unchecked = bool(
            args.precommit_unchecked_drafts
            and (uses_latent_drafter or ngram_unchecked_eligible)
            and effective_commit_policy in {"whole_block", "draft_margin", "ngram_prefix"}
        )
        if precommit_unchecked:
            unrestricted_ngram_prefix = None
            if (
                effective_commit_policy == "ngram_prefix"
                and ngram_unchecked_draft_allowed is not None
            ):
                unrestricted_ngram_prefix = precommit_unchecked_prefix_length(
                    effective_commit_policy,
                    block_margins,
                    generated_start,
                    args.min_draft_margin,
                    args.max_accepted_unchecked_per_sequence,
                    stats["accepted_unchecked_tokens"],
                    max_unchecked_per_window=args.max_accepted_unchecked_per_window,
                    unchecked_budget_window_tokens=args.unchecked_budget_window_tokens,
                    accepted_unchecked_positions=stats["accepted_unchecked_positions"],
                )
            precommit_prefix = precommit_unchecked_prefix_length(
                effective_commit_policy,
                block_margins,
                generated_start,
                args.min_draft_margin,
                args.max_accepted_unchecked_per_sequence,
                stats["accepted_unchecked_tokens"],
                max_unchecked_per_window=args.max_accepted_unchecked_per_window,
                unchecked_budget_window_tokens=args.unchecked_budget_window_tokens,
                accepted_unchecked_positions=stats["accepted_unchecked_positions"],
                margin_threshold_for_position=lambda position: min_draft_margin_for_position(
                    args, position
                ),
                ngram_draft_allowed=ngram_unchecked_draft_allowed,
            )
            if precommit_prefix is None:
                raise RuntimeError("Unchecked precommit selected an unsupported commit policy.")
            precommit_prefix = max(1, min(precommit_prefix, len(block_tokens)))
            if uses_latent_drafter and latent_unchecked_categories and precommit_prefix > 1:
                stats["latent_unchecked_category_gate_evaluations"] += 1
                candidate_categories = [
                    token_category_from_text(decode_token(token) or "")
                    for token in block_tokens[1:precommit_prefix]
                ]
                allowed_drafts = allowed_category_prefix_length(
                    candidate_categories,
                    latent_unchecked_categories,
                )
                category_prefix = 1 + allowed_drafts
                if category_prefix < precommit_prefix:
                    stats["latent_unchecked_category_gate_rejects"] += 1
                    stats["latent_unchecked_category_trimmed_tokens"] += (
                        precommit_prefix - category_prefix
                    )
                    precommit_prefix = category_prefix
                    if args.latent_unchecked_category_reject_closes_sequence:
                        latent_unchecked_category_closed = True
                        stats["latent_unchecked_category_budget_closes"] += 1
            if unrestricted_ngram_prefix is not None:
                category_trimmed_tokens = max(
                    0,
                    min(unrestricted_ngram_prefix, len(block_tokens)) - precommit_prefix,
                )
                stats["ngram_unchecked_category_trimmed_tokens"] += category_trimmed_tokens
                if (
                    category_trimmed_tokens > 0
                    and args.ngram_unchecked_category_reject_closes_sequence
                    and not ngram_unchecked_category_closed
                ):
                    ngram_unchecked_category_closed = True
                    stats["ngram_unchecked_category_budget_closes"] += 1
            trimmed_drafts = len(block_tokens) - precommit_prefix
            if trimmed_drafts:
                stats["precommit_trimmed_draft_tokens"] += trimmed_drafts
                stats["rejected_draft_tokens"] += trimmed_drafts
            block_tokens = block_tokens[:precommit_prefix]
            block_margins = block_margins[: max(0, precommit_prefix - 1)]
            block_scores = block_scores[: max(0, precommit_prefix - 1)]
            block_thresholds = block_thresholds[: max(0, precommit_prefix - 1)]
            stats["precommit_unchecked_blocks"] += 1

        block_tensor = torch.tensor([block_tokens], dtype=torch.long, device=device)
        stats["verification_intervals"].append(len(block_tokens))
        verify_cache = target_cache if args.inplace_draft_cache else clone_cache(target_cache)
        ngram_tree_path_indices: tuple[int, ...] | None = None
        ngram_tree_matched_drafts = 0
        future_generated_start = generated_start + len(block_tokens)
        future_remaining = args.max_new_tokens - future_generated_start
        future_block_limit = (
            max(1, min(choose_block_limit(args, future_generated_start), future_remaining))
            if future_remaining > 0
            else 0
        )
        lookahead_state: dict[str, Any] = {}
        preemptive_margin_gate = bool(
            args.preemptive_min_draft_margin <= 0.0
            or (
                block_margins
                and all(
                    margin is not None and margin >= args.preemptive_min_draft_margin
                    for margin in block_margins
                )
            )
        )
        use_split_unchecked = bool(
            args.split_unchecked_target_forward
            and precommit_unchecked
            and len(block_tokens) > 1
            and generated_start >= args.split_unchecked_min_position
            and (
                args.split_unchecked_min_draft_margin is None
                or (
                    block_margins
                    and all(
                        margin is not None
                        and margin >= args.split_unchecked_min_draft_margin
                        for margin in block_margins
                    )
                )
            )
        )
        if (
            args.split_unchecked_target_forward
            and precommit_unchecked
            and len(block_tokens) > 1
            and not use_split_unchecked
        ):
            stats["split_unchecked_gate_skips"] += 1
        if ngram_tree is not None:
            verify_outputs, elapsed = timed(
                lambda: tracked_target_tree_forward(
                    ngram_tree,
                    past_key_values=verify_cache,
                    start_position=old_length,
                ),
                device,
            )
            previous_draft_count = max(0, len(block_tokens) - 1)
            (
                verify_outputs,
                block_tokens,
                ngram_tree_matched_drafts,
                ngram_tree_path_indices,
            ), gather_elapsed = timed(
                lambda: select_tree_path_outputs(
                    verify_outputs,
                    ngram_tree,
                    old_length,
                ),
                device,
            )
            selected_draft_count = max(0, len(block_tokens) - 1)
            draft_delta = selected_draft_count - previous_draft_count
            stats["draft_tokens"] += draft_delta
            stats["ngram_draft_tokens"] += draft_delta
            stats["ngram_tree_cycles"] += 1
            stats["ngram_tree_branches"] += len(ngram_tree_branches)
            stats["ngram_tree_nodes"] += len(ngram_tree.nodes)
            stats["ngram_tree_selected_nodes"] += len(ngram_tree_path_indices)
            stats["ngram_tree_target_matched_drafts"] += ngram_tree_matched_drafts
            timings["ngram_tree_gather_s"] += gather_elapsed
            block_limit = len(block_tokens)
            original_proposed_len = len(block_tokens)
            block_margins = [None] * selected_draft_count
            block_scores = [None] * selected_draft_count
            block_thresholds = [None] * selected_draft_count
            ngram_wide_cycle = bool(
                selected_draft_count > args.ngram_wide_auto_route_fallback_width
            )
        elif use_split_unchecked:
            if cycle_initial_anchor is None:
                raise RuntimeError("Split unchecked execution is missing its initial T anchor.")
            split_anchor_cache = clone_cache(true_t_cache)
            predicted_anchor_block, anchor_elapsed = timed(
                lambda: rollout_recurrent_anchor_block(
                    recurrent,
                    model,
                    cycle_initial_anchor,
                    split_anchor_cache,
                    block_tokens,
                ),
                device,
            )
            timings["split_anchor_s"] += anchor_elapsed
            verify_outputs, elapsed = wall_timed(
                lambda: split_unchecked_target_forward(
                    model,
                    metadata,
                    block_tensor,
                    predicted_anchor_block,
                    verify_cache,
                    old_length,
                    device,
                    projection_stream=split_projection_stream,
                ),
                device,
            )
            timings["split_forward_s"] += elapsed
            stats["split_unchecked_blocks"] += 1
            stats["split_unchecked_tokens"] += len(block_tokens)
            stats["split_parallel_blocks"] += int(split_projection_stream is not None)
        elif lookahead_stream is not None and future_block_limit > 0 and preemptive_margin_gate:
            stats["preemptive_attempts"] += 1
            verify_outputs, elapsed = timed(
                lambda: target_forward_with_preemptive_lookahead(
                    model,
                    recurrent,
                    metadata,
                    block_tensor,
                    past_key_values=verify_cache,
                    start_position=old_length,
                    true_t_cache=true_t_cache,
                    block_tokens=block_tokens,
                    future_block_limit=future_block_limit,
                    omit_attention_mask=args.omit_target_attention_mask,
                    device=device,
                    lookahead_stream=lookahead_stream,
                ),
                device,
            )
            verify_outputs, lookahead_state = verify_outputs
        else:
            if lookahead_stream is not None and future_block_limit > 0:
                stats["preemptive_gate_skips"] += 1
            verify_outputs, elapsed = timed(
                lambda: tracked_target_forward(
                    model,
                    block_tensor,
                    past_key_values=verify_cache,
                    start_position=old_length,
                    output_hidden_states=recurrent_active,
                    omit_attention_mask=args.omit_target_attention_mask,
                    logits_to_keep=1 if precommit_unchecked else None,
                ),
                device,
            )
            if precommit_unchecked:
                stats["last_token_logits_only_blocks"] += 1
        timings["verify_s"] += elapsed
        stats["target_calls"] += 1

        if agreement_shadow_requested:
            if agreement_shadow_probe is None:
                decision = agreement_gate_decision(
                    verification_agreement_streak,
                    [],
                    [],
                    [],
                    min_consecutive_matches=args.verification_agreement_warmup,
                    min_draft_margin=args.verification_agreement_min_draft_margin,
                    min_target_margin=args.verification_agreement_min_target_margin,
                )
                shadow_target_token = None
                shadow_target_margin = None
                shadow_match = False
            else:
                shadow_target_logits = verify_outputs.logits[0, -1].float()
                shadow_target_top2 = torch.topk(shadow_target_logits, k=2).values
                shadow_target_token = int(shadow_target_logits.argmax(dim=-1).item())
                shadow_target_margin = float(
                    (shadow_target_top2[0] - shadow_target_top2[1]).item()
                )
                shadow_match = agreement_shadow_probe["token"] == shadow_target_token
                decision = agreement_gate_decision(
                    verification_agreement_streak,
                    [shadow_match],
                    [agreement_shadow_probe["draft_margin"]],
                    [shadow_target_margin],
                    min_consecutive_matches=args.verification_agreement_warmup,
                    min_draft_margin=args.verification_agreement_min_draft_margin,
                    min_target_margin=args.verification_agreement_min_target_margin,
                )
                stats["verification_agreement_shadow_matches"] += int(shadow_match)
            verification_agreement_streak = decision.next_streak
            verification_agreement_gate_open = decision.gate_open
            stats["verification_agreement_max_streak"] = max(
                stats["verification_agreement_max_streak"],
                verification_agreement_streak,
            )
            if decision.gate_open:
                stats["verification_agreement_gate_openings"] += 1
            stats["verification_agreement_events"].append(
                {
                    "cycle": cycle,
                    "kind": "shadow_probe",
                    "generated_position": generated_start + 1,
                    "draft_token": (
                        agreement_shadow_probe["token"]
                        if agreement_shadow_probe is not None
                        else None
                    ),
                    "target_token": shadow_target_token,
                    "match": shadow_match,
                    "draft_margin": (
                        agreement_shadow_probe["draft_margin"]
                        if agreement_shadow_probe is not None
                        else None
                    ),
                    "target_margin": shadow_target_margin,
                    "streak": verification_agreement_streak,
                    "gate_open": verification_agreement_gate_open,
                    "reason": decision.reason,
                }
            )

        acceptance_diagnostics_start = time.perf_counter()
        draft_count = max(0, len(block_tokens) - 1)
        target_logits_block = verify_outputs.logits[0, :draft_count]
        fast_strict_ngram_diagnostics = bool(
            args.fast_strict_ngram_diagnostics
            and args.compact_runtime_stats
            and effective_commit_policy == "target_match"
            and ngram_cycle
            and args.ngram_strict_target_match
            and not need_target_margin
        )
        fast_strict_ngram_diagnostics = fast_strict_ngram_diagnostics or exact_id_diagnostics_allowed(
            args, effective_commit_policy, need_target_margin
        )
        target_predictions = target_logits_block.argmax(dim=-1)
        need_target_acceptance_diagnostics = bool(
            not args.compact_runtime_stats
            or effective_commit_policy in {"target_match", "target_match_expand"}
        )
        target_metric_rows = None
        if need_target_acceptance_diagnostics and draft_count > 0:
            if fast_strict_ngram_diagnostics:
                nan = float("nan")
                target_metric_rows = [
                    [float(token), nan, nan, nan, nan, nan]
                    for token in target_predictions.cpu().tolist()
                ]
                stats["fast_strict_ngram_diagnostic_blocks" if ngram_cycle else "fast_strict_target_diagnostic_blocks"] += 1
            else:
                target_logits_block = target_logits_block.float()
        if (
            need_target_acceptance_diagnostics
            and draft_count > 0
            and not fast_strict_ngram_diagnostics
        ):
            draft_id_tensor = torch.tensor(block_tokens[1:], dtype=torch.long, device=device)
            row_indices = torch.arange(draft_count, device=device)
            draft_logits = target_logits_block[row_indices, draft_id_tensor]
            relative_supports = torch.exp(draft_logits - target_logits_block.max(dim=-1).values)

            nan_values = torch.full_like(relative_supports, float("nan"))
            target_entropies = nan_values
            draft_probabilities = nan_values
            target_top1_probabilities = nan_values
            if not args.compact_runtime_stats:
                target_probs_block = torch.softmax(target_logits_block, dim=-1)
                draft_probabilities = target_probs_block[row_indices, draft_id_tensor]
                target_top1_probabilities = target_probs_block[row_indices, target_predictions]
                target_entropies = -(
                    target_probs_block
                    * torch.log(target_probs_block.clamp_min(1e-12))
                ).sum(dim=-1)
            elif effective_commit_policy == "target_match":
                entropy_thresholds = (
                    [None] * draft_count
                    if ngram_cycle and args.ngram_strict_target_match
                    else [
                        mismatch_entropy_for_position(args, generated_start + draft_idx)
                        for draft_idx in range(1, len(block_tokens))
                    ]
                )
                if any(value is not None for value in entropy_thresholds):
                    if args.sparse_target_entropy:
                        lambda_thresholds = [
                            target_match_lambda_for_position(args, generated_start + draft_idx)
                            for draft_idx in range(1, len(block_tokens))
                        ]
                        if target_match_category_lambdas:
                            minimum_category_lambda = min(target_match_category_lambdas.values())
                            lambda_thresholds = [
                                min(value, minimum_category_lambda) for value in lambda_thresholds
                            ]
                        lambda_tensor = torch.tensor(
                            lambda_thresholds,
                            dtype=relative_supports.dtype,
                            device=device,
                        )
                        entropy_active = torch.tensor(
                            [value is not None for value in entropy_thresholds],
                            dtype=torch.bool,
                            device=device,
                        )
                        entropy_rows = (
                            entropy_active
                            & target_predictions.ne(draft_id_tensor)
                            & ((lambda_tensor <= 0.0) | (relative_supports >= lambda_tensor))
                        )
                        selected_logits = target_logits_block[entropy_rows]
                        selected_probs = torch.softmax(selected_logits, dim=-1)
                        selected_entropies = -(
                            selected_probs * torch.log(selected_probs.clamp_min(1e-12))
                        ).sum(dim=-1)
                        target_entropies = nan_values.clone()
                        target_entropies[entropy_rows] = selected_entropies
                    else:
                        target_probs_block = torch.softmax(target_logits_block, dim=-1)
                        target_entropies = -(
                            target_probs_block
                            * torch.log(target_probs_block.clamp_min(1e-12))
                        ).sum(dim=-1)

            target_top1_margins = nan_values
            if need_target_margin:
                target_top2 = torch.topk(target_logits_block, k=2, dim=-1).values
                target_top1_margins = target_top2[:, 0] - target_top2[:, 1]

            # One device-to-host synchronization for every target diagnostic in the block.
            target_metric_rows = torch.stack(
                [
                    target_predictions.float(),
                    relative_supports,
                    target_entropies,
                    draft_probabilities,
                    target_top1_probabilities,
                    target_top1_margins,
                ],
                dim=-1,
            ).cpu().tolist()
        timings["acceptance_diagnostics_s"] += (
            time.perf_counter() - acceptance_diagnostics_start
        )

        target_matches = None
        if need_target_acceptance_diagnostics:
            target_match_compare_start = time.perf_counter()
            target_matches = [
                block_tokens[draft_idx] == int(target_metric_rows[draft_idx - 1][0])
                for draft_idx in range(1, len(block_tokens))
            ]
            timings["target_match_compare_s"] += (
                time.perf_counter() - target_match_compare_start
            )

        acceptance_record_start = time.perf_counter()
        for draft_idx in range(1, len(block_tokens)):
            target_token = None
            matched = None
            target_top1_margin = None
            relative_support = None
            if need_target_acceptance_diagnostics:
                metric_row = target_metric_rows[draft_idx - 1]
                target_token = int(metric_row[0])
                matched = target_matches[draft_idx - 1]
                if not fast_strict_ngram_diagnostics:
                    relative_support = float(metric_row[1])
                stats["matched_draft_tokens"] += int(matched)
                stats["target_match_observations"] += 1
                if relative_support is not None:
                    stats["target_relative_supports"].append(relative_support)
                if need_target_margin:
                    target_top1_margin = float(metric_row[5])
            need_token_category = bool(
                not args.compact_runtime_stats
                or draft_category_gate
                or target_match_category_lambdas
            )
            draft_token_text = None
            if tokenizer is not None and need_token_category:
                draft_token_text = decode_token(block_tokens[draft_idx])
            draft_token_category = (
                token_category_from_text(draft_token_text or "")
                if draft_token_text is not None
                else None
            )
            draft_category_allowed = (
                draft_token_category in draft_category_gate
                if draft_category_gate
                else None
            )
            lambda_used = None
            if effective_commit_policy == "target_match":
                lambda_used = (
                    1.0
                    if ngram_cycle and args.ngram_strict_target_match
                    else target_match_category_lambdas.get(
                        draft_token_category or "",
                        target_match_lambda_for_position(
                            args,
                            generated_start + draft_idx,
                        ),
                    )
                )
            mismatch_entropy_threshold = (
                None
                if ngram_cycle and args.ngram_strict_target_match
                else mismatch_entropy_for_position(args, generated_start + draft_idx)
                if effective_commit_policy == "target_match"
                else None
            )
            need_entropy = bool(
                effective_commit_policy == "target_match"
                and mismatch_entropy_threshold is not None
                and matched is False
                and draft_category_allowed is not False
                and lambda_used is not None
                and (
                    lambda_used <= 0.0
                    or (relative_support is not None and relative_support >= lambda_used)
                )
            )
            target_entropy = None
            target_top1_prob = None
            draft_prob = None
            if not args.compact_runtime_stats or need_entropy:
                metric_row = target_metric_rows[draft_idx - 1]
                target_entropy = float(metric_row[2])
                draft_prob = float(metric_row[3])
                target_top1_prob = float(metric_row[4])
            block_draft_records.append(
                {
                    "cycle": cycle,
                    "draft_index": draft_idx,
                    "generated_position": generated_start + draft_idx,
                    "sequence_position": old_length + draft_idx,
                    "token": block_tokens[draft_idx],
                    "token_text": draft_token_text,
                    "token_category": draft_token_category,
                    "draft_source": "ngram" if ngram_cycle else "latent",
                    "draft_category_gate": sorted(draft_category_gate) if draft_category_gate else None,
                    "draft_category_allowed": draft_category_allowed,
                    "target_match_category_lambdas": (
                        target_match_category_lambdas if target_match_category_lambdas else None
                    ),
                    "target_match_lambda_used": lambda_used,
                    "mismatch_min_target_entropy_used": mismatch_entropy_threshold,
                    "target_token": target_token,
                    "margin": block_margins[draft_idx - 1],
                    "min_draft_margin_used": min_draft_margin_for_position(
                        args, generated_start + draft_idx
                    ),
                    "verifier_score": block_scores[draft_idx - 1],
                    "verifier_threshold": block_thresholds[draft_idx - 1],
                    "match": matched,
                    "target_relative_support": relative_support,
                    "target_entropy": target_entropy,
                    "target_top1_prob": target_top1_prob,
                    "target_top1_margin": target_top1_margin,
                    "draft_prob": draft_prob,
                }
            )
        timings["acceptance_record_s"] += time.perf_counter() - acceptance_record_start

        acceptance_policy_start = time.perf_counter()
        strict_safe_drafts = 0
        expanded_safe_drafts = 0
        if effective_commit_policy == "refine_block":
            refined_tokens = [block_tokens[0]]
            refined_tokens.extend(
                int(target_predictions[draft_idx - 1].item())
                for draft_idx in range(1, len(block_tokens))
            )
            if refined_tokens != block_tokens:
                stats["refinement_passes"] += 1
                for token_idx, token in enumerate(refined_tokens):
                    if token in eos_ids:
                        refined_tokens = refined_tokens[: token_idx + 1]
                        break
                verify_outputs.past_key_values.crop(old_length)
                refined_tensor = torch.tensor([refined_tokens], dtype=torch.long, device=device)
                verify_outputs, elapsed = timed(
                    lambda: tracked_target_forward(
                        model,
                        refined_tensor,
                        past_key_values=verify_outputs.past_key_values,
                        start_position=old_length,
                        omit_attention_mask=args.omit_target_attention_mask,
                    ),
                    device,
                )
                timings["refine_s"] += elapsed
                stats["target_calls"] += 1
                block_tokens = refined_tokens
            accepted = len(block_tokens)
        elif effective_commit_policy in {"whole_block", "ngram_prefix"}:
            accepted = len(block_tokens)
        elif effective_commit_policy == "category_block":
            accepted = 1
            unchecked_budget_used = stats["accepted_unchecked_tokens"]
            pending_unchecked_positions: list[int] = []
            for record in block_draft_records:
                if record.get("draft_category_allowed") is False:
                    break
                if (
                    args.max_accepted_unchecked_per_sequence is not None
                    and unchecked_budget_used >= args.max_accepted_unchecked_per_sequence
                ):
                    break
                if args.max_accepted_unchecked_per_window is not None:
                    generated_position = int(record["generated_position"])
                    window = generated_position // args.unchecked_budget_window_tokens
                    used_in_window = sum(
                        int(position) // args.unchecked_budget_window_tokens == window
                        for position in (
                            stats["accepted_unchecked_positions"] + pending_unchecked_positions
                        )
                    )
                    if used_in_window >= args.max_accepted_unchecked_per_window:
                        break
                accepted += 1
                unchecked_budget_used += 1
                pending_unchecked_positions.append(int(record["generated_position"]))
        elif effective_commit_policy == "draft_margin":
            accepted = 1 + count_accepted_draft_margin_records(
                block_draft_records,
                args.min_draft_margin,
                args.max_accepted_unchecked_per_sequence,
                stats["accepted_unchecked_tokens"],
                max_unchecked_per_window=args.max_accepted_unchecked_per_window,
                unchecked_budget_window_tokens=args.unchecked_budget_window_tokens,
                accepted_unchecked_positions=stats["accepted_unchecked_positions"],
                margin_threshold_for_position=lambda position: min_draft_margin_for_position(
                    args, position
                ),
            )
        elif effective_commit_policy == "target_match_expand":
            strict_accepted = 1
            for record in block_draft_records:
                if record.get("draft_category_allowed") is False:
                    break
                if not record["match"]:
                    break
                strict_accepted += 1

            strict_safe_drafts = max(0, strict_accepted - 1)
            expanded_safe_drafts = strict_safe_drafts
            if (
                generated_start >= args.expand_after_position
                and strict_safe_drafts >= args.expand_min_safe_drafts
            ):
                category_safe_drafts = 0
                for record in block_draft_records:
                    if record.get("draft_category_allowed") is False:
                        break
                    category_safe_drafts += 1
                expanded_safe_drafts = min(
                    category_safe_drafts,
                    args.expand_max_accepted_drafts,
                    max(strict_safe_drafts, int(math.ceil(strict_safe_drafts * args.expand_multiplier))),
                )
                expanded_safe_drafts = limit_target_match_expand_records(
                    block_draft_records,
                    expanded_safe_drafts,
                    stats["accepted_mismatch_tokens"],
                    args.max_accepted_mismatches_per_sequence,
                )
            if expanded_safe_drafts > strict_safe_drafts:
                stats["target_match_expand_events"] += 1
                stats["target_match_expand_extra_tokens"] += expanded_safe_drafts - strict_safe_drafts
            accepted = 1 + expanded_safe_drafts
        else:
            accepted = 1
            mismatch_budget_used = stats["accepted_mismatch_tokens"]
            for draft_idx in range(1, len(block_tokens)):
                record = block_draft_records[draft_idx - 1]
                if record.get("draft_category_allowed") is False:
                    break
                if (
                    args.serial_fallback_margin is not None
                    and record["target_top1_margin"] <= args.serial_fallback_margin
                ):
                    break
                lambda_for_token = float(record["target_match_lambda_used"])
                accept_draft = target_match_accepts_candidate(
                    bool(record["match"]),
                    record["target_relative_support"],
                    lambda_for_token,
                )
                if (
                    accept_draft
                    and not record["match"]
                    and record["mismatch_min_target_entropy_used"] is not None
                ):
                    if record["target_entropy"] is None:
                        raise RuntimeError("Target entropy was not computed for an accepted mismatch.")
                    if record["target_entropy"] < record["mismatch_min_target_entropy_used"]:
                        accept_draft = False
                if accept_draft and not record["match"]:
                    if (
                        args.max_accepted_mismatches_per_sequence is not None
                        and mismatch_budget_used >= args.max_accepted_mismatches_per_sequence
                    ):
                        accept_draft = False
                    else:
                        mismatch_budget_used += 1
                if not accept_draft:
                    break
                accepted += 1
        timings["acceptance_policy_s"] += time.perf_counter() - acceptance_policy_start

        if precommit_unchecked and accepted != len(block_tokens):
            raise RuntimeError(
                "Unchecked precommit must make the post-forward commit consume the full trimmed block."
            )

        accepted_tokens = block_tokens[:accepted]
        wasted_drafts = max(0, len(block_tokens) - accepted)
        if wasted_drafts:
            stats["verification_rollbacks"] += 1
            stats["verification_wasted_drafts"] += wasted_drafts
            stats["verification_first_divergence_events"].append(
                {
                    "cycle": cycle,
                    "block_offset": accepted,
                    "generated_position": generated_start + accepted,
                    "sequence_position": old_length + accepted,
                    "proposed_block_tokens": len(block_tokens),
                    "wasted_drafts": wasted_drafts,
                }
            )
        if agreement_gate_active_cycle:
            active_matches = list(target_matches or [])
            decision = agreement_gate_decision(
                verification_agreement_streak,
                active_matches,
                [record.get("margin") for record in block_draft_records],
                [record.get("target_top1_margin") for record in block_draft_records],
                min_consecutive_matches=args.verification_agreement_warmup,
                min_draft_margin=args.verification_agreement_min_draft_margin,
                min_target_margin=args.verification_agreement_min_target_margin,
            )
            verification_agreement_streak = decision.next_streak
            verification_agreement_gate_open = decision.gate_open
            stats["verification_agreement_max_streak"] = max(
                stats["verification_agreement_max_streak"],
                verification_agreement_streak,
            )
            if not verification_agreement_gate_open:
                stats["verification_agreement_gate_closures"] += 1
            stats["verification_agreement_events"].append(
                {
                    "cycle": cycle,
                    "kind": "active_block",
                    "generated_position": generated_start,
                    "drafts": len(active_matches),
                    "all_match": bool(active_matches) and all(active_matches),
                    "accepted_len": accepted,
                    "proposed_len": len(block_tokens),
                    "streak": verification_agreement_streak,
                    "gate_open": verification_agreement_gate_open,
                    "reason": decision.reason,
                }
            )
        prefetched_sync_cache = None
        prefetched_sync_anchor = None
        if lookahead_state:
            caller_stream = torch.cuda.current_stream(device)
            caller_stream.wait_stream(lookahead_stream)
            lookahead_tokens = lookahead_state["tokens"]
            lookahead_tokens.record_stream(caller_stream)
            lookahead_state["synced_next_anchor"].record_stream(caller_stream)
            record_cache_stream(lookahead_state["synced_t_cache"], caller_stream)
            lookahead_token_list = [int(token) for token in lookahead_tokens[0].cpu().tolist()]
            timings["preemptive_compute_s"] += (
                lookahead_state["start_event"].elapsed_time(lookahead_state["end_event"]) / 1000.0
            )
            if accepted == len(block_tokens):
                prefetched_sync_cache = lookahead_state["synced_t_cache"]
                prefetched_sync_anchor = lookahead_state["synced_next_anchor"]
                target_next_token = int(verify_outputs.logits[0, -1].argmax(dim=-1).item())
                if lookahead_token_list and lookahead_token_list[0] == target_next_token:
                    for token_index, token in enumerate(lookahead_token_list):
                        if token in eos_ids:
                            lookahead_token_list = lookahead_token_list[: token_index + 1]
                            break
                    pending_lookahead_tokens = lookahead_token_list
                    stats["preemptive_token_hits"] += 1
                else:
                    stats["preemptive_token_misses"] += 1
            else:
                stats["preemptive_branch_discards"] += 1
        boundary_margin = None
        use_serial_fallback = False
        if args.serial_fallback_margin is not None or not args.compact_runtime_stats:
            boundary_logits = verify_outputs.logits[0, accepted - 1].float()
            boundary_top2 = torch.topk(boundary_logits, k=2).values
            boundary_margin = float((boundary_top2[0] - boundary_top2[1]).item())
            use_serial_fallback = bool(
                args.serial_fallback_margin is not None
                and boundary_margin <= args.serial_fallback_margin
            )
        generated.extend(accepted_tokens)
        stats["accepted_draft_tokens"] += max(0, accepted - 1)
        if ngram_cycle:
            stats["accepted_ngram_tokens"] += max(0, accepted - 1)
        if ngram_wide_cycle:
            stats["ngram_wide_route_observations"] += 1
            stats["ngram_wide_route_extra_accepted_tokens"] += max(
                0,
                accepted - 1 - args.ngram_wide_auto_route_fallback_width,
            )
            if not ngram_wide_route_decided:
                disable_wide, route_score = ngram_wide_route_decision(
                    stats["ngram_wide_route_observations"],
                    stats["ngram_wide_route_extra_accepted_tokens"],
                    args.ngram_wide_auto_route_after_hits,
                    args.ngram_wide_auto_route_min_extra_per_hit,
                )
                if route_score is not None:
                    ngram_wide_route_decided = True
                    ngram_wide_route_disabled = disable_wide
                    stats["ngram_wide_route_evaluations"] += 1
                    stats["ngram_wide_route_score"] = route_score
                    if disable_wide:
                        stats["ngram_wide_route_disables"] += 1
                        stats["ngram_wide_route_disable_position"] = generated_start
        attempted_latent_drafts = max(0, len(block_tokens) - 1) if uses_latent_drafter else 0
        if attempted_latent_drafts > 0:
            recent_drafts.append((max(0, accepted - 1), attempted_latent_drafts))
        if args.draft_cooldown_after_failures > 0 and attempted_latent_drafts > 0:
            if accepted <= 1:
                consecutive_failed_draft_cycles += 1
                if consecutive_failed_draft_cycles >= args.draft_cooldown_after_failures:
                    cooldown_remaining = args.draft_cooldown_cycles
                    consecutive_failed_draft_cycles = 0
                    stats["draft_cooldown_activations"] += 1
            else:
                consecutive_failed_draft_cycles = 0
        if uses_latent_drafter:
            stats["auto_route_latent_drafts"] += attempted_latent_drafts
            stats["auto_route_accepted_drafts"] += max(0, accepted - 1)
            stats["auto_route_latent_candidate_cycles"] += 1
            if attempted_latent_drafts > 0:
                stats["auto_route_confident_candidate_cycles"] += 1
            route_on_proposal_rate = effective_commit_policy in {
                "draft_margin",
                "whole_block",
                "category_block",
            }
            if route_on_proposal_rate:
                route_observations = stats["auto_route_latent_candidate_cycles"]
                route_successes = stats["auto_route_confident_candidate_cycles"]
                stats["auto_route_score_kind"] = "confident_candidate_rate"
            else:
                route_observations = stats["auto_route_latent_drafts"]
                route_successes = stats["auto_route_accepted_drafts"]
                stats["auto_route_score_kind"] = "target_acceptance"
            if (
                args.auto_route_early_zero_after > 0
                and not auto_route_early_decided
                and route_observations >= args.auto_route_early_zero_after
            ):
                auto_route_early_decided = True
                stats["auto_route_early_evaluations"] += 1
                if route_successes == 0:
                    auto_route_decided = True
                    recurrent_active = False
                    true_t_cache = DynamicCache()
                    true_next_anchor = None
                    deferred_t_prompt_anchors = None
                    pending_t_sync_anchors.clear()
                    pending_t_sync_tokens.clear()
                    stats["auto_route_evaluations"] += 1
                    stats["auto_route_t_disables"] += 1
                    stats["auto_route_early_t_disables"] += 1
                    stats["auto_route_observed_acceptance"] = 0.0
                    stats["auto_route_disable_position"] = generated_start
            route_evaluated, disable_route, next_evaluation, observed_acceptance = (
                auto_route_monitor_decision(
                    route_observations,
                    route_successes,
                    auto_route_next_evaluation,
                    args.auto_route_min_latent_acceptance,
                    args.auto_route_reevaluate_every,
                )
            )
            if not auto_route_decided and route_evaluated:
                auto_route_next_evaluation = next_evaluation
                stats["auto_route_evaluations"] += 1
                stats["auto_route_observed_acceptance"] = observed_acceptance
                if disable_route:
                    auto_route_decided = True
                    recurrent_active = False
                    true_t_cache = DynamicCache()
                    true_next_anchor = None
                    deferred_t_prompt_anchors = None
                    pending_t_sync_anchors.clear()
                    pending_t_sync_tokens.clear()
                    stats["auto_route_t_disables"] += 1
                    stats["auto_route_disable_position"] = generated_start
                elif auto_route_next_evaluation <= 0:
                    auto_route_decided = True
        stats["block_hist"][str(len(block_tokens))] = stats["block_hist"].get(str(len(block_tokens)), 0) + 1
        stats["accepted_hist"][str(accepted)] = stats["accepted_hist"].get(str(accepted), 0) + 1
        phase = (
            "early"
            if generated_start < args.medium_position
            else "mid"
            if generated_start < args.long_position
            else "late"
        )
        phase_block_hist = stats["block_hist_by_phase"][phase]
        phase_block_hist[str(len(block_tokens))] = phase_block_hist.get(str(len(block_tokens)), 0) + 1
        phase_accepted_hist = stats["accepted_hist_by_phase"][phase]
        phase_accepted_hist[str(accepted)] = phase_accepted_hist.get(str(accepted), 0) + 1
        stats["target_generated_positions"].append(
            {
                "cycle": cycle,
                "generated_position": generated_start,
                "sequence_position": old_length,
                "source": "target_next",
            }
        )
        accelerated_generated_positions = []
        accelerated_sequence_positions = []
        for record in block_draft_records:
            accepted_draft = int(record["draft_index"]) < accepted
            accelerated = accepted_draft and effective_commit_policy != "refine_block"
            record["accepted"] = accepted_draft
            record["accelerated"] = accelerated
            record["strict_safe_drafts"] = (
                strict_safe_drafts if effective_commit_policy == "target_match_expand" else None
            )
            record["expanded_safe_drafts"] = (
                expanded_safe_drafts if effective_commit_policy == "target_match_expand" else None
            )
            record["committed_source"] = (
                "draft" if accelerated else "target_refined" if accepted_draft else "not_committed"
            )
            if not args.compact_runtime_stats:
                stats["draft_records"].append(record)
            if accelerated:
                if record["match"] is False:
                    stats["accepted_mismatch_tokens"] += 1
                    stats["accepted_mismatch_positions"].append(record["generated_position"])
                    stats["accepted_mismatch_events"].append(
                        {
                            "generated_position": record["generated_position"],
                            "draft_margin": record.get("margin"),
                            "target_margin": cheap_decision.get("target_top1_margin"),
                            "token": record.get("token"),
                            "token_category": record.get("token_category"),
                            "draft_source": record.get("draft_source"),
                        }
                    )
                    source_key = (
                        "accepted_ngram_mismatch_tokens"
                        if record["draft_source"] == "ngram"
                        else "accepted_latent_mismatch_tokens"
                    )
                    stats[source_key] += 1
                elif record["match"] is None:
                    stats["accepted_unchecked_tokens"] += 1
                    stats["accepted_unchecked_positions"].append(record["generated_position"])
                    stats["accepted_unchecked_events"].append(
                        {
                            "generated_position": record["generated_position"],
                            "draft_margin": record.get("margin"),
                            "target_margin": cheap_decision.get("target_top1_margin"),
                            "token": record.get("token"),
                            "token_category": record.get("token_category"),
                            "draft_source": record.get("draft_source"),
                        }
                    )
                accelerated_generated_positions.append(record["generated_position"])
                accelerated_sequence_positions.append(record["sequence_position"])
                if not args.compact_runtime_stats:
                    stats["accelerated_positions"].append(
                        {
                            "cycle": cycle,
                            "draft_index": record["draft_index"],
                            "generated_position": record["generated_position"],
                            "sequence_position": record["sequence_position"],
                            "token": record["token"],
                            "token_text": record.get("token_text"),
                            "token_category": record.get("token_category"),
                            "verifier_score": record["verifier_score"],
                            "verifier_threshold": record["verifier_threshold"],
                            "margin": record["margin"],
                            "match": record["match"],
                            "target_relative_support": record["target_relative_support"],
                        }
                    )
        stats["block_records"].append(
            {
                "cycle": cycle,
                "generated_start": generated_start,
                "sequence_start": old_length,
                "block_limit": block_limit,
                "proposed_len": len(block_tokens),
                "original_proposed_len": original_proposed_len,
                "draft_confidence_stopped": draft_confidence_stopped,
                "draft_step_margins": list(block_margins),
                "recent_acceptance_before_cycle": recent_rate,
                "history_draft_budget": history_budget,
                "budget_before_cooldown": budget_before_cooldown,
                "gate_target_margin": target_top1_margin,
                "gate_margin_skip": bool(cheap_decision["skip"]),
                "t_cache_clone_skipped": not uses_latent_drafter,
                "accepted_len": accepted,
                "precommit_unchecked": precommit_unchecked,
                "target_boundary_margin": boundary_margin,
                "serial_fallback": use_serial_fallback,
                "policy": effective_commit_policy,
                "target_match_lambda": (
                    args.target_match_lambda if effective_commit_policy == "target_match" else None
                ),
                "target_match_category_lambdas": (
                    target_match_category_lambdas
                    if effective_commit_policy == "target_match" and target_match_category_lambdas
                    else None
                ),
                "strict_safe_drafts": strict_safe_drafts
                if effective_commit_policy == "target_match_expand"
                else None,
                "expanded_safe_drafts": expanded_safe_drafts
                if effective_commit_policy == "target_match_expand"
                else None,
                "expand_after_position": args.expand_after_position
                if effective_commit_policy == "target_match_expand"
                else None,
                "expand_min_safe_drafts": args.expand_min_safe_drafts
                if effective_commit_policy == "target_match_expand"
                else None,
                "expand_multiplier": args.expand_multiplier
                if effective_commit_policy == "target_match_expand"
                else None,
                "mode": args.mode,
                "verifier_threshold": args.verifier_threshold if args.mode == "verifier" else None,
                "cheap_policy": cheap_decision if args.cheap_verifier_policy != "none" else None,
                "latent_target_margin": target_top1_margin,
                "latent_target_margin_gated": latent_target_margin_gated,
                "preemptive_reused": prefetched_cycle,
                "ngram_reused": ngram_cycle,
                "ngram_size": ngram_size,
                "ngram_occurrence_policy": active_ngram_occurrence_policy,
                "ngram_tree_verify": ngram_tree is not None,
                "ngram_tree_branch_limit": ngram_tree_branch_limit,
                "ngram_tree_branches": len(ngram_tree_branches),
                "ngram_tree_nodes": len(ngram_tree.nodes) if ngram_tree is not None else 0,
                "ngram_tree_selected_path": (
                    list(ngram_tree_path_indices)
                    if ngram_tree_path_indices is not None
                    else None
                ),
                "ngram_tree_target_matched_drafts": ngram_tree_matched_drafts,
                "ngram_predecode_occurrence_reason": ngram_predecode_occurrence_reason,
                "ngram_scheduled_width": ngram_scheduled_width,
                "ngram_active_width": active_ngram_max_draft_tokens,
                "ngram_wide_cycle": ngram_wide_cycle,
                "ngram_wide_route_disabled": ngram_wide_route_disabled,
                "ngram_wide_route_score": stats["ngram_wide_route_score"],
                "ngram_predecode_route_width": ngram_predecode_width,
                "ngram_predecode_route_reason": ngram_predecode_reason,
                "ngram_unchecked_eligible": ngram_unchecked_eligible,
                "ngram_unchecked_min_match_size": args.ngram_unchecked_min_match_size,
                "ngram_unchecked_min_position": args.ngram_unchecked_min_position,
                "ngram_unchecked_token_categories": sorted(ngram_unchecked_categories),
                "ngram_unchecked_draft_allowed": ngram_unchecked_draft_allowed,
                "ngram_unchecked_category_closed": ngram_unchecked_category_closed,
                "draft_cooldown": cooldown_cycle,
                "allow_zero_draft": args.allow_zero_draft,
                "verification_agreement_gate_active": agreement_gate_active_cycle,
                "verification_agreement_shadow_probe": agreement_shadow_probe,
                "verification_agreement_streak": verification_agreement_streak,
                "verification_agreement_gate_open_next": verification_agreement_gate_open,
                "target_generated_position": generated_start,
                "target_sequence_position": old_length,
                "accelerated_generated_positions": accelerated_generated_positions,
                "accelerated_sequence_positions": accelerated_sequence_positions,
                "draft_records": [] if args.compact_runtime_stats else block_draft_records,
                "rejected_draft_records": [] if args.compact_runtime_stats else block_rejected_records,
            }
        )
        if any(token in eos_ids for token in accepted_tokens):
            break

        if accepted == len(block_tokens):
            if args.replay_accepted_target_cache or use_serial_fallback:
                if (
                    use_serial_fallback
                    and args.serial_fallback_full_replay_margin is not None
                    and boundary_margin is not None
                    and boundary_margin <= args.serial_fallback_full_replay_margin
                ):
                    committed_anchor, next_logits = rebuild_target_cache_serial(
                        generated,
                        len(accepted_tokens),
                    )
                else:
                    committed_anchor, next_logits = commit_target_tokens(
                        accepted_tokens,
                        old_length,
                        count_as_replay=True,
                        force_serial=use_serial_fallback,
                    )
                if use_serial_fallback:
                    stats["serial_fallback_events"] += 1
            else:
                committed_anchor = (
                    verify_outputs.hidden_states[anchor_idx][:, :accepted, :]
                    if recurrent_active
                    else None
                )
                target_cache = verify_outputs.past_key_values
                next_logits = verify_outputs.logits[:, -1, :]
            if not recurrent_active:
                pass
            elif prefetched_sync_cache is not None:
                true_t_cache = prefetched_sync_cache
                true_next_anchor = prefetched_sync_anchor
                stats["preemptive_sync_reuses"] += 1
            elif args.lazy_t_sync and not uses_latent_drafter:
                queue_t_sync(committed_anchor, accepted_tokens)
            else:
                (true_t_output, true_t_cache), elapsed = timed(
                    lambda: recurrent.forward_with_cache(
                        committed_anchor,
                        model=get_base_causal_lm(model),
                        past_key_value=true_t_cache,
                        token_ids=torch.tensor([accepted_tokens], dtype=torch.long, device=device),
                    ),
                    device,
                )
                timings["t_sync_s"] += elapsed
                true_next_anchor = true_t_output[:, -1:, :]
            continue

        block_corrective = (
            int(target_metric_rows[accepted - 1][0])
            if target_metric_rows is not None
            else int(target_predictions[accepted - 1].item())
        )
        committed_prefix_anchor = None
        serial_boundary_logits = None
        if use_serial_fallback:
            if (
                args.serial_fallback_full_replay_margin is not None
                and boundary_margin is not None
                and boundary_margin <= args.serial_fallback_full_replay_margin
            ):
                committed_prefix_anchor, serial_boundary_logits = rebuild_target_cache_serial(
                    generated,
                    len(accepted_tokens),
                )
            else:
                committed_prefix_anchor, serial_boundary_logits = commit_target_tokens(
                    accepted_tokens,
                    old_length,
                    count_as_replay=True,
                    force_serial=True,
                )
            corrective = int(serial_boundary_logits.argmax(dim=-1).item())
            stats["serial_fallback_events"] += 1
            stats["serial_fallback_token_changes"] += int(corrective != block_corrective)
        else:
            corrective = block_corrective
        stats["block_records"][-1]["block_corrective_token"] = block_corrective
        stats["block_records"][-1]["committed_corrective_token"] = corrective
        defer_correction = bool(
            args.defer_correction_to_next_verify
            and not use_serial_fallback
            and accepted - 1 >= args.defer_correction_min_accepted_drafts
            and generated_start >= args.defer_correction_after_position
            and (
                args.defer_correction_before_position is None
                or generated_start < args.defer_correction_before_position
            )
        )
        if defer_correction:
            # Keep the corrective token one step ahead of the live target cache. The next cycle
            # verifies it together with its drafts, avoiding a standalone full-model forward.
            target_cache = verify_outputs.past_key_values
            target_cache.crop(old_length + len(accepted_tokens))
            next_logits = verify_outputs.logits[:, accepted - 1, :]
            committed_anchor = (
                verify_outputs.hidden_states[anchor_idx][:, : len(accepted_tokens), :]
                if recurrent_active
                else None
            )
            if recurrent_active:
                if args.lazy_t_sync:
                    queue_t_sync(committed_anchor, accepted_tokens)
                else:
                    (true_t_output, true_t_cache), elapsed = timed(
                        lambda: recurrent.forward_with_cache(
                            committed_anchor,
                            model=get_base_causal_lm(model),
                            past_key_value=true_t_cache,
                            token_ids=torch.tensor(
                                [accepted_tokens], dtype=torch.long, device=device
                            ),
                        ),
                        device,
                    )
                    timings["t_sync_s"] += elapsed
                    true_next_anchor = true_t_output[:, -1:, :]
            stats["deferred_corrective_tokens"] += 1
            stats["block_records"][-1]["corrective_deferred"] = True
            continue

        commit_tokens = accepted_tokens + [corrective]
        if len(generated) >= args.max_new_tokens:
            break
        corrective_generated_position = len(generated)
        generated.append(corrective)
        stats["corrective_tokens"] += 1
        stats["target_generated_positions"].append(
            {
                "cycle": cycle,
                "generated_position": corrective_generated_position,
                "sequence_position": input_ids.size(1) + corrective_generated_position,
                "source": "target_correction",
                "token": corrective,
            }
        )
        if corrective in eos_ids:
            break

        corrective_anchor = None
        committed_anchor = None
        if use_serial_fallback:
            corrective_anchor, next_logits = commit_target_tokens(
                [corrective],
                old_length + len(accepted_tokens),
                force_serial=True,
            )
            committed_anchor = torch.cat([committed_prefix_anchor, corrective_anchor], dim=1)
        elif args.reuse_verify_cache_for_correction:
            target_cache = verify_outputs.past_key_values
            target_cache.crop(old_length + len(accepted_tokens))
            accepted_anchor = (
                verify_outputs.hidden_states[anchor_idx][:, : len(accepted_tokens), :]
                if recurrent_active
                else None
            )
            if correction_sync_stream is not None:
                (corrective_outputs, _, true_t_cache), elapsed = timed(
                    lambda: parallel_corrective_target_and_t_sync(
                        model,
                        recurrent,
                        accepted_anchor,
                        accepted_tokens,
                        target_cache,
                        true_t_cache,
                        corrective,
                        old_length + len(accepted_tokens),
                        args.omit_target_attention_mask,
                        device,
                        correction_sync_stream,
                    ),
                    device,
                )
                timings["correction_tsync_parallel_s"] += elapsed
            else:
                corrective_tensor = torch.tensor([[corrective]], dtype=torch.long, device=device)
                corrective_outputs, elapsed = timed(
                    lambda: tracked_target_forward(
                        model,
                        corrective_tensor,
                        past_key_values=target_cache,
                        start_position=old_length + len(accepted_tokens),
                        output_hidden_states=recurrent_active,
                        omit_attention_mask=args.omit_target_attention_mask,
                    ),
                    device,
                )
                timings["correction_s"] += elapsed
            stats["target_calls"] += 1
            stats["fast_correction_cache_reuses"] += 1
            target_cache = corrective_outputs.past_key_values
            if recurrent_active:
                corrective_anchor = corrective_outputs.hidden_states[anchor_idx]
                committed_anchor = torch.cat([accepted_anchor, corrective_anchor], dim=1)
            next_logits = corrective_outputs.logits[:, -1, :]
        else:
            committed_anchor, next_logits = commit_target_tokens(commit_tokens, old_length)
        t_sync_anchor = corrective_anchor if correction_sync_stream is not None and not use_serial_fallback else committed_anchor
        t_sync_tokens = [corrective] if correction_sync_stream is not None and not use_serial_fallback else commit_tokens
        if recurrent_active and args.lazy_t_sync and not uses_latent_drafter:
            queue_t_sync(committed_anchor, commit_tokens)
        elif recurrent_active:
            (true_t_output, true_t_cache), elapsed = timed(
                lambda: recurrent.forward_with_cache(
                    t_sync_anchor,
                    model=get_base_causal_lm(model),
                    past_key_value=true_t_cache,
                    token_ids=torch.tensor([t_sync_tokens], dtype=torch.long, device=device),
                ),
                device,
            )
            timings["t_sync_s"] += elapsed
            true_next_anchor = true_t_output[:, -1:, :]

    if anchor_hook is not None:
        anchor_hook.remove()
    return {"token_ids": generated[: args.max_new_tokens], "timings": timings, **stats}


def main() -> None:
    global _SYNC_COMPONENT_TIMING
    args = parse_args()
    _SYNC_COMPONENT_TIMING = not args.production_async_timing
    if args.verifier_score_temperature <= 0.0:
        raise ValueError("--verifier-score-temperature must be positive.")
    if not 0.0 <= args.verification_risk_budget <= 1.0:
        raise ValueError("--verification-risk-budget must be between 0 and 1.")
    if args.max_block_tokens <= 0:
        raise ValueError("--max-block-tokens must be positive.")
    if args.min_block_tokens <= 0 or args.min_block_tokens > args.max_block_tokens:
        raise ValueError("--min-block-tokens must be between 1 and --max-block-tokens.")
    if args.verification_frequency_policy == "risk_budget":
        if args.mode != "verifier":
            raise ValueError("Risk-budgeted verification frequency requires --mode verifier.")
        if args.ngram_draft_mode != "off" or args.ngram_tree_verify:
            raise ValueError("Risk-budgeted verification frequency currently requires latent-only drafting.")
        if args.precommit_unchecked_drafts:
            raise ValueError("Risk-budgeted verification frequency is incompatible with unchecked commits.")
        if args.draft_commit_policy != "target_match" or args.latent_draft_commit_policy not in {
            None,
            "target_match",
        }:
            raise ValueError("Risk-budgeted verification frequency requires strict target_match commits.")
    if args.verification_frequency_policy == "agreement_gated":
        if args.mode != "verifier":
            raise ValueError("Agreement-gated verification frequency requires --mode verifier.")
        if args.ngram_draft_mode != "off" or args.ngram_tree_verify:
            raise ValueError("Agreement-gated verification frequency currently requires latent-only drafting.")
        if args.precommit_unchecked_drafts:
            raise ValueError("Agreement-gated verification frequency is incompatible with unchecked commits.")
        if args.draft_commit_policy != "target_match" or args.latent_draft_commit_policy not in {
            None,
            "target_match",
        }:
            raise ValueError("Agreement-gated verification frequency requires strict target_match commits.")
        if args.verification_agreement_warmup <= 0:
            raise ValueError("--verification-agreement-warmup must be positive.")
        if args.verification_agreement_min_draft_margin < 0.0:
            raise ValueError("--verification-agreement-min-draft-margin must be non-negative.")
        if args.verification_agreement_min_target_margin < 0.0:
            raise ValueError("--verification-agreement-min-target-margin must be non-negative.")
    if not 0.0 <= args.target_match_lambda <= 1.0:
        raise ValueError("--target-match-lambda must be between 0 and 1.")
    for name in (
        "target_match_lambda_early",
        "target_match_lambda_mid",
        "target_match_lambda_late",
    ):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")
    for name in (
        "mismatch_min_target_entropy",
        "mismatch_min_target_entropy_early",
        "mismatch_min_target_entropy_mid",
        "mismatch_min_target_entropy_late",
    ):
        value = getattr(args, name)
        if value is not None and value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if (
        args.max_accepted_mismatches_per_sequence is not None
        and args.max_accepted_mismatches_per_sequence < 0
    ):
        raise ValueError("--max-accepted-mismatches-per-sequence must be non-negative.")
    if (
        args.max_accepted_unchecked_per_sequence is not None
        and args.max_accepted_unchecked_per_sequence < 0
    ):
        raise ValueError("--max-accepted-unchecked-per-sequence must be non-negative.")
    if (
        args.max_accepted_unchecked_per_window is None
    ) != (
        args.unchecked_budget_window_tokens is None
    ):
        raise ValueError(
            "--max-accepted-unchecked-per-window and --unchecked-budget-window-tokens "
            "must be provided together."
        )
    if (
        args.max_accepted_unchecked_per_window is not None
        and args.max_accepted_unchecked_per_window < 0
    ):
        raise ValueError("--max-accepted-unchecked-per-window must be non-negative.")
    if (
        args.unchecked_budget_window_tokens is not None
        and args.unchecked_budget_window_tokens <= 0
    ):
        raise ValueError("--unchecked-budget-window-tokens must be positive.")
    if args.precommit_unchecked_drafts and not args.compact_runtime_stats:
        raise ValueError("--precommit-unchecked-drafts requires --compact-runtime-stats.")
    if args.precommit_unchecked_drafts and args.serial_fallback_margin is not None:
        raise ValueError("--precommit-unchecked-drafts is incompatible with serial fallback.")
    latent_unchecked_categories = parse_category_set(args.latent_unchecked_token_categories)
    unknown_latent_categories = latent_unchecked_categories - VALID_TOKEN_CATEGORIES
    if unknown_latent_categories:
        raise ValueError(
            "Unknown --latent-unchecked-token-categories values: "
            f"{sorted(unknown_latent_categories)}."
        )
    if latent_unchecked_categories and not args.precommit_unchecked_drafts:
        raise ValueError(
            "--latent-unchecked-token-categories requires --precommit-unchecked-drafts."
        )
    if (
        args.latent_unchecked_category_reject_closes_sequence
        and not latent_unchecked_categories
    ):
        raise ValueError(
            "--latent-unchecked-category-reject-closes-sequence requires latent token categories."
        )
    if args.split_unchecked_target_forward and not args.precommit_unchecked_drafts:
        raise ValueError(
            "--split-unchecked-target-forward requires --precommit-unchecked-drafts."
        )
    if args.split_unchecked_parallel and not args.split_unchecked_target_forward:
        raise ValueError(
            "--split-unchecked-parallel requires --split-unchecked-target-forward."
        )
    if args.split_unchecked_parallel and args.device != "cuda":
        raise ValueError("--split-unchecked-parallel requires --device cuda.")
    if args.split_unchecked_min_position < 0:
        raise ValueError("--split-unchecked-min-position must be non-negative.")
    if (
        args.split_unchecked_min_draft_margin is not None
        and args.split_unchecked_min_draft_margin < 0.0
    ):
        raise ValueError("--split-unchecked-min-draft-margin must be non-negative.")
    if args.split_unchecked_target_forward and (
        args.replay_accepted_target_cache
        or args.serial_target_cache_commit
        or args.preemptive_lookahead
        or args.single_gpu_parallel_draft
        or args.single_gpu_parallel_correction_sync
    ):
        raise ValueError(
            "Split unchecked execution is incompatible with cache replay, serial commit, "
            "preemptive lookahead, and the older single-GPU overlap flags."
        )
    if (
        args.latent_budget_short_circuit
        and args.max_accepted_unchecked_per_sequence is None
        and args.max_accepted_unchecked_per_window is None
    ):
        raise ValueError(
            "--latent-budget-short-circuit requires an unchecked sequence or window budget."
        )
    if args.merge_target_lora and args.target_lora_removal != "none":
        raise ValueError("--merge-target-lora is incompatible with target LoRA removal ablations.")
    if args.merge_target_lora and args.target_lora_dtype != "auto":
        raise ValueError("Use either --merge-target-lora or --target-lora-dtype, not both.")
    if args.batched_draft_boundary and args.draft_logit_source != "boundary":
        raise ValueError("--batched-draft-boundary requires --draft-logit-source boundary.")
    if args.defer_correction_min_accepted_drafts < 0:
        raise ValueError("--defer-correction-min-accepted-drafts must be non-negative.")
    if args.defer_correction_after_position < 0:
        raise ValueError("--defer-correction-after-position must be non-negative.")
    if (
        args.defer_correction_before_position is not None
        and args.defer_correction_before_position < 0
    ):
        raise ValueError("--defer-correction-before-position must be non-negative.")
    if (
        args.defer_correction_before_position is not None
        and args.defer_correction_before_position <= args.defer_correction_after_position
    ):
        raise ValueError(
            "--defer-correction-before-position must exceed --defer-correction-after-position."
        )
    if args.cheap_verifier_margin_skip_below < 0.0:
        raise ValueError("--cheap-verifier-margin-skip-below must be non-negative.")
    if args.cheap_verifier_margin_block2_below < 0.0:
        raise ValueError("--cheap-verifier-margin-block2-below must be non-negative.")
    if args.latent_min_target_margin < 0.0:
        raise ValueError("--latent-min-target-margin must be non-negative.")
    if args.latent_margin_gate_close_after < 0:
        raise ValueError("--latent-margin-gate-close-after must be non-negative.")
    if args.latent_margin_gate_close_after > 0 and args.latent_min_target_margin <= 0.0:
        raise ValueError(
            "--latent-margin-gate-close-after requires --latent-min-target-margin."
        )
    if args.serial_fallback_margin is not None and args.serial_fallback_margin < 0.0:
        raise ValueError("--serial-fallback-margin must be non-negative.")
    if (
        args.serial_fallback_full_replay_margin is not None
        and args.serial_fallback_full_replay_margin < 0.0
    ):
        raise ValueError("--serial-fallback-full-replay-margin must be non-negative.")
    if args.serial_fallback_full_replay_margin is not None and args.serial_fallback_margin is None:
        raise ValueError("--serial-fallback-full-replay-margin requires --serial-fallback-margin.")
    if (
        args.serial_fallback_full_replay_margin is not None
        and args.serial_fallback_full_replay_margin > args.serial_fallback_margin
    ):
        raise ValueError("--serial-fallback-full-replay-margin cannot exceed --serial-fallback-margin.")
    for category, value in parse_category_float_map(
        args.target_match_category_lambdas,
        flag_name="--target-match-category-lambdas",
    ).items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Lambda override for category {category!r} must be between 0 and 1.")
    device = torch.device(args.device)
    if args.single_gpu_parallel_draft and device.type != "cuda":
        raise ValueError("--single-gpu-parallel-draft requires a CUDA device.")
    if args.anchor_hook_hidden_states and args.preemptive_lookahead:
        raise ValueError("--anchor-hook-hidden-states is incompatible with preemptive lookahead.")
    if args.anchor_hook_hidden_states and args.single_gpu_parallel_correction_sync:
        raise ValueError(
            "--anchor-hook-hidden-states is incompatible with parallel correction sync."
        )
    if args.single_gpu_parallel_correction_sync and not args.reuse_verify_cache_for_correction:
        raise ValueError(
            "--single-gpu-parallel-correction-sync requires --reuse-verify-cache-for-correction."
        )
    if args.preemptive_lookahead:
        if device.type != "cuda":
            raise ValueError("--preemptive-lookahead requires a CUDA device.")
        if args.draft_logit_source != "boundary":
            raise ValueError("--preemptive-lookahead requires --draft-logit-source boundary.")
        if args.mode not in {"fixed", "schedule"}:
            raise ValueError("--preemptive-lookahead currently supports fixed or schedule mode.")
        if args.draft_commit_policy != "target_match" or args.latent_draft_commit_policy not in {
            None,
            "target_match",
        }:
            raise ValueError("--preemptive-lookahead currently requires target_match.")
        if not args.compact_runtime_stats:
            raise ValueError("--preemptive-lookahead requires --compact-runtime-stats.")
        if args.cheap_verifier_policy != "none":
            raise ValueError("--preemptive-lookahead does not yet support a cheap verifier policy.")
        if args.replay_accepted_target_cache or args.serial_target_cache_commit:
            raise ValueError("--preemptive-lookahead is incompatible with target-cache replay.")
        if args.serial_fallback_margin is not None:
            raise ValueError("--preemptive-lookahead is incompatible with serial fallback.")
    if args.preemptive_min_draft_margin < 0.0:
        raise ValueError("--preemptive-min-draft-margin must be non-negative.")
    for flag_name in (
        "min_draft_margin",
        "min_draft_margin_early",
        "min_draft_margin_mid",
        "min_draft_margin_late",
    ):
        value = getattr(args, flag_name)
        if value is not None and value < 0.0:
            raise ValueError(f"--{flag_name.replace('_', '-')} must be non-negative.")
    if args.ngram_min_size < 1:
        raise ValueError("--ngram-min-size must be positive.")
    if args.ngram_max_size < args.ngram_min_size:
        raise ValueError("--ngram-max-size must be >= --ngram-min-size.")
    if args.ngram_max_draft_tokens < 0:
        raise ValueError("--ngram-max-draft-tokens must be non-negative.")
    parse_nonnegative_int_map(
        args.ngram_draft_tokens_by_match_size,
        flag_name="--ngram-draft-tokens-by-match-size",
    )
    parse_position_width_schedule(
        args.ngram_max_draft_tokens_by_position,
        flag_name="--ngram-max-draft-tokens-by-position",
    )
    if args.ngram_wide_auto_route_after_hits < 0:
        raise ValueError("--ngram-wide-auto-route-after-hits must be non-negative.")
    if args.ngram_wide_auto_route_min_extra_per_hit < 0.0:
        raise ValueError("--ngram-wide-auto-route-min-extra-per-hit must be non-negative.")
    if args.ngram_wide_auto_route_fallback_width <= 0:
        raise ValueError("--ngram-wide-auto-route-fallback-width must be positive.")
    if args.ngram_predecode_medium_prompt_tokens <= 0:
        raise ValueError("--ngram-predecode-medium-prompt-tokens must be positive.")
    if args.ngram_predecode_long_prompt_tokens < args.ngram_predecode_medium_prompt_tokens:
        raise ValueError(
            "--ngram-predecode-long-prompt-tokens must be >= the medium prompt threshold."
        )
    if args.ngram_predecode_short_output_tokens <= 0:
        raise ValueError("--ngram-predecode-short-output-tokens must be positive.")
    if (
        args.ngram_predecode_narrow_width <= 0
        or args.ngram_predecode_medium_width <= 0
        or args.ngram_predecode_wide_width <= 0
    ):
        raise ValueError("Predecode n-gram widths must be positive.")
    if not (
        args.ngram_predecode_narrow_width
        <= args.ngram_predecode_medium_width
        <= args.ngram_predecode_wide_width
    ):
        raise ValueError("Predecode n-gram widths must satisfy narrow <= medium <= wide.")
    if args.ngram_unchecked_min_match_size < 0:
        raise ValueError("--ngram-unchecked-min-match-size must be non-negative.")
    if args.ngram_unchecked_min_position < 0:
        raise ValueError("--ngram-unchecked-min-position must be non-negative.")
    ngram_unchecked_categories = parse_category_set(args.ngram_unchecked_token_categories)
    unknown_ngram_categories = ngram_unchecked_categories - {
        "word",
        "number",
        "space",
        "newline",
        "symbol",
        "other",
    }
    if unknown_ngram_categories:
        raise ValueError(
            "Unknown --ngram-unchecked-token-categories: "
            + ", ".join(sorted(unknown_ngram_categories))
        )
    if ngram_unchecked_categories and args.ngram_unchecked_min_match_size <= 0:
        raise ValueError(
            "--ngram-unchecked-token-categories requires unchecked n-gram commits."
        )
    if (
        args.ngram_unchecked_category_reject_closes_sequence
        and not ngram_unchecked_categories
    ):
        raise ValueError(
            "--ngram-unchecked-category-reject-closes-sequence requires token categories."
        )
    if args.ngram_unchecked_min_match_size > 0:
        if args.ngram_draft_mode == "off":
            raise ValueError("Unchecked n-gram commits require n-gram drafting.")
        if not args.precommit_unchecked_drafts:
            raise ValueError("Unchecked n-gram commits require --precommit-unchecked-drafts.")
        if (
            args.max_accepted_unchecked_per_sequence is None
            and args.max_accepted_unchecked_per_window is None
        ):
            raise ValueError("Unchecked n-gram commits require a sequence or window budget.")
    if args.indexed_ngram_lookup and args.ngram_draft_mode == "off":
        raise ValueError("--indexed-ngram-lookup requires n-gram drafting.")
    if args.ngram_tree_verify:
        if not args.indexed_ngram_lookup:
            raise ValueError("--ngram-tree-verify requires --indexed-ngram-lookup.")
        if args.ngram_draft_mode != "only":
            raise ValueError("--ngram-tree-verify currently requires n-gram-only drafting.")
        if not args.ngram_strict_target_match or args.draft_commit_policy != "target_match":
            raise ValueError("--ngram-tree-verify requires strict target-match decoding.")
        if args.ngram_occurrence_policy != "latest" or args.ngram_predecode_occurrence_route:
            raise ValueError("--ngram-tree-verify currently supports latest occurrences only.")
        if args.precommit_unchecked_drafts:
            raise ValueError("--ngram-tree-verify is incompatible with unchecked precommit.")
        if args.replay_accepted_target_cache or args.serial_target_cache_commit:
            raise ValueError("--ngram-tree-verify requires direct selected-path KV adoption.")
        if min(
            args.ngram_tree_early_branches,
            args.ngram_tree_mid_branches,
            args.ngram_tree_late_branches,
        ) < 1:
            raise ValueError("N-gram tree branch budgets must be positive.")
        if args.ngram_tree_max_nodes < 2:
            raise ValueError("--ngram-tree-max-nodes must be at least two.")
    if args.ngram_occurrence_policy != "latest" and not args.indexed_ngram_lookup:
        raise ValueError("Non-latest n-gram occurrence policies require indexed lookup.")
    if not 0.0 < args.ngram_occurrence_recency_decay <= 1.0:
        raise ValueError("--ngram-occurrence-recency-decay must be in (0, 1].")
    if args.ngram_predecode_occurrence_route:
        if args.ngram_draft_mode == "off":
            raise ValueError("The predecode occurrence route requires n-gram drafting.")
        if not args.indexed_ngram_lookup:
            raise ValueError("The predecode occurrence route requires indexed lookup.")
    if args.ngram_strict_target_match and args.ngram_draft_mode == "off":
        raise ValueError("--ngram-strict-target-match requires n-gram drafting.")
    if args.ngram_strict_target_match and args.draft_commit_policy != "target_match":
        raise ValueError("--ngram-strict-target-match requires target_match commit policy.")
    if args.latent_draft_min_position < 0:
        raise ValueError("--latent-draft-min-position must be non-negative.")
    if args.draft_cooldown_after_failures < 0 or args.draft_cooldown_cycles < 0:
        raise ValueError("Draft cooldown settings must be non-negative.")
    if bool(args.draft_cooldown_after_failures) != bool(args.draft_cooldown_cycles):
        raise ValueError(
            "--draft-cooldown-after-failures and --draft-cooldown-cycles must be enabled together."
        )
    if args.auto_route_after_latent_drafts < 0:
        raise ValueError("--auto-route-after-latent-drafts must be non-negative.")
    if args.auto_route_reevaluate_every < 0:
        raise ValueError("--auto-route-reevaluate-every must be non-negative.")
    if args.auto_route_reevaluate_every > 0 and args.auto_route_after_latent_drafts <= 0:
        raise ValueError(
            "--auto-route-reevaluate-every requires --auto-route-after-latent-drafts."
        )
    if args.auto_route_early_zero_after < 0:
        raise ValueError("--auto-route-early-zero-after must be non-negative.")
    if not 0.0 <= args.auto_route_min_latent_acceptance <= 1.0:
        raise ValueError("--auto-route-min-latent-acceptance must be between 0 and 1.")
    if bool(args.auto_route_after_latent_drafts) != bool(args.auto_route_min_latent_acceptance):
        raise ValueError(
            "Auto-route warmup and minimum acceptance must be enabled together."
        )
    if (
        args.auto_route_after_latent_drafts > 0 or args.auto_route_early_zero_after > 0
    ) and args.ngram_draft_mode != "prefer":
        raise ValueError("Request-level auto-routing requires --ngram-draft-mode prefer.")
    if args.preemptive_lookahead and args.ngram_draft_mode != "off":
        raise ValueError("--preemptive-lookahead and n-gram drafting cannot yet be combined.")
    if args.ngram_draft_mode == "only" and args.single_gpu_parallel_correction_sync:
        raise ValueError("N-gram-only drafting does not use recurrent correction synchronization.")
    if args.lazy_t_sync and args.preemptive_lookahead:
        raise ValueError("--lazy-t-sync and preemptive lookahead cannot yet be combined.")
    if args.lazy_t_sync and args.single_gpu_parallel_correction_sync:
        raise ValueError("--lazy-t-sync and parallel correction sync cannot yet be combined.")
    if args.defer_t_init_until_latent and not args.lazy_t_sync:
        raise ValueError("--defer-t-init-until-latent requires --lazy-t-sync.")
    if args.defer_t_init_until_latent and args.ngram_draft_mode == "only":
        raise ValueError("N-gram-only decoding never initializes recurrent T.")
    if (
        args.draft_commit_policy == "category_block"
        or args.latent_draft_commit_policy == "category_block"
    ) and not parse_category_set(args.draft_token_category_gate):
        raise ValueError("category_block requires a non-empty --draft-token-category-gate.")
    checkpoint = Path(args.checkpoint)
    metadata = json.loads((checkpoint / RECURFT_CONFIG_NAME).read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype(args.dtype),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).to(device)
    recurrent = build_recurrent_module(base_model, checkpoint, metadata).to(device).eval()
    if args.draft_logit_source == "boundary" and not recurrent.has_boundary_head():
        raise ValueError(
            "--draft-logit-source boundary requires a checkpoint trained with `recurft_boundary_head_rank > 0`."
        )
    peft_model = PeftModel.from_pretrained(base_model, checkpoint).to(device).eval()
    removed_target_lora_modules = apply_target_lora_removal(
        peft_model, metadata, args.target_lora_removal
    )
    cast_target_lora_names = (
        cast_target_lora_parameters(peft_model, torch_dtype(args.target_lora_dtype))
        if args.target_lora_dtype != "auto"
        else []
    )
    model = (
        peft_model.merge_and_unload(safe_merge=True).to(device).eval()
        if args.merge_target_lora
        else peft_model
    )

    if args.mode == "verifier":
        if not args.verifier_weights:
            raise ValueError("--verifier-weights is required in verifier mode.")
        from safetensors.torch import load_file

        recurrent.load_state_dict(load_file(args.verifier_weights, device="cpu"), strict=False)
        recurrent.to(device).eval()

    merged_recurrent_lora_modules = (
        recurrent.merge_lora_for_inference() if args.merge_recurrent_lora else 0
    )

    rows = load_rows(args.data_file, args.start_index, args.max_samples)
    results = []
    for sample_idx, row in enumerate(rows):
        prompt_ids = make_prompt(
            tokenizer,
            row["question"],
            args.question_prefix,
            args.question_suffix,
            args.disable_thinking and not args.enable_thinking,
        )
        prompt_ids = add_synthetic_prefix(
            tokenizer,
            prompt_ids,
            args.synthetic_prefix_tokens,
            args.synthetic_prefix_text,
        )[:, -args.max_prompt_tokens :].to(device)
        def run_baseline() -> dict[str, Any]:
            baseline_result, baseline_wall_time = wall_timed(
                lambda: greedy_decode(
                    model,
                    prompt_ids,
                    args.max_new_tokens,
                    tokenizer.eos_token_id,
                    metadata.get("recurrent_hidden_state_index", metadata["anchor_layer"] + 1),
                    device,
                    args.omit_target_attention_mask,
                ),
                device,
            )
            baseline_result["wall_time_s"] = baseline_wall_time
            return baseline_result

        def run_adaptive() -> dict[str, Any]:
            adaptive_result, adaptive_wall_time = wall_timed(
                lambda: speculative_decode(
                    model,
                    recurrent,
                    metadata,
                    prompt_ids,
                    args,
                    tokenizer.eos_token_id,
                    device,
                    tokenizer=tokenizer,
                ),
                device,
            )
            adaptive_result["wall_time_s"] = adaptive_wall_time
            return adaptive_result

        baseline = None
        adaptive_first = args.decode_order == "adaptive_first" or (
            args.decode_order == "alternate" and sample_idx % 2 == 1
        )
        if adaptive_first:
            adaptive = run_adaptive()
            if not args.no_baseline:
                baseline = run_baseline()
        else:
            if not args.no_baseline:
                baseline = run_baseline()
            adaptive = run_adaptive()
        baseline_ids = baseline["token_ids"] if baseline is not None else None
        exact_match = baseline_ids == adaptive["token_ids"] if baseline_ids is not None else None
        output_ids = adaptive["token_ids"]
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        pred_answer, ref_answer, quality_score = score_response(text, row, args.answer_metric)
        baseline_text = tokenizer.decode(baseline_ids, skip_special_tokens=True) if baseline_ids is not None else None
        if baseline_text is not None:
            baseline_answer, _, baseline_quality_score = score_response(
                baseline_text,
                row,
                args.answer_metric,
            )
        else:
            baseline_answer = None
            baseline_quality_score = None
        correct = bool(quality_score == 1.0) if args.answer_metric == "math" else None
        baseline_correct = (
            bool(baseline_quality_score == 1.0)
            if args.answer_metric == "math" and baseline_quality_score is not None
            else None
        )
        result = {
            "sample": args.start_index + sample_idx,
            "question": row["question"],
            "reference_answer": ref_answer,
            "predicted_answer": pred_answer,
            "answer_metric": args.answer_metric,
            "quality_score": quality_score,
            "correct": correct,
            "response": text,
            "baseline_predicted_answer": baseline_answer,
            "baseline_quality_score": baseline_quality_score,
            "baseline_correct": baseline_correct,
            "generated_tokens": len(output_ids),
            "prompt_tokens": int(prompt_ids.size(1)),
            "exact_baseline_match": exact_match,
            "baseline": baseline,
            "adaptive": adaptive,
        }
        results.append(result)
        print(
            json.dumps(
                {
                    "sample": result["sample"],
                    "correct": result["correct"],
                    "quality_score": result["quality_score"],
                    "tokens": result["generated_tokens"],
                    "exact": exact_match,
                    "cycles": adaptive["cycles"],
                    "accepted_drafts": adaptive["accepted_draft_tokens"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    baseline_time = sum(
        result["baseline"]["wall_time_s"] for result in results if result["baseline"] is not None
    )
    adaptive_time = sum(result["adaptive"]["wall_time_s"] for result in results)
    total_tokens = sum(result["generated_tokens"] for result in results)
    baseline_total_tokens = (
        sum(len(result["baseline"]["token_ids"]) for result in results if result["baseline"] is not None)
        if not args.no_baseline
        else None
    )
    total_drafts = sum(result["adaptive"]["draft_tokens"] for result in results)
    accepted_drafts = sum(result["adaptive"]["accepted_draft_tokens"] for result in results)
    matched_drafts = sum(result["adaptive"]["matched_draft_tokens"] for result in results)
    target_match_observations = sum(
        result["adaptive"].get("target_match_observations", result["adaptive"]["draft_tokens"])
        for result in results
    )
    rejected_drafts = sum(result["adaptive"].get("rejected_draft_tokens", 0) for result in results)
    accepted_mismatches = sum(result["adaptive"].get("accepted_mismatch_tokens", 0) for result in results)
    accepted_ngram_mismatches = sum(
        result["adaptive"].get("accepted_ngram_mismatch_tokens", 0) for result in results
    )
    accepted_latent_mismatches = sum(
        result["adaptive"].get("accepted_latent_mismatch_tokens", 0) for result in results
    )
    accepted_unchecked = sum(
        result["adaptive"].get("accepted_unchecked_tokens", 0) for result in results
    )
    precommit_unchecked_blocks = sum(
        result["adaptive"].get("precommit_unchecked_blocks", 0) for result in results
    )
    precommit_trimmed_draft_tokens = sum(
        result["adaptive"].get("precommit_trimmed_draft_tokens", 0) for result in results
    )
    last_token_logits_only_blocks = sum(
        result["adaptive"].get("last_token_logits_only_blocks", 0) for result in results
    )
    split_unchecked_blocks = sum(
        result["adaptive"].get("split_unchecked_blocks", 0) for result in results
    )
    split_unchecked_tokens = sum(
        result["adaptive"].get("split_unchecked_tokens", 0) for result in results
    )
    split_parallel_blocks = sum(
        result["adaptive"].get("split_parallel_blocks", 0) for result in results
    )
    split_unchecked_gate_skips = sum(
        result["adaptive"].get("split_unchecked_gate_skips", 0) for result in results
    )
    expand_events = sum(result["adaptive"].get("target_match_expand_events", 0) for result in results)
    expand_extra_tokens = sum(result["adaptive"].get("target_match_expand_extra_tokens", 0) for result in results)
    target_cache_replays = sum(result["adaptive"].get("target_cache_replays", 0) for result in results)
    fast_correction_cache_reuses = sum(
        result["adaptive"].get("fast_correction_cache_reuses", 0) for result in results
    )
    deferred_corrective_tokens = sum(
        result["adaptive"].get("deferred_corrective_tokens", 0) for result in results
    )
    serial_target_commits = sum(result["adaptive"].get("serial_target_commits", 0) for result in results)
    serial_fallback_events = sum(result["adaptive"].get("serial_fallback_events", 0) for result in results)
    serial_fallback_token_changes = sum(
        result["adaptive"].get("serial_fallback_token_changes", 0) for result in results
    )
    full_prefix_replays = sum(result["adaptive"].get("full_prefix_replays", 0) for result in results)
    cycle_start_fallback_events = sum(
        result["adaptive"].get("cycle_start_fallback_events", 0) for result in results
    )
    cycle_start_fallback_token_changes = sum(
        result["adaptive"].get("cycle_start_fallback_token_changes", 0) for result in results
    )
    cheap_policy_skips = sum(result["adaptive"].get("cheap_policy_skips", 0) for result in results)
    cheap_policy_records = sum(len(result["adaptive"].get("cheap_policy_records", [])) for result in results)
    cheap_policy_evaluations = sum(
        result["adaptive"].get(
            "cheap_policy_evaluations",
            len(result["adaptive"].get("cheap_policy_records", [])),
        )
        for result in results
    )
    verifier_threshold_skips = sum(
        result["adaptive"].get("verifier_threshold_skips", 0) for result in results
    )
    preemptive_attempts = sum(result["adaptive"].get("preemptive_attempts", 0) for result in results)
    preemptive_sync_reuses = sum(
        result["adaptive"].get("preemptive_sync_reuses", 0) for result in results
    )
    preemptive_token_hits = sum(
        result["adaptive"].get("preemptive_token_hits", 0) for result in results
    )
    preemptive_token_misses = sum(
        result["adaptive"].get("preemptive_token_misses", 0) for result in results
    )
    preemptive_branch_discards = sum(
        result["adaptive"].get("preemptive_branch_discards", 0) for result in results
    )
    preemptive_gate_skips = sum(
        result["adaptive"].get("preemptive_gate_skips", 0) for result in results
    )
    preemptive_reused_draft_tokens = sum(
        result["adaptive"].get("preemptive_reused_draft_tokens", 0) for result in results
    )
    ngram_attempts = sum(result["adaptive"].get("ngram_attempts", 0) for result in results)
    ngram_hits = sum(result["adaptive"].get("ngram_hits", 0) for result in results)
    fast_strict_ngram_diagnostic_blocks = sum(
        result["adaptive"].get("fast_strict_ngram_diagnostic_blocks", 0)
        for result in results
    )
    ngram_draft_tokens = sum(result["adaptive"].get("ngram_draft_tokens", 0) for result in results)
    accepted_ngram_tokens = sum(
        result["adaptive"].get("accepted_ngram_tokens", 0) for result in results
    )
    ngram_tree_cycles = sum(
        result["adaptive"].get("ngram_tree_cycles", 0) for result in results
    )
    ngram_tree_branches = sum(
        result["adaptive"].get("ngram_tree_branches", 0) for result in results
    )
    ngram_tree_nodes = sum(
        result["adaptive"].get("ngram_tree_nodes", 0) for result in results
    )
    ngram_tree_selected_nodes = sum(
        result["adaptive"].get("ngram_tree_selected_nodes", 0) for result in results
    )
    ngram_tree_target_matched_drafts = sum(
        result["adaptive"].get("ngram_tree_target_matched_drafts", 0)
        for result in results
    )
    ngram_wide_route_evaluations = sum(
        result["adaptive"].get("ngram_wide_route_evaluations", 0) for result in results
    )
    ngram_wide_route_disables = sum(
        result["adaptive"].get("ngram_wide_route_disables", 0) for result in results
    )
    ngram_wide_route_observations = sum(
        result["adaptive"].get("ngram_wide_route_observations", 0) for result in results
    )
    ngram_wide_route_extra_accepted_tokens = sum(
        result["adaptive"].get("ngram_wide_route_extra_accepted_tokens", 0)
        for result in results
    )
    ngram_wide_route_fallback_cycles = sum(
        result["adaptive"].get("ngram_wide_route_fallback_cycles", 0)
        for result in results
    )
    ngram_wide_route_scores = [
        float(score)
        for result in results
        if (score := result["adaptive"].get("ngram_wide_route_score")) is not None
    ]
    ngram_predecode_route_widths: dict[str, int] = {}
    ngram_predecode_route_reasons: dict[str, int] = {}
    ngram_predecode_occurrence_policies: dict[str, int] = {}
    ngram_predecode_occurrence_reasons: dict[str, int] = {}
    for result in results:
        width = result["adaptive"].get("ngram_predecode_route_width")
        reason = str(result["adaptive"].get("ngram_predecode_route_reason") or "unknown")
        width_key = "schedule" if width is None else str(int(width))
        ngram_predecode_route_widths[width_key] = (
            ngram_predecode_route_widths.get(width_key, 0) + 1
        )
        ngram_predecode_route_reasons[reason] = (
            ngram_predecode_route_reasons.get(reason, 0) + 1
        )
        occurrence_policy = str(
            result["adaptive"].get("ngram_predecode_occurrence_policy") or "unknown"
        )
        occurrence_reason = str(
            result["adaptive"].get("ngram_predecode_occurrence_reason") or "unknown"
        )
        ngram_predecode_occurrence_policies[occurrence_policy] = (
            ngram_predecode_occurrence_policies.get(occurrence_policy, 0) + 1
        )
        ngram_predecode_occurrence_reasons[occurrence_reason] = (
            ngram_predecode_occurrence_reasons.get(occurrence_reason, 0) + 1
        )
    ngram_unchecked_gate_evaluations = sum(
        result["adaptive"].get("ngram_unchecked_gate_evaluations", 0)
        for result in results
    )
    ngram_unchecked_gate_eligible = sum(
        result["adaptive"].get("ngram_unchecked_gate_eligible", 0)
        for result in results
    )
    ngram_unchecked_budget_skips = sum(
        result["adaptive"].get("ngram_unchecked_budget_skips", 0)
        for result in results
    )
    ngram_unchecked_category_gate_evaluations = sum(
        result["adaptive"].get("ngram_unchecked_category_gate_evaluations", 0)
        for result in results
    )
    ngram_unchecked_category_gate_rejects = sum(
        result["adaptive"].get("ngram_unchecked_category_gate_rejects", 0)
        for result in results
    )
    ngram_unchecked_category_trimmed_tokens = sum(
        result["adaptive"].get("ngram_unchecked_category_trimmed_tokens", 0)
        for result in results
    )
    ngram_unchecked_category_budget_closes = sum(
        result["adaptive"].get("ngram_unchecked_category_budget_closes", 0)
        for result in results
    )
    ngram_unchecked_category_closed_skips = sum(
        result["adaptive"].get("ngram_unchecked_category_closed_skips", 0)
        for result in results
    )
    lazy_t_sync_queued_tokens = sum(
        result["adaptive"].get("lazy_t_sync_queued_tokens", 0) for result in results
    )
    lazy_t_sync_flushes = sum(
        result["adaptive"].get("lazy_t_sync_flushes", 0) for result in results
    )
    lazy_t_sync_flushed_tokens = sum(
        result["adaptive"].get("lazy_t_sync_flushed_tokens", 0) for result in results
    )
    deferred_t_initializations = sum(
        result["adaptive"].get("deferred_t_initializations", 0) for result in results
    )
    deferred_t_init_tokens = sum(
        result["adaptive"].get("deferred_t_init_tokens", 0) for result in results
    )
    latent_position_gate_skips = sum(
        result["adaptive"].get("latent_position_gate_skips", 0) for result in results
    )
    latent_target_margin_gate_evaluations = sum(
        result["adaptive"].get("latent_target_margin_gate_evaluations", 0)
        for result in results
    )
    latent_target_margin_gate_skips = sum(
        result["adaptive"].get("latent_target_margin_gate_skips", 0)
        for result in results
    )
    latent_target_margin_gate_closes = sum(
        result["adaptive"].get("latent_target_margin_gate_closes", 0)
        for result in results
    )
    latent_target_margin_gate_close_positions = [
        int(position)
        for result in results
        if (
            position := result["adaptive"].get(
                "latent_target_margin_gate_close_position"
            )
        )
        is not None
    ]
    latent_target_margins = sorted(
        float(margin)
        for result in results
        for margin in result["adaptive"].get("latent_target_margins", [])
    )
    latent_unchecked_budget_skips = sum(
        result["adaptive"].get("latent_unchecked_budget_skips", 0) for result in results
    )
    latent_unchecked_category_gate_evaluations = sum(
        result["adaptive"].get("latent_unchecked_category_gate_evaluations", 0)
        for result in results
    )
    latent_unchecked_category_gate_rejects = sum(
        result["adaptive"].get("latent_unchecked_category_gate_rejects", 0)
        for result in results
    )
    latent_unchecked_category_trimmed_tokens = sum(
        result["adaptive"].get("latent_unchecked_category_trimmed_tokens", 0)
        for result in results
    )
    latent_unchecked_category_budget_closes = sum(
        result["adaptive"].get("latent_unchecked_category_budget_closes", 0)
        for result in results
    )
    latent_unchecked_category_closed_skips = sum(
        result["adaptive"].get("latent_unchecked_category_closed_skips", 0)
        for result in results
    )
    draft_cooldown_skips = sum(
        result["adaptive"].get("draft_cooldown_skips", 0) for result in results
    )
    draft_cooldown_activations = sum(
        result["adaptive"].get("draft_cooldown_activations", 0) for result in results
    )
    auto_route_evaluations = sum(
        result["adaptive"].get("auto_route_evaluations", 0) for result in results
    )
    auto_route_t_disables = sum(
        result["adaptive"].get("auto_route_t_disables", 0) for result in results
    )
    auto_route_early_evaluations = sum(
        result["adaptive"].get("auto_route_early_evaluations", 0) for result in results
    )
    auto_route_early_t_disables = sum(
        result["adaptive"].get("auto_route_early_t_disables", 0) for result in results
    )
    auto_route_latent_drafts = sum(
        result["adaptive"].get("auto_route_latent_drafts", 0) for result in results
    )
    auto_route_accepted_drafts = sum(
        result["adaptive"].get("auto_route_accepted_drafts", 0) for result in results
    )
    auto_route_latent_candidate_cycles = sum(
        result["adaptive"].get("auto_route_latent_candidate_cycles", 0) for result in results
    )
    auto_route_confident_candidate_cycles = sum(
        result["adaptive"].get("auto_route_confident_candidate_cycles", 0)
        for result in results
    )
    route_on_proposal_rate = args.latent_draft_commit_policy in {
        "draft_margin",
        "whole_block",
        "category_block",
    }
    missing_t_anchor_fallbacks = sum(
        result["adaptive"].get("missing_t_anchor_fallbacks", 0) for result in results
    )
    skipped_terminal_t_steps = sum(
        result["adaptive"].get("skipped_terminal_t_steps", 0) for result in results
    )
    verification_intervals = [
        int(interval)
        for result in results
        for interval in result["adaptive"].get("verification_intervals", [])
    ]
    verification_risk_budget_stops = sum(
        result["adaptive"].get("verification_risk_budget_stops", 0) for result in results
    )
    verification_risk_at_stops = [
        float(risk)
        for result in results
        for risk in result["adaptive"].get("verification_risk_at_stops", [])
    ]
    verification_rollbacks = sum(
        result["adaptive"].get("verification_rollbacks", 0) for result in results
    )
    verification_wasted_drafts = sum(
        result["adaptive"].get("verification_wasted_drafts", 0) for result in results
    )
    verification_first_divergence_events = [
        event
        for result in results
        for event in result["adaptive"].get("verification_first_divergence_events", [])
    ]
    verification_agreement_shadow_probes = sum(
        result["adaptive"].get("verification_agreement_shadow_probes", 0)
        for result in results
    )
    verification_agreement_shadow_matches = sum(
        result["adaptive"].get("verification_agreement_shadow_matches", 0)
        for result in results
    )
    verification_agreement_gate_openings = sum(
        result["adaptive"].get("verification_agreement_gate_openings", 0)
        for result in results
    )
    verification_agreement_gate_closures = sum(
        result["adaptive"].get("verification_agreement_gate_closures", 0)
        for result in results
    )
    verification_agreement_active_cycles = sum(
        result["adaptive"].get("verification_agreement_active_cycles", 0)
        for result in results
    )
    verification_agreement_max_streak = max(
        (
            int(result["adaptive"].get("verification_agreement_max_streak", 0))
            for result in results
        ),
        default=0,
    )
    verification_agreement_events = [
        event
        for result in results
        for event in result["adaptive"].get("verification_agreement_events", [])
    ]
    auto_route_disable_positions = [
        int(position)
        for result in results
        if (position := result["adaptive"].get("auto_route_disable_position")) is not None
    ]
    cheap_policy_thresholds = [
        threshold
        for result in results
        for threshold in result["adaptive"].get("cheap_policy_dynamic_thresholds", [])
    ]
    cheap_policy_target_margins = sorted(
        margin
        for result in results
        for margin in result["adaptive"].get("cheap_policy_target_margins", [])
    )

    def cheap_margin_quantile(fraction: float) -> float | None:
        if not cheap_policy_target_margins:
            return None
        index = round(fraction * (len(cheap_policy_target_margins) - 1))
        return float(cheap_policy_target_margins[index])

    def latent_margin_quantile(fraction: float) -> float | None:
        if not latent_target_margins:
            return None
        index = round(fraction * (len(latent_target_margins) - 1))
        return float(latent_target_margins[index])

    adaptive_timing_sums: dict[str, float] = {}
    for result in results:
        for key, value in result["adaptive"].get("timings", {}).items():
            adaptive_timing_sums[key] = adaptive_timing_sums.get(key, 0.0) + float(value)

    baseline_target_calls = (
        sum(result["baseline"]["target_calls"] for result in results if result["baseline"] is not None)
        if not args.no_baseline
        else None
    )
    adaptive_target_calls = sum(result["adaptive"]["target_calls"] for result in results)
    adaptive_verification_rounds = max(1, adaptive_target_calls - len(results))
    baseline_prefill_time = (
        sum(
            float(result["baseline"]["timings"].get("prefill_s", 0.0))
            for result in results
            if result["baseline"] is not None
        )
        if not args.no_baseline
        else None
    )
    adaptive_prefill_time = sum(
        float(result["adaptive"].get("timings", {}).get("prefill_s", 0.0))
        for result in results
    )
    adaptive_t_init_time = sum(
        float(result["adaptive"].get("timings", {}).get("t_init_s", 0.0))
        for result in results
    )
    baseline_decode_only_time = (
        max(1e-12, baseline_time - baseline_prefill_time)
        if baseline_prefill_time is not None
        else None
    )
    adaptive_decode_only_time = max(1e-12, adaptive_time - adaptive_prefill_time)
    # Domino's public benchmark excludes target prefill and the first draft-model
    # prefill. T's prompt-wide initialization is the closest RecurFT analogue.
    adaptive_steady_decode_time = max(
        1e-12,
        adaptive_time - adaptive_prefill_time - adaptive_t_init_time,
    )
    # Optimistic bound: assume recurrent rollout, verifier probe, and draft-tail projection are perfectly
    # overlapped on other devices, while target prefill/verify/correction/refine remain on the critical path.
    if args.production_async_timing:
        parallelizable_aux_time = None
        optimistic_parallel_time = None
    else:
        parallelizable_aux_time = sum(
            adaptive_timing_sums.get(key, 0.0)
            for key in (
                "t_step_s",
                "tail_s",
                "probe_s",
                "draft_parallel_s",
                "split_anchor_s",
            )
        )
        optimistic_parallel_time = max(1e-12, adaptive_time - parallelizable_aux_time)
    summary = {
        "mode": args.mode,
        "answer_metric": args.answer_metric,
        "checkpoint": str(checkpoint),
        "verifier_weights": args.verifier_weights,
        "verifier_tag": infer_verifier_tag(args.verifier_weights),
        "target_lora_removal": args.target_lora_removal,
        "target_lora_merged": args.merge_target_lora,
        "recurrent_lora_merged": recurrent.recurrent_lora_is_merged(),
        "merged_recurrent_lora_modules": merged_recurrent_lora_modules,
        "target_lora_dtype": args.target_lora_dtype,
        "cast_target_lora_parameters": len(cast_target_lora_names),
        "removed_target_lora_modules": len(removed_target_lora_modules),
        "samples": len(results),
        "accuracy": (
            sum(bool(result["correct"]) for result in results) / max(1, len(results))
            if args.answer_metric == "math"
            else None
        ),
        "baseline_accuracy": (
            sum(bool(result["baseline_correct"]) for result in results) / max(1, len(results))
            if not args.no_baseline and args.answer_metric == "math"
            else None
        ),
        "mean_quality_score": sum(result["quality_score"] for result in results)
        / max(1, len(results)),
        "baseline_mean_quality_score": (
            sum(float(result["baseline_quality_score"]) for result in results)
            / max(1, len(results))
            if not args.no_baseline
            else None
        ),
        "exact_baseline_rate": (
            sum(bool(result["exact_baseline_match"]) for result in results) / max(1, len(results))
            if not args.no_baseline
            else None
        ),
        "generated_tokens": total_tokens,
        "adaptive_generated_tokens": total_tokens,
        "baseline_generated_tokens": baseline_total_tokens,
        "output_length_ratio": (
            total_tokens / baseline_total_tokens
            if baseline_total_tokens is not None and baseline_total_tokens > 0
            else None
        ),
        "mean_prompt_tokens": sum(result["prompt_tokens"] for result in results)
        / max(1, len(results)),
        "min_prompt_tokens": min((result["prompt_tokens"] for result in results), default=0),
        "max_prompt_tokens_observed": max(
            (result["prompt_tokens"] for result in results), default=0
        ),
        "baseline_time_s": baseline_time if not args.no_baseline else None,
        "adaptive_time_s": adaptive_time,
        "wall_clock_speedup": baseline_time / adaptive_time if baseline_time > 0 else None,
        "baseline_prefill_time_s": baseline_prefill_time,
        "adaptive_prefill_time_s": adaptive_prefill_time,
        "adaptive_t_init_time_s": adaptive_t_init_time,
        "baseline_decode_only_time_s": baseline_decode_only_time,
        "adaptive_decode_only_time_s": adaptive_decode_only_time,
        "decode_only_wall_speedup": (
            baseline_decode_only_time / adaptive_decode_only_time
            if baseline_decode_only_time is not None
            else None
        ),
        "adaptive_steady_decode_time_s": adaptive_steady_decode_time,
        "domino_style_decode_speedup": (
            baseline_decode_only_time / adaptive_steady_decode_time
            if baseline_decode_only_time is not None
            else None
        ),
        "baseline_tokens_per_s": (
            baseline_total_tokens / baseline_time
            if baseline_total_tokens is not None and baseline_time > 0
            else None
        ),
        "adaptive_tokens_per_s": total_tokens / adaptive_time if adaptive_time > 0 else None,
        "token_throughput_speedup": (
            (total_tokens / adaptive_time) / (baseline_total_tokens / baseline_time)
            if baseline_total_tokens is not None
            and baseline_total_tokens > 0
            and baseline_time > 0
            and adaptive_time > 0
            else None
        ),
        "adaptive_timing_sums": adaptive_timing_sums,
        "component_timing_mode": (
            "host_enqueue" if args.production_async_timing else "cuda_synchronized"
        ),
        "parallelizable_aux_time_s": parallelizable_aux_time,
        "optimistic_parallel_adaptive_time_s": optimistic_parallel_time,
        "optimistic_parallel_wall_speedup": (
            baseline_time / optimistic_parallel_time
            if baseline_time > 0 and optimistic_parallel_time is not None
            else None
        ),
        "draft_tokens": total_drafts,
        "accepted_draft_tokens": accepted_drafts,
        "rejected_draft_tokens": rejected_drafts,
        "matched_draft_tokens": matched_drafts,
        "target_match_observations": target_match_observations,
        "accepted_mismatch_tokens": accepted_mismatches,
        "accepted_ngram_mismatch_tokens": accepted_ngram_mismatches,
        "accepted_latent_mismatch_tokens": accepted_latent_mismatches,
        "accepted_unchecked_tokens": accepted_unchecked,
        "precommit_unchecked_blocks": precommit_unchecked_blocks,
        "precommit_trimmed_draft_tokens": precommit_trimmed_draft_tokens,
        "last_token_logits_only_blocks": last_token_logits_only_blocks,
        "split_unchecked_blocks": split_unchecked_blocks,
        "split_unchecked_tokens": split_unchecked_tokens,
        "split_parallel_blocks": split_parallel_blocks,
        "split_unchecked_gate_skips": split_unchecked_gate_skips,
        "accepted_mismatch_share": accepted_mismatches / max(1, accepted_drafts),
        "target_match_expand_events": expand_events,
        "target_match_expand_extra_tokens": expand_extra_tokens,
        "target_cache_replays": target_cache_replays,
        "fast_correction_cache_reuses": fast_correction_cache_reuses,
        "deferred_corrective_tokens": deferred_corrective_tokens,
        "serial_target_commits": serial_target_commits,
        "serial_fallback_events": serial_fallback_events,
        "serial_fallback_token_changes": serial_fallback_token_changes,
        "full_prefix_replays": full_prefix_replays,
        "cycle_start_fallback_events": cycle_start_fallback_events,
        "cycle_start_fallback_token_changes": cycle_start_fallback_token_changes,
        "cheap_policy_skips": cheap_policy_skips,
        "cheap_policy_records": cheap_policy_records,
        "cheap_policy_evaluations": cheap_policy_evaluations,
        "cheap_policy_skip_rate": cheap_policy_skips / max(1, cheap_policy_evaluations),
        "cheap_policy_target_margin_mean": (
            sum(cheap_policy_target_margins) / len(cheap_policy_target_margins)
            if cheap_policy_target_margins
            else None
        ),
        "cheap_policy_target_margin_p10": cheap_margin_quantile(0.10),
        "cheap_policy_target_margin_p50": cheap_margin_quantile(0.50),
        "cheap_policy_target_margin_p90": cheap_margin_quantile(0.90),
        "verifier_threshold_skips": verifier_threshold_skips,
        "verifier_threshold_skip_rate": verifier_threshold_skips / max(1, adaptive_target_calls),
        "preemptive_attempts": preemptive_attempts,
        "preemptive_sync_reuses": preemptive_sync_reuses,
        "preemptive_token_hits": preemptive_token_hits,
        "preemptive_token_misses": preemptive_token_misses,
        "preemptive_branch_discards": preemptive_branch_discards,
        "preemptive_gate_skips": preemptive_gate_skips,
        "preemptive_token_hit_rate": preemptive_token_hits / max(1, preemptive_attempts),
        "preemptive_reused_draft_tokens": preemptive_reused_draft_tokens,
        "ngram_attempts": ngram_attempts,
        "ngram_hits": ngram_hits,
        "ngram_hit_rate": ngram_hits / max(1, ngram_attempts),
        "fast_strict_ngram_diagnostic_blocks": fast_strict_ngram_diagnostic_blocks,
        "ngram_draft_tokens": ngram_draft_tokens,
        "accepted_ngram_tokens": accepted_ngram_tokens,
        "ngram_draft_acceptance": accepted_ngram_tokens / max(1, ngram_draft_tokens),
        "ngram_tree_cycles": ngram_tree_cycles,
        "ngram_tree_branches": ngram_tree_branches,
        "ngram_tree_mean_branches": ngram_tree_branches / max(1, ngram_tree_cycles),
        "ngram_tree_nodes": ngram_tree_nodes,
        "ngram_tree_mean_nodes": ngram_tree_nodes / max(1, ngram_tree_cycles),
        "ngram_tree_selected_nodes": ngram_tree_selected_nodes,
        "ngram_tree_mean_selected_nodes": (
            ngram_tree_selected_nodes / max(1, ngram_tree_cycles)
        ),
        "ngram_tree_target_matched_drafts": ngram_tree_target_matched_drafts,
        "ngram_wide_route_evaluations": ngram_wide_route_evaluations,
        "ngram_wide_route_disables": ngram_wide_route_disables,
        "ngram_wide_route_disable_rate": (
            ngram_wide_route_disables / max(1, ngram_wide_route_evaluations)
        ),
        "ngram_wide_route_observations": ngram_wide_route_observations,
        "ngram_wide_route_extra_accepted_tokens": ngram_wide_route_extra_accepted_tokens,
        "ngram_wide_route_score_mean": (
            sum(ngram_wide_route_scores) / len(ngram_wide_route_scores)
            if ngram_wide_route_scores
            else None
        ),
        "ngram_wide_route_fallback_cycles": ngram_wide_route_fallback_cycles,
        "ngram_predecode_route_widths": ngram_predecode_route_widths,
        "ngram_predecode_route_reasons": ngram_predecode_route_reasons,
        "ngram_predecode_occurrence_policies": ngram_predecode_occurrence_policies,
        "ngram_predecode_occurrence_reasons": ngram_predecode_occurrence_reasons,
        "ngram_unchecked_gate_evaluations": ngram_unchecked_gate_evaluations,
        "ngram_unchecked_gate_eligible": ngram_unchecked_gate_eligible,
        "ngram_unchecked_gate_rate": (
            ngram_unchecked_gate_eligible / max(1, ngram_unchecked_gate_evaluations)
        ),
        "ngram_unchecked_budget_skips": ngram_unchecked_budget_skips,
        "ngram_unchecked_category_gate_evaluations": (
            ngram_unchecked_category_gate_evaluations
        ),
        "ngram_unchecked_category_gate_rejects": ngram_unchecked_category_gate_rejects,
        "ngram_unchecked_category_gate_reject_rate": (
            ngram_unchecked_category_gate_rejects
            / max(1, ngram_unchecked_category_gate_evaluations)
        ),
        "ngram_unchecked_category_trimmed_tokens": ngram_unchecked_category_trimmed_tokens,
        "ngram_unchecked_category_budget_closes": ngram_unchecked_category_budget_closes,
        "ngram_unchecked_category_closed_skips": ngram_unchecked_category_closed_skips,
        "lazy_t_sync_queued_tokens": lazy_t_sync_queued_tokens,
        "lazy_t_sync_flushes": lazy_t_sync_flushes,
        "lazy_t_sync_flushed_tokens": lazy_t_sync_flushed_tokens,
        "deferred_t_initializations": deferred_t_initializations,
        "deferred_t_init_skips": (
            len(results) - deferred_t_initializations if args.defer_t_init_until_latent else 0
        ),
        "deferred_t_init_tokens": deferred_t_init_tokens,
        "latent_position_gate_skips": latent_position_gate_skips,
        "latent_target_margin_gate_evaluations": latent_target_margin_gate_evaluations,
        "latent_target_margin_gate_skips": latent_target_margin_gate_skips,
        "latent_target_margin_gate_skip_rate": (
            latent_target_margin_gate_skips / max(1, latent_target_margin_gate_evaluations)
        ),
        "latent_target_margin_gate_closes": latent_target_margin_gate_closes,
        "latent_target_margin_gate_close_rate": (
            latent_target_margin_gate_closes / max(1, len(results))
        ),
        "latent_target_margin_gate_mean_close_position": (
            sum(latent_target_margin_gate_close_positions)
            / len(latent_target_margin_gate_close_positions)
            if latent_target_margin_gate_close_positions
            else None
        ),
        "latent_target_margin_mean": (
            sum(latent_target_margins) / len(latent_target_margins)
            if latent_target_margins
            else None
        ),
        "latent_target_margin_p10": latent_margin_quantile(0.10),
        "latent_target_margin_p50": latent_margin_quantile(0.50),
        "latent_target_margin_p90": latent_margin_quantile(0.90),
        "latent_unchecked_budget_skips": latent_unchecked_budget_skips,
        "latent_unchecked_category_gate_evaluations": (
            latent_unchecked_category_gate_evaluations
        ),
        "latent_unchecked_category_gate_rejects": latent_unchecked_category_gate_rejects,
        "latent_unchecked_category_gate_reject_rate": (
            latent_unchecked_category_gate_rejects
            / max(1, latent_unchecked_category_gate_evaluations)
        ),
        "latent_unchecked_category_trimmed_tokens": (
            latent_unchecked_category_trimmed_tokens
        ),
        "latent_unchecked_category_budget_closes": (
            latent_unchecked_category_budget_closes
        ),
        "latent_unchecked_category_closed_skips": latent_unchecked_category_closed_skips,
        "draft_cooldown_skips": draft_cooldown_skips,
        "draft_cooldown_activations": draft_cooldown_activations,
        "auto_route_evaluations": auto_route_evaluations,
        "auto_route_t_disables": auto_route_t_disables,
        "auto_route_disable_rate": auto_route_t_disables / max(1, auto_route_evaluations),
        "auto_route_early_evaluations": auto_route_early_evaluations,
        "auto_route_early_t_disables": auto_route_early_t_disables,
        "auto_route_latent_drafts": auto_route_latent_drafts,
        "auto_route_accepted_drafts": auto_route_accepted_drafts,
        "auto_route_latent_candidate_cycles": auto_route_latent_candidate_cycles,
        "auto_route_confident_candidate_cycles": auto_route_confident_candidate_cycles,
        "auto_route_score_kind": (
            "confident_candidate_rate" if route_on_proposal_rate else "target_acceptance"
        ),
        "missing_t_anchor_fallbacks": missing_t_anchor_fallbacks,
        "skipped_terminal_t_steps": skipped_terminal_t_steps,
        "auto_route_observed_acceptance": (
            auto_route_confident_candidate_cycles / max(1, auto_route_latent_candidate_cycles)
            if route_on_proposal_rate
            else auto_route_accepted_drafts / max(1, auto_route_latent_drafts)
        ),
        "auto_route_mean_disable_position": (
            sum(auto_route_disable_positions) / len(auto_route_disable_positions)
            if auto_route_disable_positions
            else None
        ),
        "cheap_policy_mean_threshold": (
            sum(cheap_policy_thresholds) / len(cheap_policy_thresholds)
            if cheap_policy_thresholds
            else None
        ),
        "draft_acceptance": accepted_drafts / max(1, total_drafts),
        "draft_target_match": (
            matched_drafts / target_match_observations if target_match_observations > 0 else None
        ),
        "verifier_rejection_rate": rejected_drafts / max(1, total_drafts + rejected_drafts),
        "baseline_target_calls": baseline_target_calls,
        "target_calls": adaptive_target_calls,
        "target_call_speedup": (
            baseline_target_calls / adaptive_target_calls
            if baseline_target_calls is not None and adaptive_target_calls > 0
            else None
        ),
        "verification_rounds": adaptive_verification_rounds,
        "verification_frequency_policy": args.verification_frequency_policy,
        "verification_risk_budget": (
            args.verification_risk_budget
            if args.verification_frequency_policy == "risk_budget"
            else None
        ),
        "verification_agreement_warmup": (
            args.verification_agreement_warmup
            if args.verification_frequency_policy == "agreement_gated"
            else None
        ),
        "verification_agreement_min_draft_margin": (
            args.verification_agreement_min_draft_margin
            if args.verification_frequency_policy == "agreement_gated"
            else None
        ),
        "verification_agreement_min_target_margin": (
            args.verification_agreement_min_target_margin
            if args.verification_frequency_policy == "agreement_gated"
            else None
        ),
        "verification_risk_budget_stops": verification_risk_budget_stops,
        "verification_risk_at_stops": verification_risk_at_stops,
        "verification_agreement_shadow_probes": verification_agreement_shadow_probes,
        "verification_agreement_shadow_matches": verification_agreement_shadow_matches,
        "verification_agreement_shadow_match_rate": (
            verification_agreement_shadow_matches
            / max(1, verification_agreement_shadow_probes)
        ),
        "verification_agreement_gate_openings": verification_agreement_gate_openings,
        "verification_agreement_gate_closures": verification_agreement_gate_closures,
        "verification_agreement_active_cycles": verification_agreement_active_cycles,
        "verification_agreement_max_streak": verification_agreement_max_streak,
        "verification_agreement_events": verification_agreement_events,
        "verification_intervals": verification_intervals,
        "mean_verification_interval": (
            sum(verification_intervals) / len(verification_intervals)
            if verification_intervals
            else None
        ),
        "max_verification_interval": max(verification_intervals, default=0),
        "verification_rollbacks": verification_rollbacks,
        "verification_rollback_rate": (
            verification_rollbacks / len(verification_intervals)
            if verification_intervals
            else 0.0
        ),
        "verification_wasted_drafts": verification_wasted_drafts,
        "verification_first_divergence_events": verification_first_divergence_events,
        "target_tokens_per_call": total_tokens / max(1, adaptive_target_calls),
        "mean_advance_per_verification_round": total_tokens / adaptive_verification_rounds,
        "mean_accepted_drafts_per_verification_round": accepted_drafts
        / adaptive_verification_rounds,
        "args": vars(args),
    }
    output = {"summary": summary, "results": results}
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "output_json": str(output_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
