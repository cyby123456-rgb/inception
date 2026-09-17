# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class FreezeArguments:
    r"""Arguments pertaining to the freeze (partial-parameter) training."""

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
    freeze_extra_modules: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
            )
        },
    )


@dataclass
class LoraArguments:
    r"""Arguments pertaining to the LoRA training."""

    additional_target: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    lora_alpha: int | None = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    loraplus_lr_ratio: float | None = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize a PiSSA adapter."},
    )
    pissa_iter: int = field(
        default=16,
        metadata={"help": "The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it."},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class OFTArguments:
    r"""Arguments pertaining to the OFT training."""

    additional_target: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    module_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the OFT fine-tuning."},
    )
    oft_rank: int = field(
        default=0,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_block_size: int = field(
        default=32,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply OFT. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class RecurFTArguments:
    r"""Arguments pertaining to RecurFT hidden-space recurrent training."""

    use_recurft: bool = field(
        default=False,
        metadata={"help": "Whether or not to use RecurFT hidden-space recurrent training in SFT."},
    )
    recurft_anchor_layer: int = field(
        default=-2,
        metadata={
            "help": (
                "The decoder layer whose output is treated as the new hidden space. "
                "Negative values are counted from the end, e.g. -2 means the penultimate layer."
            )
        },
    )
    recurft_pre_lora_rank: int = field(
        default=4,
        metadata={"help": "LoRA rank for layers up to and including the RecurFT anchor layer."},
    )
    recurft_last_lora_rank: int = field(
        default=32,
        metadata={
            "help": (
                "LoRA rank for post/projection layers from the recurrent hidden space to the model output, "
                "used to preserve final hidden states and logits."
            )
        },
    )
    recurft_t_lora_rank: int = field(
        default=32,
        metadata={"help": "LoRA rank for the recurrent T module copied from the final decoder tail block."},
    )
    recurft_tail_layers: int = field(
        default=1,
        metadata={
            "help": (
                "Number of final decoder layers used as the high-rank tail block and copied into T. "
                "Used only when `recurft_loop_start_layer` and `recurft_loop_end_layer` are unset. "
                "Set to 2 with `recurft_anchor_layer: -3` to use the last two layers."
            )
        },
    )
    recurft_loop_start_layer: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional first decoder layer copied into the recurrent T module. "
                "Negative values are counted from the end. When set, the recurrent hidden space is the input "
                "to this layer and `recurft_anchor_layer`/`recurft_tail_layers` are ignored."
            )
        },
    )
    recurft_loop_end_layer: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional last decoder layer copied into the recurrent T module, inclusive. "
                "Must be set together with `recurft_loop_start_layer`."
            )
        },
    )
    recurft_pre_lora_alpha: int | None = field(
        default=None,
        metadata={"help": "LoRA alpha for layers up to the RecurFT anchor layer (default: rank * 2)."},
    )
    recurft_last_lora_alpha: int | None = field(
        default=None,
        metadata={"help": "LoRA alpha for the final decoder tail block (default: rank * 2)."},
    )
    recurft_t_lora_alpha: int | None = field(
        default=None,
        metadata={"help": "LoRA alpha for the recurrent T module (default: rank * 2)."},
    )
    recurft_t_lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the recurrent T module LoRA."},
    )
    recurft_boundary_head_rank: int = field(
        default=0,
        metadata={
            "help": (
                "Rank of the optional lightweight hidden-to-token boundary adapter. "
                "Set to 0 to keep tail-layer draft projection."
            )
        },
    )
    recurft_token_conditioning_rank: int = field(
        default=0,
        metadata={
            "help": (
                "Rank of the optional token-embedding conditioner applied before recurrent T. "
                "Set to 0 to keep hidden-only recurrence."
            )
        },
    )
    recurft_joint_mode: str = field(
        default="legacy",
        metadata={"help": "legacy preserves existing frozen/staged behavior; trainable_target jointly trains target LoRA, T and boundary head."},
    )
    recurft_boundary_teacher_source: str = field(
        default="reference",
        metadata={"help": "Direct boundary KL teacher: reference base or detached current adapted target."},
    )
    recurft_stage1_heads_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Freeze existing model/T LoRA weights and train only the optional boundary head and "
                "token conditioner. Intended for cheap Stage-1 continuation pilots."
            )
        },
    )
    recurft_recurrent_trainable_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Freeze the target-model adapters and train only parameters inside the recurrent module "
                "that are already marked trainable (T LoRA, boundary/token heads, and rollout residuals). "
                "The copied T base layers and verifier remain frozen."
            )
        },
    )
    recurft_boundary_token_ce_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight of next-token CE for the optional lightweight boundary head."},
    )
    recurft_boundary_logit_kl_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight of teacher-logit KL for the optional lightweight boundary head."},
    )
    recurft_multistep_boundary_token_ce_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight of boundary-head next-token CE on recurrent rollout states."},
    )
    recurft_multistep_boundary_logit_kl_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight of boundary-head teacher-logit KL on recurrent rollout states."},
    )
    recurft_multistep_boundary_teacher_top1_labels: bool = field(
        default=False,
        metadata={"help": "Use frozen target argmax labels for rollout boundary CE instead of dataset next-token labels."},
    )
    recurft_boundary_loss_stride: int = field(
        default=8,
        metadata={"help": "Token stride used to subsample positions for boundary-head CE/KL."},
    )
    recurft_boundary_loss_max_tokens: int = field(
        default=64,
        metadata={
            "help": "Maximum boundary-head training positions per sequence; set to 0 to use every strided token."
        },
    )
    recurft_scheduled_sampling_max_ratio: float = field(
        default=0.0,
        metadata={
            "help": (
                "Maximum probability of conditioning multi-step T rollout on boundary-head draft tokens. "
                "Step 1 always uses the target token available from the target model."
            )
        },
    )
    recurft_scheduled_sampling_warmup_steps: int = field(
        default=0,
        metadata={"help": "Steps used to linearly warm up RecurFT token-conditioning scheduled sampling."},
    )
    recurft_scheduled_sampling_start_step: int = field(
        default=0,
        metadata={"help": "Global step where RecurFT scheduled-sampling warmup starts."},
    )
    recurft_hidden_loss_weight: float = field(
        default=10.0,
        metadata={"help": "Weight of the final hidden-state alignment loss."},
    )
    recurft_hidden_relative_mse_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight of the optional relative-MSE final hidden-state alignment loss. "
                "This is scale-normalized and is disabled by default."
            )
        },
    )
    recurft_hidden_cosine_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight of the optional cosine-direction final hidden-state alignment loss. "
                "The loss is 1 - cosine similarity and is disabled by default."
            )
        },
    )
    recurft_recurrent_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Final weight of the recurrent hidden autoregressive loss."},
    )
    recurft_recurrent_relative_mse_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight of the optional relative-MSE recurrent hidden prediction loss. "
                "It is multiplied by the recurrent warmup weight and disabled by default."
            )
        },
    )
    recurft_recurrent_cosine_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight of the optional cosine-direction recurrent hidden prediction loss. "
                "The loss is multiplied by the recurrent warmup weight and disabled by default."
            )
        },
    )
    recurft_recurrent_delta_mse_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight of the optional recurrent hidden transition-vector MSE loss. "
                "This compares T(h_t)-h_t with h_{t+1}-h_t and is disabled by default."
            )
        },
    )
    recurft_recurrent_delta_cosine_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight of the optional recurrent hidden transition-direction loss. "
                "The loss is 1 - cosine(T(h_t)-h_t, h_{t+1}-h_t) and is disabled by default."
            )
        },
    )
    recurft_recurrent_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of optimizer steps used to linearly warm up the recurrent loss weight."},
    )
    recurft_multistep_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight of the optional multi-step recurrent rollout loss. "
                "Set to 0 to keep the original one-step teacher-forced RecurFT objective."
            )
        },
    )
    recurft_multistep_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of optimizer steps used to linearly warm up the multi-step rollout loss."},
    )
    recurft_multistep_warmup_start_step: int = field(
        default=0,
        metadata={
            "help": (
                "Global optimizer step where multi-step rollout warmup starts. "
                "Use the resumed checkpoint step when adding multi-step loss to an already trained run."
            )
        },
    )
    recurft_multistep_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Maximum rollout horizon for the optional multi-step recurrent loss. "
                "The extra loss uses rollout targets from step 2 through this value because step 1 is "
                "already covered by `recurft_recurrent_loss_weight`."
            )
        },
    )
    recurft_multistep_stride: int = field(
        default=16,
        metadata={"help": "Token stride between anchor positions sampled for the optional multi-step loss."},
    )
    recurft_multistep_max_starts: int = field(
        default=8,
        metadata={
            "help": (
                "Maximum number of anchor positions per sequence used by the optional multi-step loss. "
                "Set to 0 to use all positions selected by `recurft_multistep_stride`."
            )
        },
    )
    recurft_multistep_start_selection: Literal["linspace", "cyclic"] = field(
        default="linspace",
        metadata={
            "help": (
                "How to subsample rollout anchors when their count exceeds "
                "`recurft_multistep_max_starts`. `linspace` preserves the legacy fixed anchors; "
                "`cyclic` rotates evenly spaced anchors with the global optimizer step."
            )
        },
    )
    recurft_multistep_residual_rank: int = field(
        default=0,
        metadata={
            "help": (
                "Rank of an optional zero-initialized recurrent rollout residual. Its first active step is "
                "controlled by `recurft_multistep_residual_start_step`; set the rank to 0 to preserve the "
                "historical shared-T architecture."
            )
        },
    )
    recurft_multistep_residual_alpha: int | None = field(
        default=None,
        metadata={"help": "Scaling alpha for the optional multi-step residual; defaults to twice its rank."},
    )
    recurft_multistep_residual_start_step: int = field(
        default=2,
        metadata={
            "help": (
                "First recurrent rollout step that uses the optional multi-step residual. "
                "The legacy value is 2; set to 1 when directly supervising the first deployable draft."
            )
        },
    )
    recurft_multistep_residual_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Freeze the target model and original recurrent T, training only the optional multi-step residual."
            )
        },
    )
    recurft_multistep_step1_residual_rank: int = field(
        default=0,
        metadata={
            "help": (
                "Rank of an optional zero-initialized residual used only for rollout step 1. "
                "This keeps first-step supervision separate from the residual used by later recurrent steps."
            )
        },
    )
    recurft_multistep_step1_residual_alpha: int | None = field(
        default=None,
        metadata={"help": "Scaling alpha for the optional step-1 residual; defaults to twice its rank."},
    )
    recurft_multistep_step1_residual_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Freeze the target model, recurrent T, and later-step residual, training only the step-1 residual."
            )
        },
    )
    recurft_multistep_context_tokens: int = field(
        default=128,
        metadata={
            "help": (
                "Number of ground-truth hidden tokens kept before each multi-step rollout anchor. "
                "Set to 0 to use the full prefix."
            )
        },
    )
    recurft_multistep_step_decay: float = field(
        default=1.0,
        metadata={"help": "Multiplicative decay applied to farther multi-step rollout losses."},
    )
    recurft_multistep_huber_beta: float = field(
        default=0.0,
        metadata={
            "help": (
                "Use SmoothL1/Huber as the base multi-step hidden loss when positive. "
                "Raw multi-step MSE is still logged for comparability."
            )
        },
    )
    recurft_multistep_mse_loss_clip: float = field(
        default=0.0,
        metadata={
            "help": (
                "Clip each token's base multi-step MSE loss before averaging when positive. "
                "Ignored when `recurft_multistep_huber_beta` is positive."
            )
        },
    )
    recurft_multistep_trim_ratio: float = field(
        default=0.0,
        metadata={
            "help": (
                "Drop the largest fraction of per-anchor-step multi-step training losses before aggregation. "
                "This only affects the optimized multi-step loss; raw MSE metrics are still averaged over all terms."
            )
        },
    )
    recurft_multistep_relative_mse_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Auxiliary relative-MSE weight inside the optional multi-step rollout loss. "
                "This is applied in addition to the base multi-step MSE and is disabled by default."
            )
        },
    )
    recurft_multistep_cosine_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Auxiliary cosine-direction weight inside the optional multi-step rollout loss. "
                "This is applied in addition to the base multi-step MSE and is disabled by default."
            )
        },
    )
    recurft_multistep_delta_mse_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Auxiliary transition-vector MSE weight inside the optional multi-step rollout loss. "
                "This compares predicted hidden deltas with teacher hidden deltas and is disabled by default."
            )
        },
    )
    recurft_multistep_delta_cosine_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Auxiliary transition-direction weight inside the optional multi-step rollout loss. "
                "The loss is 1 - cosine between predicted and teacher hidden deltas, disabled by default."
            )
        },
    )
    recurft_multistep_contrastive_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Auxiliary hidden-space contrastive weight inside the optional multi-step rollout loss. "
                "It asks each rollout hidden state to identify its correct future teacher hidden among nearby "
                "future hidden candidates. Disabled by default."
            )
        },
    )
    recurft_multistep_contrastive_temperature: float = field(
        default=0.1,
        metadata={"help": "Temperature for the optional hidden-space multi-step contrastive loss."},
    )
    recurft_multistep_contrastive_window: int = field(
        default=0,
        metadata={
            "help": (
                "Number of future teacher hidden states used as local contrastive candidates. "
                "Set to 0 to use `recurft_multistep_steps` candidates."
            )
        },
    )
    recurft_multistep_token_ce_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Auxiliary next-token cross-entropy weight inside the optional multi-step rollout loss. "
                "It projects T-rollout hidden states through the model tail and directly trains draft-token logits. "
                "Disabled by default because it adds extra tail-forward compute."
            )
        },
    )
    recurft_multistep_logit_kl_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Auxiliary KL weight from T-rollout draft logits to teacher target-model logits. "
                "This is closer to target-match acceptance than hidden MSE and is disabled by default."
            )
        },
    )
    recurft_multistep_logit_tv_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Auxiliary total-variation weight between T-rollout draft and teacher token distributions. "
                "This directly optimizes distribution overlap (1 - TV is the speculative-acceptance probability) "
                "and is disabled by default."
            )
        },
    )
    recurft_multistep_logit_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for the optional multi-step draft-logit KL and TV losses."},
    )
    recurft_multistep_logit_context_tokens: int = field(
        default=0,
        metadata={
            "help": (
                "Maximum hidden context length used when projecting rollout hidden states to draft logits. "
                "Set to 0 to use the full multi-step context; small values reduce the extra tail-forward cost."
            )
        },
    )
    recurft_multistep_step1_logit_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Additional weight for the token-CE and teacher-logit KL objective at the first T rollout step. "
                "Use it to preserve one-step draft quality while optimizing later rollout steps."
            )
        },
    )
    recurft_multistep_detach_rollout: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to stop gradients through previously predicted hidden states in multi-step rollout. "
                "This gives a stable off-manifold training signal without full backpropagation through time."
            )
        },
    )
    recurft_kl_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight of the optional output KL loss. Set to 0 to disable."},
    )
    recurft_kl_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature used by the optional output KL loss."},
    )
    recurft_sft_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight of the ordinary SFT cross-entropy loss when RecurFT is enabled."},
    )
    recurft_loss_on_labels_only: bool = field(
        default=False,
        metadata={"help": "Whether recurrent and KL losses should only use non-ignored label positions."},
    )
    recurft_verifier_lora_rank: int = field(
        default=32,
        metadata={"help": "LoRA rank reserved for the future RecurFT verifier adapter on T."},
    )
    recurft_verifier_lora_alpha: int | None = field(
        default=None,
        metadata={"help": "LoRA alpha for the future RecurFT verifier adapter on T (default: rank * 2)."},
    )
    recurft_verifier_lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the future RecurFT verifier adapter on T."},
    )
    recurft_verifier_probe_init_std: float = field(
        default=0.02,
        metadata={"help": "Initialization std for the learned hidden-space RecurFT verifier probe."},
    )


@dataclass
class RLHFArguments:
    r"""Arguments pertaining to the PPO, DPO and KTO training."""

    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    pref_bco_weight: float = field(
        default=0.0,
        metadata={"help": "The Binary Classifier Optimization coefficient in DPO training."},
    )
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the desirable losses in KTO training."},
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the undesirable losses in KTO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of epochs to perform in a PPO optimization step."},
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."},
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training."},
    )
    ref_model: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to the frozen reference model used for preference training or RecurFT hidden/KL "
                "preservation. For RecurFT continuation, pair it with `ref_model_adapters` pointing to "
                "the starting checkpoint's target adapter."
            )
        },
    )
    ref_model_adapters: str | None = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    ref_model_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    reward_model: str | None = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    reward_model_adapters: str | None = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    reward_model_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )
    ld_alpha: float | None = field(
        default=None,
        metadata={
            "help": (
                "Alpha parameter from the LD-DPO paper, which controls the weighting of"
                " the verbose token log-probabilities in responses."
            )
        },
    )


@dataclass
class GaloreArguments:
    r"""Arguments pertaining to the GaLore algorithm."""

    use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the gradient low-Rank projection (GaLore)."},
    )
    galore_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply GaLore. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients."},
    )
    galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection."},
    )
    galore_scale: float = field(
        default=2.0,
        metadata={"help": "GaLore scaling coefficient."},
    )
    galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection."},
    )
    galore_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )


@dataclass
class ApolloArguments:
    r"""Arguments pertaining to the APOLLO algorithm."""

    use_apollo: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the APOLLO optimizer."},
    )
    apollo_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply APOLLO. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    apollo_rank: int = field(
        default=16,
        metadata={"help": "The rank of APOLLO gradients."},
    )
    apollo_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the APOLLO projection."},
    )
    apollo_scale: float = field(
        default=32.0,
        metadata={"help": "APOLLO scaling coefficient."},
    )
    apollo_proj: Literal["svd", "random"] = field(
        default="random",
        metadata={"help": "Type of APOLLO low-rank projection algorithm (svd or random)."},
    )
    apollo_proj_type: Literal["std", "right", "left"] = field(
        default="std",
        metadata={"help": "Type of APOLLO projection."},
    )
    apollo_scale_type: Literal["channel", "tensor"] = field(
        default="channel",
        metadata={"help": "Type of APOLLO scaling (channel or tensor)."},
    )
    apollo_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )
    apollo_scale_front: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the norm-growth limiter in front of gradient scaling."},
    )


@dataclass
class BAdamArgument:
    r"""Arguments pertaining to the BAdam optimizer."""

    use_badam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the BAdam optimizer."},
    )
    badam_mode: Literal["layer", "ratio"] = field(
        default="layer",
        metadata={"help": "Whether to use layer-wise or ratio-wise BAdam optimizer."},
    )
    badam_start_block: int | None = field(
        default=None,
        metadata={"help": "The starting block index for layer-wise BAdam."},
    )
    badam_switch_mode: Literal["ascending", "descending", "random", "fixed"] | None = field(
        default="ascending",
        metadata={"help": "the strategy of picking block to update for layer-wise BAdam."},
    )
    badam_switch_interval: int | None = field(
        default=50,
        metadata={
            "help": "Number of steps to update the block for layer-wise BAdam. Use -1 to disable the block update."
        },
    )
    badam_update_ratio: float = field(
        default=0.05,
        metadata={"help": "The ratio of the update for ratio-wise BAdam."},
    )
    badam_mask_mode: Literal["adjacent", "scatter"] = field(
        default="adjacent",
        metadata={
            "help": (
                "The mode of the mask for BAdam optimizer. "
                "`adjacent` means that the trainable parameters are adjacent to each other, "
                "`scatter` means that trainable parameters are randomly choosed from the weight."
            )
        },
    )
    badam_verbose: int = field(
        default=0,
        metadata={
            "help": (
                "The verbosity level of BAdam optimizer. "
                "0 for no print, 1 for print the block prefix, 2 for print trainable parameters."
            )
        },
    )


@dataclass
class SwanLabArguments:
    use_swanlab: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the SwanLab (an experiment tracking and visualization tool)."},
    )
    swanlab_project: str | None = field(
        default="llamafactory",
        metadata={"help": "The project name in SwanLab."},
    )
    swanlab_workspace: str | None = field(
        default=None,
        metadata={"help": "The workspace name in SwanLab."},
    )
    swanlab_run_name: str | None = field(
        default=None,
        metadata={"help": "The experiment name in SwanLab."},
    )
    swanlab_mode: Literal["cloud", "local"] = field(
        default="cloud",
        metadata={"help": "The mode of SwanLab."},
    )
    swanlab_api_key: str | None = field(
        default=None,
        metadata={"help": "The API key for SwanLab."},
    )
    swanlab_logdir: str | None = field(
        default=None,
        metadata={"help": "The log directory for SwanLab."},
    )
    swanlab_lark_webhook_url: str | None = field(
        default=None,
        metadata={"help": "The Lark(飞书) webhook URL for SwanLab."},
    )
    swanlab_lark_secret: str | None = field(
        default=None,
        metadata={"help": "The Lark(飞书) secret for SwanLab."},
    )


@dataclass
class FinetuningArguments(
    SwanLabArguments,
    BAdamArgument,
    ApolloArguments,
    GaloreArguments,
    RLHFArguments,
    RecurFTArguments,
    LoraArguments,
    OFTArguments,
    FreezeArguments,
):
    r"""Arguments pertaining to which techniques we are going to fine-tuning with."""

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Literal["lora", "oft", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    use_adam_mini: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Adam-mini optimizer."},
    )
    use_mca: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use MCA (Megatron Core Adapter) training. "
                "Controlled by USE_MCA environment variable."
            )
        },
    )
    use_hyper_parallel: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use HyperParallel distributed training backend (FSDP/TP). "
                "Only supported for the 'pt' and 'sft' stages with full fine-tuning."
            )
        },
    )
    hyper_parallel_args: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to a JSON file containing HyperParallel strategy arguments "
                "(e.g., tp_size, param_dtype). Used when use_hyper_parallel=True."
            )
        },
    )
    use_muon: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Muon optimizer."},
    )
    use_dft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the DFT loss."},
    )
    use_asft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the ASFT loss."},
    )
    asft_alpha: float = field(
        default=0.1,
        metadata={"help": "The alpha parameter for ASFT loss to control the power of adaptive weight."},
    )
    use_eaft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the EAFT loss."},
    )
    eaft_alpha: float = field(
        default=1.0,
        metadata={"help": "The alpha parameter for EAFT loss to control the power of adaptive weight."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether ot not to freeze the vision tower in MLLM training."},
    )
    freeze_multi_modal_projector: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the multi modal projector in MLLM training."},
    )
    freeze_language_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the language model in MLLM training."},
    )
    compute_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the token-level accuracy at evaluation."},
    )
    disable_shuffling: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the shuffling of the training set."},
    )
    early_stopping_steps: int | None = field(
        default=None,
        metadata={"help": "Number of steps to stop training if the `metric_for_best_model` does not improve."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
    include_effective_tokens_per_second: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute effective tokens per second."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: list[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: list[str] | None = split_arg(self.freeze_extra_modules)
        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: list[str] = split_arg(self.lora_target)
        self.oft_target: list[str] = split_arg(self.oft_target)
        self.additional_target: list[str] | None = split_arg(self.additional_target)
        self.galore_target: list[str] = split_arg(self.galore_target)
        self.apollo_target: list[str] = split_arg(self.apollo_target)
        self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]

        assert self.finetuning_type in ["lora", "oft", "freeze", "full"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("`reward_model` is necessary for PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "oft" and self.finetuning_type != "oft":
            raise ValueError("`reward_model_type` cannot be oft for Freeze/Full PPO training.")

        if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
            raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")

        if self.finetuning_type == "lora" and (self.use_galore or self.use_apollo or self.use_badam):
            raise ValueError("Cannot use LoRA with GaLore, APOLLO or BAdam together.")

        if int(self.use_galore) + int(self.use_apollo) + (self.use_badam) > 1:
            raise ValueError("Cannot use GaLore, APOLLO or BAdam together.")

        if self.pissa_init and (self.stage in ["ppo", "kto"] or self.use_ref_model):
            raise ValueError("Cannot use PiSSA for current training stage.")

        if self.use_recurft:
            if self.stage != "sft":
                raise ValueError("RecurFT is currently only supported for the SFT stage.")

            if self.finetuning_type != "lora":
                raise ValueError("RecurFT currently requires `finetuning_type: lora`.")

            if (
                self.recurft_pre_lora_rank <= 0
                or self.recurft_last_lora_rank <= 0
                or self.recurft_t_lora_rank <= 0
                or self.recurft_verifier_lora_rank <= 0
            ):
                raise ValueError("RecurFT LoRA ranks must be positive.")

            if self.recurft_tail_layers <= 0:
                raise ValueError("`recurft_tail_layers` must be positive.")

            if (self.recurft_loop_start_layer is None) != (self.recurft_loop_end_layer is None):
                raise ValueError("`recurft_loop_start_layer` and `recurft_loop_end_layer` must be set together.")

            if self.recurft_kl_temperature <= 0:
                raise ValueError("`recurft_kl_temperature` must be positive.")

            if self.recurft_boundary_head_rank < 0 or self.recurft_token_conditioning_rank < 0:
                raise ValueError("RecurFT boundary-head and token-conditioning ranks must be non-negative.")

            if self.recurft_boundary_token_ce_loss_weight < 0 or self.recurft_boundary_logit_kl_loss_weight < 0:
                raise ValueError("RecurFT boundary-head loss weights must be non-negative.")

            if (
                self.recurft_multistep_boundary_token_ce_loss_weight < 0
                or self.recurft_multistep_boundary_logit_kl_loss_weight < 0
            ):
                raise ValueError("RecurFT multi-step boundary-head loss weights must be non-negative.")

            if self.recurft_boundary_loss_stride <= 0 or self.recurft_boundary_loss_max_tokens < 0:
                raise ValueError("RecurFT boundary-head stride must be positive and max tokens non-negative.")

            if not 0.0 <= self.recurft_scheduled_sampling_max_ratio <= 1.0:
                raise ValueError("`recurft_scheduled_sampling_max_ratio` must be in [0, 1].")

            if self.recurft_scheduled_sampling_warmup_steps < 0:
                raise ValueError("`recurft_scheduled_sampling_warmup_steps` must be non-negative.")

            if self.recurft_scheduled_sampling_start_step < 0:
                raise ValueError("`recurft_scheduled_sampling_start_step` must be non-negative.")

            if (
                self.recurft_boundary_token_ce_loss_weight > 0.0
                or self.recurft_boundary_logit_kl_loss_weight > 0.0
                or self.recurft_multistep_boundary_token_ce_loss_weight > 0.0
                or self.recurft_multistep_boundary_logit_kl_loss_weight > 0.0
            ) and self.recurft_boundary_head_rank <= 0:
                raise ValueError("Boundary-head losses require a positive `recurft_boundary_head_rank`.")

            if self.recurft_scheduled_sampling_max_ratio > 0.0 and (
                self.recurft_boundary_head_rank <= 0 or self.recurft_token_conditioning_rank <= 0
            ):
                raise ValueError("Scheduled sampling requires both boundary-head and token-conditioning ranks.")

            if self.recurft_joint_mode not in ("legacy", "trainable_target"):
                raise ValueError("Unknown recurft_joint_mode")
            if self.recurft_boundary_teacher_source not in ("reference", "target"):
                raise ValueError("Unknown recurft_boundary_teacher_source")
            if self.recurft_joint_mode == "trainable_target":
                if (self.recurft_stage1_heads_only or self.recurft_recurrent_trainable_only
                        or self.recurft_multistep_residual_only or self.recurft_multistep_step1_residual_only):
                    raise ValueError("trainable_target joint mode cannot freeze target/T")
                if self.recurft_boundary_head_rank <= 0 or self.recurft_boundary_teacher_source != "target":
                    raise ValueError("trainable_target requires a boundary head and current target teacher")
                if self.recurft_kl_loss_weight <= 0:
                    raise ValueError("trainable_target requires target-to-base KL regularization")
                if self.recurft_recurrent_loss_weight <= 0 or self.recurft_t_lora_rank <= 0:
                    raise ValueError("trainable_target requires an active recurrent loss gate and T LoRA")
                if not (self.recurft_multistep_loss_weight > 0 and self.recurft_multistep_steps > 1
                        and (self.recurft_multistep_boundary_logit_kl_loss_weight > 0
                             or self.recurft_multistep_boundary_token_ce_loss_weight > 0)):
                    raise ValueError("trainable_target requires an active rollout boundary objective")
                if (self.recurft_token_conditioning_rank or self.recurft_multistep_residual_rank
                        or self.recurft_multistep_step1_residual_rank):
                    raise ValueError("trainable_target supports target LoRA, T LoRA and boundary head only")

            if self.recurft_stage1_heads_only and (
                self.recurft_boundary_head_rank <= 0 and self.recurft_token_conditioning_rank <= 0
            ):
                raise ValueError("`recurft_stage1_heads_only` requires a boundary head or token conditioner.")

            if self.recurft_hidden_relative_mse_loss_weight < 0:
                raise ValueError("`recurft_hidden_relative_mse_loss_weight` must be non-negative.")

            if self.recurft_hidden_cosine_loss_weight < 0:
                raise ValueError("`recurft_hidden_cosine_loss_weight` must be non-negative.")

            if self.recurft_recurrent_relative_mse_loss_weight < 0:
                raise ValueError("`recurft_recurrent_relative_mse_loss_weight` must be non-negative.")

            if self.recurft_recurrent_cosine_loss_weight < 0:
                raise ValueError("`recurft_recurrent_cosine_loss_weight` must be non-negative.")

            if self.recurft_recurrent_delta_mse_loss_weight < 0:
                raise ValueError("`recurft_recurrent_delta_mse_loss_weight` must be non-negative.")

            if self.recurft_recurrent_delta_cosine_loss_weight < 0:
                raise ValueError("`recurft_recurrent_delta_cosine_loss_weight` must be non-negative.")

            if self.recurft_multistep_loss_weight < 0:
                raise ValueError("`recurft_multistep_loss_weight` must be non-negative.")

            if self.recurft_multistep_warmup_steps < 0:
                raise ValueError("`recurft_multistep_warmup_steps` must be non-negative.")

            if self.recurft_multistep_warmup_start_step < 0:
                raise ValueError("`recurft_multistep_warmup_start_step` must be non-negative.")

            if self.recurft_multistep_steps <= 0:
                raise ValueError("`recurft_multistep_steps` must be positive.")

            if self.recurft_multistep_stride <= 0:
                raise ValueError("`recurft_multistep_stride` must be positive.")

            if self.recurft_multistep_max_starts < 0:
                raise ValueError("`recurft_multistep_max_starts` must be non-negative.")

            if self.recurft_multistep_start_selection not in {"linspace", "cyclic"}:
                raise ValueError("`recurft_multistep_start_selection` must be `linspace` or `cyclic`.")

            if self.recurft_multistep_residual_rank < 0:
                raise ValueError("`recurft_multistep_residual_rank` must be non-negative.")

            if self.recurft_multistep_residual_alpha is not None and self.recurft_multistep_residual_alpha <= 0:
                raise ValueError("`recurft_multistep_residual_alpha` must be positive when provided.")

            if self.recurft_multistep_residual_start_step <= 0:
                raise ValueError("`recurft_multistep_residual_start_step` must be positive.")

            if self.recurft_multistep_residual_only and self.recurft_multistep_residual_rank <= 0:
                raise ValueError(
                    "`recurft_multistep_residual_only` requires a positive `recurft_multistep_residual_rank`."
                )

            if self.recurft_multistep_residual_only and self.recurft_stage1_heads_only:
                raise ValueError("Multi-step residual-only and Stage-1 heads-only training are mutually exclusive.")

            if self.recurft_multistep_step1_residual_rank < 0:
                raise ValueError("`recurft_multistep_step1_residual_rank` must be non-negative.")

            if (
                self.recurft_multistep_step1_residual_alpha is not None
                and self.recurft_multistep_step1_residual_alpha <= 0
            ):
                raise ValueError("`recurft_multistep_step1_residual_alpha` must be positive when provided.")

            if self.recurft_multistep_step1_residual_only and self.recurft_multistep_step1_residual_rank <= 0:
                raise ValueError(
                    "`recurft_multistep_step1_residual_only` requires a positive "
                    "`recurft_multistep_step1_residual_rank`."
                )

            residual_only_modes = sum(
                (
                    self.recurft_multistep_residual_only,
                    self.recurft_multistep_step1_residual_only,
                    self.recurft_stage1_heads_only,
                    self.recurft_recurrent_trainable_only,
                )
            )
            if residual_only_modes > 1:
                raise ValueError("RecurFT restricted trainable-scope modes are mutually exclusive.")

            if self.recurft_multistep_context_tokens < 0:
                raise ValueError("`recurft_multistep_context_tokens` must be non-negative.")

            if self.recurft_multistep_step_decay <= 0:
                raise ValueError("`recurft_multistep_step_decay` must be positive.")

            if self.recurft_multistep_huber_beta < 0:
                raise ValueError("`recurft_multistep_huber_beta` must be non-negative.")

            if self.recurft_multistep_mse_loss_clip < 0:
                raise ValueError("`recurft_multistep_mse_loss_clip` must be non-negative.")

            if not 0 <= self.recurft_multistep_trim_ratio < 1:
                raise ValueError("`recurft_multistep_trim_ratio` must be in [0, 1).")

            if self.recurft_multistep_relative_mse_loss_weight < 0:
                raise ValueError("`recurft_multistep_relative_mse_loss_weight` must be non-negative.")

            if self.recurft_multistep_cosine_loss_weight < 0:
                raise ValueError("`recurft_multistep_cosine_loss_weight` must be non-negative.")

            if self.recurft_multistep_delta_mse_loss_weight < 0:
                raise ValueError("`recurft_multistep_delta_mse_loss_weight` must be non-negative.")

            if self.recurft_multistep_delta_cosine_loss_weight < 0:
                raise ValueError("`recurft_multistep_delta_cosine_loss_weight` must be non-negative.")

            if self.recurft_multistep_contrastive_loss_weight < 0:
                raise ValueError("`recurft_multistep_contrastive_loss_weight` must be non-negative.")

            if self.recurft_multistep_contrastive_temperature <= 0:
                raise ValueError("`recurft_multistep_contrastive_temperature` must be positive.")

            if self.recurft_multistep_contrastive_window < 0:
                raise ValueError("`recurft_multistep_contrastive_window` must be non-negative.")

            if self.recurft_multistep_token_ce_loss_weight < 0:
                raise ValueError("`recurft_multistep_token_ce_loss_weight` must be non-negative.")

            if self.recurft_multistep_logit_kl_loss_weight < 0:
                raise ValueError("`recurft_multistep_logit_kl_loss_weight` must be non-negative.")

            if self.recurft_multistep_logit_tv_loss_weight < 0:
                raise ValueError("`recurft_multistep_logit_tv_loss_weight` must be non-negative.")

            if self.recurft_multistep_logit_temperature <= 0:
                raise ValueError("`recurft_multistep_logit_temperature` must be positive.")

            if self.recurft_multistep_logit_context_tokens < 0:
                raise ValueError("`recurft_multistep_logit_context_tokens` must be non-negative.")

            if self.recurft_multistep_step1_logit_loss_weight < 0:
                raise ValueError("`recurft_multistep_step1_logit_loss_weight` must be non-negative.")

            if self.recurft_verifier_probe_init_std <= 0:
                raise ValueError("`recurft_verifier_probe_init_std` must be positive.")

        if self.finetuning_type != "lora":
            if self.loraplus_lr_ratio is not None:
                raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")

            if self.use_rslora:
                raise ValueError("`use_rslora` is only valid for LoRA training.")

            if self.use_dora:
                raise ValueError("`use_dora` is only valid for LoRA training.")

            if self.pissa_init:
                raise ValueError("`pissa_init` is only valid for LoRA training.")

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("api_key") else v for k, v in args.items()}
        return args
