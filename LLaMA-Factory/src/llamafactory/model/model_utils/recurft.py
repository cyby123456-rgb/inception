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

import copy
import inspect
import json
import os
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import nn

from ...extras import logging
from ...extras.packages import is_safetensors_available


if is_safetensors_available():
    from safetensors.torch import load_file, save_file


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)

RECURFT_CONFIG_NAME = "recurft_config.json"
RECURFT_SAFE_WEIGHTS_NAME = "recurft_recurrent.safetensors"
RECURFT_WEIGHTS_NAME = "recurft_recurrent.bin"

_DECODER_LAYER_PATHS = (
    "model.layers",
    "language_model.model.layers",
    "model.language_model.layers",
    "model.decoder.layers",
    "decoder.layers",
    "transformer.h",
    "transformer.blocks",
    "transformer.encoder.layers",
    "gpt_neox.layers",
)


def _get_nested_attr(module: nn.Module, path: str) -> Any:
    current = module
    for key in path.split("."):
        current = getattr(current, key, None)
        if current is None:
            return None

    return current


def find_decoder_layers(model: "PreTrainedModel") -> tuple[nn.ModuleList | list[nn.Module], str]:
    r"""Find the decoder block list for common causal LM architectures."""
    for path in _DECODER_LAYER_PATHS:
        layers = _get_nested_attr(model, path)
        if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
            return layers, path

    raise ValueError(
        "RecurFT cannot locate decoder layers. Please add this model's layer path to `_DECODER_LAYER_PATHS`."
    )


def _resolve_layer_index(num_layers: int, layer_index: int, name: str) -> int:
    resolved = num_layers + layer_index if layer_index < 0 else layer_index
    if resolved < 0 or resolved >= num_layers:
        raise ValueError(f"`{name}` {layer_index} is out of range for {num_layers} layers.")

    return resolved


def resolve_recurft_layer_index(num_layers: int, layer_index: int, tail_layers: int = 1) -> int:
    if tail_layers <= 0 or tail_layers >= num_layers:
        raise ValueError(f"`recurft_tail_layers` {tail_layers} is out of range for {num_layers} layers.")

    resolved = _resolve_layer_index(num_layers, layer_index, "recurft_anchor_layer")
    tail_start = num_layers - tail_layers
    if resolved >= tail_start:
        raise ValueError(
            "`recurft_anchor_layer` must be before the RecurFT tail block. "
            f"For `recurft_tail_layers: {tail_layers}`, use a layer index <= {tail_start - 1} "
            f"(for example, {-tail_layers - 1})."
        )

    return resolved


def resolve_recurft_layout(num_layers: int, finetuning_args: "FinetuningArguments") -> dict[str, Any]:
    r"""Resolve the recurrent hidden boundary, loop block, and projection block.

    In the explicit layout, `loop_start_layer..loop_end_layer` is the decoder
    layer range copied into T. The recurrent hidden space is the input to
    `loop_start_layer`, i.e. `hidden_states[loop_start_layer]`.
    """

    last_idx = num_layers - 1
    explicit_loop = finetuning_args.recurft_loop_start_layer is not None
    if explicit_loop:
        loop_start = _resolve_layer_index(num_layers, finetuning_args.recurft_loop_start_layer, "recurft_loop_start_layer")
        loop_end = _resolve_layer_index(num_layers, finetuning_args.recurft_loop_end_layer, "recurft_loop_end_layer")
        if loop_start > loop_end:
            raise ValueError("`recurft_loop_start_layer` must be <= `recurft_loop_end_layer`.")

        recurrent_hidden_state_index = loop_start
        anchor_idx = loop_start - 1
        tail_layers = None
        tail_start = None
    else:
        tail_layers = finetuning_args.recurft_tail_layers
        anchor_idx = resolve_recurft_layer_index(num_layers, finetuning_args.recurft_anchor_layer, tail_layers)
        loop_start = num_layers - tail_layers
        loop_end = last_idx
        recurrent_hidden_state_index = anchor_idx + 1
        tail_start = loop_start

    return {
        "anchor_layer": anchor_idx,
        "last_layer": last_idx,
        "tail_layers": tail_layers,
        "tail_layer_start": tail_start,
        "tail_layer_ids": list(range(loop_start, num_layers)) if loop_end == last_idx else None,
        "loop_start_layer": loop_start,
        "loop_end_layer": loop_end,
        "loop_layer_ids": list(range(loop_start, loop_end + 1)),
        "recurrent_hidden_state_index": recurrent_hidden_state_index,
        "projection_layer_ids": list(range(recurrent_hidden_state_index, num_layers)),
    }


def _is_linear_module(module: nn.Module) -> bool:
    return "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__


def _target_matches(full_name: str, local_name: str, target_modules: list[str]) -> bool:
    if len(target_modules) == 1 and target_modules[0] == "all":
        return True

    leaf_name = local_name.rsplit(".", maxsplit=1)[-1]
    return any(
        target == leaf_name or local_name.endswith(target) or full_name.endswith(target) for target in target_modules
    )


def collect_recurft_lora_targets(
    model: "PreTrainedModel", finetuning_args: "FinetuningArguments"
) -> tuple[list[str], dict[str, int], dict[str, int], dict[str, Any]]:
    r"""Collect exact LoRA target module names and per-layer rank/alpha patterns."""
    layers, layers_path = find_decoder_layers(model)
    num_layers = len(layers)
    layout = resolve_recurft_layout(num_layers, finetuning_args)
    anchor_idx = layout["anchor_layer"]
    loop_start = layout["loop_start_layer"]
    loop_end = layout["loop_end_layer"]

    pre_rank = finetuning_args.recurft_pre_lora_rank
    last_rank = finetuning_args.recurft_last_lora_rank
    pre_alpha = finetuning_args.recurft_pre_lora_alpha or pre_rank * 2
    last_alpha = finetuning_args.recurft_last_lora_alpha or last_rank * 2

    target_modules: list[str] = []
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    pre_layer_ids = set(range(layout["recurrent_hidden_state_index"]))
    post_layer_ids = set(layout["projection_layer_ids"])
    trainable_layer_ids = sorted(pre_layer_ids | post_layer_ids)
    for layer_idx in trainable_layer_ids:
        for local_name, module in layers[layer_idx].named_modules():
            if local_name == "" or not _is_linear_module(module):
                continue

            full_name = f"{layers_path}.{layer_idx}.{local_name}"
            if not _target_matches(full_name, local_name, finetuning_args.lora_target):
                continue

            target_modules.append(full_name)
            if layer_idx in pre_layer_ids:
                rank_pattern[full_name] = pre_rank
                alpha_pattern[full_name] = pre_alpha
            else:
                rank_pattern[full_name] = last_rank
                alpha_pattern[full_name] = last_alpha

    if not target_modules:
        raise ValueError("RecurFT found no LoRA target modules. Please check `lora_target` and the model structure.")

    metadata = {
        **layout,
        "layers_path": layers_path,
        "pre_lora_rank": pre_rank,
        "last_lora_rank": last_rank,
        "t_lora_rank": finetuning_args.recurft_t_lora_rank,
        "pre_lora_alpha": pre_alpha,
        "last_lora_alpha": last_alpha,
        "t_lora_alpha": finetuning_args.recurft_t_lora_alpha or finetuning_args.recurft_t_lora_rank * 2,
        "boundary_head_rank": finetuning_args.recurft_boundary_head_rank,
        "token_conditioning_rank": finetuning_args.recurft_token_conditioning_rank,
        "multistep_residual_rank": finetuning_args.recurft_multistep_residual_rank,
        "multistep_residual_alpha": (
            finetuning_args.recurft_multistep_residual_alpha
            or finetuning_args.recurft_multistep_residual_rank * 2
        ),
        "multistep_residual_start_step": finetuning_args.recurft_multistep_residual_start_step,
        "multistep_step1_residual_rank": finetuning_args.recurft_multistep_step1_residual_rank,
        "multistep_step1_residual_alpha": (
            finetuning_args.recurft_multistep_step1_residual_alpha
            or finetuning_args.recurft_multistep_step1_residual_rank * 2
        ),
        "verifier_lora_rank": finetuning_args.recurft_verifier_lora_rank,
        "verifier_lora_alpha": (
            finetuning_args.recurft_verifier_lora_alpha or finetuning_args.recurft_verifier_lora_rank * 2
        ),
        "verifier_probe_init_std": finetuning_args.recurft_verifier_probe_init_std,
        "lora_target": finetuning_args.lora_target,
    }
    pre_desc = f"0-{anchor_idx}" if anchor_idx >= 0 else "none"
    post_desc = "{}-{}".format(min(post_layer_ids), max(post_layer_ids)) if post_layer_ids else "none"
    logger.info_rank0(
        "RecurFT applies LoRA to pre layers {} with rank {} and post/projection layers {} with rank {}; "
        "T copies loop layers {}-{}.".format(
            pre_desc, pre_rank, post_desc, last_rank, loop_start, loop_end
        )
    )
    return target_modules, rank_pattern, alpha_pattern, metadata


class RecurFTLoraLinear(nn.Module):
    r"""A minimal LoRA wrapper used by the recurrent T module."""

    def __init__(self, base_layer: nn.Module, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        if not hasattr(base_layer, "in_features") or not hasattr(base_layer, "out_features"):
            raise ValueError(f"Cannot inject RecurFT LoRA into {base_layer.__class__.__name__}.")

        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.verifier_enabled = False
        self.verifier_scaling = 0.0
        self.verifier_dropout = nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        self.verifier_lora_A: Optional[nn.Linear] = None
        self.verifier_lora_B: Optional[nn.Linear] = None
        self._lora_merged = False
        self.lora_A.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.lora_B.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        for param in self.base_layer.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_weight = getattr(self.base_layer, "weight", None)
        base_input = x.to(base_weight.dtype) if base_weight is not None else x
        result = self.base_layer(base_input).to(x.dtype)
        if not self._lora_merged:
            lora_result = self.lora_B(self.lora_A(self.dropout(x.to(self.lora_A.weight.dtype))))
            result = result + lora_result.to(x.dtype) * self.scaling
        if self.verifier_enabled and self.verifier_lora_A is not None and self.verifier_lora_B is not None:
            verifier_input = self.verifier_dropout(x.to(self.verifier_lora_A.weight.dtype))
            verifier_result = self.verifier_lora_B(self.verifier_lora_A(verifier_input))
            result = result + verifier_result.to(x.dtype) * self.verifier_scaling

        return result

    @torch.no_grad()
    def merge_lora_(self) -> None:
        """Materialize the recurrent adapter into its private base weight for inference."""
        if self._lora_merged:
            return

        weight = getattr(self.base_layer, "weight", None)
        if weight is None or not torch.is_floating_point(weight):
            raise ValueError("RecurFT can only merge LoRA into a floating-point base weight.")

        delta = self.lora_B.weight.float() @ self.lora_A.weight.float()
        weight.add_(delta.to(device=weight.device, dtype=weight.dtype), alpha=self.scaling)
        self._lora_merged = True

    @torch.no_grad()
    def unmerge_lora_(self) -> None:
        if not self._lora_merged:
            return

        weight = getattr(self.base_layer, "weight", None)
        if weight is None or not torch.is_floating_point(weight):
            raise ValueError("RecurFT can only unmerge LoRA from a floating-point base weight.")

        delta = self.lora_B.weight.float() @ self.lora_A.weight.float()
        weight.add_(delta.to(device=weight.device, dtype=weight.dtype), alpha=-self.scaling)
        self._lora_merged = False

    def add_verifier_adapter(self, rank: int, alpha: int, dropout: float) -> None:
        self.verifier_scaling = alpha / rank
        self.verifier_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.verifier_lora_A = nn.Linear(self.base_layer.in_features, rank, bias=False)
        self.verifier_lora_B = nn.Linear(rank, self.base_layer.out_features, bias=False)
        self.verifier_lora_A.to(device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype)
        self.verifier_lora_B.to(device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype)
        nn.init.kaiming_uniform_(self.verifier_lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.verifier_lora_B.weight)
        self.set_verifier_trainable(False)

    def set_verifier_enabled(self, enabled: bool) -> None:
        self.verifier_enabled = enabled

    def set_verifier_trainable(self, trainable: bool) -> None:
        if self.verifier_lora_A is not None:
            for param in self.verifier_lora_A.parameters():
                param.requires_grad_(trainable)

        if self.verifier_lora_B is not None:
            for param in self.verifier_lora_B.parameters():
                param.requires_grad_(trainable)


def _set_submodule(module: nn.Module, name: str, child: nn.Module) -> None:
    parent = module
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)

    setattr(parent, parts[-1], child)


def inject_recurft_lora(
    module: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: int,
    dropout: float,
) -> None:
    replacements: list[tuple[str, RecurFTLoraLinear]] = []
    for local_name, child in module.named_modules():
        if local_name == "" or not _is_linear_module(child):
            continue

        if _target_matches(local_name, local_name, target_modules):
            replacements.append((local_name, RecurFTLoraLinear(child, rank=rank, alpha=alpha, dropout=dropout)))

    if not replacements:
        raise ValueError("RecurFT could not inject LoRA into the recurrent T module.")

    for local_name, wrapped in replacements:
        _set_submodule(module, local_name, wrapped)


class RecurFTRecurrentModule(nn.Module):
    r"""A copy of the final decoder tail block with its own LoRA weights."""

    def __init__(
        self,
        layers: list[nn.Module],
        target_modules: list[str],
        rank: int,
        alpha: int,
        dropout: float,
        metadata: dict[str, Any],
        verifier_rank: int,
        verifier_alpha: int,
        verifier_dropout: float,
        verifier_probe_init_std: float,
        boundary_head_rank: int = 0,
        token_conditioning_rank: int = 0,
        multistep_residual_rank: int = 0,
        multistep_residual_alpha: int = 0,
        multistep_residual_start_step: int = 2,
        multistep_step1_residual_rank: int = 0,
        multistep_step1_residual_alpha: int = 0,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("RecurFT recurrent T requires at least one decoder layer.")

        self.metadata = metadata
        if len(layers) == 1:
            # Keep the historical `layer.*` state_dict keys for one-layer T checkpoints.
            self.layer = copy.deepcopy(layers[0])
            self.layers: Optional[nn.ModuleList] = None
        else:
            self.layers = nn.ModuleList(copy.deepcopy(layer) for layer in layers)

        for layer in self._iter_layers():
            for param in layer.parameters():
                param.requires_grad_(False)

            inject_recurft_lora(layer, target_modules=target_modules, rank=rank, alpha=alpha, dropout=dropout)

        self._layer_forward_parameter_names = {
            id(layer): frozenset(inspect.signature(layer.forward).parameters)
            for layer in self._iter_layers()
        }
        self._cached_causal_lm_source_id: Optional[int] = None
        self._cached_rotary_source_id: Optional[int] = None
        object.__setattr__(self, "_cached_causal_lm", None)
        object.__setattr__(self, "_cached_rotary_emb", None)

        self.hidden_size = self._infer_hidden_size()
        device, dtype = self._infer_probe_device_dtype()
        self.boundary_head_rank = boundary_head_rank
        self.token_conditioning_rank = token_conditioning_rank
        self.multistep_residual_rank = multistep_residual_rank
        self.multistep_residual_start_step = multistep_residual_start_step
        self.multistep_step1_residual_rank = multistep_step1_residual_rank
        if boundary_head_rank > 0:
            self.boundary_norm: Optional[nn.Module] = nn.LayerNorm(self.hidden_size)
            self.boundary_A: Optional[nn.Linear] = nn.Linear(self.hidden_size, boundary_head_rank, bias=False)
            self.boundary_B: Optional[nn.Linear] = nn.Linear(boundary_head_rank, self.hidden_size, bias=False)
            self.boundary_norm.to(device=device, dtype=dtype)
            self.boundary_A.to(device=device, dtype=dtype)
            self.boundary_B.to(device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.boundary_A.weight, a=5**0.5)
            nn.init.zeros_(self.boundary_B.weight)
        else:
            self.boundary_norm = None
            self.boundary_A = None
            self.boundary_B = None

        if token_conditioning_rank > 0:
            self.token_conditioning_norm: Optional[nn.Module] = nn.LayerNorm(self.hidden_size)
            self.token_conditioning_A: Optional[nn.Linear] = nn.Linear(
                self.hidden_size, token_conditioning_rank, bias=False
            )
            self.token_conditioning_B: Optional[nn.Linear] = nn.Linear(
                token_conditioning_rank, self.hidden_size, bias=False
            )
            self.token_conditioning_norm.to(device=device, dtype=dtype)
            self.token_conditioning_A.to(device=device, dtype=dtype)
            self.token_conditioning_B.to(device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.token_conditioning_A.weight, a=5**0.5)
            nn.init.zeros_(self.token_conditioning_B.weight)
        else:
            self.token_conditioning_norm = None
            self.token_conditioning_A = None
            self.token_conditioning_B = None

        if multistep_residual_rank > 0:
            residual_alpha = multistep_residual_alpha or multistep_residual_rank * 2
            self.multistep_residual_scaling = residual_alpha / multistep_residual_rank
            self.multistep_residual_norm: Optional[nn.Module] = nn.LayerNorm(self.hidden_size)
            self.multistep_residual_A: Optional[nn.Linear] = nn.Linear(
                self.hidden_size, multistep_residual_rank, bias=False
            )
            self.multistep_residual_B: Optional[nn.Linear] = nn.Linear(
                multistep_residual_rank, self.hidden_size, bias=False
            )
            self.multistep_residual_norm.to(device=device, dtype=dtype)
            self.multistep_residual_A.to(device=device, dtype=dtype)
            self.multistep_residual_B.to(device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.multistep_residual_A.weight, a=5**0.5)
            nn.init.zeros_(self.multistep_residual_B.weight)
        else:
            self.multistep_residual_scaling = 0.0
            self.multistep_residual_norm = None
            self.multistep_residual_A = None
            self.multistep_residual_B = None

        if multistep_step1_residual_rank > 0:
            step1_alpha = multistep_step1_residual_alpha or multistep_step1_residual_rank * 2
            self.multistep_step1_residual_scaling = step1_alpha / multistep_step1_residual_rank
            self.multistep_step1_residual_norm: Optional[nn.Module] = nn.LayerNorm(self.hidden_size)
            self.multistep_step1_residual_A: Optional[nn.Linear] = nn.Linear(
                self.hidden_size, multistep_step1_residual_rank, bias=False
            )
            self.multistep_step1_residual_B: Optional[nn.Linear] = nn.Linear(
                multistep_step1_residual_rank, self.hidden_size, bias=False
            )
            self.multistep_step1_residual_norm.to(device=device, dtype=dtype)
            self.multistep_step1_residual_A.to(device=device, dtype=dtype)
            self.multistep_step1_residual_B.to(device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.multistep_step1_residual_A.weight, a=5**0.5)
            nn.init.zeros_(self.multistep_step1_residual_B.weight)
        else:
            self.multistep_step1_residual_scaling = 0.0
            self.multistep_step1_residual_norm = None
            self.multistep_step1_residual_A = None
            self.multistep_step1_residual_B = None

        self.verifier_probe = nn.Parameter(torch.empty(self.hidden_size, device=device, dtype=dtype))
        self.verifier_head = nn.Sequential(nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, 1))
        self.verifier_head.to(device=device, dtype=dtype)
        self._reset_verifier_parameters(verifier_probe_init_std)
        self.add_verifier_adapter(verifier_rank, verifier_alpha, verifier_dropout)
        self.set_verifier_trainable(False)

    def _iter_layers(self):
        if self.layers is not None:
            return self.layers

        return (self.layer,)

    def _infer_hidden_size(self) -> int:
        for layer in self._iter_layers():
            for module in layer.modules():
                if isinstance(module, RecurFTLoraLinear):
                    return module.base_layer.in_features

                if hasattr(module, "in_features"):
                    return module.in_features

        raise ValueError("RecurFT cannot infer hidden size for the verifier probe.")

    def _infer_probe_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        for layer in self._iter_layers():
            for param in layer.parameters():
                return param.device, param.dtype

        return torch.device("cpu"), torch.float32

    def _reset_verifier_parameters(self, probe_init_std: float) -> None:
        nn.init.normal_(self.verifier_probe, mean=0.0, std=probe_init_std)
        for module in self.verifier_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def add_verifier_adapter(self, rank: int, alpha: int, dropout: float) -> None:
        for layer in self._iter_layers():
            for module in layer.modules():
                if isinstance(module, RecurFTLoraLinear):
                    module.add_verifier_adapter(rank, alpha, dropout)

    def set_verifier_enabled(self, enabled: bool) -> None:
        for layer in self._iter_layers():
            for module in layer.modules():
                if isinstance(module, RecurFTLoraLinear):
                    module.set_verifier_enabled(enabled)

    def set_verifier_trainable(self, trainable: bool) -> None:
        self.verifier_probe.requires_grad_(trainable)
        for param in self.verifier_head.parameters():
            param.requires_grad_(trainable)

        for layer in self._iter_layers():
            for module in layer.modules():
                if isinstance(module, RecurFTLoraLinear):
                    module.set_verifier_trainable(trainable)

    def merge_lora_for_inference(self) -> int:
        if self.training:
            raise ValueError("RecurFT recurrent LoRA can only be merged in eval mode.")

        merged = 0
        for layer in self._iter_layers():
            for module in layer.modules():
                if isinstance(module, RecurFTLoraLinear):
                    module.merge_lora_()
                    merged += 1

        return merged

    def unmerge_lora_for_inference(self) -> int:
        unmerged = 0
        for layer in self._iter_layers():
            for module in layer.modules():
                if isinstance(module, RecurFTLoraLinear) and module._lora_merged:
                    module.unmerge_lora_()
                    unmerged += 1

        return unmerged

    def recurrent_lora_is_merged(self) -> bool:
        adapters = [
            module
            for layer in self._iter_layers()
            for module in layer.modules()
            if isinstance(module, RecurFTLoraLinear)
        ]
        return bool(adapters) and all(module._lora_merged for module in adapters)

    def _make_position_ids(self, attention_mask: Optional[torch.Tensor], seq_len: int, device: torch.device):
        if attention_mask is None:
            return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        return position_ids.masked_fill(attention_mask == 0, 0)

    def _make_causal_mask(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len <= 1 and attention_mask is None:
            return None

        min_dtype = torch.finfo(hidden_states.dtype).min
        causal_mask = torch.full((seq_len, seq_len), min_dtype, dtype=hidden_states.dtype, device=hidden_states.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len).clone()
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].eq(0)
            causal_mask = causal_mask.masked_fill(padding_mask, min_dtype)

        return causal_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        token_ids: Optional[torch.Tensor] = None,
        rollout_step: int = 1,
    ) -> torch.Tensor:
        output = self._run_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            model=model,
            token_ids=token_ids,
            rollout_step=rollout_step,
            use_cache=False,
        )
        return output[0]

    def has_boundary_head(self) -> bool:
        return self.boundary_A is not None and self.boundary_B is not None and self.boundary_norm is not None

    def has_token_conditioning(self) -> bool:
        return (
            self.token_conditioning_A is not None
            and self.token_conditioning_B is not None
            and self.token_conditioning_norm is not None
        )

    def has_multistep_residual(self) -> bool:
        return (
            self.multistep_residual_norm is not None
            and self.multistep_residual_A is not None
            and self.multistep_residual_B is not None
        )

    def has_multistep_step1_residual(self) -> bool:
        return (
            self.multistep_step1_residual_norm is not None
            and self.multistep_step1_residual_A is not None
            and self.multistep_step1_residual_B is not None
        )

    def _apply_low_rank_residual(
        self,
        hidden_states: torch.Tensor,
        norm: nn.Module,
        down: nn.Linear,
        up: nn.Linear,
        scaling: float,
    ) -> torch.Tensor:
        residual_input = hidden_states.to(norm.weight.dtype)
        residual_hidden = norm(residual_input)
        residual_hidden = down(residual_hidden.to(down.weight.dtype))
        residual = up(torch.nn.functional.silu(residual_hidden).to(up.weight.dtype))
        return hidden_states + residual.to(hidden_states.dtype) * scaling

    def _apply_multistep_residual(self, hidden_states: torch.Tensor, rollout_step: int) -> torch.Tensor:
        if rollout_step == 1 and self.has_multistep_step1_residual():
            return self._apply_low_rank_residual(
                hidden_states,
                self.multistep_step1_residual_norm,
                self.multistep_step1_residual_A,
                self.multistep_step1_residual_B,
                self.multistep_step1_residual_scaling,
            )

        if rollout_step < self.multistep_residual_start_step or not self.has_multistep_residual():
            return hidden_states

        return self._apply_low_rank_residual(
            hidden_states,
            self.multistep_residual_norm,
            self.multistep_residual_A,
            self.multistep_residual_B,
            self.multistep_residual_scaling,
        )

    def _resolve_causal_lm(self, model: nn.Module) -> nn.Module:
        if self._cached_causal_lm_source_id == id(model):
            cached = self._cached_causal_lm
            if cached is not None:
                return cached

        current = model
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "get_input_embeddings") and hasattr(current, "get_output_embeddings"):
                self._cached_causal_lm_source_id = id(model)
                object.__setattr__(self, "_cached_causal_lm", current)
                return current

            base_model = getattr(current, "base_model", None)
            current = base_model if base_model is not None and base_model is not current else getattr(current, "model", None)

        raise ValueError("RecurFT cannot resolve the causal LM for boundary/token conditioning.")

    def boundary_logits(self, hidden_states: torch.Tensor, model: nn.Module) -> torch.Tensor:
        if not self.has_boundary_head():
            raise ValueError("RecurFT boundary head is not enabled for this checkpoint.")

        boundary_input = hidden_states.to(self.boundary_norm.weight.dtype)
        boundary_hidden = self.boundary_norm(boundary_input)
        boundary_hidden = self.boundary_A(boundary_hidden.to(self.boundary_A.weight.dtype))
        residual = self.boundary_B(boundary_hidden.to(self.boundary_B.weight.dtype))
        adapted_hidden = hidden_states + residual.to(hidden_states.dtype)
        causal_lm = self._resolve_causal_lm(model)
        decoder = getattr(causal_lm, "model", causal_lm)
        final_norm = getattr(decoder, "norm", None) or getattr(decoder, "ln_f", None)
        if final_norm is not None:
            final_norm_weight = getattr(final_norm, "weight", None)
            if final_norm_weight is not None:
                adapted_hidden = adapted_hidden.to(final_norm_weight.dtype)

            adapted_hidden = final_norm(adapted_hidden)

        lm_head = causal_lm.get_output_embeddings()
        adapted_hidden = adapted_hidden.to(lm_head.weight.dtype)
        return lm_head(adapted_hidden)

    def _apply_token_conditioning(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor],
        model: Optional[nn.Module],
    ) -> torch.Tensor:
        if not self.has_token_conditioning():
            return hidden_states
        if token_ids is None or model is None:
            raise ValueError("Token-conditioned RecurFT requires both `token_ids` and `model`.")
        if token_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"RecurFT token_ids shape {tuple(token_ids.shape)} does not match hidden states "
                f"{tuple(hidden_states.shape[:2])}."
            )

        causal_lm = self._resolve_causal_lm(model)
        token_embeddings = causal_lm.get_input_embeddings()(token_ids).detach()
        token_embeddings = token_embeddings.to(self.token_conditioning_norm.weight.dtype)
        token_hidden = self.token_conditioning_norm(token_embeddings)
        token_hidden = self.token_conditioning_A(token_hidden.to(self.token_conditioning_A.weight.dtype))
        condition = self.token_conditioning_B(token_hidden.to(self.token_conditioning_B.weight.dtype))
        return hidden_states + condition.to(hidden_states.dtype)

    def probe(
        self,
        batch_size: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        past_key_value: Optional[Any] = None,
        cache_position: Optional[torch.Tensor] = None,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        r"""Run a learned hidden-space verifier probe through T with verifier LoRA enabled.

        `past_key_value` and `cache_position` are accepted for future cached T rollout
        inference. The probe pass is meant to be run on a temporary cache fork and
        should not append the probe token to the main recurrent rollout cache.
        """
        probe_hidden = self.verifier_probe.view(1, 1, -1).expand(batch_size, 1, -1)
        self.set_verifier_enabled(True)
        try:
            output_hidden, present_key_value = self._run_layer(
                hidden_states=probe_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                model=model,
                past_key_value=past_key_value,
                use_cache=return_cache,
                cache_position=cache_position,
            )
        finally:
            self.set_verifier_enabled(False)

        score = self.verifier_head(output_hidden[:, -1, :]).squeeze(-1)
        if return_cache:
            return score, present_key_value

        return score

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        past_key_value: Optional[Any] = None,
        cache_position: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        rollout_step: int = 1,
    ) -> tuple[torch.Tensor, Any]:
        r"""Run T while updating a cache.

        This is used by latent speculative decoding. The cache stores the
        recurrent context, while the returned final token predicts the next
        hidden state.
        """
        return self._run_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            model=model,
            past_key_value=past_key_value,
            use_cache=True,
            cache_position=cache_position,
            token_ids=token_ids,
            rollout_step=rollout_step,
        )

    def _cache_seq_length(self, past_key_value: Optional[Any]) -> int:
        if past_key_value is None or not hasattr(past_key_value, "get_seq_length"):
            return 0

        layer_ids = self.metadata.get("loop_layer_ids") or self.metadata.get("tail_layer_ids") or []
        lengths = []
        for layer_idx in layer_ids:
            try:
                lengths.append(int(past_key_value.get_seq_length(layer_idx)))
            except (IndexError, TypeError):
                pass

        if lengths:
            return max(lengths)

        try:
            return int(past_key_value.get_seq_length())
        except (IndexError, TypeError):
            return 0

    def _make_cached_causal_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_length: int,
    ) -> Optional[torch.Tensor]:
        batch_size, query_length, _ = hidden_states.shape
        key_length = past_length + query_length
        if query_length == 1 and attention_mask is None:
            return None

        min_dtype = torch.finfo(hidden_states.dtype).min
        query_positions = torch.arange(
            past_length, past_length + query_length, device=hidden_states.device
        ).view(query_length, 1)
        key_positions = torch.arange(key_length, device=hidden_states.device).view(1, key_length)
        allowed = key_positions <= query_positions
        causal_mask = torch.zeros(
            (query_length, key_length), dtype=hidden_states.dtype, device=hidden_states.device
        ).masked_fill(~allowed, min_dtype)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, query_length, key_length).clone()
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :key_length].eq(0)
            causal_mask = causal_mask.masked_fill(padding_mask, min_dtype)

        return causal_mask

    def _run_layer(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        past_key_value: Optional[Any] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        rollout_step: int = 1,
    ) -> tuple[torch.Tensor, Any]:
        hidden_states = self._apply_token_conditioning(hidden_states, token_ids, model)
        seq_len = hidden_states.size(1)
        past_length = self._cache_seq_length(past_key_value)
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + seq_len, device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0)

        if position_ids.size(0) == 1 and hidden_states.size(0) > 1:
            position_ids = position_ids.expand(hidden_states.size(0), -1)

        causal_mask = self._make_cached_causal_mask(hidden_states, attention_mask, past_length)
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + seq_len, device=hidden_states.device, dtype=torch.long
            )

        present_key_value = past_key_value
        for layer in self._iter_layers():
            kwargs = {
                "attention_mask": causal_mask,
                "position_ids": position_ids,
                "output_attentions": False,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
            parameter_names = self._layer_forward_parameter_names[id(layer)]
            if "past_key_values" in parameter_names:
                kwargs["past_key_values"] = past_key_value
            elif "past_key_value" in parameter_names:
                kwargs["past_key_value"] = past_key_value

            position_embeddings = self._maybe_make_position_embeddings(model, hidden_states, position_ids)
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings

            kwargs = {key: value for key, value in kwargs.items() if key in parameter_names}
            try:
                output = layer(hidden_states, **kwargs)
            except TypeError:
                fallback_kwargs = {key: value for key, value in kwargs.items() if key != "position_embeddings"}
                output = layer(hidden_states, **fallback_kwargs)

            if isinstance(output, tuple):
                hidden_states = output[0]
                present_key_value = output[-1] if use_cache and len(output) > 1 else present_key_value
            else:
                hidden_states = output

        hidden_states = self._apply_multistep_residual(hidden_states, rollout_step)

        return hidden_states, present_key_value

    def _maybe_make_position_embeddings(
        self, model: Optional[nn.Module], hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if model is None:
            return None

        if self._cached_rotary_source_id == id(model):
            rotary_emb = self._cached_rotary_emb
            if rotary_emb is None:
                return None
            try:
                return rotary_emb(hidden_states, position_ids)
            except TypeError:
                return None

        rotary_emb = None
        current = model
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            rotary_emb = getattr(current, "rotary_emb", None)
            if rotary_emb is not None:
                break

            current = getattr(current, "model", None) or getattr(current, "base_model", None)

        if rotary_emb is None:
            self._cached_rotary_source_id = id(model)
            object.__setattr__(self, "_cached_rotary_emb", None)
            return None

        self._cached_rotary_source_id = id(model)
        object.__setattr__(self, "_cached_rotary_emb", rotary_emb)

        try:
            return rotary_emb(hidden_states, position_ids)
        except TypeError:
            return None


def build_recurft_recurrent_module(
    model: "PreTrainedModel", finetuning_args: "FinetuningArguments", metadata: dict[str, Any]
) -> RecurFTRecurrentModule:
    layers, _ = find_decoder_layers(model)
    t_rank = finetuning_args.recurft_t_lora_rank
    t_alpha = finetuning_args.recurft_t_lora_alpha or t_rank * 2
    verifier_rank = finetuning_args.recurft_verifier_lora_rank
    verifier_alpha = finetuning_args.recurft_verifier_lora_alpha or verifier_rank * 2
    loop_layer_ids = metadata.get("loop_layer_ids") or metadata.get("tail_layer_ids") or [metadata["last_layer"]]
    module = RecurFTRecurrentModule(
        layers=[layers[layer_idx] for layer_idx in loop_layer_ids],
        target_modules=finetuning_args.lora_target,
        rank=t_rank,
        alpha=t_alpha,
        dropout=finetuning_args.recurft_t_lora_dropout,
        metadata=metadata,
        verifier_rank=verifier_rank,
        verifier_alpha=verifier_alpha,
        verifier_dropout=finetuning_args.recurft_verifier_lora_dropout,
        verifier_probe_init_std=finetuning_args.recurft_verifier_probe_init_std,
        boundary_head_rank=finetuning_args.recurft_boundary_head_rank,
        token_conditioning_rank=finetuning_args.recurft_token_conditioning_rank,
        multistep_residual_rank=finetuning_args.recurft_multistep_residual_rank,
        multistep_residual_alpha=(
            finetuning_args.recurft_multistep_residual_alpha
            or finetuning_args.recurft_multistep_residual_rank * 2
        ),
        multistep_residual_start_step=finetuning_args.recurft_multistep_residual_start_step,
        multistep_step1_residual_rank=finetuning_args.recurft_multistep_step1_residual_rank,
        multistep_step1_residual_alpha=(
            finetuning_args.recurft_multistep_step1_residual_alpha
            or finetuning_args.recurft_multistep_step1_residual_rank * 2
        ),
    )
    logger.info_rank0(
        "RecurFT creates recurrent T from loop layers {} with rank {}.".format(loop_layer_ids, t_rank)
    )
    return module


def attach_recurft_recurrent_module(model: nn.Module, recurrent_module: RecurFTRecurrentModule) -> None:
    setattr(model, "recurft_recurrent", recurrent_module)
    setattr(model, "recurft_metadata", recurrent_module.metadata)


@contextmanager
def disable_adapter(model: nn.Module):
    # PEFT enable_adapter_layers() may re-enable gradients on frozen adapters.
    # Reference forwards must leave the caller's trainable scope unchanged.
    flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        with model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext():
            yield
    finally:
        for param, requires_grad in flags:
            if param.requires_grad != requires_grad:
                param.requires_grad_(requires_grad)


def _unwrap_module(model: nn.Module) -> nn.Module:
    return _unwrap_module(model.module) if hasattr(model, "module") else model


def save_recurft_recurrent_module(model: nn.Module, output_dir: str, safe_serialization: bool = True) -> None:
    model = _unwrap_module(model)
    recurrent_module = getattr(model, "recurft_recurrent", None)
    metadata = getattr(model, "recurft_metadata", None)
    if recurrent_module is None or metadata is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, RECURFT_CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    state_dict = {
        name: param
        for name, param in recurrent_module.state_dict().items()
        if "lora_" in name
        or name.startswith("verifier_")
        or name.startswith("boundary_")
        or name.startswith("token_conditioning_")
        or name.startswith("multistep_residual_")
        or name.startswith("multistep_step1_residual_")
    }
    if safe_serialization and is_safetensors_available():
        save_file(state_dict, os.path.join(output_dir, RECURFT_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(state_dict, os.path.join(output_dir, RECURFT_WEIGHTS_NAME))

    logger.info_rank0(f"RecurFT recurrent module saved at: {output_dir}")


def load_recurft_recurrent_module(model: nn.Module, input_dir: str) -> None:
    model = _unwrap_module(model)
    recurrent_module = getattr(model, "recurft_recurrent", None)
    if recurrent_module is None:
        return

    safe_path = os.path.join(input_dir, RECURFT_SAFE_WEIGHTS_NAME)
    bin_path = os.path.join(input_dir, RECURFT_WEIGHTS_NAME)
    if os.path.exists(safe_path) and is_safetensors_available():
        state_dict = load_file(safe_path, device="cpu")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        logger.warning_rank0(f"RecurFT recurrent checkpoint is not found at: {input_dir}")
        return

    recurrent_module.load_state_dict(state_dict, strict=False)
    logger.info_rank0(f"RecurFT recurrent module loaded from: {input_dir}")
