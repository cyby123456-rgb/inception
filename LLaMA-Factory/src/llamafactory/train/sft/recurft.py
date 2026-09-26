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

import inspect
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F

from ...extras.constants import IGNORE_INDEX
from ...model.model_utils.recurft import disable_adapter, find_decoder_layers


if TYPE_CHECKING:
    from torch import nn

    from ...hparams import FinetuningArguments


def _unwrap_model(model: "nn.Module") -> "nn.Module":
    return _unwrap_model(model.module) if hasattr(model, "module") else model


def _forward_with_hidden_states(model: "nn.Module", inputs: dict[str, torch.Tensor], include_loss: bool = False):
    model_inputs = dict(inputs)
    if not include_loss:
        model_inputs.pop("labels", None)

    model_inputs["output_hidden_states"] = True
    model_inputs["return_dict"] = True
    model_inputs["use_cache"] = False
    return model(**model_inputs)


def _forward_recurft_reference(
    model: "nn.Module",
    inputs: dict[str, torch.Tensor],
    ref_model: Optional["nn.Module"] = None,
):
    """Run the frozen checkpoint teacher when supplied, otherwise preserve legacy base-model behavior."""
    if ref_model is not None:
        with torch.no_grad():
            return _forward_with_hidden_states(ref_model, inputs, include_loss=False)

    unwrapped_model = _unwrap_model(model)
    with torch.no_grad(), disable_adapter(unwrapped_model):
        return _forward_with_hidden_states(model, inputs, include_loss=False)


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(loss.device, dtype=loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_hidden_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_clip: float = 0.0
) -> torch.Tensor:
    token_loss = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)
    if loss_clip > 0.0:
        token_loss = token_loss.clamp_max(loss_clip)

    return _masked_mean(token_loss, mask)


def _masked_hidden_smooth_l1(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float
) -> torch.Tensor:
    token_loss = F.smooth_l1_loss(pred.float(), target.float(), reduction="none", beta=beta).mean(dim=-1)
    return _masked_mean(token_loss, mask)


def _masked_hidden_relative_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    token_loss = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)
    token_scale = target.float().pow(2).mean(dim=-1).clamp_min(eps)
    return _masked_mean(token_loss / token_scale, mask)


def _masked_hidden_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_cosine = F.cosine_similarity(pred.float(), target.float(), dim=-1)
    return _masked_mean(token_cosine, mask)


def _masked_hidden_cosine_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return 1.0 - _masked_hidden_cosine(pred, target, mask)


def _masked_token_ce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_loss = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view(labels.shape)
    return _masked_mean(token_loss, mask)


def _masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_correct = logits.detach().float().argmax(dim=-1).eq(labels)
    return _masked_mean(token_correct.to(logits.dtype), mask)


def _masked_top1_match(logits: torch.Tensor, ref_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_correct = logits.detach().float().argmax(dim=-1).eq(ref_logits.detach().float().argmax(dim=-1))
    return _masked_mean(token_correct.to(logits.dtype), mask)


def _hidden_contrastive_logits(
    pred: torch.Tensor,
    candidates: torch.Tensor,
    candidate_mask: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    pred = F.normalize(pred.float().squeeze(1), dim=-1)
    candidates = F.normalize(candidates.float(), dim=-1)
    logits = torch.einsum("bh,bnh->bn", pred, candidates) / temperature
    if candidate_mask is not None:
        min_value = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~candidate_mask.to(device=logits.device, dtype=torch.bool), min_value)

    return logits


def _masked_hidden_contrastive_loss(
    pred: torch.Tensor,
    candidates: torch.Tensor,
    correct_index: int,
    mask: torch.Tensor,
    candidate_mask: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    logits = _hidden_contrastive_logits(pred, candidates, candidate_mask, temperature)
    labels = torch.full((logits.size(0),), correct_index, dtype=torch.long, device=logits.device)
    token_loss = F.cross_entropy(logits, labels, reduction="none")
    return _masked_mean(token_loss, mask.squeeze(1))


def _masked_hidden_contrastive_accuracy(
    pred: torch.Tensor,
    candidates: torch.Tensor,
    correct_index: int,
    mask: torch.Tensor,
    candidate_mask: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    logits = _hidden_contrastive_logits(pred.detach(), candidates.detach(), candidate_mask, temperature)
    correct = logits.argmax(dim=-1).eq(correct_index)
    return _masked_mean(correct.to(logits.dtype), mask.squeeze(1))


def _masked_hidden_delta_mse(
    pred: torch.Tensor,
    pred_prev: torch.Tensor,
    target: torch.Tensor,
    target_prev: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_delta = pred.float() - pred_prev.float()
    target_delta = target.float() - target_prev.float()
    token_loss = F.mse_loss(pred_delta, target_delta, reduction="none").mean(dim=-1)
    return _masked_mean(token_loss, mask)


def _masked_hidden_delta_cosine(
    pred: torch.Tensor,
    pred_prev: torch.Tensor,
    target: torch.Tensor,
    target_prev: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_delta = pred.float() - pred_prev.float()
    target_delta = target.float() - target_prev.float()
    token_cosine = F.cosine_similarity(pred_delta, target_delta, dim=-1)
    return _masked_mean(token_cosine, mask)


def _make_token_mask(
    inputs: dict[str, torch.Tensor], finetuning_args: "FinetuningArguments", shifted: bool = False
) -> torch.Tensor:
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()

    if shifted:
        attention_mask = attention_mask[:, 1:]

    if finetuning_args.recurft_loss_on_labels_only and "labels" in inputs:
        label_mask = inputs["labels"].ne(IGNORE_INDEX)
        label_mask = label_mask[:, 1:] if shifted else label_mask
        attention_mask = attention_mask & label_mask

    return attention_mask


def _compute_kl_loss(
    logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = logits.float() / temperature
    ref_logits = ref_logits.float() / temperature
    log_probs = F.log_softmax(logits, dim=-1)
    ref_probs = F.softmax(ref_logits, dim=-1)
    token_loss = F.kl_div(log_probs, ref_probs, reduction="none").sum(dim=-1) * (temperature**2)
    return _masked_mean(token_loss, mask)


def _compute_tv_loss(
    logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    probs = F.softmax(logits.float() / temperature, dim=-1)
    ref_probs = F.softmax(ref_logits.float() / temperature, dim=-1)
    token_loss = 0.5 * (probs - ref_probs).abs().sum(dim=-1)
    return _masked_mean(token_loss, mask)


def _get_base_causal_lm(model: "nn.Module") -> "nn.Module":
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model

    return model


def _get_decoder(model: "nn.Module") -> "nn.Module":
    base = _get_base_causal_lm(model)
    return getattr(base, "model", base)


def _make_projection_causal_mask(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    min_dtype = torch.finfo(hidden_states.dtype).min
    mask = torch.full((seq_len, seq_len), min_dtype, dtype=hidden_states.dtype, device=hidden_states.device)
    mask = torch.triu(mask, diagonal=1)
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len).clone()
    if attention_mask is not None:
        mask = mask.masked_fill(attention_mask[:, None, None, :].eq(0), min_dtype)

    return mask


def _maybe_make_position_embeddings(
    model: "nn.Module", hidden_states: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    decoder = _get_decoder(model)
    rotary_emb = getattr(decoder, "rotary_emb", None)
    if rotary_emb is None:
        return None

    try:
        return rotary_emb(hidden_states, position_ids)
    except TypeError:
        return None


def _call_projection_layer(
    layer: "nn.Module",
    hidden_states: torch.Tensor,
    model: "nn.Module",
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    for param in layer.parameters():
        if hidden_states.dtype != param.dtype:
            hidden_states = hidden_states.to(param.dtype)
        break

    kwargs: dict[str, Any] = {
        "attention_mask": _make_projection_causal_mask(hidden_states, attention_mask),
        "position_ids": position_ids,
        "output_attentions": False,
        "use_cache": False,
        "cache_position": torch.arange(hidden_states.size(1), device=hidden_states.device, dtype=torch.long),
    }
    position_embeddings = _maybe_make_position_embeddings(model, hidden_states, position_ids)
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings

    signature = inspect.signature(layer.forward)
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    try:
        output = layer(hidden_states, **kwargs)
    except TypeError:
        kwargs.pop("position_embeddings", None)
        output = layer(hidden_states, **kwargs)

    return output[0] if isinstance(output, tuple) else output


def _project_anchor_logits(
    model: "nn.Module",
    metadata: dict[str, Any],
    anchor_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    base = _get_base_causal_lm(model)
    layers, _ = find_decoder_layers(base)
    hidden_states = anchor_hidden
    projection_layer_ids = metadata.get("projection_layer_ids") or metadata.get("tail_layer_ids") or [
        metadata["last_layer"]
    ]
    for layer_idx in projection_layer_ids:
        hidden_states = _call_projection_layer(layers[layer_idx], hidden_states, base, attention_mask, position_ids)

    decoder = _get_decoder(base)
    norm = getattr(decoder, "norm", None) or getattr(decoder, "ln_f", None)
    if norm is not None:
        hidden_states = norm(hidden_states)

    lm_head = base.get_output_embeddings()
    return lm_head(hidden_states)


def _full_token_mask(inputs: dict[str, torch.Tensor], finetuning_args: "FinetuningArguments") -> torch.Tensor:
    return _make_token_mask(inputs, finetuning_args, shifted=False)


def _get_recurrent_weight(finetuning_args: "FinetuningArguments", global_step: int) -> float:
    recurrent_weight = finetuning_args.recurft_recurrent_loss_weight
    warmup_steps = finetuning_args.recurft_recurrent_warmup_steps
    if warmup_steps > 0:
        recurrent_weight *= min(1.0, float(global_step + 1) / float(warmup_steps))

    return recurrent_weight


def _get_multistep_weight(finetuning_args: "FinetuningArguments", global_step: int) -> float:
    multistep_weight = finetuning_args.recurft_multistep_loss_weight
    warmup_steps = finetuning_args.recurft_multistep_warmup_steps
    if warmup_steps > 0:
        progress = max(0, global_step + 1 - finetuning_args.recurft_multistep_warmup_start_step)
        multistep_weight *= min(1.0, float(progress) / float(warmup_steps))

    return multistep_weight


def _get_scheduled_sampling_ratio(finetuning_args: "FinetuningArguments", global_step: int) -> float:
    max_ratio = finetuning_args.recurft_scheduled_sampling_max_ratio
    warmup_steps = finetuning_args.recurft_scheduled_sampling_warmup_steps
    if warmup_steps <= 0:
        return max_ratio

    progress = max(0, global_step + 1 - finetuning_args.recurft_scheduled_sampling_start_step)
    return max_ratio * min(1.0, float(progress) / float(warmup_steps))


def _metric_value(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())

    return float(value)


def _select_multistep_starts(
    seq_len: int,
    steps: int,
    stride: int,
    max_starts: int,
    device: torch.device,
    selection: str = "linspace",
    offset: int = 0,
) -> list[int]:
    if seq_len <= steps:
        return []

    starts = list(range(0, seq_len - steps, max(1, stride)))
    if max_starts > 0 and len(starts) > max_starts:
        if selection == "cyclic":
            candidate_count = len(starts)
            offset %= candidate_count
            indices = [
                (offset + index * candidate_count // max_starts) % candidate_count
                for index in range(max_starts)
            ]
            starts = sorted(starts[index] for index in indices)
        else:
            indices = torch.linspace(0, len(starts) - 1, steps=max_starts, device=device).round().long().unique()
            starts = [starts[int(index.item())] for index in indices]

    return starts


def _compute_multistep_recurrent_loss(
    recurrent_module: "nn.Module",
    anchor_hidden: torch.Tensor,
    teacher_logits: torch.Tensor,
    inputs: dict[str, torch.Tensor],
    finetuning_args: "FinetuningArguments",
    model: "nn.Module",
    metadata: dict[str, Any],
    global_step: int,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    steps = finetuning_args.recurft_multistep_steps
    if steps <= 1 or anchor_hidden.size(1) <= steps:
        return None, {}

    starts = _select_multistep_starts(
        seq_len=anchor_hidden.size(1),
        steps=steps,
        stride=finetuning_args.recurft_multistep_stride,
        max_starts=finetuning_args.recurft_multistep_max_starts,
        device=anchor_hidden.device,
        selection=finetuning_args.recurft_multistep_start_selection,
        offset=global_step,
    )
    if not starts:
        return None, {}

    input_attention_mask = inputs.get("attention_mask")
    if input_attention_mask is None:
        input_attention_mask = torch.ones(anchor_hidden.shape[:2], dtype=torch.long, device=anchor_hidden.device)
    else:
        input_attention_mask = input_attention_mask.to(anchor_hidden.device)

    input_position_ids = inputs.get("position_ids")
    if input_position_ids is None:
        input_position_ids = torch.arange(anchor_hidden.size(1), device=anchor_hidden.device, dtype=torch.long)
        input_position_ids = input_position_ids.unsqueeze(0).expand(anchor_hidden.size(0), -1)
    else:
        input_position_ids = input_position_ids.to(anchor_hidden.device)

    target_mask = _full_token_mask(inputs, finetuning_args).to(anchor_hidden.device)
    context_tokens = finetuning_args.recurft_multistep_context_tokens
    detach_rollout = finetuning_args.recurft_multistep_detach_rollout
    step_decay = finetuning_args.recurft_multistep_step_decay

    weighted_losses = []
    weighted_base_losses = []
    weighted_mse_losses = []
    weighted_relative_mse_losses = []
    weighted_cosine_values = []
    weighted_delta_mse_losses = []
    weighted_delta_cosine_values = []
    weighted_contrastive_losses = []
    weighted_contrastive_acc_values = []
    weighted_token_ce_losses = []
    weighted_logit_kl_losses = []
    weighted_logit_tv_losses = []
    weighted_token_acc_values = []
    weighted_teacher_top1_values = []
    weighted_boundary_rollout_losses = []
    weighted_boundary_token_ce_losses = []
    weighted_boundary_logit_kl_losses = []
    weighted_boundary_token_acc_values = []
    weighted_boundary_teacher_top1_values = []
    boundary_rollout_weights = []
    step1_logit_losses = []
    weights = []
    per_step_losses: dict[int, list[torch.Tensor]] = {step: [] for step in range(2, steps + 1)}
    per_step_relative_mse: dict[int, list[torch.Tensor]] = {step: [] for step in range(2, steps + 1)}
    per_step_cosine: dict[int, list[torch.Tensor]] = {step: [] for step in range(2, steps + 1)}
    per_step_delta_mse: dict[int, list[torch.Tensor]] = {step: [] for step in range(2, steps + 1)}
    per_step_delta_cosine: dict[int, list[torch.Tensor]] = {step: [] for step in range(2, steps + 1)}
    per_step_contrastive_loss: dict[int, list[torch.Tensor]] = {step: [] for step in range(2, steps + 1)}
    per_step_contrastive_acc: dict[int, list[torch.Tensor]] = {step: [] for step in range(2, steps + 1)}
    per_step_token_ce: dict[int, list[torch.Tensor]] = {step: [] for step in range(1, steps + 1)}
    per_step_logit_kl: dict[int, list[torch.Tensor]] = {step: [] for step in range(1, steps + 1)}
    per_step_logit_tv: dict[int, list[torch.Tensor]] = {step: [] for step in range(1, steps + 1)}
    per_step_token_acc: dict[int, list[torch.Tensor]] = {step: [] for step in range(1, steps + 1)}
    per_step_teacher_top1: dict[int, list[torch.Tensor]] = {step: [] for step in range(1, steps + 1)}
    per_step_boundary_token_ce: dict[int, list[torch.Tensor]] = {step: [] for step in range(1, steps + 1)}
    per_step_boundary_logit_kl: dict[int, list[torch.Tensor]] = {step: [] for step in range(1, steps + 1)}
    per_step_boundary_token_acc: dict[int, list[torch.Tensor]] = {step: [] for step in range(1, steps + 1)}
    per_step_boundary_teacher_top1: dict[int, list[torch.Tensor]] = {step: [] for step in range(1, steps + 1)}
    contrastive_loss_enabled = finetuning_args.recurft_multistep_contrastive_loss_weight > 0.0
    contrastive_window = finetuning_args.recurft_multistep_contrastive_window or steps
    contrastive_window = max(steps, contrastive_window)
    logit_tv_loss_weight = getattr(finetuning_args, "recurft_multistep_logit_tv_loss_weight", 0.0)
    logit_loss_enabled = (
        finetuning_args.recurft_multistep_token_ce_loss_weight > 0.0
        or finetuning_args.recurft_multistep_logit_kl_loss_weight > 0.0
        or logit_tv_loss_weight > 0.0
    )
    boundary_rollout_loss_enabled = getattr(recurrent_module, "has_boundary_head", lambda: False)() and (
        finetuning_args.recurft_multistep_boundary_token_ce_loss_weight > 0.0
        or finetuning_args.recurft_multistep_boundary_logit_kl_loss_weight > 0.0
    )
    scheduled_sampling_ratio = _get_scheduled_sampling_ratio(finetuning_args, global_step)

    for start in starts:
        context_start = 0 if context_tokens == 0 else max(0, start - context_tokens + 1)
        rollout_context = anchor_hidden[:, context_start : start + 1, :]
        rollout_attention = input_attention_mask[:, context_start : start + 1]
        rollout_position_ids = input_position_ids[:, context_start : start + 1]
        rollout_token_ids = inputs["input_ids"][:, context_start : start + 1]

        for step in range(1, steps + 1):
            pred_seq = recurrent_module(
                rollout_context,
                attention_mask=rollout_attention,
                position_ids=rollout_position_ids,
                model=model,
                token_ids=rollout_token_ids,
                rollout_step=step,
            )
            next_hidden = pred_seq[:, -1:, :]
            target_index = start + step
            target_hidden = anchor_hidden[:, target_index : target_index + 1, :].detach()
            teacher_prev_hidden = anchor_hidden[:, target_index - 1 : target_index, :].detach()
            loss_mask = target_mask[:, target_index : target_index + 1]

            boundary_label_index = target_index + 1
            if boundary_rollout_loss_enabled and boundary_label_index < anchor_hidden.size(1):
                boundary_rollout_logits = recurrent_module.boundary_logits(next_hidden, model)
                boundary_rollout_labels = inputs["input_ids"][:, boundary_label_index : boundary_label_index + 1]
                boundary_rollout_mask = target_mask[:, boundary_label_index : boundary_label_index + 1]
                boundary_rollout_teacher_logits = teacher_logits[
                    :, target_index : target_index + 1, :
                ].detach()
                if getattr(finetuning_args, "recurft_multistep_boundary_teacher_top1_labels", False):
                    boundary_rollout_labels = boundary_rollout_teacher_logits.argmax(dim=-1)
                boundary_rollout_token_ce = _masked_token_ce_loss(
                    boundary_rollout_logits, boundary_rollout_labels, boundary_rollout_mask
                )
                boundary_rollout_logit_kl = _compute_kl_loss(
                    logits=boundary_rollout_logits,
                    ref_logits=boundary_rollout_teacher_logits,
                    mask=boundary_rollout_mask,
                    temperature=finetuning_args.recurft_multistep_logit_temperature,
                )
                boundary_rollout_token_acc = _masked_token_accuracy(
                    boundary_rollout_logits, boundary_rollout_labels, boundary_rollout_mask
                )
                boundary_rollout_teacher_top1 = _masked_top1_match(
                    boundary_rollout_logits, boundary_rollout_teacher_logits, boundary_rollout_mask
                )
                boundary_rollout_weight = step_decay ** (step - 1)
                boundary_rollout_loss = (
                    boundary_rollout_token_ce
                    * finetuning_args.recurft_multistep_boundary_token_ce_loss_weight
                    + boundary_rollout_logit_kl
                    * finetuning_args.recurft_multistep_boundary_logit_kl_loss_weight
                )
                weighted_boundary_rollout_losses.append(boundary_rollout_loss * boundary_rollout_weight)
                weighted_boundary_token_ce_losses.append(
                    boundary_rollout_token_ce.detach() * boundary_rollout_weight
                )
                weighted_boundary_logit_kl_losses.append(
                    boundary_rollout_logit_kl.detach() * boundary_rollout_weight
                )
                weighted_boundary_token_acc_values.append(
                    boundary_rollout_token_acc.detach() * boundary_rollout_weight
                )
                weighted_boundary_teacher_top1_values.append(
                    boundary_rollout_teacher_top1.detach() * boundary_rollout_weight
                )
                boundary_rollout_weights.append(boundary_rollout_weight)
                per_step_boundary_token_ce[step].append(boundary_rollout_token_ce.detach())
                per_step_boundary_logit_kl[step].append(boundary_rollout_logit_kl.detach())
                per_step_boundary_token_acc[step].append(boundary_rollout_token_acc.detach())
                per_step_boundary_teacher_top1[step].append(boundary_rollout_teacher_top1.detach())

            if (
                step == 1
                and logit_loss_enabled
                and finetuning_args.recurft_multistep_step1_logit_loss_weight > 0.0
                and target_index + 1 < anchor_hidden.size(1)
            ):
                projection_context = torch.cat([rollout_context, next_hidden], dim=1)
                projection_attention = torch.cat(
                    [rollout_attention, input_attention_mask[:, target_index : target_index + 1]], dim=1
                )
                projection_position_ids = torch.cat(
                    [rollout_position_ids, input_position_ids[:, target_index : target_index + 1]], dim=1
                )
                logit_context_tokens = finetuning_args.recurft_multistep_logit_context_tokens
                if logit_context_tokens > 0 and projection_context.size(1) > logit_context_tokens:
                    projection_context = projection_context[:, -logit_context_tokens:, :]
                    projection_attention = projection_attention[:, -logit_context_tokens:]
                    projection_position_ids = projection_position_ids[:, -logit_context_tokens:]

                pred_logits = _project_anchor_logits(
                    model=model,
                    metadata=metadata,
                    anchor_hidden=projection_context,
                    attention_mask=projection_attention,
                    position_ids=projection_position_ids,
                )[:, -1:, :]
                label_index = target_index + 1
                labels = inputs["input_ids"][:, label_index : label_index + 1]
                logit_mask = target_mask[:, label_index : label_index + 1]
                step1_token_ce_loss = _masked_token_ce_loss(pred_logits, labels, logit_mask)
                step1_teacher_logits = teacher_logits[:, target_index : target_index + 1, :].detach()
                step1_logit_kl_loss = _compute_kl_loss(
                    logits=pred_logits,
                    ref_logits=step1_teacher_logits,
                    mask=logit_mask,
                    temperature=finetuning_args.recurft_multistep_logit_temperature,
                )
                step1_logit_tv_loss = (
                    _compute_tv_loss(
                        logits=pred_logits,
                        ref_logits=step1_teacher_logits,
                        mask=logit_mask,
                        temperature=finetuning_args.recurft_multistep_logit_temperature,
                    )
                    if logit_tv_loss_weight > 0.0
                    else pred_logits.new_zeros(())
                )
                step1_token_acc = _masked_token_accuracy(pred_logits, labels, logit_mask)
                step1_teacher_top1 = _masked_top1_match(pred_logits, step1_teacher_logits, logit_mask)
                step1_logit_losses.append(
                    step1_token_ce_loss * finetuning_args.recurft_multistep_token_ce_loss_weight
                    + step1_logit_kl_loss * finetuning_args.recurft_multistep_logit_kl_loss_weight
                    + step1_logit_tv_loss * logit_tv_loss_weight
                )
                per_step_token_ce[1].append(step1_token_ce_loss.detach())
                per_step_logit_kl[1].append(step1_logit_kl_loss.detach())
                if logit_tv_loss_weight > 0.0:
                    per_step_logit_tv[1].append(step1_logit_tv_loss.detach())
                per_step_token_acc[1].append(step1_token_acc.detach())
                per_step_teacher_top1[1].append(step1_teacher_top1.detach())

            if step >= 2:
                step_mse_loss = _masked_hidden_mse(next_hidden, target_hidden, loss_mask)
                if finetuning_args.recurft_multistep_huber_beta > 0.0:
                    step_base_loss = _masked_hidden_smooth_l1(
                        next_hidden, target_hidden, loss_mask, finetuning_args.recurft_multistep_huber_beta
                    )
                else:
                    step_base_loss = _masked_hidden_mse(
                        next_hidden,
                        target_hidden,
                        loss_mask,
                        loss_clip=finetuning_args.recurft_multistep_mse_loss_clip,
                    )
                step_relative_mse_loss = _masked_hidden_relative_mse(next_hidden, target_hidden, loss_mask)
                step_cosine = _masked_hidden_cosine(next_hidden, target_hidden, loss_mask)
                step_delta_mse_loss = _masked_hidden_delta_mse(
                    next_hidden, rollout_context[:, -1:, :].detach(), target_hidden, teacher_prev_hidden, loss_mask
                )
                step_delta_cosine = _masked_hidden_delta_cosine(
                    next_hidden, rollout_context[:, -1:, :].detach(), target_hidden, teacher_prev_hidden, loss_mask
                )
                step_loss = step_base_loss
                step_loss = (
                    step_loss
                    + step_relative_mse_loss * finetuning_args.recurft_multistep_relative_mse_loss_weight
                )
                step_loss = step_loss + (1.0 - step_cosine) * finetuning_args.recurft_multistep_cosine_loss_weight
                step_loss = step_loss + step_delta_mse_loss * finetuning_args.recurft_multistep_delta_mse_loss_weight
                step_loss = (
                    step_loss
                    + (1.0 - step_delta_cosine) * finetuning_args.recurft_multistep_delta_cosine_loss_weight
                )
                if contrastive_loss_enabled:
                    candidate_end = min(anchor_hidden.size(1), start + contrastive_window + 1)
                    candidate_hidden = anchor_hidden[:, start + 1 : candidate_end, :].detach()
                    candidate_mask = input_attention_mask[:, start + 1 : candidate_end].bool().clone()
                    correct_index = step - 1
                    if correct_index < candidate_hidden.size(1):
                        candidate_mask[:, correct_index] = True
                        step_contrastive_loss = _masked_hidden_contrastive_loss(
                            pred=next_hidden,
                            candidates=candidate_hidden,
                            correct_index=correct_index,
                            mask=loss_mask,
                            candidate_mask=candidate_mask,
                            temperature=finetuning_args.recurft_multistep_contrastive_temperature,
                        )
                        step_contrastive_acc = _masked_hidden_contrastive_accuracy(
                            pred=next_hidden,
                            candidates=candidate_hidden,
                            correct_index=correct_index,
                            mask=loss_mask,
                            candidate_mask=candidate_mask,
                            temperature=finetuning_args.recurft_multistep_contrastive_temperature,
                        )
                        step_loss = (
                            step_loss
                            + step_contrastive_loss * finetuning_args.recurft_multistep_contrastive_loss_weight
                        )
                    else:
                        step_contrastive_loss = next_hidden.new_zeros(())
                        step_contrastive_acc = next_hidden.new_zeros(())
                else:
                    step_contrastive_loss = next_hidden.new_zeros(())
                    step_contrastive_acc = next_hidden.new_zeros(())

                label_index = target_index + 1
                if logit_loss_enabled and label_index < anchor_hidden.size(1):
                    projection_context = torch.cat([rollout_context, next_hidden], dim=1)
                    projection_attention = torch.cat(
                        [rollout_attention, input_attention_mask[:, target_index : target_index + 1]], dim=1
                    )
                    projection_position_ids = torch.cat(
                        [rollout_position_ids, input_position_ids[:, target_index : target_index + 1]], dim=1
                    )
                    logit_context_tokens = finetuning_args.recurft_multistep_logit_context_tokens
                    if logit_context_tokens > 0 and projection_context.size(1) > logit_context_tokens:
                        projection_context = projection_context[:, -logit_context_tokens:, :]
                        projection_attention = projection_attention[:, -logit_context_tokens:]
                        projection_position_ids = projection_position_ids[:, -logit_context_tokens:]

                    pred_logits = _project_anchor_logits(
                        model=model,
                        metadata=metadata,
                        anchor_hidden=projection_context,
                        attention_mask=projection_attention,
                        position_ids=projection_position_ids,
                    )[:, -1:, :]
                    labels = inputs["input_ids"][:, label_index : label_index + 1]
                    logit_mask = target_mask[:, label_index : label_index + 1]
                    step_token_ce_loss = _masked_token_ce_loss(pred_logits, labels, logit_mask)
                    step_teacher_logits = teacher_logits[:, target_index : target_index + 1, :].detach()
                    step_logit_kl_loss = _compute_kl_loss(
                        logits=pred_logits,
                        ref_logits=step_teacher_logits,
                        mask=logit_mask,
                        temperature=finetuning_args.recurft_multistep_logit_temperature,
                    )
                    step_logit_tv_loss = (
                        _compute_tv_loss(
                            logits=pred_logits,
                            ref_logits=step_teacher_logits,
                            mask=logit_mask,
                            temperature=finetuning_args.recurft_multistep_logit_temperature,
                        )
                        if logit_tv_loss_weight > 0.0
                        else pred_logits.new_zeros(())
                    )
                    step_token_acc = _masked_token_accuracy(pred_logits, labels, logit_mask)
                    step_teacher_top1 = _masked_top1_match(pred_logits, step_teacher_logits, logit_mask)
                    step_loss = (
                        step_loss
                        + step_token_ce_loss * finetuning_args.recurft_multistep_token_ce_loss_weight
                    )
                    step_loss = step_loss + step_logit_kl_loss * finetuning_args.recurft_multistep_logit_kl_loss_weight
                    step_loss = step_loss + step_logit_tv_loss * logit_tv_loss_weight
                else:
                    step_token_ce_loss = next_hidden.new_zeros(())
                    step_logit_kl_loss = next_hidden.new_zeros(())
                    step_logit_tv_loss = next_hidden.new_zeros(())
                    step_token_acc = next_hidden.new_zeros(())
                    step_teacher_top1 = next_hidden.new_zeros(())

                weight = step_decay ** (step - 2)
                weighted_losses.append(step_loss * weight)
                weighted_base_losses.append(step_base_loss.detach() * weight)
                weighted_mse_losses.append(step_mse_loss.detach() * weight)
                weighted_relative_mse_losses.append(step_relative_mse_loss.detach() * weight)
                weighted_cosine_values.append(step_cosine.detach() * weight)
                weighted_delta_mse_losses.append(step_delta_mse_loss.detach() * weight)
                weighted_delta_cosine_values.append(step_delta_cosine.detach() * weight)
                weighted_contrastive_losses.append(step_contrastive_loss.detach() * weight)
                weighted_contrastive_acc_values.append(step_contrastive_acc.detach() * weight)
                weighted_token_ce_losses.append(step_token_ce_loss.detach() * weight)
                weighted_logit_kl_losses.append(step_logit_kl_loss.detach() * weight)
                if logit_tv_loss_weight > 0.0:
                    weighted_logit_tv_losses.append(step_logit_tv_loss.detach() * weight)
                weighted_token_acc_values.append(step_token_acc.detach() * weight)
                weighted_teacher_top1_values.append(step_teacher_top1.detach() * weight)
                weights.append(weight)
                per_step_losses[step].append(step_mse_loss.detach())
                per_step_relative_mse[step].append(step_relative_mse_loss.detach())
                per_step_cosine[step].append(step_cosine.detach())
                per_step_delta_mse[step].append(step_delta_mse_loss.detach())
                per_step_delta_cosine[step].append(step_delta_cosine.detach())
                per_step_contrastive_loss[step].append(step_contrastive_loss.detach())
                per_step_contrastive_acc[step].append(step_contrastive_acc.detach())
                per_step_token_ce[step].append(step_token_ce_loss.detach())
                per_step_logit_kl[step].append(step_logit_kl_loss.detach())
                if logit_tv_loss_weight > 0.0:
                    per_step_logit_tv[step].append(step_logit_tv_loss.detach())
                per_step_token_acc[step].append(step_token_acc.detach())
                per_step_teacher_top1[step].append(step_teacher_top1.detach())

            context_next = next_hidden.detach() if detach_rollout else next_hidden
            teacher_conditioning_token = inputs["input_ids"][:, target_index : target_index + 1]
            conditioning_token = teacher_conditioning_token
            if (
                step >= 2
                and scheduled_sampling_ratio > 0.0
                and recurrent_module.has_boundary_head()
            ):
                with torch.no_grad():
                    draft_conditioning_token = recurrent_module.boundary_logits(
                        rollout_context[:, -1:, :], model
                    ).argmax(dim=-1)
                    sample_draft = torch.rand(
                        teacher_conditioning_token.shape,
                        device=teacher_conditioning_token.device,
                    ) < scheduled_sampling_ratio
                    conditioning_token = torch.where(
                        sample_draft,
                        draft_conditioning_token,
                        teacher_conditioning_token,
                    )

            rollout_context = torch.cat([rollout_context, context_next], dim=1)
            rollout_attention = torch.cat(
                [rollout_attention, input_attention_mask[:, target_index : target_index + 1]], dim=1
            )
            rollout_position_ids = torch.cat(
                [rollout_position_ids, input_position_ids[:, target_index : target_index + 1]], dim=1
            )
            rollout_token_ids = torch.cat([rollout_token_ids, conditioning_token], dim=1)

    if not weighted_losses:
        return None, {}

    denominator = max(sum(weights), 1e-8)
    loss_values = torch.stack(weighted_losses)
    trim_ratio = finetuning_args.recurft_multistep_trim_ratio
    kept_terms = len(weighted_losses)
    if trim_ratio > 0.0 and len(weighted_losses) > 1:
        weight_values = torch.tensor(weights, device=loss_values.device, dtype=loss_values.dtype)
        keep_count = max(1, int(round(len(weighted_losses) * (1.0 - trim_ratio))))
        keep_count = min(keep_count, len(weighted_losses))
        if keep_count < len(weighted_losses):
            sort_values = loss_values.detach() / weight_values.clamp_min(1e-8)
            kept_indices = torch.argsort(sort_values)[:keep_count]
            loss = loss_values.index_select(0, kept_indices).sum() / weight_values.index_select(0, kept_indices).sum()
            kept_terms = keep_count
        else:
            loss = loss_values.sum() / denominator
    else:
        loss = loss_values.sum() / denominator

    if step1_logit_losses:
        step1_logit_loss = torch.stack(step1_logit_losses).mean()
        loss = loss + step1_logit_loss * finetuning_args.recurft_multistep_step1_logit_loss_weight
    else:
        step1_logit_loss = loss.new_zeros(())

    if weighted_boundary_rollout_losses:
        boundary_rollout_denominator = max(sum(boundary_rollout_weights), 1e-8)
        boundary_rollout_loss = (
            torch.stack(weighted_boundary_rollout_losses).sum() / boundary_rollout_denominator
        )
        loss = loss + boundary_rollout_loss
        boundary_rollout_token_ce = (
            torch.stack(weighted_boundary_token_ce_losses).sum() / boundary_rollout_denominator
        )
        boundary_rollout_logit_kl = (
            torch.stack(weighted_boundary_logit_kl_losses).sum() / boundary_rollout_denominator
        )
        boundary_rollout_token_acc = (
            torch.stack(weighted_boundary_token_acc_values).sum() / boundary_rollout_denominator
        )
        boundary_rollout_teacher_top1 = (
            torch.stack(weighted_boundary_teacher_top1_values).sum() / boundary_rollout_denominator
        )
    else:
        boundary_rollout_loss = loss.new_zeros(())
        boundary_rollout_token_ce = loss.new_zeros(())
        boundary_rollout_logit_kl = loss.new_zeros(())
        boundary_rollout_token_acc = loss.new_zeros(())
        boundary_rollout_teacher_top1 = loss.new_zeros(())

    base_loss = torch.stack(weighted_base_losses).sum() / denominator
    mse_loss = torch.stack(weighted_mse_losses).sum() / denominator
    relative_mse_loss = torch.stack(weighted_relative_mse_losses).sum() / denominator
    cosine = torch.stack(weighted_cosine_values).sum() / denominator
    delta_mse_loss = torch.stack(weighted_delta_mse_losses).sum() / denominator
    delta_cosine = torch.stack(weighted_delta_cosine_values).sum() / denominator
    contrastive_loss = torch.stack(weighted_contrastive_losses).sum() / denominator
    contrastive_acc = torch.stack(weighted_contrastive_acc_values).sum() / denominator
    token_ce_loss = torch.stack(weighted_token_ce_losses).sum() / denominator
    logit_kl_loss = torch.stack(weighted_logit_kl_losses).sum() / denominator
    logit_tv_loss = (
        torch.stack(weighted_logit_tv_losses).sum() / denominator
        if weighted_logit_tv_losses
        else loss.new_zeros(())
    )
    token_acc = torch.stack(weighted_token_acc_values).sum() / denominator
    teacher_top1 = torch.stack(weighted_teacher_top1_values).sum() / denominator
    metrics = {
        "recurft_multistep_loss": _metric_value(loss),
        "recurft_multistep_base_loss": _metric_value(base_loss),
        "recurft_multistep_mse_loss": _metric_value(mse_loss),
        "recurft_multistep_huber_beta": _metric_value(finetuning_args.recurft_multistep_huber_beta),
        "recurft_multistep_mse_loss_clip": _metric_value(finetuning_args.recurft_multistep_mse_loss_clip),
        "recurft_multistep_trim_ratio": _metric_value(finetuning_args.recurft_multistep_trim_ratio),
        "recurft_multistep_kept_terms": float(kept_terms),
        "recurft_multistep_relative_mse_loss": _metric_value(relative_mse_loss),
        "recurft_multistep_cosine": _metric_value(cosine),
        "recurft_multistep_cosine_loss": _metric_value(1.0 - cosine),
        "recurft_multistep_delta_mse_loss": _metric_value(delta_mse_loss),
        "recurft_multistep_delta_cosine": _metric_value(delta_cosine),
        "recurft_multistep_delta_cosine_loss": _metric_value(1.0 - delta_cosine),
        "recurft_multistep_contrastive_loss": _metric_value(contrastive_loss),
        "recurft_multistep_contrastive_acc": _metric_value(contrastive_acc),
        "recurft_multistep_contrastive_loss_weight": _metric_value(
            finetuning_args.recurft_multistep_contrastive_loss_weight
        ),
        "recurft_multistep_contrastive_temperature": _metric_value(
            finetuning_args.recurft_multistep_contrastive_temperature
        ),
        "recurft_multistep_contrastive_window": float(finetuning_args.recurft_multistep_contrastive_window),
        "recurft_multistep_token_ce_loss": _metric_value(token_ce_loss),
        "recurft_multistep_token_ce_loss_weight": _metric_value(
            finetuning_args.recurft_multistep_token_ce_loss_weight
        ),
        "recurft_multistep_logit_kl_loss": _metric_value(logit_kl_loss),
        "recurft_multistep_logit_kl_loss_weight": _metric_value(
            finetuning_args.recurft_multistep_logit_kl_loss_weight
        ),
        "recurft_multistep_logit_tv_loss": _metric_value(logit_tv_loss),
        "recurft_multistep_logit_tv_loss_weight": _metric_value(logit_tv_loss_weight),
        "recurft_multistep_logit_temperature": _metric_value(finetuning_args.recurft_multistep_logit_temperature),
        "recurft_multistep_logit_context_tokens": float(finetuning_args.recurft_multistep_logit_context_tokens),
        "recurft_multistep_step1_logit_loss": _metric_value(step1_logit_loss),
        "recurft_multistep_step1_logit_loss_weight": _metric_value(
            finetuning_args.recurft_multistep_step1_logit_loss_weight
        ),
        "recurft_multistep_token_acc": _metric_value(token_acc),
        "recurft_multistep_teacher_top1_acc": _metric_value(teacher_top1),
        "recurft_multistep_boundary_loss": _metric_value(boundary_rollout_loss),
        "recurft_multistep_boundary_token_ce_loss": _metric_value(boundary_rollout_token_ce),
        "recurft_multistep_boundary_token_ce_loss_weight": _metric_value(
            finetuning_args.recurft_multistep_boundary_token_ce_loss_weight
        ),
        "recurft_multistep_boundary_logit_kl_loss": _metric_value(boundary_rollout_logit_kl),
        "recurft_multistep_boundary_logit_kl_loss_weight": _metric_value(
            finetuning_args.recurft_multistep_boundary_logit_kl_loss_weight
        ),
        "recurft_multistep_boundary_token_acc": _metric_value(boundary_rollout_token_acc),
        "recurft_multistep_boundary_teacher_top1_acc": _metric_value(boundary_rollout_teacher_top1),
        "recurft_multistep_steps": float(steps),
        "recurft_multistep_starts": float(len(starts)),
        "recurft_multistep_start_min": float(min(starts)),
        "recurft_multistep_start_max": float(max(starts)),
        "recurft_multistep_start_selection_cyclic": float(
            finetuning_args.recurft_multistep_start_selection == "cyclic"
        ),
        "recurft_scheduled_sampling_ratio": _metric_value(scheduled_sampling_ratio),
        "recurft_multistep_relative_mse_loss_weight": _metric_value(
            finetuning_args.recurft_multistep_relative_mse_loss_weight
        ),
        "recurft_multistep_cosine_loss_weight": _metric_value(finetuning_args.recurft_multistep_cosine_loss_weight),
        "recurft_multistep_delta_mse_loss_weight": _metric_value(
            finetuning_args.recurft_multistep_delta_mse_loss_weight
        ),
        "recurft_multistep_delta_cosine_loss_weight": _metric_value(
            finetuning_args.recurft_multistep_delta_cosine_loss_weight
        ),
    }
    for step, losses in per_step_losses.items():
        if losses:
            metrics[f"recurft_multistep_mse_k{step}"] = _metric_value(torch.stack(losses).mean())
    for step, losses in per_step_relative_mse.items():
        if losses:
            metrics[f"recurft_multistep_relative_mse_k{step}"] = _metric_value(torch.stack(losses).mean())
    for step, cosines in per_step_cosine.items():
        if cosines:
            metrics[f"recurft_multistep_cosine_k{step}"] = _metric_value(torch.stack(cosines).mean())
    for step, losses in per_step_delta_mse.items():
        if losses:
            metrics[f"recurft_multistep_delta_mse_k{step}"] = _metric_value(torch.stack(losses).mean())
    for step, cosines in per_step_delta_cosine.items():
        if cosines:
            metrics[f"recurft_multistep_delta_cosine_k{step}"] = _metric_value(torch.stack(cosines).mean())
    for step, losses in per_step_contrastive_loss.items():
        if losses:
            metrics[f"recurft_multistep_contrastive_loss_k{step}"] = _metric_value(torch.stack(losses).mean())
    for step, values in per_step_contrastive_acc.items():
        if values:
            metrics[f"recurft_multistep_contrastive_acc_k{step}"] = _metric_value(torch.stack(values).mean())
    for step, losses in per_step_token_ce.items():
        if losses:
            metrics[f"recurft_multistep_token_ce_k{step}"] = _metric_value(torch.stack(losses).mean())
    for step, losses in per_step_logit_kl.items():
        if losses:
            metrics[f"recurft_multistep_logit_kl_k{step}"] = _metric_value(torch.stack(losses).mean())
    for step, losses in per_step_logit_tv.items():
        if losses:
            metrics[f"recurft_multistep_logit_tv_k{step}"] = _metric_value(torch.stack(losses).mean())
    for step, values in per_step_token_acc.items():
        if values:
            metrics[f"recurft_multistep_token_acc_k{step}"] = _metric_value(torch.stack(values).mean())
    for step, values in per_step_teacher_top1.items():
        if values:
            metrics[f"recurft_multistep_teacher_top1_acc_k{step}"] = _metric_value(torch.stack(values).mean())
    for step, losses in per_step_boundary_token_ce.items():
        if losses:
            metrics[f"recurft_multistep_boundary_token_ce_k{step}"] = _metric_value(
                torch.stack(losses).mean()
            )
    for step, losses in per_step_boundary_logit_kl.items():
        if losses:
            metrics[f"recurft_multistep_boundary_logit_kl_k{step}"] = _metric_value(
                torch.stack(losses).mean()
            )
    for step, values in per_step_boundary_token_acc.items():
        if values:
            metrics[f"recurft_multistep_boundary_token_acc_k{step}"] = _metric_value(
                torch.stack(values).mean()
            )
    for step, values in per_step_boundary_teacher_top1.items():
        if values:
            metrics[f"recurft_multistep_boundary_teacher_top1_k{step}"] = _metric_value(
                torch.stack(values).mean()
            )

    return loss, metrics


def compute_recurft_loss(
    model: "nn.Module",
    inputs: dict[str, torch.Tensor],
    finetuning_args: "FinetuningArguments",
    global_step: int,
    ref_model: Optional["nn.Module"] = None,
) -> tuple[torch.Tensor, Any, dict[str, float]]:
    if getattr(finetuning_args, "recurft_joint_mode", "legacy") == "trainable_target":
        from .trainable_target_joint import compute_trainable_target_joint_loss

        return compute_trainable_target_joint_loss(
            _compute_recurft_loss_impl, model, inputs, finetuning_args, global_step, ref_model
        )
    return _compute_recurft_loss_impl(model, inputs, finetuning_args, global_step, ref_model)


def _compute_recurft_loss_impl(model, inputs, finetuning_args, global_step, ref_model=None):
    unwrapped_model = _unwrap_model(model)
    recurrent_module = getattr(unwrapped_model, "recurft_recurrent", None)
    metadata = getattr(unwrapped_model, "recurft_metadata", None)
    if recurrent_module is None or metadata is None:
        raise ValueError("RecurFT is enabled but the model does not have a recurrent T module.")

    ref_outputs = _forward_recurft_reference(model, inputs, ref_model=ref_model)

    outputs = _forward_with_hidden_states(
        model, inputs, include_loss=finetuning_args.recurft_sft_loss_weight > 0.0
    )
    hidden_mask = inputs.get("attention_mask")
    if hidden_mask is None:
        hidden_mask = torch.ones_like(inputs["input_ids"], dtype=torch.bool)
    else:
        hidden_mask = hidden_mask.bool()

    hidden_loss = _masked_hidden_mse(outputs.hidden_states[-1], ref_outputs.hidden_states[-1], hidden_mask)
    hidden_relative_mse_loss = _masked_hidden_relative_mse(
        outputs.hidden_states[-1], ref_outputs.hidden_states[-1], hidden_mask
    )
    hidden_cosine = _masked_hidden_cosine(outputs.hidden_states[-1], ref_outputs.hidden_states[-1], hidden_mask)
    total_loss = hidden_loss * finetuning_args.recurft_hidden_loss_weight
    total_loss = total_loss + hidden_relative_mse_loss * finetuning_args.recurft_hidden_relative_mse_loss_weight
    total_loss = total_loss + (1.0 - hidden_cosine) * finetuning_args.recurft_hidden_cosine_loss_weight
    sft_loss = None
    kl_loss = None
    recurrent_loss = None
    recurrent_relative_mse_loss = None
    recurrent_cosine = None
    recurrent_delta_mse_loss = None
    recurrent_delta_cosine = None
    boundary_token_ce_loss = None
    boundary_logit_kl_loss = None
    boundary_token_acc = None
    boundary_teacher_top1 = None
    boundary_positions = 0
    multistep_loss = None
    multistep_metrics: dict[str, float] = {}

    if finetuning_args.recurft_sft_loss_weight > 0.0:
        if outputs.loss is None:
            raise ValueError("The model did not return SFT loss while `recurft_sft_loss_weight` is positive.")

        sft_loss = outputs.loss
        total_loss = total_loss + sft_loss * finetuning_args.recurft_sft_loss_weight

    if finetuning_args.recurft_kl_loss_weight > 0.0:
        kl_mask = _make_token_mask(inputs, finetuning_args, shifted=False)
        kl_loss = _compute_kl_loss(
            logits=outputs.logits,
            ref_logits=ref_outputs.logits,
            mask=kl_mask,
            temperature=finetuning_args.recurft_kl_temperature,
        )
        total_loss = total_loss + kl_loss * finetuning_args.recurft_kl_loss_weight

    recurrent_weight = _get_recurrent_weight(finetuning_args, global_step)
    if recurrent_weight > 0.0:
        anchor_hidden_idx = metadata.get("recurrent_hidden_state_index", metadata["anchor_layer"] + 1)
        anchor_hidden = outputs.hidden_states[anchor_hidden_idx]
        if anchor_hidden.size(1) > 1:
            current_hidden = anchor_hidden[:, :-1, :]
            target_hidden = anchor_hidden[:, 1:, :].detach()
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask[:, :-1]

            position_ids = inputs.get("position_ids")
            if position_ids is not None:
                position_ids = position_ids[:, :-1]

            pred_hidden = recurrent_module(
                current_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                model=unwrapped_model,
                token_ids=inputs["input_ids"][:, :-1],
            )
            recurrent_mask = _make_token_mask(inputs, finetuning_args, shifted=True)
            recurrent_loss = _masked_hidden_mse(pred_hidden, target_hidden, recurrent_mask)
            recurrent_relative_mse_loss = _masked_hidden_relative_mse(pred_hidden, target_hidden, recurrent_mask)
            recurrent_cosine = _masked_hidden_cosine(pred_hidden, target_hidden, recurrent_mask)
            recurrent_delta_mse_loss = _masked_hidden_delta_mse(
                pred_hidden, current_hidden.detach(), target_hidden, current_hidden.detach(), recurrent_mask
            )
            recurrent_delta_cosine = _masked_hidden_delta_cosine(
                pred_hidden, current_hidden.detach(), target_hidden, current_hidden.detach(), recurrent_mask
            )
            total_loss = total_loss + recurrent_loss * recurrent_weight
            total_loss = (
                total_loss
                + recurrent_relative_mse_loss
                * recurrent_weight
                * finetuning_args.recurft_recurrent_relative_mse_loss_weight
            )
            total_loss = (
                total_loss
                + (1.0 - recurrent_cosine)
                * recurrent_weight
                * finetuning_args.recurft_recurrent_cosine_loss_weight
            )
            total_loss = (
                total_loss
                + recurrent_delta_mse_loss
                * recurrent_weight
                * finetuning_args.recurft_recurrent_delta_mse_loss_weight
            )
            total_loss = (
                total_loss
                + (1.0 - recurrent_delta_cosine)
                * recurrent_weight
                * finetuning_args.recurft_recurrent_delta_cosine_loss_weight
            )

            multistep_weight = _get_multistep_weight(finetuning_args, global_step)
            if multistep_weight > 0.0 and finetuning_args.recurft_multistep_steps > 1:
                multistep_loss, multistep_metrics = _compute_multistep_recurrent_loss(
                    recurrent_module=recurrent_module,
                    anchor_hidden=anchor_hidden,
                    teacher_logits=outputs.logits,
                    inputs=inputs,
                    finetuning_args=finetuning_args,
                    model=unwrapped_model,
                    metadata=metadata,
                    global_step=global_step,
                )
                if multistep_loss is not None:
                    total_loss = total_loss + multistep_loss * multistep_weight

            if recurrent_module.has_boundary_head() and (
                finetuning_args.recurft_boundary_token_ce_loss_weight > 0.0
                or finetuning_args.recurft_boundary_logit_kl_loss_weight > 0.0
            ):
                selected_positions = _select_multistep_starts(
                    seq_len=anchor_hidden.size(1),
                    steps=1,
                    stride=finetuning_args.recurft_boundary_loss_stride,
                    max_starts=finetuning_args.recurft_boundary_loss_max_tokens,
                    device=anchor_hidden.device,
                )
                position_index = torch.tensor(selected_positions, device=anchor_hidden.device, dtype=torch.long)
                boundary_positions = len(selected_positions)
                boundary_hidden = anchor_hidden.index_select(1, position_index)
                boundary_logits = recurrent_module.boundary_logits(boundary_hidden, unwrapped_model)
                boundary_labels = inputs["input_ids"][:, 1:].index_select(1, position_index)
                boundary_mask = _make_token_mask(inputs, finetuning_args, shifted=True).index_select(
                    1, position_index
                )
                teacher_outputs = outputs if getattr(finetuning_args, "recurft_boundary_teacher_source", "reference") == "target" else ref_outputs
                boundary_teacher_logits = teacher_outputs.logits[:, :-1, :].index_select(1, position_index).detach()
                boundary_token_ce_loss = _masked_token_ce_loss(
                    boundary_logits, boundary_labels, boundary_mask
                )
                boundary_logit_kl_loss = _compute_kl_loss(
                    logits=boundary_logits,
                    ref_logits=boundary_teacher_logits,
                    mask=boundary_mask,
                    temperature=finetuning_args.recurft_multistep_logit_temperature,
                )
                boundary_token_acc = _masked_token_accuracy(
                    boundary_logits, boundary_labels, boundary_mask
                )
                boundary_teacher_top1 = _masked_top1_match(
                    boundary_logits, boundary_teacher_logits, boundary_mask
                )
                total_loss = (
                    total_loss
                    + boundary_token_ce_loss * finetuning_args.recurft_boundary_token_ce_loss_weight
                    + boundary_logit_kl_loss * finetuning_args.recurft_boundary_logit_kl_loss_weight
                )

    metrics = {
        "recurft_total_loss": _metric_value(total_loss),
        "recurft_frozen_reference_model": float(ref_model is not None),
        "recurft_hidden_loss": _metric_value(hidden_loss),
        "recurft_hidden_loss_weight": _metric_value(finetuning_args.recurft_hidden_loss_weight),
        "recurft_hidden_relative_mse_loss": _metric_value(hidden_relative_mse_loss),
        "recurft_hidden_relative_mse_loss_weight": _metric_value(
            finetuning_args.recurft_hidden_relative_mse_loss_weight
        ),
        "recurft_hidden_cosine": _metric_value(hidden_cosine),
        "recurft_hidden_cosine_loss": _metric_value(1.0 - hidden_cosine),
        "recurft_hidden_cosine_loss_weight": _metric_value(finetuning_args.recurft_hidden_cosine_loss_weight),
        "recurft_recurrent_loss": _metric_value(recurrent_loss) if recurrent_loss is not None else 0.0,
        "recurft_recurrent_loss_weight": _metric_value(recurrent_weight),
        "recurft_recurrent_relative_mse_loss": (
            _metric_value(recurrent_relative_mse_loss) if recurrent_relative_mse_loss is not None else 0.0
        ),
        "recurft_recurrent_relative_mse_loss_weight": _metric_value(
            finetuning_args.recurft_recurrent_relative_mse_loss_weight
        ),
        "recurft_recurrent_cosine": _metric_value(recurrent_cosine) if recurrent_cosine is not None else 0.0,
        "recurft_recurrent_cosine_loss": _metric_value(1.0 - recurrent_cosine)
        if recurrent_cosine is not None
        else 0.0,
        "recurft_recurrent_cosine_loss_weight": _metric_value(
            finetuning_args.recurft_recurrent_cosine_loss_weight
        ),
        "recurft_recurrent_delta_mse_loss": (
            _metric_value(recurrent_delta_mse_loss) if recurrent_delta_mse_loss is not None else 0.0
        ),
        "recurft_recurrent_delta_mse_loss_weight": _metric_value(
            finetuning_args.recurft_recurrent_delta_mse_loss_weight
        ),
        "recurft_recurrent_delta_cosine": (
            _metric_value(recurrent_delta_cosine) if recurrent_delta_cosine is not None else 0.0
        ),
        "recurft_recurrent_delta_cosine_loss": _metric_value(1.0 - recurrent_delta_cosine)
        if recurrent_delta_cosine is not None
        else 0.0,
        "recurft_recurrent_delta_cosine_loss_weight": _metric_value(
            finetuning_args.recurft_recurrent_delta_cosine_loss_weight
        ),
        "recurft_multistep_loss": _metric_value(multistep_loss) if multistep_loss is not None else 0.0,
        "recurft_multistep_loss_weight": _metric_value(finetuning_args.recurft_multistep_loss_weight),
        "recurft_multistep_effective_loss_weight": _metric_value(
            _get_multistep_weight(finetuning_args, global_step)
        ),
        "recurft_kl_loss": _metric_value(kl_loss) if kl_loss is not None else 0.0,
        "recurft_kl_loss_weight": _metric_value(finetuning_args.recurft_kl_loss_weight),
        "recurft_sft_loss": _metric_value(sft_loss) if sft_loss is not None else 0.0,
        "recurft_sft_loss_weight": _metric_value(finetuning_args.recurft_sft_loss_weight),
        "recurft_boundary_token_ce_loss": (
            _metric_value(boundary_token_ce_loss) if boundary_token_ce_loss is not None else 0.0
        ),
        "recurft_boundary_token_ce_loss_weight": _metric_value(
            finetuning_args.recurft_boundary_token_ce_loss_weight
        ),
        "recurft_boundary_logit_kl_loss": (
            _metric_value(boundary_logit_kl_loss) if boundary_logit_kl_loss is not None else 0.0
        ),
        "recurft_boundary_logit_kl_loss_weight": _metric_value(
            finetuning_args.recurft_boundary_logit_kl_loss_weight
        ),
        "recurft_boundary_token_acc": (
            _metric_value(boundary_token_acc) if boundary_token_acc is not None else 0.0
        ),
        "recurft_boundary_teacher_top1_acc": (
            _metric_value(boundary_teacher_top1) if boundary_teacher_top1 is not None else 0.0
        ),
        "recurft_boundary_positions": float(boundary_positions),
        "recurft_scheduled_sampling_ratio": _metric_value(
            _get_scheduled_sampling_ratio(finetuning_args, global_step)
        ),
    }
    metrics.update(multistep_metrics)
    return total_loss, outputs, metrics
