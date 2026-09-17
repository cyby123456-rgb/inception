"""Opt-in target-LoRA + T + boundary training; legacy frozen training is unchanged.

The base reference regularizes target drift. Draft hidden/logit labels come from
the current adapted target and are detached. Target input features retain their
gradient, allowing target LoRA and T to co-adapt. Original vocabulary/base weights
remain frozen, including Qwen3-4B's tied embedding and output matrix.
"""


def compute_trainable_target_joint_loss(implementation, model, inputs, args, step, ref_model):
    from .recurft import _unwrap_model

    unwrapped = _unwrap_model(model)
    groups = set()
    for name, parameter in unwrapped.named_parameters():
        if not parameter.requires_grad:
            continue
        if "boundary_" in name:
            groups.add("head")
        elif "recurft_recurrent" in name and "lora_" in name:
            groups.add("T")
        elif "lora_" in name:
            groups.add("target")
        else:
            raise ValueError(f"Unexpected trainable base parameter: {name}")
    if groups != {"target", "T", "head"}:
        raise ValueError(f"trainable_target joint scope mismatch: {groups}")
    if args.recurft_boundary_teacher_source != "target":
        raise ValueError("Draft KL must learn the current trained target")
    loss, outputs, metrics = implementation(model, inputs, args, step, ref_model)
    metrics["recurft_trainable_target_joint"] = 1.0
    metrics["recurft_draft_teacher_current_target"] = 1.0
    return loss, outputs, metrics
