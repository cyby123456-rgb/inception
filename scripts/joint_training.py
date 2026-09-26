"""Build the experimental joint stage without importing the ML runtime."""

import json
from pathlib import Path


def joint_config(source, recipe, checkpoint, steps, model_config):
    checkpoint = Path(checkpoint)
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("Joint training requires --steps with a positive number of added updates.")
    metadata = json.loads((checkpoint / "recurft_config.json").read_text(encoding="utf-8"))
    if metadata.get("boundary_head_rank", 0) <= 0:
        raise ValueError("Joint training needs a boundary checkpoint. Run boundary-warmup first.")
    for name in ("optimizer.pt", "optimizer.bin", "scheduler.pt", "optimizer_safe.json", "optimizer_safe.safetensors"):
        if (checkpoint / name).exists():
            raise ValueError("Joint training changes optimizer groups; use a model-only checkpoint "
                             "saved with save_only_model: true. Keep the original checkpoint intact.")
    if source.get("adapter_name_or_path"):
        raise ValueError("Use resume_from_checkpoint for RecurFT, not adapter_name_or_path.")
    if not source.get("use_recurft") or source.get("finetuning_type") != "lora":
        raise ValueError("--source-config must be the RecurFT LoRA train.yaml for this checkpoint.")
    for key in ("loop_start_layer", "loop_end_layer", "t_lora_rank", "boundary_head_rank",
                "token_conditioning_rank", "multistep_residual_rank", "multistep_step1_residual_rank"):
        expected = metadata.get(key, 0)
        if source.get("recurft_" + key, 0) != expected:
            raise ValueError(f"Source config/checkpoint mismatch for recurft_{key}: "
                             f"{source.get('recurft_' + key, 0)} != {expected}.")
    layers = model_config.get("num_hidden_layers")
    if layers is None or metadata.get("last_layer") != layers - 1:
        raise ValueError("Base-model layer count does not match the checkpoint layout.")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    start = state["global_step"]
    if not isinstance(start, int) or start < 0:
        raise ValueError("Checkpoint global_step must be a non-negative integer.")
    config = dict(source)
    config.update(recipe)
    # This launcher always selects the frozen-target continuation recipe, even
    # when its source checkpoint came from trainable-target joint training.
    config["recurft_joint_mode"] = "legacy"
    config.update(resume_from_checkpoint=str(checkpoint), max_steps=start + steps,
                  recurft_multistep_warmup_start_step=start,
                  recurft_scheduled_sampling_start_step=start)
    return config
