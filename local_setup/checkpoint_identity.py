"""Behavior-relevant target adapter settings for paired checkpoint comparisons."""
import json
from pathlib import Path


def target_adapter_signature(checkpoint):
    config = json.loads((Path(checkpoint)/'adapter_config.json').read_text())
    # Paths can change when exporting a model. All scaling, rank, module and
    # adapter-type settings must still match; weights alone are insufficient.
    for key in ('base_model_name_or_path', 'revision', 'inference_mode'):
        config.pop(key, None)
    return config
