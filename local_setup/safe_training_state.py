"""Tensor/JSON optimizer and RNG checkpoints; no pickle deserialization."""
from pathlib import Path
import hashlib
import json
import random
import numpy as np
import torch
from safetensors.torch import load_file, save_file


def save_state(prefix, value):
    prefix = Path(prefix)
    tensors = {}

    def encode(x):
        if isinstance(x, torch.Tensor):
            name = f'tensor_{len(tensors)}'
            tensors[name] = x.detach().cpu().contiguous().clone()
            return {'kind': 'tensor', 'name': name}
        if isinstance(x, np.ndarray):
            return {'kind': 'ndarray', 'dtype': x.dtype.str, 'shape': list(x.shape), 'values': x.tolist()}
        if isinstance(x, np.generic):
            return encode(x.item())
        if isinstance(x, dict):
            return {'kind': 'dict', 'items': [[encode(k), encode(v)] for k, v in x.items()]}
        if isinstance(x, (list, tuple)):
            return {'kind': 'tuple' if isinstance(x, tuple) else 'list', 'items': [encode(v) for v in x]}
        if x is None or isinstance(x, (str, int, float, bool)):
            return {'kind': 'scalar', 'value': x}
        raise TypeError(f'Unsupported checkpoint type: {type(x)}')

    payload = encode(value)
    tensor_path = prefix.with_suffix('.safetensors')
    tensor_temp = prefix.with_suffix('.safetensors.tmp')
    save_file(tensors, str(tensor_temp))
    digest = hashlib.sha256(tensor_temp.read_bytes()).hexdigest()
    tensor_temp.replace(tensor_path)
    metadata = {'format': 'joint_safe_state_v1', 'tensor_sha256': digest, 'payload': payload}
    temp = prefix.with_suffix('.json.tmp')
    temp.write_text(json.dumps(metadata, allow_nan=False))
    temp.replace(prefix.with_suffix('.json'))


def load_state(prefix):
    prefix = Path(prefix)
    metadata = json.loads(prefix.with_suffix('.json').read_text())
    assert metadata['format'] == 'joint_safe_state_v1'
    tensor_path = prefix.with_suffix('.safetensors')
    assert hashlib.sha256(tensor_path.read_bytes()).hexdigest() == metadata['tensor_sha256']
    tensors = load_file(str(tensor_path))

    def decode(x):
        kind = x['kind']
        if kind == 'tensor':
            return tensors[x['name']]
        if kind == 'scalar':
            return x['value']
        if kind == 'ndarray':
            return np.asarray(x['values'], dtype=x['dtype']).reshape(x['shape'])
        if kind == 'dict':
            return {decode(k): decode(v) for k, v in x['items']}
        if kind in ('tuple', 'list'):
            values = [decode(v) for v in x['items']]
            return tuple(values) if kind == 'tuple' else values
        raise ValueError(f'Unsupported checkpoint tag: {kind}')

    return decode(metadata['payload'])


def rng_state():
    result = {'python': random.getstate(), 'numpy': np.random.get_state(), 'cpu': torch.random.get_rng_state()}
    if torch.cuda.is_available():
        result['cuda'] = torch.cuda.random.get_rng_state_all()
    return result


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['cpu'])
    if 'cuda' in state:
        torch.cuda.random.set_rng_state_all(state['cuda'])


def install_safe_resume():
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    def save_optimizer(self, output_dir):
        assert self.args.world_size == 1 and not self.is_deepspeed_enabled and not self.is_fsdp_enabled
        save_state(Path(output_dir) / 'optimizer_safe', {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None})

    def load_optimizer(self, checkpoint):
        if checkpoint is None:
            return
        assert self.args.world_size == 1 and not self.is_deepspeed_enabled and not self.is_fsdp_enabled
        state = load_state(Path(checkpoint) / 'optimizer_safe')
        self.optimizer.load_state_dict(state['optimizer'])
        if state['scheduler'] is not None:
            self.lr_scheduler.load_state_dict(state['scheduler'])

    def save_rng(self, output_dir):
        assert self.args.world_size == 1
        save_state(Path(output_dir) / 'rng_safe', rng_state())

    def load_rng(self, checkpoint):
        if checkpoint is not None:
            restore_rng(load_state(Path(checkpoint) / 'rng_safe'))

    CustomSeq2SeqTrainer._save_optimizer_and_scheduler = save_optimizer
    CustomSeq2SeqTrainer._load_optimizer_and_scheduler = load_optimizer
    CustomSeq2SeqTrainer._save_rng_state = save_rng
    CustomSeq2SeqTrainer._load_rng_state = load_rng
