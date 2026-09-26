import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM, Seq2SeqTrainer
from llamafactory.hparams import FinetuningArguments
from llamafactory.model.model_utils.recurft import (
    resolve_recurft_layout, build_recurft_recurrent_module, attach_recurft_recurrent_module)
from llamafactory.train.sft import recurft
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'local_setup'))
from qwen3_4b_recipe import architecture, budgets, split_data, stage_config
from benchmark_trained_target import command


def recipe():
    return yaml.safe_load((ROOT/'configs/experimental/qwen3_4b_matrix.yaml').read_text())


def config():
    return dict(model_type='qwen3', num_hidden_layers=36, hidden_size=2560,
                intermediate_size=9728, num_attention_heads=32, num_key_value_heads=8,
                head_dim=128, tie_word_embeddings=True, vocab_size=151936)


def test_model_capacity_and_three_training_scopes():
    r = recipe()
    arch = architecture(config(), r)
    assert (arch['t_rank'], arch['boundary_rank']) == (80, 160)
    assert arch['t_trainable_parameters'] == 9175040
    assert arch['boundary_trainable_parameters'] == 824320
    assert (arch['loop_start'], arch['loop_end']) == (33, 34)
    with pytest.raises(ValueError):
        architecture(dict(config(), hidden_size=4096), r)
    for seed in (42, 43, 44):
        for stage, trainable in [('core', True), ('multistep', True), ('head', False), ('joint', False), ('joint', True)]:
            c = stage_config('model', 'data', 'out', seed, stage, 4000, arch, r, trainable)
            fields = FinetuningArguments.__dataclass_fields__
            args = FinetuningArguments(**{k:v for k,v in c.items() if k in fields})
            assert args.recurft_stage1_heads_only == (stage == 'head')
            assert args.recurft_recurrent_trainable_only == (stage == 'joint' and not trainable)
            assert args.recurft_joint_mode == ('trainable_target' if stage == 'joint' and trainable else 'legacy')
            assert args.recurft_boundary_teacher_source == 'target'
    b = budgets(394000, r)
    assert b['full_joint'] == b['non_joint'] + b['joint']
    assert b['head'] == b['head_main'] + b['head_warmup']


def test_split_excludes_every_duplicate_of_validation_query():
    rows = [dict(query=f'question {i}', response=str(i)) for i in range(20)]
    rows += [dict(query='question   3', response='alternative')]
    train, valid, audit = split_data(rows, 19, 'test')
    assert not {' '.join(x['query'].split()) for x in train} & {' '.join(x['query'].split()) for x in valid}
    assert len(valid) == 19
    assert audit == split_data(rows, 19, 'test')[2]


def tiny_model(mode='trainable_target'):
    torch.manual_seed(17)
    args = FinetuningArguments(use_recurft=True, finetuning_type='lora', lora_target='all',
        recurft_loop_start_layer=1, recurft_loop_end_layer=1,
        recurft_t_lora_rank=2, recurft_boundary_head_rank=4,
        recurft_joint_mode=mode, recurft_boundary_teacher_source='target',
        recurft_hidden_loss_weight=20., recurft_kl_loss_weight=.2,
        recurft_recurrent_loss_weight=.1, recurft_recurrent_warmup_steps=0,
        recurft_multistep_steps=2, recurft_multistep_loss_weight=.1,
        recurft_multistep_boundary_logit_kl_loss_weight=10.,
        recurft_multistep_stride=1, recurft_multistep_max_starts=2,
        recurft_multistep_context_tokens=0, recurft_multistep_detach_rollout=False,
        recurft_boundary_logit_kl_loss_weight=.2, recurft_boundary_loss_stride=1,
        recurft_boundary_loss_max_tokens=2)
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
        attention_dropout=0.0, use_cache=False))
    meta = resolve_recurft_layout(3, args)
    module = build_recurft_recurrent_module(base, args, meta)
    model = get_peft_model(base, LoraConfig(r=2, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    attach_recurft_recurrent_module(model, module)
    # Nonzero target adapter makes adapted and original teachers distinguishable.
    for name, p in model.named_parameters():
        if 'lora_B' in name and 'recurft' not in name:
            torch.nn.init.normal_(p, std=.2)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11]])
    return model, args, dict(input_ids=ids, labels=ids.clone(), attention_mask=torch.ones_like(ids))


def test_joint_updates_three_groups_preserves_base_and_detaches_teacher():
    model, args, inputs = tiny_model()
    model.train()
    captured = []
    original = recurft._compute_kl_loss
    def capture(*a, **kw):
        if 'ref_logits' in kw:
            captured.append(kw['ref_logits'])
            assert not kw['ref_logits'].requires_grad
        return original(*a, **kw)
    with patch.object(recurft, '_compute_kl_loss', capture):
        loss, outputs, metrics = recurft.compute_recurft_loss(model, inputs, args, 0)
    assert torch.isfinite(loss)
    assert metrics['recurft_trainable_target_joint'] == 1
    # Direct boundary KL is last; select the same positions in current target logits.
    positions = recurft._select_multistep_starts(seq_len=6, steps=1, stride=1, max_starts=2, device=outputs.logits.device)
    assert torch.equal(captured[-1], outputs.logits[:, :-1, :][:, positions].detach())
    loss.backward()
    groups = {'target': 0., 'T': 0., 'head': 0.}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None
        elif p.grad is not None:
            key = 'head' if 'boundary_' in name else 'T' if 'recurft' in name else 'target'
            groups[key] += p.grad.float().square().sum().item()
    assert all(v > 0 for v in groups.values()), groups
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    assert model.get_input_embeddings().weight.grad is None


def test_legacy_frozen_path_unchanged_and_new_mode_rejects_frozen_target():
    model, args, inputs = tiny_model('legacy')
    for name, p in model.named_parameters():
        if 'recurft' not in name:
            p.requires_grad_(False)
    model.eval()
    args.recurft_boundary_teacher_source = 'reference'
    with torch.no_grad():
        result = recurft.compute_recurft_loss(model, inputs, args, 0)
        impl = recurft._compute_recurft_loss_impl(model, inputs, args, 0)
    assert torch.equal(result[0], impl[0]) and result[2] == impl[2]
    args.recurft_joint_mode = 'trainable_target'
    with pytest.raises(ValueError, match='scope mismatch'):
        recurft.compute_recurft_loss(model, inputs, args, 0)


def test_eval_metrics_do_not_pollute_training_windows():
    trainer = object.__new__(CustomSeq2SeqTrainer)
    trainer.finetuning_args = SimpleNamespace(use_recurft=True)
    trainer.state = SimpleNamespace(global_step=0)
    trainer.ref_model = None
    model = torch.nn.Linear(1, 1)
    with patch('llamafactory.train.sft.trainer.compute_recurft_loss',
               side_effect=[(torch.tensor(1.), None, {'component': 2.}),
                            (torch.tensor(1.), None, {'component': 100.}),
                            (torch.tensor(1.), None, {'component': 4.})]), \
         patch.object(Seq2SeqTrainer, 'log') as log:
        trainer.compute_loss(model, {})
        model.eval()
        trainer.compute_loss(model, {})
        trainer.log({'eval_loss': 1.})
        assert log.call_args.args[0]['eval_component'] == 100.
        model.train()
        trainer.compute_loss(model, {})
        trainer.log({'loss': 1.})
        assert log.call_args.args[0]['component'] == 3.
        assert 'eval_component' not in log.call_args.args[0]


def test_inference_uses_single_checkpoint_for_both_targets(tmp_path):
    for name in ['adapter_model.safetensors', 'recurft_recurrent.safetensors', 'recurft_config.json']:
        (tmp_path/name).touch()
    cmd = command(['--checkpoint', str(tmp_path), '--model', 'model'])
    assert cmd[cmd.index('--before')+1] == cmd[cmd.index('--after')+1] == str(tmp_path)
    with pytest.raises(SystemExit):
        command(['--checkpoint', str(tmp_path), '--after=other'])
