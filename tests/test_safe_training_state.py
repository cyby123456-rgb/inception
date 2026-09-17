import importlib.util
from pathlib import Path
import random
import numpy as np
import torch
import pytest

PATH = Path(__file__).resolve().parents[1] / 'local_setup/safe_training_state.py'
spec = importlib.util.spec_from_file_location('safe_training_state', PATH)
safe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safe)


def step(model, optimizer, scheduler):
    x = torch.randn(4, 3) + float(np.random.normal()) + random.random()
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def test_resumed_optimizer_scheduler_rng_match_uninterrupted(tmp_path):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: 1 / (1 + n))
    for _ in range(2):
        step(model, optimizer, scheduler)
    safe.save_state(tmp_path / 'model', model.state_dict())
    safe.save_state(tmp_path / 'optimizer', {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()})
    safe.save_state(tmp_path / 'rng', safe.rng_state())
    for _ in range(2):
        step(model, optimizer, scheduler)
    resumed = torch.nn.Linear(3, 2)
    resumed.load_state_dict(safe.load_state(tmp_path / 'model'))
    opt2 = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda n: 1 / (1 + n))
    state = safe.load_state(tmp_path / 'optimizer')
    opt2.load_state_dict(state['optimizer'])
    sched2.load_state_dict(state['scheduler'])
    safe.restore_rng(safe.load_state(tmp_path / 'rng'))
    for _ in range(2):
        step(resumed, opt2, sched2)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), resumed.parameters()))
    assert scheduler.get_last_lr() == sched2.get_last_lr()
    for key, item in optimizer.state_dict()['state'].items():
        assert all(torch.equal(value, opt2.state_dict()['state'][key][name]) for name, value in item.items())


def test_corrupt_tensor_file_is_rejected(tmp_path):
    safe.save_state(tmp_path / 'state', {'x': torch.arange(3)})
    p = tmp_path / 'state.safetensors'
    p.write_bytes(p.read_bytes() + b'corruption')
    with pytest.raises(AssertionError):
        safe.load_state(tmp_path / 'state')


def test_actual_trainer_resume_preserves_updates_and_data_order(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM, TrainerCallback, default_data_collator
    from llamafactory.hparams import FinetuningArguments, TrainingArguments
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    safe.install_safe_resume()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=8)
    initial = Qwen3ForCausalLM(cfg).state_dict()
    data = [{'input_ids': torch.tensor([1, i + 2, 10, 11, 12]),
             'labels': torch.tensor([1, i + 2, 10, 11, 12])} for i in range(8)]

    class StopAtTwo(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_save = control.should_training_stop = True
            return control

    def make(name, callbacks=()):
        model = Qwen3ForCausalLM(cfg)
        model.load_state_dict(initial)
        seen = []
        model.register_forward_pre_hook(lambda module, args, kwargs: seen.append(kwargs['input_ids'].tolist()), with_kwargs=True)
        args = TrainingArguments(output_dir=str(tmp_path / name), use_cpu=True,
            max_steps=4, per_device_train_batch_size=1, gradient_accumulation_steps=1,
            learning_rate=1e-3, save_steps=2, logging_steps=1, report_to=[],
            disable_tqdm=True, seed=42, data_seed=42, ignore_data_skip=False)
        trainer = CustomSeq2SeqTrainer(model=model, args=args, tokenizer=None, processor=None,
            finetuning_args=FinetuningArguments(finetuning_type='full'),
            train_dataset=data, data_collator=default_data_collator, callbacks=list(callbacks))
        return trainer, seen

    full, full_seen = make('full')
    full.train()
    part, first_seen = make('split', [StopAtTwo()])
    part.train()
    checkpoint = tmp_path / 'split/checkpoint-2'
    assert (checkpoint / 'optimizer_safe.json').exists()
    assert (checkpoint / 'rng_safe.json').exists()
    assert not (checkpoint / 'optimizer.pt').exists()
    resumed, second_seen = make('resumed')
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 4
    assert full_seen == first_seen + second_seen
    assert all(torch.equal(v, resumed.model.state_dict()[k]) for k, v in full.model.state_dict().items())
    assert full.lr_scheduler.get_last_lr() == resumed.lr_scheduler.get_last_lr()
