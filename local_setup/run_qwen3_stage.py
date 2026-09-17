"""Run one Qwen3 RecurFT stage with telemetry and explicit weights-only transitions."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import environment, gpu_snapshot, write_json, sha256
from training_contract import make_contract, validate_resume, validate_initialization_layout

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--initialize-from', type=Path)
    parser.add_argument('--resume-from', type=Path, help='Resume this stage including optimizer/RNG from a full checkpoint.')
    parser.add_argument('--allow-shared-gpu', action='store_true', help='Explicit shared-GPU training; preserve other processes.')
    parser.add_argument('--stop-after-step', type=int, help='Save and stop at this global step without changing the full optimizer/scheduler budget.')
    parser.add_argument('--export-initial', type=Path, help='Save the untrained adapter/T/head once, before the first optimizer update.')
    parser.add_argument('--min-free-mib', type=int, help='Model-specific memory preflight; default retains the 8B thresholds.')
    parser.add_argument('--memory-fraction', type=float, help='Optional CUDA allocator fraction for this process.')
    parser.add_argument('--trainable-scope', choices=['target_t', 'head', 't_head', 'target_t_head'])
    parser.add_argument('--require-tied-embeddings', action='store_true')
    args = parser.parse_args()
    output = args.config.resolve().parent
    if args.resume_from and args.initialize_from:
        raise ValueError('Choose stage initialization or exact within-stage resume, not both.')
    cfg = yaml.safe_load(args.config.read_text())
    contract = make_contract(cfg, args.trainable_scope)
    resume_audit = None
    if args.resume_from:
        resume_audit = validate_resume(args.resume_from, contract, required=bool(args.trainable_scope))
    if args.initialize_from and args.trainable_scope:
        validate_initialization_layout(args.initialize_from, cfg, args.trainable_scope)
    if (output / 'execution.json').exists():
        if not args.resume_from:raise RuntimeError('Refusing to overwrite an existing execution.')
        previous=json.loads((output/'execution.json').read_text())
        if previous.get('status')=='running' and Path('/proc',str(previous.get('pid'))).exists():
            raise RuntimeError('Recorded stage process is still alive; refusing concurrent resume.')
        (output/'execution.json').rename(output/('execution_previous_'+str(time.time_ns())+'.json'))
    before = gpu_snapshot(args.gpu)
    minimum_free = args.min_free_mib or (45000 if args.allow_shared_gpu else 60000)
    if minimum_free <= 0 or (args.memory_fraction is not None and not 0 < args.memory_fraction <= 1):
        raise ValueError('Invalid GPU memory preflight/fraction')
    # nvidia-smi utilization is a trailing sample and can outlive the preceding stage.
    for _ in range(0 if args.allow_shared_gpu else 15):
        if before['free_mib'] >= minimum_free and before['util'] <= 10:
            break
        time.sleep(2)
        before = gpu_snapshot(args.gpu)
    if args.allow_shared_gpu:
        assert before['free_mib'] >= minimum_free, before
    else:
        assert before['free_mib'] >= minimum_free and before['util'] <= 10, before
    import subprocess
    daemons = []
    for pid in ([] if args.allow_shared_gpu else before['compute_pids']):
        proc = Path('/proc') / str(pid)
        assert Path(os.readlink(proc / 'exe')).name == 'nvidia-cuda-mps-server', before
        entries = (proc / 'environ').read_bytes().split(b'\0')
        pipe = next(x.split(b'=', 1)[1].decode() for x in entries if x.startswith(b'CUDA_MPS_PIPE_DIRECTORY='))
        clients = subprocess.run(['nvidia-cuda-mps-control'], input=f'get_client_list {pid}\n',
                                 env=dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=pipe), text=True,
                                 capture_output=True, timeout=10, check=True)
        assert not clients.stdout.strip() and not clients.stderr.strip(), clients
        daemons.append({'pid': pid, 'clients_at_preflight': clients.stdout})
    os.environ.update(environment(args.gpu))
    os.environ.pop('CUDA_MPS_PIPE_DIRECTORY', None)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path[:0] = os.environ['PYTHONPATH'].split(os.pathsep)
    import torch
    if args.memory_fraction is not None or args.allow_shared_gpu:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction or .50)
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp
    from safe_training_state import install_safe_resume
    install_safe_resume()

    if args.stop_after_step is not None:
        assert 0 < args.stop_after_step <= cfg['max_steps']
    if args.resume_from:
        assert (args.resume_from/'optimizer_safe.json').exists() and (args.resume_from/'rng_safe.json').exists() and (args.resume_from/'trainer_state.json').exists()
        cfg['resume_from_checkpoint']=str(args.resume_from.resolve())
    record = {'status': 'running', 'pid': os.getpid(), 'config': str(args.config.resolve()),
              'config_sha256': sha256(args.config), 'gpu_before': before, 'idle_mps': daemons,
              'training_regime': contract['description'], 'resume_contract_audit': resume_audit,
              'timing_scope': 'shared-GPU training, other processes left running' if args.allow_shared_gpu else 'training; idle MPS left running, not an exclusive benchmark',
              'initialization_checkpoint': str(args.initialize_from.resolve()) if args.initialize_from else None,
              'resume_checkpoint': str(args.resume_from.resolve()) if args.resume_from else None,
              'transition_policy': 'load adapter and recurrent safetensors; fresh optimizer, scheduler, stage-local step',
              'gradient_accumulation_fix': 'RecurFT model_accepts_loss_kwargs=False; Trainer divides microbatch mean by actual accumulation length',
              'model': cfg['model_name_or_path'], 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    if args.resume_from:record['transition_policy']='Trainer restores adapter, recurrent, optimizer, scheduler and RNG; original stage-local step retained.'
    write_json(output / 'execution.json', record)
    started = time.monotonic()

    class Telemetry(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            record['starting_global_step'] = state.global_step
            record['ignore_data_skip'] = args.ignore_data_skip
            optimizer = kwargs.get('optimizer')
            optimizer_steps = [float(v['step']) for v in optimizer.state.values() if 'step' in v] if optimizer is not None else []
            record['optimizer_steps_at_start'] = sorted(set(optimizer_steps))
            if record['resume_checkpoint']:
                assert not args.ignore_data_skip, 'Resume must preserve data progress'
                assert optimizer_steps and all(x == state.global_step for x in optimizer_steps), 'Optimizer update count was not restored'
            record['observed_max_sequence_length']=0
            def observe_shape(_model,_args,kwargs):
                ids=kwargs.get('input_ids')
                if ids is not None:
                    record['observed_max_sequence_length']=max(record['observed_max_sequence_length'],ids.shape[-1])
            self.shape_hook=model.register_forward_pre_hook(observe_shape,with_kwargs=True)
            if initial_export is not None:
                assert state.global_step == 0 and not record['initialization_checkpoint'] and not record['resume_checkpoint']
                from peft import get_peft_model_state_dict
                from llamafactory.model.model_utils.recurft import save_recurft_recurrent_module
                initial_export.mkdir(parents=True, exist_ok=False)
                adapter = get_peft_model_state_dict(model)
                assert all(torch.count_nonzero(v).item() == 0 for k, v in adapter.items() if 'lora_B' in k), 'Target must start as the unchanged base model'
                model.save_pretrained(initial_export, safe_serialization=True)
                save_recurft_recurrent_module(model, str(initial_export), safe_serialization=True)
                processor = kwargs.get('processing_class') or kwargs.get('tokenizer')
                assert processor is not None, 'Missing training tokenizer for initial export'
                processor.save_pretrained(initial_export)
                state.save_to_json(str(initial_export/'trainer_state.json'))
                record['initial_export'] = str(initial_export)
                record['initial_target_lora_B_all_zero'] = True
            if record['initialization_checkpoint']:
                from peft import get_peft_model_state_dict, set_peft_model_state_dict
                from safetensors.torch import load_file
                source = Path(record['initialization_checkpoint'])
                adapter = load_file(str(source/'adapter_model.safetensors'))
                assert set(adapter) == set(get_peft_model_state_dict(model)), 'Adapter layout mismatch'
                set_peft_model_state_dict(model, adapter)
                current = get_peft_model_state_dict(model)
                assert all(torch.equal(current[k].detach().cpu(), v.to(current[k].dtype)) for k,v in adapter.items())
                recurrent = load_file(str(source/'recurft_recurrent.safetensors'))
                # Frozen T LoRA must retain checkpoint precision too; casting it to
                # BF16 merely because it is frozen changes the branch before training.
                preserved=[]
                for name,p in model.recurft_recurrent.named_parameters():
                    if name in recurrent and p.dtype!=recurrent[name].dtype:
                        p.data=p.data.to(recurrent[name].dtype);preserved.append(name)
                record['restored_checkpoint_parameter_dtypes']=preserved
                current_recurrent = model.recurft_recurrent.state_dict()
                assert set(recurrent) <= set(current_recurrent), 'Unexpected recurrent checkpoint keys'
                model.recurft_recurrent.load_state_dict(recurrent, strict=False)
                assert all(torch.equal(current_recurrent[k].detach().cpu(), v.to(current_recurrent[k].dtype)) for k,v in recurrent.items())
                record['restored_adapter_tensors'] = len(adapter)
                record['restored_recurrent_tensors'] = len(recurrent)
                record['new_recurrent_tensor_names'] = [k for k in current_recurrent if k not in recurrent and k.startswith('boundary_')]
                record['weights_reload_exact_after_dtype_cast'] = True
                del current, current_recurrent, adapter, recurrent
            self.initial = {}
            groups = set()
            for name, p in model.named_parameters():
                if p.requires_grad and ('lora_B' in name or 'boundary_' in name):
                    group = 'boundary_head' if 'boundary_' in name else ('recurrent' if 'recurft' in name else 'base_adapter')
                    if group not in groups:
                        self.initial[name] = p.detach().float().cpu().clone()
                        groups.add(group)
            record['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            record['total_parameters_including_recurrent_copy'] = sum(p.numel() for p in model.parameters())
            record['model_class'] = type(model).__name__
            record['hidden_size'] = model.config.hidden_size
            record['num_hidden_layers'] = model.config.num_hidden_layers
            import llamafactory.train.sft.recurft as loss_module
            record['loaded_loss_module'] = loss_module.__file__
            assert str(ROOT) in record['loaded_loss_module']
            record['trainable_groups'] = {}
            for name,p in model.named_parameters():
                group = 'boundary' if 'boundary_' in name else 'T' if 'recurft' in name else 'target'
                if p.requires_grad:record['trainable_groups'][group]=record['trainable_groups'].get(group,0)+p.numel()
            expected = {'target_t': {'target', 'T'}, 'head': {'boundary'},
                        't_head': {'T', 'boundary'}, 'target_t_head': {'target', 'T', 'boundary'}}
            if cli_scope:
                assert set(record['trainable_groups']) == expected[cli_scope], record['trainable_groups']
            if require_tied:
                inp, out = model.get_input_embeddings().weight, model.get_output_embeddings().weight
                assert inp.data_ptr() == out.data_ptr(), 'Qwen3-4B tied embedding/output weights were detached'
                assert not inp.requires_grad and not out.requires_grad, 'Original tied vocabulary weights must remain frozen'
                record['tied_vocabulary_weights_frozen'] = True
            if cfg.get('recurft_recurrent_trainable_only') or cfg.get('recurft_stage1_heads_only'):
                assert record['trainable_groups'].get('target',0)==0,record['trainable_groups']
            record['gradient_checks']=[]
            write_json(output / 'execution.json', record)

        def on_step_end(self, args, state, control, **kwargs):
            if stop_after_step is not None and state.global_step >= stop_after_step:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            write_json(Path(args.output_dir)/f'checkpoint-{state.global_step}'/'training_contract.json', contract)

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            if state.global_step>=3:return
            groups={}
            for name,p in model.named_parameters():
                if p.grad is None:continue
                group='boundary' if 'boundary_' in name else 'T' if 'recurft' in name else 'target'
                groups[group]=groups.get(group,0.0)+p.grad.detach().float().square().sum().item()
            if cfg.get('recurft_recurrent_trainable_only') or cfg.get('recurft_stage1_heads_only'):
                assert groups.get('target',0.0)==0.0,'Frozen target unexpectedly received gradients'
            record['gradient_checks'].append({'step':state.global_step,'gradient_l2':{k:v**.5 for k,v in groups.items()}})
            write_json(output/'execution.json',record)

        def on_log(self, args, state, control, logs=None, **kwargs):
            row = dict(logs or {}, global_step=state.global_step,
                       observed_max_sequence_length=record['observed_max_sequence_length'],
                       allocated_gib=torch.cuda.memory_allocated()/2**30,
                       reserved_gib=torch.cuda.memory_reserved()/2**30,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            with (output / 'telemetry.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            import math
            for key in ('loss', 'eval_loss', 'grad_norm'):
                if key in row and not math.isfinite(row[key]):
                    raise FloatingPointError(f'Nonfinite {key} at step {state.global_step}')

        def on_train_end(self, args, state, control, model=None, **kwargs):
            self.shape_hook.remove()
            parameters = dict(model.named_parameters())
            record['sampled_adapter_changes'] = {name: {
                'delta_l2': (parameters[name].detach().float().cpu()-initial).norm().item(),
                'final_l2': parameters[name].detach().float().norm().item()
            } for name, initial in self.initial.items()}
            record['global_step'] = state.global_step
            record['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
            record['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
            write_json(Path(args.output_dir)/'training_contract.json', contract)

    stop_after_step = args.stop_after_step
    cli_scope = args.trainable_scope
    require_tied = args.require_tied_embeddings
    initial_export = args.export_initial
    record['requested_stop_after_step'] = stop_after_step
    record['full_training_max_steps'] = cfg['max_steps']
    record['checkpoint_format'] = 'safetensors+JSON for optimizer, scheduler and RNG; no pickle loading'
    if not args.initialize_from and not args.resume_from:
        record['transition_policy'] = 'Fresh base-model weights and new T/head adapters; no previous training checkpoint loaded.'
    try:
        run_exp(args=cfg, callbacks=[Telemetry()])
        record['status'] = 'completed'
    except BaseException as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        record['wall_seconds_including_loading_and_saving'] = time.monotonic()-started
        record['ended_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        write_json(output / 'execution.json', record)

if __name__ == '__main__':
    main()
