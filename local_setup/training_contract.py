"""Explicit training scopes and immutable within-stage resume identity.

This file does not import the model runtime. A transition may change the
objective/scope; an exact resume may not. Historical runners without an explicit
scope remain usable, but are marked unverified if their checkpoint lacks this
contract. Prepared 4B runs require the contract.
"""
import hashlib
import json
from pathlib import Path

REGIMES = {'target_t': 'staged_target_t', 'head': 'boundary_only',
           't_head': 'frozen_target_joint', 'target_t_head': 'trainable_target_joint'}
NON_SEMANTIC = {'output_dir', 'overwrite_output_dir', 'resume_from_checkpoint',
                'logging_steps', 'save_steps', 'save_total_limit', 'report_to', 'plot_loss'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_scope(config, scope):
    if scope not in REGIMES:
        raise ValueError(f'Unknown training scope: {scope}')
    if not config.get('use_recurft') or config.get('finetuning_type') != 'lora':
        raise ValueError('Named training scopes require RecurFT LoRA training')
    heads = bool(config.get('recurft_stage1_heads_only', False))
    recurrent_only = bool(config.get('recurft_recurrent_trainable_only', False))
    if (heads, recurrent_only) != ((scope=='head'), (scope=='t_head')):
        raise ValueError(f'Freeze flags disagree with training scope {scope}')
    for name in ('multistep_residual_rank', 'multistep_step1_residual_rank', 'token_conditioning_rank'):
        if config.get('recurft_'+name, 0):
            raise ValueError(f'{name} is a separate experimental regime, not {scope}')
    for name in ('multistep_residual_only', 'multistep_step1_residual_only'):
        if config.get('recurft_'+name, False):
            raise ValueError(f'{name} conflicts with {scope}')
    mode = config.get('recurft_joint_mode', 'legacy')
    if mode != ('trainable_target' if scope=='target_t_head' else 'legacy'):
        raise ValueError(f'Joint mode disagrees with training scope {scope}')
    if config.get('recurft_t_lora_rank', 0) <= 0 or config.get('recurft_recurrent_loss_weight', 0) <= 0:
        raise ValueError('Named scopes require a positive T rank and recurrent gate weight')
    rank = config.get('recurft_boundary_head_rank', 0)
    if (rank > 0) != (scope != 'target_t'):
        raise ValueError(f'Boundary head presence disagrees with {scope}')
    direct = sum(config.get('recurft_boundary_'+kind+'_loss_weight', 0) for kind in ['token_ce','logit_kl'])
    rollout = (config.get('recurft_multistep_loss_weight', 0)>0 and
               config.get('recurft_multistep_steps', 1)>1 and
               sum(config.get('recurft_multistep_boundary_'+kind+'_loss_weight', 0) for kind in ['token_ce','logit_kl'])>0)
    if scope=='head' and direct <= 0:
        raise ValueError('Boundary-only training needs an active direct head objective')
    if scope in ('t_head','target_t_head') and not rollout:
        raise ValueError('Joint T/head training needs an active rollout head objective')
    return dict(regime=REGIMES[scope], scope=scope,
                original_base_trainable=False, target_lora_trainable=scope in ('target_t','target_t_head'),
                T_lora_trainable=scope!='head', boundary_trainable=scope!='target_t',
                direct_boundary_teacher=config.get('recurft_boundary_teacher_source','reference'),
                rollout_teacher='detached_current_target', target_sft_ce=config.get('recurft_sft_loss_weight',0))


def make_contract(config, scope):
    description = validate_scope(config, scope) if scope else {'regime':'legacy_unclassified'}
    files = {}
    model_config = Path(config['model_name_or_path'])/'config.json'
    if model_config.is_file():
        files[str(model_config.resolve())] = sha(model_config)
    directory = Path(config.get('dataset_dir', 'data'))
    info = directory/'dataset_info.json'
    if info.is_file():
        files[str(info.resolve())] = sha(info)
        mapping = json.loads(info.read_text())
        for key in ('dataset','eval_dataset'):
            names = config.get(key) or []
            names = names.split(',') if isinstance(names,str) else names
            for name in names:
                entry = mapping.get(name.strip(), {})
                if 'file_name' in entry:
                    path = directory/entry['file_name']
                    if path.is_file():
                        files[str(path.resolve())] = sha(path)
    return dict(format='recurft_training_contract_v1', description=description,
                config={k:v for k,v in config.items() if k not in NON_SEMANTIC}, input_sha256=files)


def validate_resume(checkpoint, current, required):
    path = Path(checkpoint)/'training_contract.json'
    if not path.exists():
        if required:
            raise ValueError('Exact resume requires training_contract.json; use the original source snapshot for a legacy checkpoint')
        return 'legacy_checkpoint_without_contract'
    old = json.loads(path.read_text())
    if old != current:
        changed = [k for k in sorted(set(old.get('config',{})) | set(current['config']))
                   if old.get('config',{}).get(k) != current['config'].get(k)]
        raise ValueError(f'Exact resume changes training contract ({changed}); use a new stage and --initialize-from')
    return 'exact_contract_match'


def validate_initialization_layout(checkpoint, config, scope):
    metadata = json.loads((Path(checkpoint)/'recurft_config.json').read_text())
    for key in ('loop_start_layer','loop_end_layer','pre_lora_rank','last_lora_rank','t_lora_rank'):
        requested = config.get('recurft_'+key)
        if requested is not None and requested != metadata.get(key):
            raise ValueError(f'Initialization layout mismatch: {key}')
    for prefix in ('pre','last','t'):
        rank = config.get(f'recurft_{prefix}_lora_rank')
        if rank is not None:
            alpha = config.get(f'recurft_{prefix}_lora_alpha') or rank*2
            if alpha != metadata.get(f'{prefix}_lora_alpha', alpha):
                raise ValueError(f'Initialization LoRA scaling mismatch: {prefix}_lora_alpha')
    old_head, new_head = metadata.get('boundary_head_rank',0), config.get('recurft_boundary_head_rank',0)
    if old_head != new_head and not (scope=='head' and old_head==0 and new_head>0):
        raise ValueError('Only head-stage initialization may add a new boundary head')
