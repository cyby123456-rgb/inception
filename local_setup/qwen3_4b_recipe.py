"""Model-aware recipes and deterministic, query-disjoint validation split."""
import hashlib
import math
from pathlib import Path


def architecture(model_config, recipe):
    expected = dict(model_type='qwen3', num_hidden_layers=36, hidden_size=2560,
                    intermediate_size=9728, num_attention_heads=32,
                    num_key_value_heads=8, head_dim=128, tie_word_embeddings=True)
    for key, value in expected.items():
        if model_config.get(key) != value:
            raise ValueError(f'Expected Qwen3-4B {key}={value}, got {model_config.get(key)!r}')
    cap = recipe['capacity']
    width, multiple = model_config['hidden_size'], cap['rank_multiple']
    rank = lambda ratio: max(multiple, round(width * ratio / multiple) * multiple)
    end = model_config['num_hidden_layers'] - cap['target_tail_layers'] - 1
    start = end - cap['t_layers'] + 1
    tr, br = rank(cap['t_rank_per_hidden']), rank(cap['boundary_rank_per_hidden'])
    q = model_config['num_attention_heads'] * model_config['head_dim']
    kv = model_config['num_key_value_heads'] * model_config['head_dim']
    ff = model_config['intermediate_size']
    t_parameters = cap['t_layers'] * tr * (2 * (width + q) + 2 * (width + kv) + 3 * (width + ff))
    return dict(**expected, vocab_size=model_config['vocab_size'], loop_start=start,
                loop_end=end, anchor_hidden_index=start, t_rank=tr, t_alpha=2*tr,
                boundary_rank=br, t_trainable_parameters=t_parameters,
                boundary_trainable_parameters=2*width*br + 2*width)


def query_key(row):
    return ' '.join(row['query'].split())


def split_data(rows, n, salt):
    groups = {}
    for i, row in enumerate(rows):
        if not isinstance(row.get('query'), str) or not isinstance(row.get('response'), str):
            raise ValueError(f'Row {i} must have string query/response fields')
        key = query_key(row)
        if not key or not row['response'].strip():
            raise ValueError(f'Empty training row {i}')
        groups.setdefault(key, i)
    if len(groups) <= n:
        raise ValueError('Not enough distinct queries for train and validation')
    chosen = set(sorted(groups, key=lambda k: hashlib.sha256((salt+'\0'+k).encode()).digest())[:n])
    validation = [rows[groups[k]] for k in sorted(chosen)]
    training = [row for row in rows if query_key(row) not in chosen]
    return training, validation, dict(input_rows=len(rows), training_rows=len(training),
        validation_rows=len(validation), excluded_duplicate_validation_rows=len(rows)-len(training)-len(validation),
        validation_query_sha256=hashlib.sha256('\n'.join(sorted(chosen)).encode()).hexdigest())


def budgets(training_rows, recipe):
    if recipe['effective_batch'] % recipe['micro_batch']:
        raise ValueError('effective_batch must be divisible by micro_batch')
    result = {k: math.ceil(training_rows * recipe[k+'_epochs'] / recipe['effective_batch'])
              for k in ['core', 'multistep', 'head']}
    result['non_joint'] = sum(result.values())
    result['head_warmup'] = max(1, math.ceil(result['head'] * .225))
    result['head_main'] = result['head'] - result['head_warmup']
    result['joint'] = int(recipe['joint_steps'])
    if result['joint'] < 2000:
        raise ValueError('Keep at least the prespecified 2,000-step joint checkpoint')
    result['full_joint'] = result['non_joint'] + result['joint']
    return result


def stage_config(model, dataset_dir, output, seed, stage, steps, arch, recipe, target_trainable=False):
    joint = stage == 'joint'
    head = stage in ('head', 'head_warmup')
    core = stage == 'core'
    cfg = dict(model_name_or_path=str(Path(model).resolve()), trust_remote_code=False,
        stage='sft', do_train=True, do_eval=True, finetuning_type='lora', lora_target='all',
        use_recurft=True, recurft_loop_start_layer=arch['loop_start'], recurft_loop_end_layer=arch['loop_end'],
        recurft_pre_lora_rank=recipe['capacity']['target_lora_rank'],
        recurft_last_lora_rank=recipe['capacity']['target_lora_rank'],
        recurft_t_lora_rank=arch['t_rank'], recurft_t_lora_alpha=arch['t_alpha'],
        recurft_boundary_head_rank=arch['boundary_rank'] if (joint or head) else 0,
        recurft_token_conditioning_rank=0, recurft_stage1_heads_only=head,
        recurft_joint_mode='trainable_target' if joint and target_trainable else 'legacy',
        recurft_boundary_teacher_source='target',
        recurft_recurrent_trainable_only=not target_trainable and not head,
        recurft_hidden_loss_weight=20.0 if target_trainable else 0.0,
        recurft_hidden_relative_mse_loss_weight=1.0 if target_trainable else 0.0,
        recurft_hidden_cosine_loss_weight=1.0 if target_trainable else 0.0,
        recurft_kl_loss_weight=0.2 if target_trainable else 0.0, recurft_sft_loss_weight=0.0,
        recurft_recurrent_loss_weight=0.1 if joint else 1.0,
        recurft_recurrent_relative_mse_loss_weight=0.05, recurft_recurrent_cosine_loss_weight=0.05,
        recurft_recurrent_warmup_steps=recipe['core_recurrent_warmup_steps'] if core else 0,
        recurft_multistep_steps=4 if stage=='multistep' else 2,
        recurft_multistep_loss_weight=0.1 if joint else 0.0005 if stage=='multistep' else 0.0,
        recurft_multistep_boundary_logit_kl_loss_weight=10.0 if joint else 0.0,
        recurft_multistep_boundary_token_ce_loss_weight=0.0,
        recurft_boundary_token_ce_loss_weight=1.0 if head else 0.0,
        recurft_boundary_logit_kl_loss_weight=0.2 if head else 0.0,
        recurft_boundary_loss_stride=8, recurft_boundary_loss_max_tokens=64,
        recurft_multistep_warmup_steps=200 if joint else 1000 if stage=='multistep' else 0,
        recurft_multistep_warmup_start_step=0, recurft_multistep_stride=64,
        recurft_multistep_max_starts=4, recurft_multistep_context_tokens=128 if stage=='multistep' else 0,
        recurft_multistep_step_decay=0.7 if stage=='multistep' else 1.0, recurft_multistep_huber_beta=0.1,
        recurft_multistep_delta_cosine_loss_weight=recipe['multistep_delta_cosine_weight'] if stage=='multistep' else 0.0,
        recurft_multistep_start_selection='linspace', recurft_multistep_detach_rollout=stage=='multistep',
        recurft_scheduled_sampling_max_ratio=0.0, recurft_loss_on_labels_only=True,
        dataset='matrix_train', eval_dataset='matrix_validation', dataset_dir=str(dataset_dir),
        template='qwen3', enable_thinking=False, cutoff_len=recipe['cutoff_len'],
        output_dir=str(output), overwrite_output_dir=False, overwrite_cache=False,
        max_steps=steps, per_device_train_batch_size=recipe['micro_batch'],
        gradient_accumulation_steps=recipe['effective_batch']//recipe['micro_batch'],
        per_device_eval_batch_size=1, prediction_loss_only=True,
        learning_rate=recipe['learning_rates'][stage], lr_scheduler_type='constant',
        warmup_steps=0, warmup_ratio=0.0, optim='adamw_torch', weight_decay=0.0,
        max_grad_norm=1.0, bf16=True, flash_attn='sdpa', disable_gradient_checkpointing=False,
        seed=seed, data_seed=seed, ignore_data_skip=False, logging_steps=25,
        save_steps=500, save_total_limit=3, save_only_model=False, report_to='none',
        plot_loss=False, dataloader_num_workers=0, preprocessing_num_workers=2,
        eval_strategy='steps', eval_steps=500 if joint else 1000)
    if recipe['target_policy'] != 'original_staged':
        raise ValueError('This matrix follows the user-selected original staged target policy')
    return cfg
