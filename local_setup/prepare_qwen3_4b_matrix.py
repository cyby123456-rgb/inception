"""Materialize the three-arm, three-seed Qwen3-4B experiment without launching jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import yaml
from safetensors import safe_open
from qwen3_4b_recipe import architecture, budgets, split_data, stage_config
from rescore_numeric import extract_numeric
from training_contract import validate_scope
from matrix_audit import write_protocol_manifest, audit_suite

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def audit_model(model, recipe):
    config = json.loads((model/'config.json').read_text())
    arch = architecture(config, recipe)
    index = json.loads((model/'model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    shapes, shards = {}, []
    for name in sorted(set(weight_map.values())):
        path = model/name
        with safe_open(path, framework='pt', device='cpu') as f:
            keys = set(f.keys())
            required = {k for k, v in weight_map.items() if v == name}
            if not required <= keys:
                raise ValueError(f'Missing tensors in {name}')
            for k in keys:
                if k.endswith(('embed_tokens.weight', 'layers.33.self_attn.q_proj.weight',
                               'layers.33.self_attn.k_proj.weight', 'layers.33.mlp.up_proj.weight')):
                    shapes[k] = f.get_slice(k).get_shape()
        shards.append(dict(file=name, bytes=path.stat().st_size))
    if shapes.get('model.embed_tokens.weight') != [arch['vocab_size'], arch['hidden_size']]:
        raise ValueError('Embedding shape does not match config')
    if shapes.get('model.layers.33.self_attn.q_proj.weight') != [4096, 2560]:
        raise ValueError('4B attention width must come from head_dim, not hidden_size / num_heads')
    return dict(architecture=arch, tensor_shapes=shapes, shards=shards,
                config_sha256=hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),
                tied_lm_head_omitted_from_shards='lm_head.weight' not in weight_map,
                official_config='https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json')


def stops(stage, steps, arm, non_joint):
    if stage != 'joint':
        return [steps]
    values = list(range(500, steps+1, 500)) if arm == 'post_joint' else [1000, 2000, 4000, *range(5000, steps+1, 5000), non_joint]
    return sorted({x for x in [*values, steps] if 0 < x <= steps})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'train-data', 'test-data', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--recipe', type=Path, default=ROOT/'configs/experimental/qwen3_4b_matrix.yaml')
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--joint-steps', type=int, help='Optional common continuation budget; preserves the 2,000-step milestone')
    a = p.parse_args()
    recipe = yaml.safe_load(a.recipe.read_text())
    seeds = [int(x) for x in a.seeds.split(',')]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        p.error('Exactly three distinct seeds are required')
    recipe['seeds'] = seeds
    if a.joint_steps is not None:
        recipe['joint_steps'] = a.joint_steps
    model, source, test, run = (x.resolve() for x in [a.model, a.train_data, a.test_data, a.output])
    if run.exists():
        p.error('Output must be a new directory')
    audit = audit_model(model, recipe)
    rows = json.loads(source.read_text())
    train, valid, split = split_data(rows, recipe['validation_queries'], recipe['split_salt'])
    budget = budgets(len(train), recipe)
    tests = [json.loads(s) for s in test.read_text().splitlines() if s.strip()]
    if len(tests) < max(recipe['final']['start']+recipe['final']['samples'], recipe['final']['samples']+1):
        p.error('Not enough final test questions')
    if any(not isinstance(x.get('question'), str) or 'answer' not in x for x in tests):
        p.error('Test JSONL requires question and answer')
    run.mkdir(parents=True, exist_ok=False)
    dataset = run/'data'
    dataset.mkdir()
    dump(dataset/'train.json', train)
    dump(dataset/'validation.json', valid)
    dump(dataset/'dataset_info.json', {name: {'file_name': str(dataset/file),
        'columns': {'prompt': 'query', 'response': 'response'}}
        for name, file in [('matrix_train', 'train.json'), ('matrix_validation', 'validation.json')]})
    numeric_validation_rows = 0
    with (dataset/'validation_inference.jsonl').open('w') as f:
        for row in valid:
            answer, _ = extract_numeric(row['response'])
            if answer is None:
                continue
            f.write(json.dumps(dict(question=row['query'], answer=answer), ensure_ascii=False)+'\n')
            numeric_validation_rows += 1
    if numeric_validation_rows < recipe['periodic']['samples']+1:
        raise ValueError('Not enough numeric validation examples for monitoring')
    split['numeric_validation_rows'] = numeric_validation_rows
    split.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 test_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
                 train_sha256=hashlib.sha256((dataset/'train.json').read_bytes()).hexdigest(),
                 validation_sha256=hashlib.sha256((dataset/'validation.json').read_bytes()).hexdigest())
    dump(run/'data_audit.json', split)
    dump(run/'model_audit.json', audit)
    for folder in ['LLaMA-Factory', 'scripts', 'local_setup', 'configs']:
        shutil.copytree(ROOT/folder, run/'source_snapshot'/folder,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache', '.git'))
    hashes = {str(f.relative_to(run/'source_snapshot')): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in (run/'source_snapshot').rglob('*') if f.is_file()}
    dump(run/'source_manifest.json', hashes)
    experiments = []
    for seed in seeds:
        non_joint_ckpt = run/f'no_joint/seed_{seed}/head/checkpoints'
        for arm in ['no_joint', 'post_joint', 'full_joint']:
            specs = [('core', budget['core'], True, 'target_t'),
                     ('multistep', budget['multistep'], True, 'target_t'),
                     ('head_warmup', budget['head_warmup'], False, 'head'),
                     ('head', budget['head_main'], False, 'head')] if arm=='no_joint' else [
                         ('joint', budget[arm] if arm=='full_joint' else budget['joint'],
                          arm=='full_joint', 'target_t_head' if arm=='full_joint' else 't_head')]
            previous = non_joint_ckpt if arm=='post_joint' else None
            stages = []
            for stage, count, target_trainable, scope in specs:
                directory = run/f'{arm}/seed_{seed}'/stage
                directory.mkdir(parents=True)
                cfg = stage_config(model, dataset, directory/'checkpoints', seed, stage, count,
                                   audit['architecture'], recipe, target_trainable)
                (directory/'train.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
                stages.append(dict(name=stage, directory=str(directory), config=str(directory/'train.yaml'),
                    steps=count, scope=scope, initialize_from=str(previous) if previous else None,
                    training_regime=validate_scope(cfg, scope),
                    stop_steps=stops(stage, count, arm, budget['non_joint']),
                    benchmark=stage=='head' or stage=='joint'))
                previous = directory/'checkpoints'
            experiments.append(dict(arm=arm, seed=seed, stages=stages,
                family={'no_joint':'staged_target_then_boundary', 'post_joint':'trained_target_frozen_joint',
                        'full_joint':'from_base_trainable_target_joint'}[arm],
                dependency=f'no_joint/seed_{seed}' if arm=='post_joint' else None,
                total_updates=budget['non_joint'] if arm=='no_joint' else budget['full_joint']))
    plan = dict(created_unix=time.time(), recipe=recipe, model=str(model), test_data=str(test),
        validation_inference=str(dataset/'validation_inference.jsonl'), budget=budget,
        experiments=experiments, status='prepared_not_started',
        target_policy='Original target weights frozen throughout. Target LoRA+T train in staged core/multistep; head-only afterwards. Post-checkpoint joint freezes target LoRA. Full joint trains target LoRA+T+head from step one.',
        evaluation_policy='Held-out MetaMath for monitoring; GSM8K test only at prespecified final endpoints. Each checkpoint has its own paired greedy.',
        budget_limit='A is a shorter baseline; B and C have equal update budgets. C also saves the A-budget checkpoint. Equal updates do not imply equal FLOPs or training wall time.')
    dump(run/'plan.json', plan)
    write_protocol_manifest(run, plan)
    dump(run/'training_regimes.json', audit_suite(run, plan))
    for arm in ['all', 'no_joint', 'post_joint', 'full_joint']:
        script = run/f'run_{arm}.sh'
        script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nSUITE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)\n'
            'exec "${PYTHON:-python}" "$SUITE/source_snapshot/local_setup/run_qwen3_4b_matrix.py" '
            f'--suite "$SUITE" --arms {arm} "$@"\n')
        script.chmod(0o755)
    print(json.dumps(dict(suite=str(run), model=audit['architecture'], budget=budget,
                         experiments=len(experiments), data=split), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
