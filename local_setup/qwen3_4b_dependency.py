"""Publish/import an immutable staged checkpoint for a remote continuation."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import tempfile

import yaml
from training_contract import NON_SEMANTIC, sha, validate_initialization_layout

REQUIRED = {
    'adapter_model.safetensors', 'adapter_config.json',
    'recurft_recurrent.safetensors', 'recurft_config.json',
    'tokenizer.json', 'tokenizer_config.json', 'training_contract.json', 'audit.json',
}


def identity(suite, plan, seed):
    """Ignore machine-specific paths, retain objectives, data, seed and training code."""
    suite = Path(suite)
    exp = next(e for e in plan['experiments'] if e['arm'] == 'no_joint' and e['seed'] == seed)
    ignored = NON_SEMANTIC | {'model_name_or_path', 'dataset_dir'}
    configs = []
    for stage in exp['stages']:
        config = yaml.safe_load(Path(stage['config']).read_text())
        configs.append({k: v for k, v in config.items() if k not in ignored})
    sources = json.loads((suite/'source_manifest.json').read_text())
    code = {k: v for k, v in sources.items() if k.startswith('LLaMA-Factory/src/') or k in (
        'local_setup/run_qwen3_stage.py', 'local_setup/training_contract.py',
        'local_setup/safe_training_state.py')}
    return dict(seed=seed, total_updates=exp['total_updates'], configs=configs,
                model_config=sha(Path(plan['model'])/'config.json'),
                train=sha(suite/'data/train.json'), validation=sha(suite/'data/validation.json'),
                training_sources=code)


def bundle_path(directory, seed):
    return Path(directory)/f'no_joint_seed_{seed}'


def verify_bundle(bundle):
    bundle = Path(bundle)
    info = json.loads((bundle/'READY.json').read_text())
    if info.get('format') != 'qwen3_4b_staged_dependency_v1':
        raise ValueError('Unknown checkpoint handoff format')
    files = info.get('files', {})
    if not REQUIRED <= set(files):
        raise ValueError('Checkpoint handoff is missing required assets')
    for name, digest in files.items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Checkpoint handoff filenames must be basenames')
        path = bundle/'checkpoint'/name
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError(f'Checkpoint handoff checksum mismatch: {name}')
    return info


def publish(suite, plan, seed, directory):
    suite = Path(suite)
    home = suite/'no_joint'/f'seed_{seed}'
    marker = json.loads((home/'COMPLETE.json').read_text())
    if marker.get('status') != 'completed' or marker.get('seed') != seed or marker.get('arm') != 'no_joint':
        raise ValueError('Only a completed same-seed staged run may be published')
    source = Path(marker['final_checkpoint']).resolve()
    if not source.is_relative_to(home.resolve()):
        raise ValueError('Final checkpoint must belong to the staged experiment')
    checkpoint_audit = json.loads((source/'audit.json').read_text())
    exp = next(e for e in plan['experiments'] if e['arm'] == 'no_joint' and e['seed'] == seed)
    if checkpoint_audit.get('stage') != 'head' or checkpoint_audit.get('global_step') != exp['stages'][-1]['steps']:
        raise ValueError('Only the prescribed final head checkpoint may be published')
    for name, digest in checkpoint_audit['weight_sha256'].items():
        if Path(name).name != name or sha(source/name) != digest:
            raise ValueError('Final checkpoint differs from its training audit')
    expected = identity(suite, plan, seed)
    destination = bundle_path(directory, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        old = verify_bundle(destination)
        if old['identity'] != expected or any(sha(source/name) != digest for name, digest in old['files'].items()):
            raise ValueError('Refusing to replace a different published checkpoint')
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=destination.name+'.tmp-', dir=destination.parent))
    try:
        output = temporary/'checkpoint'
        output.mkdir()
        for p in source.iterdir():
            if p.is_file() and p.suffix in ('.json', '.safetensors', '.jinja', '.txt'):
                if p.is_symlink() or p.name.startswith(('optimizer', 'rng_', 'scheduler')):
                    continue
                shutil.copy2(p, output/p.name)
        info = dict(format='qwen3_4b_staged_dependency_v1', identity=expected,
                    source_suite=str(suite.resolve()), source_checkpoint=str(source),
                    files={p.name: sha(p) for p in output.iterdir()})
        (temporary/'READY.json').write_text(json.dumps(info, indent=2)+'\n')
        verify_bundle(temporary)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def import_bundle(suite, plan, seed, directory):
    """Copy verified model-only assets, then publish COMPLETE last. Never resume optimizer state."""
    suite = Path(suite)
    bundle = bundle_path(directory, seed)
    if not (bundle/'READY.json').is_file():
        return False
    info = verify_bundle(bundle)
    if info['identity'] != identity(suite, plan, seed):
        raise ValueError('External checkpoint seed/data/model/training recipe or code mismatch')
    home = suite/'no_joint'/f'seed_{seed}'
    with (home/'experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = home/'COMPLETE.json'
        destination = home/'head/checkpoints'
        imported = home/'external_checkpoint.json'
        if marker.exists():
            if not imported.exists() or json.loads(imported.read_text())['files'] != info['files']:
                raise ValueError('A different local completion already exists')
            for name, digest in info['files'].items():
                if sha(destination/name) != digest:
                    raise ValueError('Imported checkpoint was modified')
            return True
        if list(home.glob('*/execution.json')):
            raise ValueError('Refusing to import over a locally started staged experiment')
        post = next(e for e in plan['experiments'] if e['arm'] == 'post_joint' and e['seed'] == seed)
        validate_initialization_layout(bundle/'checkpoint', yaml.safe_load(Path(post['stages'][0]['config']).read_text()), 't_head')
        if destination.exists():
            # Recover an import interrupted after directory rename, before COMPLETE.
            if {p.name for p in destination.iterdir()} != set(info['files']):
                raise ValueError('Refusing to replace an existing checkpoint directory')
            for name, digest in info['files'].items():
                if sha(destination/name) != digest:
                    raise ValueError('Existing checkpoint differs from the external bundle')
        else:
            temporary = Path(tempfile.mkdtemp(prefix='checkpoint-import-', dir=destination.parent))
            try:
                for name, digest in info['files'].items():
                    shutil.copy2(bundle/'checkpoint'/name, temporary/name)
                    if sha(temporary/name) != digest:
                        raise ValueError('Checkpoint changed during copy')
                temporary.rename(destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        imported.write_text(json.dumps(info, indent=2)+'\n')
        completion = dict(status='completed', arm='no_joint', seed=seed,
                          final_checkpoint=str(destination), total_updates=info['identity']['total_updates'],
                          imported=True, source_checkpoint=info['source_checkpoint'])
        tmp = home/'COMPLETE.json.tmp'
        tmp.write_text(json.dumps(completion, indent=2)+'\n')
        tmp.replace(marker)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['publish', 'import'])
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    from matrix_audit import audit_suite
    plan = json.loads((args.suite/'plan.json').read_text())
    audit_suite(args.suite, plan)
    result = (publish if args.action == 'publish' else import_bundle)(args.suite, plan, args.seed, args.directory)
    print(result)
    if result is False:
        raise SystemExit('Dependency bundle is not ready')


if __name__ == '__main__':
    main()
