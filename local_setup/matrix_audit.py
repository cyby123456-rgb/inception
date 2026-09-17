"""Check prepared code, data, per-stage configs and declared training regimes."""
import json
from pathlib import Path
import yaml
from training_contract import sha, validate_scope


def write_protocol_manifest(suite, plan):
    suite = Path(suite)
    files = [suite/'plan.json', suite/'data/dataset_info.json', suite/'data/train.json',
             suite/'data/validation.json', Path(plan['validation_inference']),
             Path(plan['test_data']), Path(plan['model'])/'config.json']
    files += [Path(s['config']) for e in plan['experiments'] for s in e['stages']]
    payload = {str(p.resolve()):sha(p) for p in files}
    (suite/'protocol_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')


def audit_suite(suite, plan):
    suite = Path(suite)
    for relative, digest in json.loads((suite/'source_manifest.json').read_text()).items():
        if sha(suite/'source_snapshot'/relative) != digest:
            raise ValueError(f'Source snapshot modified: {relative}')
    path = suite/'protocol_manifest.json'
    if not path.is_file():
        raise ValueError('Legacy suite lacks protocol manifest; use its original runner or prepare an audited new suite')
    for name, digest in json.loads(path.read_text()).items():
        if sha(name) != digest:
            raise ValueError(f'Prepared protocol/data/config modified: {name}')
    stages = []
    for experiment in plan['experiments']:
        arm, seed = experiment['arm'], experiment['seed']
        expected = {'no_joint':['target_t','target_t','head','head'],
                    'post_joint':['t_head'], 'full_joint':['target_t_head']}[arm]
        if [s['scope'] for s in experiment['stages']] != expected:
            raise ValueError(f'Incorrect stage scopes for {arm}/seed_{seed}')
        for stage in experiment['stages']:
            config = yaml.safe_load(Path(stage['config']).read_text())
            description = validate_scope(config, stage['scope'])
            if description != stage['training_regime']:
                raise ValueError(f'Training regime mismatch: {stage["config"]}')
            if config['seed'] != seed or config['data_seed'] != seed or config['max_steps'] != stage['steps']:
                raise ValueError(f'Seed/budget mismatch: {stage["config"]}')
            if str(Path(config['output_dir']).resolve()) != str(Path(stage['directory'])/'checkpoints'):
                raise ValueError(f'Checkpoint output mismatch: {stage["config"]}')
            stages.append(dict(arm=arm,seed=seed,stage=stage['name'],**description))
    return dict(protocol_verified=True,source_verified=True,stages=stages)
