"""Prepare an isolated joint-training run; this command does not start training."""
import argparse
import json
from pathlib import Path
import shutil
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--train-data', type=Path, required=True)
    p.add_argument('--test-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', default='0', help='Comma-separated physical GPU indices.')
    a = p.parse_args()
    if not a.model.is_dir() or not a.train_data.is_file() or not a.test_data.is_file():
        p.error('Supply an existing local model directory and training/test data files.')
    gpus = a.gpus.split(',')
    if not gpus or any(not g.isdecimal() for g in gpus) or len(set(gpus)) != len(gpus):
        p.error('--gpus must contain unique nonnegative integer indices.')
    run = a.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    for name in ['logs', 'executions', 'milestones', 'evaluations', 'joint/dataset']:
        (run / name).mkdir(parents=True, exist_ok=True)
    for name in ['LLaMA-Factory', 'scripts', 'local_setup']:
        shutil.copytree(ROOT / name, run / 'source_snapshot' / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.pytest_cache'))
    shutil.copy2(ROOT / 'local_setup/joint_from_base_pipeline.py', run / 'run_pipeline.py')
    cfg = yaml.safe_load((ROOT / 'configs/experimental/qwen3_joint_from_base.yaml').read_text())
    cfg.update(model_name_or_path=str(a.model.resolve()), dataset_dir=str(run / 'joint/dataset'),
               output_dir=str(run / 'joint/checkpoints'))
    (run / 'joint/train.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    dataset = {'joint_metamath': {'file_name': str(a.train_data.resolve()),
                                'columns': {'prompt': 'query', 'response': 'response'}}}
    (run / 'joint/dataset/dataset_info.json').write_text(json.dumps(dataset, indent=2) + '\n')
    plan = dict(created_unix=time.time(), model=str(a.model.resolve()),
                data=str(a.train_data.resolve()), test_data=str(a.test_data.resolve()),
                max_steps=cfg['max_steps'], gpu=gpus[0], gpu_candidates=gpus,
                allowed_residents_by_gpu={g: [] for g in gpus},
                stop_steps=[2, 4, 250, 500, 750, 1000] + list(range(2000, 49001, 1000)) + [49375],
                smoke=dict(start_index=0, samples=2, repeats=1, max_new_tokens=64),
                periodic=dict(start_index=0, samples=8, repeats=1, max_new_tokens=256),
                final=dict(start_index=456, samples=64, repeats=2, max_new_tokens=256),
                variants='fixed,adaptive', draft_start_tokens='0,128', max_drafts=2)
    (run / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Prepared {run}. Start with: python {run / "run_pipeline.py"}')


if __name__ == '__main__':
    main()
