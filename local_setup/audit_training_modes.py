"""Export resolved historical/4B training modes and verify a prepared suite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('DISABLE_VERSION_CHECK','1')
sys.path.insert(0,str(ROOT/'LLaMA-Factory/src'))
import yaml
from llamafactory.hparams import FinetuningArguments
from matrix_audit import audit_suite


def resolved(path, name, overlay=None):
    config=yaml.safe_load(Path(path).read_text())
    config.update(overlay or {})
    fields=FinetuningArguments.__dataclass_fields__
    args=FinetuningArguments(**{k:v for k,v in config.items() if k in fields})
    if args.recurft_stage1_heads_only:
        scope='boundary_only'
    elif args.recurft_recurrent_trainable_only:
        scope='frozen_target_joint'
    elif args.recurft_boundary_head_rank:
        scope='trainable_target_joint'
    else:
        scope='staged_target_t'
    return dict(name=name,source=str(Path(path).resolve()),scope=scope,
        scope_source='freeze flags plus head presence; legacy is not itself a frozen-mode label',
        recurft={k:getattr(args,k) for k in fields if k.startswith('recurft_')},
        schedule={k:config.get(k) for k in ['model_name_or_path','max_steps','num_train_epochs',
            'cutoff_len','per_device_train_batch_size','gradient_accumulation_steps','learning_rate',
            'seed','data_seed','ignore_data_skip','save_only_model']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    plan=json.loads((a.suite/'plan.json').read_text())
    protocol=audit_suite(a.suite,plan)
    profiles=[resolved(path,'historical/'+path.stem) for path in sorted((ROOT/'configs/reference').glob('*.yaml'))]
    profiles.append(resolved(ROOT/'configs/experimental/qwen3_joint_from_base.yaml','8b_frozen_joint_from_base'))
    source=ROOT/'configs/reference/llama3_q3_stage1_boundary_only_continue_to_1000_20260711.yaml'
    overlay=yaml.safe_load((ROOT/'configs/experimental/recurft_joint_boundary.yaml').read_text())
    profiles.append(resolved(source,'legacy_frozen_joint_continuation',overlay))
    for e in plan['experiments']:
        for stage in e['stages']:
            profiles.append(resolved(stage['config'],f"4b/{e['arm']}/seed_{e['seed']}/{stage['name']}"))
    sources=['LLaMA-Factory/src/llamafactory/model/adapter.py',
        'LLaMA-Factory/src/llamafactory/model/model_utils/recurft.py',
        'LLaMA-Factory/src/llamafactory/train/sft/recurft.py',
        'LLaMA-Factory/src/llamafactory/train/sft/trainable_target_joint.py',
        'local_setup/training_contract.py','local_setup/matrix_audit.py',
        'local_setup/benchmark_inference_matrix.py','local_setup/compact_decode.py']
    result=dict(suite=str(a.suite.resolve()),protocol=protocol,profiles=profiles,
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                limits=['Static resolved configurations are not proof of successful training; consult gradient/weight audits.',
                        'A/B/C compare training recipes, not a single isolated joint-training switch.',
                        'The 4B matrix uses strict fixed-block neural/lookup inference; legacy adaptive and relaxed-verification routes are separate.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(profiles=len(profiles),stages=len(protocol['stages']),source_verified=True,protocol_verified=True),indent=2))


if __name__=='__main__':
    main()
