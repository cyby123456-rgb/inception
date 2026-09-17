"""CPU checks for eight independent jobs and cross-host checkpoint handoff."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'local_setup'))
from qwen3_4b_recipe import architecture, stage_config
from training_contract import sha, validate_scope
from matrix_audit import write_protocol_manifest
from qwen3_4b_dependency import identity, publish, import_bundle, verify_bundle
from run_qwen3_4b_workers import assignments, dependency_status, sync_dependency, dependency_source_parts


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def make_suite(root):
    root.mkdir(parents=True)
    recipe = yaml.safe_load((ROOT/'configs/experimental/qwen3_4b_matrix.yaml').read_text())
    model = root/'model'
    config = dict(model_type='qwen3', num_hidden_layers=36, hidden_size=2560,
                  intermediate_size=9728, num_attention_heads=32, num_key_value_heads=8,
                  head_dim=128, tie_word_embeddings=True, vocab_size=151936)
    dump(model/'config.json', config)
    arch = architecture(config, recipe)
    data = root/'data'
    for name in ['train.json', 'validation.json', 'dataset_info.json', 'dev.jsonl', 'test.jsonl']:
        dump(data/name, [])
    snap = root/'source_snapshot'
    (snap/'local_setup').mkdir(parents=True)
    (snap/'scripts').mkdir()
    for name in ['run_qwen3_4b_workers.py', 'qwen3_4b_dependency.py', 'training_contract.py', 'matrix_audit.py']:
        shutil.copy2(ROOT/'local_setup'/name, snap/'local_setup'/name)
    # Deliberately lightweight fixture: test orchestration, no GPU or model training.
    (snap/'scripts/common.py').write_text('''import json,time
from pathlib import Path
def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)
def gpu_snapshot(gpu):
 return dict(uuid='fixture-'+gpu,free_mib=80000,util=0,compute_pids=[])
''')
    (snap/'local_setup/run_qwen3_4b_matrix.py').write_text('''import os,time
from pathlib import Path
from common import write_json
def run_experiment(suite,plan,experiment,gpu,shared):
 home=suite/experiment['arm']/('seed_'+str(experiment['seed']))
 started=time.time()
 if experiment['dependency']:
  assert (suite/experiment['dependency']/'COMPLETE.json').exists()
 if experiment['arm']=='no_joint':time.sleep(.2)
 checkpoint=home/'head/checkpoints';checkpoint.mkdir(parents=True,exist_ok=True)
 for name in ['adapter_model.safetensors','recurft_recurrent.safetensors','recurft_config.json']:
  (checkpoint/name).touch()
 write_json(home/'fixture_execution.json',dict(pid=os.getpid(),gpu=gpu,started=started,ended=time.time()))
 write_json(home/'COMPLETE.json',dict(status='completed',arm=experiment['arm'],seed=experiment['seed'],final_checkpoint=str(checkpoint)))
''')
    experiments = []
    for seed in [42, 43, 44]:
        for arm in ['no_joint', 'post_joint', 'full_joint']:
            specs = [('core','target_t'), ('multistep','target_t'), ('head_warmup','head'), ('head','head')] if arm == 'no_joint' else [
                ('joint', 't_head' if arm == 'post_joint' else 'target_t_head')]
            stages = []
            for name, scope in specs:
                directory = root/arm/f'seed_{seed}'/name
                directory.mkdir(parents=True)
                cfg = stage_config(model, data, directory/'checkpoints', seed, name, 2,
                                   arch, recipe, scope in ('target_t', 'target_t_head'))
                path = directory/'train.yaml'
                path.write_text(yaml.safe_dump(cfg))
                stages.append(dict(name=name, scope=scope, steps=2, config=str(path),
                                   directory=str(directory), training_regime=validate_scope(cfg, scope)))
            experiments.append(dict(arm=arm, seed=seed, stages=stages, total_updates=8 if arm=='no_joint' else 10,
                                    dependency=f'no_joint/seed_{seed}' if arm=='post_joint' else None))
    plan = dict(experiments=experiments, model=str(model), recipe=recipe,
                validation_inference=str(data/'dev.jsonl'), test_data=str(data/'test.jsonl'))
    dump(root/'plan.json', plan)
    dump(root/'source_manifest.json', {str(p.relative_to(snap)):sha(p) for p in snap.rglob('*') if p.is_file()})
    write_protocol_manifest(root, plan)
    return plan


def complete_staged(suite, plan, seed=42):
    home = suite/'no_joint'/f'seed_{seed}'
    checkpoint = home/'milestones/head_2'
    checkpoint.mkdir(parents=True)
    for name in ['adapter_model.safetensors', 'recurft_recurrent.safetensors']:
        (checkpoint/name).write_bytes(b'fixture weights')
    for name in ['adapter_config.json', 'tokenizer_config.json', 'tokenizer.json', 'training_contract.json']:
        dump(checkpoint/name, {})
    dump(checkpoint/'recurft_config.json', dict(loop_start_layer=33, loop_end_layer=34,
          pre_lora_rank=8,last_lora_rank=8,t_lora_rank=80,t_lora_alpha=160,boundary_head_rank=160))
    dump(checkpoint/'audit.json', dict(stage='head',global_step=2,
         weight_sha256={p.name:sha(p) for p in checkpoint.glob('*.safetensors')}))
    dump(home/'COMPLETE.json', dict(status='completed',arm='no_joint',seed=seed,final_checkpoint=str(checkpoint)))
    return checkpoint


def test_eight_assignments_exclude_only_local_and_keep_all_dependencies(tmp_path):
    plan = make_suite(tmp_path/'suite')
    result = assignments(plan, list(map(str,range(8))), exclude='no_joint/seed_42')
    assert len(result) == 8
    assert {r['gpu'] for r in result} == set(map(str,range(8)))
    assert 'no_joint/seed_42' not in {r['experiment'] for r in result}
    assert len([r for r in result if r['dependency']]) == 3
    single = assignments(plan, ['2'], only='no_joint/seed_42')
    assert single == [dict(experiment='no_joint/seed_42',gpu='2',dependency=None)]
    for gpus in [['0']*8, ['0','1','2'], [str(n) for n in range(7)]+['-1']]:
        with pytest.raises(ValueError):
            assignments(plan,gpus,exclude='no_joint/seed_42')
    with pytest.raises(ValueError):
        assignments(plan,list(map(str,range(8))),exclude='no_joint/seed_99')


def test_cross_host_identity_and_model_only_handoff(tmp_path):
    source, target = tmp_path/'host1', tmp_path/'host2'
    a,b = make_suite(source),make_suite(target)
    assert identity(source,a,42) == identity(target,b,42)
    checkpoint = complete_staged(source,a)
    bundles = tmp_path/'exports'
    exported = publish(source,a,42,bundles)
    assert publish(source,a,42,bundles) == exported
    assert not import_bundle(target,b,43,bundles)
    assert import_bundle(target,b,42,bundles)
    assert import_bundle(target,b,42,bundles)
    imported = target/'no_joint/seed_42/head/checkpoints'
    assert (imported/'adapter_model.safetensors').read_bytes() == (checkpoint/'adapter_model.safetensors').read_bytes()
    assert dependency_status(target,'no_joint/seed_42')
    assert not list(imported.glob('optimizer*'))
    (imported/'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='modified'):
        import_bundle(target,b,42,bundles)


@pytest.mark.parametrize('mutation',['data','model','recipe','weights','path'])
def test_handoff_rejects_mismatches(tmp_path,mutation):
    source,target=tmp_path/'host1',tmp_path/'host2'
    a,b=make_suite(source),make_suite(target)
    complete_staged(source,a)
    bundles=tmp_path/'exports';bundle=publish(source,a,42,bundles)
    if mutation=='data':
        (target/'data/train.json').write_text('[1]')
    elif mutation=='model':
        (target/'model/config.json').write_text('{}')
    elif mutation=='recipe':
        p=Path(b['experiments'][0]['stages'][0]['config'])
        cfg=yaml.safe_load(p.read_text());cfg['learning_rate']=9;p.write_text(yaml.safe_dump(cfg))
    elif mutation=='weights':
        (bundle/'checkpoint/adapter_model.safetensors').write_bytes(b'bad transfer')
    else:
        p=bundle/'READY.json';info=json.loads(p.read_text());info['files']['../outside']='x';dump(p,info)
    with pytest.raises(ValueError):
        import_bundle(target,b,42,bundles)
    assert not (target/'no_joint/seed_42/COMPLETE.json').exists()


def test_dependency_failure_and_stale_restart_status(tmp_path):
    home=tmp_path/'no_joint/seed_42'
    assert not dependency_status(tmp_path,'no_joint/seed_42')
    dump(home/'worker.json',dict(status='failed',updated_unix=10,error='fixture failure'))
    with pytest.raises(RuntimeError,match='fixture failure'):
        dependency_status(tmp_path,'no_joint/seed_42')
    assert not dependency_status(tmp_path,'no_joint/seed_42',failed_since=20)


def test_ssh_handoff_checks_ready_then_publishes_verified_copy(tmp_path,monkeypatch):
    source=tmp_path/'local';plan=make_suite(source);complete_staged(source,plan)
    bundle=publish(source,plan,42,tmp_path/'export')
    calls=[]
    def transfer(command,**kwargs):
        calls.append(command)
        if command[0]=='scp':
            shutil.copytree(bundle,Path(command[-1])/bundle.name)
        return subprocess.CompletedProcess(command,0,stdout='',stderr='')
    monkeypatch.setattr('run_qwen3_4b_workers.subprocess.run',transfer)
    result=sync_dependency('user@host:/export',tmp_path/'incoming',42)
    assert result['returncode']==0
    assert [c[0] for c in calls]==['ssh','scp']
    assert 'BatchMode=yes' in calls[0] and calls[0][-1].startswith('test -f ')
    verify_bundle(tmp_path/'incoming/no_joint_seed_42')
    sync_dependency('user@host:/export',tmp_path/'incoming',42)
    assert len(calls)==2  # A completed checkpoint is not copied on every poll.
    for invalid in ['host:relative','-host:/root','host;bad:/root','host:/root;bad']:
        with pytest.raises(ValueError):dependency_source_parts(invalid)


def test_remote_not_ready_does_not_copy_partial_checkpoint(tmp_path,monkeypatch):
    calls=[]
    def missing(command,**kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command,1,stdout='',stderr='')
    monkeypatch.setattr('run_qwen3_4b_workers.subprocess.run',missing)
    assert sync_dependency('user@host:/export',tmp_path/'incoming',42)['returncode']==1
    assert len(calls)==1 and calls[0][0]=='ssh'
    assert not (tmp_path/'incoming/no_joint_seed_42').exists()


def test_eight_real_worker_processes_wait_and_use_distinct_gpus(tmp_path):
    """Real subprocess scheduler, fake training body: no inference/performance claim."""
    source,target=tmp_path/'local',tmp_path/'remote'
    a,b=make_suite(source),make_suite(target)
    complete_staged(source,a)
    bundles=tmp_path/'handoff'
    # Exercise external dependency waiting before the bundle is published.
    command=[sys.executable,str(target/'source_snapshot/local_setup/run_qwen3_4b_workers.py'),
             '--suite',str(target),'--exclude','no_joint/seed_42','--gpus','0,1,2,3,4,5,6,7',
             '--dependency-dir',str(bundles),'--poll-seconds','.05']
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
    with (tmp_path/'scheduler.log').open('w') as log:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                path=target/'post_joint/seed_42/worker.json'
                if path.exists() and json.loads(path.read_text())['phase']=='waiting_for_dependency':
                    break
                if proc.poll() is not None:
                    pytest.fail((tmp_path/'scheduler.log').read_text())
                time.sleep(.05)
            else:
                pytest.fail('Remote continuation never entered dependency wait')
            assert not (target/'post_joint/seed_42/fixture_execution.json').exists()
            publish(source,a,42,bundles)
            assert proc.wait(timeout=30)==0,(tmp_path/'scheduler.log').read_text()
        finally:
            if proc.poll() is None:
                proc.terminate();proc.wait(timeout=20)
    rows={str(p.parent.relative_to(target)):json.loads(p.read_text()) for p in target.glob('*/seed_*/fixture_execution.json')}
    assert len(rows)==8 and 'no_joint/seed_42' not in rows
    assert len({x['pid'] for x in rows.values()})==8
    assert {x['gpu'] for x in rows.values()}==set(map(str,range(8)))
    for seed in [43,44]:
        assert rows[f'post_joint/seed_{seed}']['started']>=rows[f'no_joint/seed_{seed}']['ended']
    status=json.loads(next((target/'worker_launches').glob('*/status.json')).read_text())
    assert status['status']=='completed' and all(x['returncode']==0 for x in status['children'])


def test_termination_stops_owned_training_descendants(tmp_path):
    target=tmp_path/'suite';make_suite(target)
    path=target/'source_snapshot/local_setup/run_qwen3_4b_matrix.py'
    path.write_text('''import subprocess,sys,time
from common import write_json
def run_experiment(suite,plan,experiment,gpu,shared):
 child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
 write_json(suite/'child.json',dict(pid=child.pid))
 child.wait()
''')
    manifest=target/'source_manifest.json';hashes=json.loads(manifest.read_text())
    hashes['local_setup/run_qwen3_4b_matrix.py']=sha(path);dump(manifest,hashes)
    command=[sys.executable,str(target/'source_snapshot/local_setup/run_qwen3_4b_workers.py'),
             '--suite',str(target),'--only','full_joint/seed_42','--gpus','0']
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
    with (tmp_path/'termination.log').open('w') as log:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+20
            while not (target/'child.json').exists() and time.monotonic()<deadline:
                assert proc.poll() is None,(tmp_path/'termination.log').read_text()
                time.sleep(.05)
            assert (target/'child.json').exists()
            child=json.loads((target/'child.json').read_text())['pid']
            proc.terminate()
            assert proc.wait(timeout=20)!=0
            stat=Path(f'/proc/{child}/stat')
            assert not stat.exists() or stat.read_text().split()[2]=='Z'
            state=json.loads(next((target/'worker_launches').glob('*/status.json')).read_text())
            assert state['status']=='interrupted'
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=20)
