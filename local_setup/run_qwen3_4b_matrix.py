"""Run prepared experiments, preserving seed dependencies and checkpoint state."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from common import environment, gpu_snapshot, write_json
from matrix_audit import audit_suite


def latest_checkpoint(stage):
    found = []
    for path in (Path(stage['directory'])/'checkpoints').glob('checkpoint-*'):
        if all((path/f).is_file() for f in ['trainer_state.json', 'optimizer_safe.json',
                                           'optimizer_safe.safetensors', 'rng_safe.json', 'rng_safe.safetensors',
                                           'training_contract.json']):
            state = json.loads((path/'trainer_state.json').read_text())
            if state['max_steps'] != stage['steps']:
                raise ValueError('Checkpoint belongs to a different training budget')
            found.append((state['global_step'], path))
    return max(found, default=(0, None), key=lambda x:x[0])


def snapshot(checkpoint, destination, stage, step):
    import torch
    from safetensors.torch import load_file
    destination.mkdir(parents=True, exist_ok=False)
    target = load_file(str(checkpoint/'adapter_model.safetensors'))
    recurrent = load_file(str(checkpoint/'recurft_recurrent.safetensors'))
    audit = dict(stage=stage['name'], global_step=step, expected_trainable_scope=stage['scope'],
                 training_regime=stage['training_regime'])
    if stage['initialize_from']:
        source = Path(stage['initialize_from'])
        previous = load_file(str(source/'adapter_model.safetensors'))
        same = target.keys() == previous.keys() and all(torch.equal(v, previous[k]) for k,v in target.items())
        audit['target_equal_to_stage_source'] = same
        if stage['scope'] in ['head', 't_head']:
            assert same, 'A frozen target adapter changed'
        if stage['scope'] == 'head':
            old = load_file(str(source/'recurft_recurrent.safetensors'))
            assert all(torch.equal(v, recurrent[k]) for k,v in old.items() if not k.startswith('boundary_')), 'Frozen T changed during head training'
            audit['T_equal_to_stage_source'] = True
    for p in checkpoint.iterdir():
        if p.is_file() and p.suffix in ['.json','.safetensors','.jinja','.txt'] and not p.stem.startswith(('optimizer_safe','rng_safe')):
            shutil.copy2(p, destination/p.name)
    audit['weight_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in destination.glob('*.safetensors')}
    write_json(destination/'audit.json', audit)


def run_experiment(suite, plan, experiment, gpu, shared):
    arm, seed = experiment['arm'], experiment['seed']
    home = suite/arm/f'seed_{seed}'
    if (home/'COMPLETE.json').exists():
        print(f'SKIP completed {arm}/seed_{seed}', flush=True)
        return
    lock = (home/'experiment.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if experiment['dependency'] and not (suite/experiment['dependency']/'COMPLETE.json').exists():
        raise RuntimeError(f"Run dependency {experiment['dependency']} first")
    snap = suite/'source_snapshot'
    env = environment(gpu)
    env['PYTHONPATH'] = os.pathsep.join(str(snap/p) for p in ['LLaMA-Factory/src','LLaMA-Factory','LLaMA-Factory/experiments/recurft_math'])
    env.pop('CUDA_MPS_PIPE_DIRECTORY', None)
    recipe = plan['recipe']
    state = dict(status='running', arm=arm, seed=seed, gpu=gpu, pid=os.getpid(), child_pid=None)
    def save():
        state['updated_unix'] = time.time()
        write_json(home/'status.json', state)
    def resources():
        state['phase'] = 'waiting_for_gpu'
        while True:
            g = gpu_snapshot(gpu)
            state['gpu_snapshot'] = g
            save()
            if g['free_mib'] >= recipe['min_free_mib'] and (shared or not g['compute_pids']):
                return
            time.sleep(10)
    def execute(name, command):
        state.update(phase=name, command=command)
        logs = home/'logs'
        logs.mkdir(exist_ok=True)
        path = logs/f'{name}_{time.time_ns()}.log'
        state['log'] = str(path)
        with path.open('x') as log:
            proc = subprocess.Popen(command, cwd=snap, env=env, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = proc.pid
            while proc.poll() is None:
                save()
                time.sleep(5)
            state['child_pid'] = None
            save()
        if proc.returncode:
            raise RuntimeError(f'{name} failed with exit {proc.returncode}; inspect {path}')
    def evaluate(name, checkpoint, final=False):
        output = home/'evaluations'/name
        output.parent.mkdir(exist_ok=True)
        if output.exists():
            status = output/'status.json'
            if status.exists() and json.loads(status.read_text()).get('status') == 'completed':
                if not (output/'analysis/aggregate.json').exists():
                    execute(name+'_analysis', [sys.executable, str(snap/'local_setup/analyze_acceleration_goal.py'),
                                              str(output), '--out', str(output/'analysis')])
                return
            output.rename(output.with_name(output.name+f'_failed_{time.time_ns()}'))
        resources()
        settings = recipe['final' if final else 'periodic']
        cmd = [sys.executable, str(snap/'local_setup/benchmark_trained_target.py'),
            '--model', plan['model'], '--checkpoint', str(checkpoint),
            '--data', plan['test_data'] if final else plan['validation_inference'], '--output', str(output),
            '--start', str(settings.get('start', 0)), '--samples', str(settings['samples']),
            '--repeats', str(settings['repeats']), '--max-new-tokens', str(settings['max_new_tokens']),
            '--merge-target', '--lower-right', '--fused-norms', '--compact', '--t-graph', '--gpu-greedy', '--lookup',
            '--variants', 'greedy_lower_fused,after_b3_lower_fused_compact_tgraph,after_b3_lower_fused_compact_tgraph_lookup,after_b3_lower_fused_compact_tgraph_lookuponly']
        execute(name, cmd)
        execute(name+'_analysis', [sys.executable, str(snap/'local_setup/analyze_acceleration_goal.py'),
                                  str(output), '--out', str(output/'analysis')])
    save()
    try:
        for stage in experiment['stages']:
            directory = Path(stage['directory'])
            for step in stage['stop_steps']:
                milestone = home/'milestones'/f"{stage['name']}_{step}"
                if milestone.exists() and not (milestone/'audit.json').exists():
                    milestone.rename(milestone.with_name(milestone.name+f'_incomplete_{time.time_ns()}'))
                if not milestone.exists():
                    current, checkpoint = latest_checkpoint(stage)
                    if current > step:
                        raise RuntimeError(f'Missing earlier milestone {milestone}; refusing to substitute later weights')
                    if current < step:
                        resources()
                        gpu_lock = open('/tmp/inception-eval-'+hashlib.sha256(gpu.encode()).hexdigest()[:16]+'.lock', 'w')
                        with gpu_lock:
                            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                            resources()
                            cmd = [sys.executable, str(snap/'local_setup/run_qwen3_stage.py'),
                                '--config', stage['config'], '--gpu', gpu, '--stop-after-step', str(step),
                                '--min-free-mib', str(recipe['min_free_mib']), '--memory-fraction', str(recipe['memory_fraction']),
                                '--trainable-scope', stage['scope'], '--require-tied-embeddings']
                            if shared:
                                cmd.append('--allow-shared-gpu')
                            if checkpoint:
                                cmd += ['--resume-from', str(checkpoint)]
                            elif stage['initialize_from']:
                                cmd += ['--initialize-from', stage['initialize_from']]
                            else:
                                cmd += ['--export-initial', str(home/'initial_step0')]
                            execute(f"train_{stage['name']}_{step}", cmd)
                        current, checkpoint = latest_checkpoint(stage)
                    assert current == step and checkpoint is not None
                    execution = json.loads((directory/'execution.json').read_text())
                    assert execution['status']=='completed' and execution['global_step']==step
                    snapshot(checkpoint, milestone, stage, step)
                if stage['benchmark']:
                    evaluate(f"dev_{stage['name']}_{step}", milestone)
                    if arm == 'full_joint' and step == plan['budget']['non_joint']:
                        evaluate('final_A_budget', milestone, final=True)
            write_json(directory/'COMPLETE.json', dict(step=stage['steps'], status='completed'))
        final_stage = experiment['stages'][-1]
        final = home/'milestones'/f"{final_stage['name']}_{final_stage['steps']}"
        evaluate('final', final, final=True)
        state.update(status='completed', phase='completed')
        save()
        write_json(home/'COMPLETE.json', dict(status='completed', final_checkpoint=str(final),
                    seed=seed, arm=arm, total_updates=experiment['total_updates']))
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        save()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--gpus', required=True, help='Physical GPUs; one worker per GPU')
    p.add_argument('--arms', default='all')
    p.add_argument('--seeds', help='Optional subset of the three prepared seeds')
    p.add_argument('--allow-shared-gpu', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    suite = a.suite.resolve()
    plan = json.loads((suite/'plan.json').read_text())
    audit_suite(suite, plan)
    arms = ['no_joint','post_joint','full_joint'] if a.arms=='all' else a.arms.split(',')
    seeds = plan['recipe']['seeds'] if a.seeds is None else [int(s) for s in a.seeds.split(',')]
    gpus = a.gpus.split(',')
    if not set(arms) <= {'no_joint','post_joint','full_joint'} or len(set(arms)) != len(arms):
        p.error('Invalid or duplicate arm')
    if not seeds or not set(seeds) <= set(plan['recipe']['seeds']) or len(set(seeds)) != len(seeds):
        p.error('Invalid or duplicate seed')
    if any(not g.isdecimal() for g in gpus) or len(set(gpus)) != len(gpus):
        p.error('Provide distinct nonnegative GPU indices')
    tasks = [[] for _ in gpus]
    for i, seed in enumerate(seeds):
        for arm in ['no_joint','post_joint','full_joint']:
            if arm in arms:
                exp = next(x for x in plan['experiments'] if x['arm']==arm and x['seed']==seed)
                tasks[i % len(gpus)].append(exp)
                print(json.dumps(dict(gpu=gpus[i % len(gpus)], arm=arm, seed=seed,
                                      dependency=exp['dependency'], stages=[(s['name'],s['steps'],s['scope']) for s in exp['stages']])), flush=True)
    if a.dry_run:
        return
    def worker(gpu, items):
        for exp in items:
            run_experiment(suite, plan, exp, gpu, a.allow_shared_gpu)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, items) for gpu,items in zip(gpus,tasks)]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
