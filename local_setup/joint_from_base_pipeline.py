"""Single-stage joint training with state-preserving pauses for paired inference."""
from pathlib import Path
import collections
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parent
SNAP = RUN / 'source_snapshot'
sys.path.insert(0, str(SNAP / 'scripts'))
from common import environment, gpu_snapshot, write_json, sha256

PLAN = json.loads((RUN / 'plan.json').read_text())
ENV = environment(PLAN['gpu'])
ENV.pop('CUDA_MPS_PIPE_DIRECTORY', None)
ENV.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
STATE = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(),
         'completed_training_steps': 0, 'max_steps': PLAN['max_steps'],
         'jobs': [], 'evaluations': []}
LOCKPATH = '/tmp/inception-eval-' + hashlib.sha256(PLAN['gpu'].encode()).hexdigest()[:16] + '.lock'


def save():
    STATE['updated_unix'] = time.time()
    write_json(RUN / 'status.json', STATE)


def wait_resources(allow_switch=True):
    global ENV, LOCKPATH
    STATE['phase'] = 'waiting_for_gpu'
    while True:
        devices = PLAN['gpu_candidates'] if allow_switch else [PLAN['gpu']]
        candidates = []
        for device in devices:
            gpu = gpu_snapshot(device)
            unexpected = set(gpu['compute_pids']) - set(PLAN['allowed_residents_by_gpu'][device])
            candidates.append({'device': device, 'unexpected_pids': sorted(unexpected), **gpu})
            if not unexpected and gpu['free_mib'] >= 45000:
                PLAN['gpu'] = device
                ENV = environment(device)
                ENV.pop('CUDA_MPS_PIPE_DIRECTORY', None)
                ENV.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                LOCKPATH = '/tmp/inception-eval-' + hashlib.sha256(device.encode()).hexdigest()[:16] + '.lock'
                STATE.update(active_gpu=device, gpu_preflight=gpu, waiting_for_gpu_pids=[])
                save()
                return
        STATE['gpu_candidates'] = candidates
        STATE['waiting_for_gpu_pids'] = sorted({pid for c in candidates for pid in c['unexpected_pids']})
        save()
        time.sleep(10)


def execute(name, command):
    item = {'name': name, 'command': command, 'gpu': PLAN['gpu'], 'started_unix': time.time()}
    STATE.update(phase=name, child_pid=None)
    STATE['jobs'].append(item)
    log_path = RUN / 'logs' / (name + '.log')
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open('x') as log:
        proc = subprocess.Popen(command, cwd=SNAP, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        item['pid'] = proc.pid
        STATE['child_pid'] = proc.pid
        save()
        while proc.poll() is None:
            try:
                gpu = gpu_snapshot(PLAN['gpu'])
                with (RUN / 'gpu_monitor.jsonl').open('a') as f:
                    f.write(json.dumps({'phase': name, **gpu}) + '\n')
                STATE['gpu_latest'] = gpu
                telemetry = RUN / 'joint/telemetry.jsonl'
                if telemetry.exists():
                    rows = [json.loads(s) for s in telemetry.read_text().splitlines() if s.strip()]
                    train = [r for r in rows if 'loss' in r]
                    if train:
                        STATE['latest_training_log'] = {k: train[-1][k] for k in
                            ['global_step', 'loss', 'recurft_recurrent_loss',
                             'recurft_multistep_boundary_logit_kl_k1', 'recurft_multistep_boundary_logit_kl_k2']}
            except Exception as exc:
                STATE['monitor_error'] = repr(exc)
            save()
            time.sleep(5)
        item.update(returncode=proc.returncode, ended_unix=time.time())
        STATE['child_pid'] = None
        save()
    if proc.returncode:
        raise RuntimeError(f'{name} failed ({proc.returncode}); see {log_path}')


def freeze_snapshot(checkpoint, step):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(1)
    initial = RUN / 'initial_step0'
    a = load_file(str(initial / 'adapter_model.safetensors'))
    b = load_file(str(checkpoint / 'adapter_model.safetensors'))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), 'Frozen target changed'
    assert all(torch.count_nonzero(v).item() == 0 for k, v in b.items() if 'lora_B' in k)
    rec_a = load_file(str(initial / 'recurft_recurrent.safetensors'))
    rec_b = load_file(str(checkpoint / 'recurft_recurrent.safetensors'))
    changed = [k for k in rec_a if not torch.equal(rec_a[k], rec_b[k])]
    assert any('boundary_' in k for k in changed), 'Head did not update'
    assert any('lora_' in k for k in changed), 'T did not update'
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert state['global_step'] == step and state['max_steps'] == PLAN['max_steps']
    assert (checkpoint / 'optimizer_safe.json').exists() and (checkpoint / 'optimizer_safe.safetensors').exists()
    assert (checkpoint / 'rng_safe.json').exists() and (checkpoint / 'rng_safe.safetensors').exists()
    dest = RUN / 'milestones' / f'joint_{step}'
    dest.mkdir(parents=True, exist_ok=False)
    for p in checkpoint.iterdir():
        if p.is_file() and p.suffix in ('.json', '.safetensors', '.jinja', '.txt') and not p.stem.startswith(('optimizer_safe', 'rng_safe')):
            shutil.copy2(p, dest / p.name)
    write_json(dest / 'audit.json', {'step': step, 'target_identical_to_initial': True,
        'target_lora_B_all_zero': True, 'changed_recurrent_tensors': changed,
        'model_sha256': {p.name: sha256(p) for p in dest.glob('*.safetensors')},
        'optimizer_scheduler_rng_checkpoint_present': True})
    return dest


def result_rows(path):
    decoder = json.JSONDecoder()
    with path.open() as f:
        buf = ''
        marker = '"results": ['
        while marker not in buf:
            part = f.read(1024 * 1024)
            if not part:
                raise ValueError('Missing results array')
            buf += part
        buf = buf.split(marker, 1)[1]
        while True:
            buf = buf.lstrip(' ,\r\n\t')
            if buf.startswith(']'):
                return
            try:
                row, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                part = f.read(1024 * 1024)
                if not part:
                    raise
                buf += part
                continue
            yield row
            buf = buf[end:]


def evaluate(step, checkpoint, kind):
    wait_resources()
    p = PLAN[kind]
    output = RUN / 'evaluations' / f'{kind}_{step}'
    output.parent.mkdir(exist_ok=True)
    command = [sys.executable, '-u', str(SNAP / 'local_setup/benchmark_two_stage_adaptive.py'),
        '--checkpoint', str(checkpoint), '--output', str(output), '--model', PLAN['model'],
        '--data', PLAN['test_data'], '--dtype', 'fp32', '--route', 'boundary',
        '--variants', PLAN['variants'], '--draft-start-tokens', PLAN['draft_start_tokens'],
        '--max-drafts', str(PLAN['max_drafts']), '--omit-baseline-mask']
    for key in ['start_index', 'samples', 'repeats', 'max_new_tokens']:
        command += ['--' + key.replace('_', '-'), str(p[key])]
    execute(f'eval_{kind}_{step}', command)
    summary = json.loads((output / 'summary.json').read_text())
    assert all(v['exact'] == v['pairs'] for v in summary.values())
    tokens = collections.Counter()
    for row in result_rows(output / 'result.json'):
        if row['phase'] != 'measurement':
            continue
        for name, value in {'greedy': row['baseline'], **row['variants']}.items():
            tokens[name] += len(value['token_ids'])
    assert len(set(tokens.values())) == 1
    record = {'step': step, 'kind': kind, 'gpu': PLAN['gpu'], 'questions': p['samples'], 'repeats': p['repeats'],
        'output_tokens_per_method': tokens['greedy'], 'summary': summary,
        'source': str(output / 'result.json'), 'ended_unix': time.time()}
    STATE['evaluations'].append(record)
    save()
    render()


def render():
    rows = []
    for result in STATE['evaluations']:
        for name, s in result['summary'].items():
            rows.append({'step': result['step'], 'kind': result['kind'], 'gpu': result['gpu'], 'variant': name,
                'questions': result['questions'], 'repeats': result['repeats'],
                'output_tokens': result['output_tokens_per_method'], **s})
    if rows:
        with (RUN / 'speed_history.csv').open('w') as f:
            writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# 从基座开始的全程联合训练', '',
        '原始 Qwen3-8B → 同时训练 T LoRA 与 boundary 输出头；目标模型冻结。没有加载此前训练的 core、multistep 或 boundary 检查点。', '',
        f"状态：{STATE['status']} / {STATE.get('phase')}。已保存训练步数：{STATE['completed_training_steps']} / {PLAN['max_steps']}。", '',
        '训练配置：一轮 MetaMath（49,375 步），有效 batch 8，学习率 3e-6，序列上限 1024，BF16。前 200 步仅对联合损失权重预热，T 和输出头始终同时开放训练。', '',
        '测速时暂停并卸载训练进程；随后恢复模型、优化器、调度器、随机状态、global_step 和数据位置。没有重启损失预热。按 GPU 0、2、3 的空闲情况调度，实际设备写入每个结果；共享环境保留遥测，跨检查点设备/负载变化可能影响速度曲线。', '',
        '正式周期测试为固定 8 题、每题最多 256 token；最终评测为独立 64 题 × 2 次。每个检查点使用同题 greedy 基线、轮换执行顺序、FP32、严格 token 验证、n-gram 关闭。step 2 smoke 仅为 2 题 × 64 token 的流程检查，不与正式速度曲线混算。', '',
        '| 联合步数 | 测试 | 方法 | 相对 greedy 速度 | 接受率 | token 一致 | 每方法输出 token |',
        '|---:|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['step']} | {r['kind']} | {r['variant']} | {r['speedup']:.4f}× | {r['draft_acceptance']:.2%} | {r['exact']}/{r['pairs']} | {r['output_tokens']} |")
    lines += ['', '此实验的 target 是原始基座；先前分阶段实验使用了训练后的 target adapter，跨实验原始耗时不是相同目标模型下的受控比较。', '',
        '[实时状态](status.json) · [实验计划](plan.json) · [训练日志](joint/telemetry.jsonl) · [计时表](speed_history.csv) · [日志目录](logs/)', '']
    (RUN / 'REPORT.md').write_text('\n'.join(lines))
    formal = [r for r in rows if r['kind'] == 'periodic']
    if formal:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for variant in dict.fromkeys(r['variant'] for r in formal):
            selected = [r for r in formal if r['variant'] == variant]
            ax.plot([r['step'] for r in selected], [r['speedup'] for r in selected], marker='o', label=variant)
        ax.axhline(1, color='black', linestyle='--', linewidth=1)
        ax.set(xlabel='Joint optimizer updates', ylabel='Greedy time / speculative time', title='Periodic neural-draft timing (shared GPU)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RUN / 'speed_curve.png', dpi=160)
        plt.close(fig)


def main():
    lock = (RUN / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RUN / 'status.json').exists():
        raise RuntimeError('This pipeline starts a fresh run. Resume a failed stage explicitly with run_qwen3_stage.py --resume-from; do not overwrite its status.')
    save()
    render()
    previous = None
    try:
        for step in PLAN['stop_steps']:
            wait_resources()
            with open(LOCKPATH, 'w') as gpu_lock:
                STATE['phase'] = 'waiting_for_training_lock'
                save()
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                wait_resources(allow_switch=False)
                command = [sys.executable, '-u', str(SNAP / 'local_setup/run_qwen3_stage.py'),
                    '--config', str(RUN / 'joint/train.yaml'), '--gpu', PLAN['gpu'],
                    '--allow-shared-gpu', '--stop-after-step', str(step)]
                if previous is None:
                    command += ['--export-initial', str(RUN / 'initial_step0')]
                else:
                    command += ['--resume-from', str(previous)]
                execute(f'train_to_{step}', command)
            checkpoint = RUN / 'joint/checkpoints' / f'checkpoint-{step}'
            execution = json.loads((RUN / 'joint/execution.json').read_text())
            assert execution['status'] == 'completed' and execution['global_step'] == step
            assert execution['trainable_groups'].get('target', 0) == 0
            assert execution['trainable_groups'].get('T', 0) > 0 and execution['trainable_groups'].get('boundary', 0) > 0
            if previous is not None:
                assert execution['starting_global_step'] == STATE['completed_training_steps']
                assert execution['optimizer_steps_at_start'] == [float(STATE['completed_training_steps'])]
            write_json(RUN / 'executions' / f'train_to_{step}.json', execution)
            frozen = freeze_snapshot(checkpoint, step)
            STATE['completed_training_steps'] = step
            previous = checkpoint
            save()
            if step == 2:
                evaluate(step, frozen, 'smoke')
            elif step == 4:
                write_json(RUN / 'resume_preflight.json', {'status': 'passed',
                    'initial_segment_steps': 2, 'resumed_segment_steps': 2,
                    'optimizer_start_steps': execution['optimizer_steps_at_start'],
                    'resumed_start_global_step': execution['starting_global_step'],
                    'end_global_step': step, 'ignore_data_skip': execution['ignore_data_skip'],
                    'full_max_steps_unchanged': PLAN['max_steps'], 'target_remained_frozen': True})
            else:
                evaluate(step, frozen, 'final' if step == PLAN['max_steps'] else 'periodic')
        STATE.update(status='completed', phase='completed', ended_unix=time.time())
        save()
        render()
        write_json(RUN / 'COMPLETION.json', {'status': 'completed', 'steps': PLAN['max_steps'],
            'ended_unix': time.time(), 'report': str(RUN / 'REPORT.md')})
    except BaseException as exc:
        STATE.update(status='failed', error=repr(exc), ended_unix=time.time())
        save()
        render()
        raise


if __name__ == '__main__':
    main()
