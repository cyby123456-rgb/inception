"""Assign one experiment to each GPU, waiting for exact same-seed dependencies."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time


def experiment_id(experiment):
    return f"{experiment['arm']}/seed_{experiment['seed']}"


def assignments(plan, gpus, only=None, exclude=None):
    experiments = plan['experiments']
    known = {experiment_id(e) for e in experiments}
    if len(known) != len(experiments):
        raise ValueError('Duplicate experiment IDs in plan')
    if only and exclude:
        raise ValueError('Choose --only or --exclude')
    if any(key and key not in known for key in (only, exclude)):
        raise ValueError('Unknown experiment ID')
    chosen = [e for e in experiments if (not only or experiment_id(e) == only)
              and experiment_id(e) != exclude]
    if len(gpus) != len(chosen) or len(set(gpus)) != len(gpus):
        raise ValueError(f'Require exactly {len(chosen)} distinct GPU IDs, one per experiment')
    if any(not re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+)', g) for g in gpus):
        raise ValueError('Use comma-separated physical GPU indices or full GPU UUIDs')
    return [dict(experiment=experiment_id(e), gpu=g, dependency=e['dependency'])
            for e, g in zip(chosen, gpus)]


def dependency_status(suite, key, failed_since=None):
    home = Path(suite)/key
    marker = home/'COMPLETE.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        if record.get('status') != 'completed' or f"{record.get('arm')}/seed_{record.get('seed')}" != key:
            raise ValueError(f'Invalid dependency completion: {key}')
        checkpoint = Path(record['final_checkpoint'])
        if not checkpoint.is_relative_to(home.resolve()):
            raise ValueError('Dependency checkpoint is outside its experiment')
        if not all((checkpoint/name).is_file() for name in (
                'adapter_model.safetensors', 'recurft_recurrent.safetensors', 'recurft_config.json')):
            raise ValueError(f'Dependency completion has no usable checkpoint: {key}')
        return True
    for name in ('worker.json', 'status.json'):
        path = home/name
        if path.exists():
            record = json.loads(path.read_text())
            if (record.get('status') in ('failed', 'interrupted') and
                    (failed_since is None or record.get('updated_unix', 0) >= failed_since)):
                raise RuntimeError(f"Dependency {key} failed: {record.get('error', 'inspect its logs')}")
    return False


def dependency_source_parts(source):
    host, separator, path = source.partition(':')
    if (not separator or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', host)
            or not re.fullmatch(r'/[A-Za-z0-9_./-]+', path)):
        raise ValueError('Use user@host:/absolute/export/root with plain path components')
    return host, path


def sync_dependency(source, directory, seed):
    """Check READY over SSH, then copy/verify a complete bundle before publishing it."""
    from qwen3_4b_dependency import verify_bundle
    name = f'no_joint_seed_{seed}'
    host, path = dependency_source_parts(source)
    destination = Path(directory)/name
    if (destination/'READY.json').exists():
        verify_bundle(destination)
        return dict(returncode=0, message='Bundle already downloaded and verified')
    remote = path.rstrip('/')+'/'+name
    options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    ready = subprocess.run(['ssh', *options, '--', host, 'test -f '+shlex.quote(remote+'/READY.json')],
                           text=True, capture_output=True, timeout=30)
    if ready.returncode:
        return dict(returncode=ready.returncode, message=ready.stderr[-1000:] or 'Local checkpoint is not ready yet')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dependency-transfer-', dir=destination.parent) as temporary:
        copied = subprocess.run(['scp', '-r', *options, '--', host+':'+remote, temporary+'/'],
                                text=True, capture_output=True, timeout=3600)
        if copied.returncode:
            return dict(returncode=copied.returncode, message=copied.stderr[-1000:])
        incoming = Path(temporary)/name
        verify_bundle(incoming)
        if destination.exists():
            raise ValueError('Incomplete incoming bundle already exists; use a clean --dependency-dir')
        incoming.rename(destination)
    return dict(returncode=0, message='SSH checkpoint transfer verified')


def worker(args, plan):
    from common import write_json
    from qwen3_4b_dependency import import_bundle, publish
    from run_qwen3_4b_matrix import run_experiment
    experiment = next(e for e in plan['experiments'] if experiment_id(e) == args.worker)
    home = args.suite/args.worker
    with (home/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='running', phase='preflight', pid=os.getpid(),
                     experiment=args.worker, gpu=args.gpus,
                     dependency=experiment['dependency'])
        def save():
            state['updated_unix'] = time.time()
            write_json(home/'worker.json', state)
        save()
        try:
            dependency = experiment['dependency']
            while dependency and not dependency_status(args.suite, dependency, args.launch_started):
                state['phase'] = 'waiting_for_dependency'
                if args.dependency_dir and dependency == args.external_dependency:
                    transfer_ok = True
                    if args.dependency_source:
                        try:
                            state['dependency_transfer'] = sync_dependency(
                                args.dependency_source, args.dependency_dir, experiment['seed'])
                            transfer_ok = state['dependency_transfer']['returncode'] == 0
                        except subprocess.TimeoutExpired:
                            state['dependency_transfer'] = dict(error='SSH/scp timed out; will retry')
                            transfer_ok = False
                    if transfer_ok and import_bundle(args.suite, plan, experiment['seed'], args.dependency_dir):
                        continue
                save()
                time.sleep(args.poll_seconds)
            state['phase'] = 'training_and_evaluation'
            save()
            run_experiment(args.suite, plan, experiment, args.gpus, args.allow_shared_gpu)
            if args.publish_dependencies and experiment['arm'] == 'no_joint':
                state['phase'] = 'publishing_dependency'
                save()
                state['published_dependency'] = str(publish(
                    args.suite, plan, experiment['seed'], args.publish_dependencies))
            state.update(status='completed', phase='completed')
            save()
        except BaseException as exc:
            state.update(status='failed', phase='failed', error=repr(exc))
            save()
            raise


def supervise(args, plan, selected):
    from common import gpu_snapshot, write_json
    from training_contract import sha
    snapshots = {row['gpu']: gpu_snapshot(row['gpu']) for row in selected}
    if len({s['uuid'] for s in snapshots.values()}) != len(selected):
        raise ValueError('GPU aliases refer to the same physical device')
    allocated = os.environ.get('CUDA_VISIBLE_DEVICES')
    if allocated is not None:
        permitted = {gpu_snapshot(g.strip())['uuid'] for g in allocated.split(',') if g.strip()}
        if not {s['uuid'] for s in snapshots.values()} <= permitted:
            raise ValueError('Requested GPUs are outside CUDA_VISIBLE_DEVICES allocation')
    launch = args.suite/'worker_launches'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    launch.mkdir(parents=True)
    state = dict(status='running', pid=os.getpid(), assignments=selected, children=[],
                 gpu_before=snapshots, source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
                 allow_shared_gpu=args.allow_shared_gpu)
    def save():
        state['updated_unix'] = time.time()
        write_json(launch/'status.json', state)
    processes = []
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for row in selected:
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--suite', str(args.suite),
                   '--worker', row['experiment'], '--gpus', row['gpu'],
                   '--poll-seconds', str(args.poll_seconds), '--launch-started', str(args.launch_started)]
            for name in ('dependency_dir', 'dependency_source', 'external_dependency', 'publish_dependencies'):
                value = getattr(args, name)
                if value:
                    cmd += ['--'+name.replace('_', '-'), str(value)]
            if args.allow_shared_gpu:
                cmd.append('--allow-shared-gpu')
            path = launch/(row['experiment'].replace('/', '_')+'.log')
            with path.open('x') as log:
                proc = subprocess.Popen(cmd, cwd=args.suite/'source_snapshot', stdout=log,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
            processes.append(proc)
            state['children'].append(dict(**row, pid=proc.pid, log=str(path), returncode=None))
            save()
        while True:
            for proc, child in zip(processes, state['children']):
                child['returncode'] = proc.poll()
            save()
            if all(proc.returncode is not None for proc in processes):
                break
            time.sleep(5)
        state['status'] = 'completed' if all(p.returncode == 0 for p in processes) else 'failed'
        save()
        return 0 if state['status'] == 'completed' else 1
    except BaseException as exc:
        state.update(status='interrupted', error=repr(exc))
        for proc in processes:
            # Each worker owns its process group, including its training/evaluation children.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    select = parser.add_mutually_exclusive_group()
    select.add_argument('--exclude', help='For eight remote workers: no_joint/seed_42')
    select.add_argument('--only', help='For the local worker: no_joint/seed_42')
    select.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--external-dependency', help=argparse.SUPPRESS)
    parser.add_argument('--launch-started', type=float, default=time.time(), help=argparse.SUPPRESS)
    parser.add_argument('--dependency-dir', type=Path, help='Directory receiving exported checkpoint bundles')
    parser.add_argument('--dependency-source', help='Optional SSH source, e.g. user@host:/absolute/export/root')
    parser.add_argument('--publish-dependencies', type=Path, help='Export each completed staged run here')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--allow-shared-gpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.suite = args.suite.resolve()
    if args.poll_seconds <= 0:
        parser.error('--poll-seconds must be positive')
    for name in ('dependency_dir', 'publish_dependencies'):
        value = getattr(args, name)
        if value:
            setattr(args, name, value.resolve())
    if args.dependency_source and not args.dependency_dir:
        parser.error('--dependency-source requires --dependency-dir')
    if args.dependency_source:
        dependency_source_parts(args.dependency_source)
    # Training always uses the audited suite snapshot, including on the original host.
    snap = args.suite/'source_snapshot'
    sys.path[:0] = [str(snap/'local_setup'), str(snap/'scripts')]
    from matrix_audit import audit_suite
    plan = json.loads((args.suite/'plan.json').read_text())
    audit_suite(args.suite, plan)
    chosen = assignments(plan, [g.strip() for g in args.gpus.split(',')],
                         args.worker or args.only, args.exclude)
    if args.exclude:
        external = {r['dependency'] for r in chosen if r['dependency'] == args.exclude}
        if external:
            args.external_dependency = args.exclude
            if not dependency_status(args.suite, args.exclude) and not args.dependency_dir:
                parser.error('Excluded staged experiment needs --dependency-dir for its checkpoint handoff')
    for row in chosen:
        print(json.dumps(row), flush=True)
    if args.dry_run:
        return
    if args.worker:
        worker(args, plan)
    else:
        raise SystemExit(supervise(args, plan, chosen))


if __name__ == '__main__':
    main()
