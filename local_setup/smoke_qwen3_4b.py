"""Short real-model checks of all scopes, validation, transitions and paired inference."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
from qwen3_4b_recipe import architecture, stage_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', required=True)
    a = p.parse_args()
    run = a.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    r = yaml.safe_load((ROOT/'configs/experimental/qwen3_4b_matrix.yaml').read_text())
    arch = architecture(json.loads((a.model/'config.json').read_text()), r)
    data = run/'data'
    data.mkdir()
    # Force padded training batches to reach the real 1024-token cutoff.
    rows = [dict(query=f'Calculate {i}+1 and explain.', response=('We add one to the integer. '*300)+f' The answer is {i+1}.') for i in range(8)]
    (data/'train.json').write_text(json.dumps(rows))
    (data/'validation.json').write_text(json.dumps(rows[:2]))
    (data/'dataset_info.json').write_text(json.dumps({n:dict(file_name=f, columns=dict(prompt='query',response='response')) for n,f in [('matrix_train','train.json'),('matrix_validation','validation.json')]}))
    (data/'inference.jsonl').write_text('\n'.join(json.dumps(row) for row in [
        dict(question='What is 2 plus 3?',answer='5'), dict(question='What is 1 plus 1?',answer='2')])+'\n')
    prev = None
    for name, stage, trainable, scope in [('core','core',True,'target_t'), ('multistep','multistep',True,'target_t'),
        ('head','head',False,'head'), ('post_joint','joint',False,'t_head'), ('full_joint','joint',True,'target_t_head')]:
        folder = run/name
        folder.mkdir()
        cfg = stage_config(a.model, data, folder/'checkpoints', 42, stage, 2, arch, r, trainable)
        cfg.update(gradient_accumulation_steps=1, logging_steps=1, save_steps=2,
                   eval_steps=2, preprocessing_num_workers=1)
        path = folder/'train.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cmd = [sys.executable, str(ROOT/'local_setup/run_qwen3_stage.py'), '--config',str(path), '--gpu',a.gpu,
               '--allow-shared-gpu','--min-free-mib','40000','--memory-fraction','0.7',
               '--trainable-scope',scope,'--require-tied-embeddings']
        if prev and name != 'full_joint':
            cmd += ['--initialize-from',str(prev)]
        lock = open('/tmp/inception-eval-'+hashlib.sha256(a.gpu.encode()).hexdigest()[:16]+'.lock','w')
        with lock, (folder/'stdout.log').open('w') as log:
            fcntl.flock(lock,fcntl.LOCK_EX)
            print('Running '+name, flush=True)
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = json.loads((folder/'execution.json').read_text())
        assert record['observed_max_sequence_length']==1024
        assert record['tied_vocabulary_weights_frozen']
        expected = set(record['trainable_groups'])
        assert all(set(x['gradient_l2'])==expected and all(v>0 for v in x['gradient_l2'].values()) for x in record['gradient_checks'])
        assert all(v['delta_l2']>0 for v in record['sampled_adapter_changes'].values())
        print(json.dumps(dict(stage=name, groups=record['trainable_groups'], peak_gib=record['peak_allocated_gib'])), flush=True)
        prev = folder/'checkpoints'
    import os
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu)
    cmd = [sys.executable, str(ROOT/'local_setup/benchmark_trained_target.py'), '--model',str(a.model),
           '--checkpoint',str(prev), '--data',str(data/'inference.jsonl'), '--output',str(run/'inference'),
           '--samples','1','--repeats','1','--max-new-tokens','32','--merge-target','--lower-right',
           '--fused-norms','--compact','--t-graph','--gpu-greedy','--lookup',
           '--variants','greedy_lower_fused,after_b3_lower_fused_compact_tgraph,after_b3_lower_fused_compact_tgraph_lookup,after_b3_lower_fused_compact_tgraph_lookuponly']
    with (run/'inference.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Smoke completed: '+str(run), flush=True)


if __name__ == '__main__':
    main()
