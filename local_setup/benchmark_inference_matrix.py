"""Paired inference ablations on frozen checkpoints; no training or checkpoint writes."""
import os
os.environ.update(DISABLE_VERSION_CHECK='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1')
os.environ.pop('CUDA_MPS_PIPE_DIRECTORY', None)
import argparse
import contextlib
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

SOURCE_AT_START = Path(__file__).read_bytes()
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TRITON_CACHE_DIR',str(ROOT/'runs/triton_cache'))
sys.path[:0] = [str(ROOT/'LLaMA-Factory/src'), str(ROOT/'LLaMA-Factory/experiments/recurft_math')]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file
import recurft_speculative_generate as dec
from recurft_rollout_eval import build_recurrent_module


def dump(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def baseline_for(name):
    return 'greedy'+('_last' if '_last' in name else '')+('_lower' if '_lower' in name else '')+('_fused' if '_fused' in name else '')


def monitor(out, stop):
    with (out/'gpu_telemetry.jsonl').open('a') as f:
        while not stop.is_set():
            row = {'time': time.time()}
            for name, query in [('gpu', '--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm'),
                                ('processes', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory')]:
                try:
                    row[name] = subprocess.check_output(['nvidia-smi', query, '--format=csv,noheader'], text=True, timeout=5)
                except Exception as e:
                    row[name] = repr(e)
            f.write(json.dumps(row)+'\n'); f.flush(); stop.wait(5)


@contextlib.contextmanager
def target_options(last_prefill=False, profiling=False):
    original = dec.target_forward
    def forward(model, ids, **kw):
        # Prefill only: never remove logits needed to verify a speculative block.
        if last_prefill and kw.get('past_key_values') is None and kw.get('start_position') == 0:
            kw['logits_to_keep'] = 1
        scope = torch.profiler.record_function('target_q'+str(ids.shape[1])) if profiling else contextlib.nullcontext()
        with scope:
            return original(model, ids, **kw)
    dec.target_forward = forward
    try:
        yield
    finally:
        dec.target_forward = original


def summarize(rows):
    rows = [r for r in rows if r['phase'] == 'measurement']
    results = {}
    if not rows:
        return results
    for name in rows[0]['values']:
        vals = [r['values'][name] for r in rows]
        bn = vals[0].get('baseline','greedy_last' if name.endswith('_last') else 'greedy')
        base = [r['values'][bn] for r in rows]
        seconds = sum(v['wall_seconds'] for v in vals); tokens = sum(len(v['token_ids']) for v in vals)
        bs = sum(v['wall_seconds'] for v in base); bt = sum(len(v['token_ids']) for v in base)
        hist = {str(i): 0 for i in range(4)}
        for v in vals:
            for block in v.get('block_records', []):
                if block['proposed_len'] > 1:
                    key=str(block['accepted_len']-1);hist[key]=hist.get(key,0)+1
        results[name] = dict(measurements=len(vals), unique_questions=len({r['sample'] for r in rows}),
            seconds=seconds, tokens=tokens, tokens_per_second=tokens/seconds,
            baseline=bn, baseline_seconds=bs, baseline_tokens=bt, wall_speedup=bs/seconds,
            throughput_speedup=(tokens/seconds)/(bt/bs),
            greedy_exact=sum(v['token_ids']==b['token_ids'] for v,b in zip(vals,base)),
            answer_correct=sum(v['answer_correct'] for v in vals), capped=sum(v['hit_token_cap'] for v in vals),
            target_calls=sum(v['target_calls'] for v in vals),
            draft_tokens=sum(v.get('draft_tokens',0) for v in vals),
            accepted_draft_tokens=sum(v.get('accepted_draft_tokens',0) for v in vals),
            accepted_draft_prefix_histogram=hist,
            fast_strict_blocks=sum(v.get('fast_strict_blocks',0) for v in vals))
    eq={}
    pairs=[('greedy','greedy_last'),('after_b3','after_b3_last'),('before_b3','before_b3_lower'),('after_b3','after_b3_lower')]
    pairs += [(n.removesuffix('_fused'),n) for n in rows[0]['values'] if n.endswith('_fused')]
    for original,changed in pairs:
        if original in rows[0]['values'] and changed in rows[0]['values']:
            eq[original+'_vs_'+changed]=sum(r['values'][original]['token_ids']==r['values'][changed]['token_ids'] for r in rows)
    results['optimization_equivalence'] = dict(exact_counts=eq,measurements=len(rows))
    return results


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=8); p.add_argument('--start', type=int, default=0)
    p.add_argument('--repeats', type=int, default=2); p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--merge-target', action='store_true')
    p.add_argument('--profile-tokens', type=int, default=0)
    p.add_argument('--lower-right',action='store_true')
    p.add_argument('--fused-norms',action='store_true')
    p.add_argument('--compact',action='store_true',help='Add GPU-resident compact fixed-strict decoder; all baselines also use last-only prefill logits')
    p.add_argument('--t-graph',action='store_true')
    p.add_argument('--cycle-graph',action='store_true')
    p.add_argument('--fused-decision',action='store_true')
    p.add_argument('--branching',action='store_true')
    p.add_argument('--lookup',action='store_true')
    p.add_argument('--shortlist-vocab',type=Path)
    p.add_argument('--gpu-greedy',action='store_true',help='Reuse GPU argmax IDs as greedy inputs too, avoiding CPU-to-GPU token reconstruction')
    p.add_argument('--variants',help='Optional subset of methods, comma separated; include matching greedy baselines')
    a = p.parse_args()
    if min(a.samples,a.repeats,a.max_new_tokens) < 1 or min(a.start,a.profile_tokens) < 0:
        p.error('Invalid limits')
    if a.t_graph and not (a.compact and a.fused_norms):p.error('--t-graph requires --compact --fused-norms')
    if a.cycle_graph and not a.t_graph:p.error('--cycle-graph requires --t-graph')
    if a.gpu_greedy and not a.compact:p.error('--gpu-greedy requires --compact for shared last-only prefill')
    a.output.mkdir(parents=True, exist_ok=False)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES','0')
    lock = open('/tmp/inception-eval-'+hashlib.sha256(visible.encode()).hexdigest()[:16]+'.lock','w')
    dump(a.output/'status.json', {'status':'waiting_for_gpu_lock','gpu':visible})
    fcntl.flock(lock,fcntl.LOCK_EX)
    dump(a.output/'status.json',{'status':'loading_and_validating','gpu':visible,'time':time.time()})
    stop = threading.Event(); threading.Thread(target=monitor,args=(a.output,stop),daemon=True).start()
    assets = {str(path):sha(path) for ck in [a.before,a.after] for path in
              [ck/'recurft_config.json',ck/'recurft_recurrent.safetensors',ck/'adapter_model.safetensors']}
    assets[str(a.data)] = sha(a.data)
    if a.shortlist_vocab:assets[str(a.shortlist_vocab)]=sha(a.shortlist_vocab)
    sources = [Path(__file__),Path(dec.__file__),ROOT/'LLaMA-Factory/src/llamafactory/model/model_utils/recurft.py',Path(__file__).with_name('lower_right_target.py')]
    if a.fused_norms:
        sources += [Path(__file__).with_name('fused_rmsnorm.py'),Path(__file__).with_name('validate_fused_rms.py')]
    if a.compact:sources.append(Path(__file__).with_name('compact_decode.py'))
    if a.t_graph:sources += [Path(__file__).with_name('recurrent_graph.py'),Path(__file__).with_name('validate_recurrent_graph.py')]
    if a.cycle_graph:sources += [Path(__file__).with_name('draft_cycle_graph.py'),Path(__file__).with_name('validate_draft_cycle_graph.py')]
    if a.shortlist_vocab:sources.append(Path(__file__).with_name('shortlist_boundary.py'))
    if a.lookup:sources.append(Path(__file__).with_name('lookup_decode.py'))
    if a.branching:sources.append(Path(__file__).with_name('branched_decode.py'))
    if a.fused_decision:sources.append(Path(__file__).with_name('strict_prefix_kernel.py'))
    for path in sources:
        if path.resolve()==Path(__file__).resolve():(a.output/path.name).write_bytes(SOURCE_AT_START)
        else:shutil.copy2(path,a.output/path.name)
    # Execute the frozen helper copies; subsequent workspace edits cannot alter a queued run.
    sys.path.insert(0,str(a.output.resolve()))
    dump(a.output/'manifest.json',dict(args=vars(a),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes={str(s):sha(a.output/s.name) for s in sources},asset_hashes=assets,
        torch=torch.__version__,transformers=__import__('transformers').__version__,
        shared_gpu=True,measurement='outer CUDA-synchronized wall; all component counters are host dispatch durations',
        precision='BF16 SDPA, no target quantization, T LoRA merged',baseline='same target model and target optimizations as speculative',
        warmup='one excluded question per repeat, up to 64 generated tokens for every method',
        order='rotate methods by question and repeat; merge states use separate processes and own baselines'))
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False; torch.cuda.set_per_process_memory_fraction(.60)
    if a.lower_right:
        from lower_right_target import validate_cuda_lower_right
        dump(a.output/'lower_right_numerical_validation.json',validate_cuda_lower_right())
    if a.t_graph:
        from validate_recurrent_graph import validate as validate_graph
        dump(a.output/'t_graph_validation.json',validate_graph())
    if a.cycle_graph:
        from validate_draft_cycle_graph import validate_cycle
        dump(a.output/'draft_cycle_graph_validation.json',validate_cycle())
    if a.fused_decision:
        from strict_prefix_kernel import validate as validate_decision
        dump(a.output/'strict_prefix_kernel_validation.json',validate_decision())
    adapter1=load_file(str(a.before/'adapter_model.safetensors'));adapter2=load_file(str(a.after/'adapter_model.safetensors'))
    assert adapter1.keys()==adapter2.keys() and all(torch.equal(adapter1[k],adapter2[k]) for k in adapter1), 'Target adapters differ'
    del adapter1,adapter2
    tokenizer=AutoTokenizer.from_pretrained(a.after,local_files_only=True)
    tok2=AutoTokenizer.from_pretrained(a.before,local_files_only=True)
    assert tokenizer.get_vocab()==tok2.get_vocab() and tokenizer.chat_template==tok2.chat_template
    base=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    recurrent={};metadata={};load_audit={}
    for branch,path in [('before',a.before),('after',a.after)]:
        meta=json.loads((path/'recurft_config.json').read_text());metadata[branch]=meta
        module=build_recurrent_module(base,path,meta).cuda().eval()
        # Checkpoints deliberately omit frozen base weights; require every saved adapter/head field.
        state=load_file(str(path/'recurft_recurrent.safetensors'))
        expected={n for n in module.state_dict() if 'lora_' in n or n.startswith(
            ('verifier_','boundary_','token_conditioning_','multistep_residual_','multistep_step1_residual_'))}
        assert set(state)==expected, {'missing_saved_fields':sorted(expected-set(state)), 'unexpected':sorted(set(state)-expected)}
        check=module.load_state_dict(state,strict=False); del state
        assert not check.unexpected_keys and not (expected & set(check.missing_keys))
        load_audit[branch]={'load':str(check),'merged_modules':module.merge_lora_for_inference()}
        recurrent[branch]=module
    model=PeftModel.from_pretrained(base,a.after).cuda().eval()
    if a.merge_target:
        model=model.merge_and_unload(safe_merge=True).eval()
        assert not any('lora_A' in n or 'lora_B' in n for n,_ in model.named_parameters())
    dump(a.output/'load_audit.json',load_audit)
    if a.fused_norms:
        from validate_fused_rms import validate
        dump(a.output/'fused_norm_numerical_validation.json',validate())
    argv=sys.argv
    try:
        sys.argv=['decoder','--model-name-or-path',str(a.model),'--checkpoint',str(a.after),'--data-file',str(a.data),'--output-json',str(a.output/'unused.json')]
        cfg0=vars(dec.parse_args())
    finally:
        sys.argv=argv
    cfg0.update(mode='fixed',draft_commit_policy='target_match',target_match_lambda=1.,ngram_draft_mode='off',draft_logit_source='boundary',
        min_block_tokens=1,max_new_tokens=a.max_new_tokens,max_prompt_tokens=1024,dtype='bf16',attn_implementation='sdpa',
        inplace_draft_cache=True,merge_recurrent_lora=True,merge_target_lora=a.merge_target,batched_draft_boundary=True,
        fast_strict_verification=True,compact_runtime_stats=True,production_async_timing=True,single_gpu_parallel_draft=False,
        defer_correction_to_next_verify=True,reuse_verify_cache_for_correction=False,lazy_t_sync=True,defer_t_init_until_latent=True,
        latent_draft_min_position=0,omit_target_attention_mask=True,anchor_hook_hidden_states=True)
    settings={}
    for branch in ['before','after']:
        for block in [2,3,4]:
            cfg=argparse.Namespace(**cfg0);cfg.max_block_tokens=cfg.short_block=cfg.medium_block=cfg.long_block=block
            cfg.checkpoint=str(a.before if branch=='before' else a.after)
            dec.validate_fast_strict_verification(cfg);settings[f'{branch}_b{block}']=cfg
    settings['after_b3_last']=argparse.Namespace(**vars(settings['after_b3']))
    if a.lower_right:
        for branch in ['before','after']:
            for block in [2,3,4]:
                key=f'{branch}_b{block}'
                settings[key+'_lower']=argparse.Namespace(**vars(settings[key]))
    available=['greedy','greedy_last',*settings]+(['greedy_lower'] if a.lower_right else [])
    if a.fused_norms:
        for name in list(settings):
            if not name.endswith('_last'):
                settings[name+'_fused']=argparse.Namespace(**vars(settings[name]))
                available.append(name+'_fused')
        available+=['greedy_fused']+(['greedy_lower_fused'] if a.lower_right else [])
    if a.gpu_greedy and a.lower_right and a.fused_norms:
        available.append('greedy_legacy_lower_fused')
    if a.compact:
        for name in list(settings):
            if not name.endswith('_last'):
                settings[name+'_compact']=argparse.Namespace(**vars(settings[name]))
                available.append(name+'_compact')
    if a.t_graph:
        for name in list(settings):
            if name.endswith('_compact') and '_fused' in name:
                cfg=argparse.Namespace(**vars(settings[name]));cfg.t_graph=True
                settings[name+'_tgraph']=cfg;available.append(name+'_tgraph')
    if a.cycle_graph:
        for name in list(settings):
            if name.endswith('_tgraph'):
                cfg=argparse.Namespace(**vars(settings[name]));cfg.cycle_graph=True
                settings[name+'_cycle']=cfg;available.append(name+'_cycle')
    if a.fused_decision:
        for name in list(settings):
            if '_compact' in name:
                cfg=argparse.Namespace(**vars(settings[name]));cfg.fused_decision=True
                settings[name+'_fdecision']=cfg;available.append(name+'_fdecision')
    if a.branching:
        for name in list(settings):
            if name.endswith('_compact_tgraph'):
                for width in (2,3):
                    cfg=argparse.Namespace(**vars(settings[name]));cfg.branch_width=width
                    settings[name+'_branch'+str(width)]=cfg;available.append(name+'_branch'+str(width))
    if a.lookup:
        for name in list(settings):
            if name.endswith('_compact_tgraph'):
                for only in (False,True):
                    cfg=argparse.Namespace(**vars(settings[name]));cfg.lookup_only=only;cfg.lookup_max_drafts=8
                    cfg.local_execution_route='lookup_only' if only else 'prompt_lookup_then_neural'
                    key=name+('_lookuponly' if only else '_lookup')
                    settings[key]=cfg;available.append(key)
    if a.shortlist_vocab:
        for name in list(settings):
            if name.endswith('_compact_tgraph'):
                for size in (2048,4096,8192):
                    cfg=argparse.Namespace(**vars(settings[name]));cfg.shortlist_size=size;cfg.shortlist_vocab=str(a.shortlist_vocab)
                    key=name+'_sl'+str(size);settings[key]=cfg;available.append(key)
    methods=a.variants.split(',') if a.variants else available
    assert set(methods)<=set(available) and len(set(methods))==len(methods)
    for name in methods:
        baseline=baseline_for(name)
        assert baseline in methods, f'Missing matching baseline {baseline}'
    dump(a.output/'settings.json',{k:vars(v) for k,v in settings.items()})
    dec._SYNC_COMPONENT_TIMING=False;device=torch.device('cuda')
    data=[dict(json.loads(line),sample=i) for i,line in enumerate(a.data.read_text().splitlines()) if line.strip()]
    chosen=data[a.start:a.start+a.samples];assert len(chosen)==a.samples
    warmup=next(r for r in data if r['sample'] not in {x['sample'] for x in chosen})
    def invoke(name,prompt,cap):
        from lower_right_target import lower_right_target
        if '_fused' in name:
            from fused_rmsnorm import fused_qwen_norms
            norm_context=fused_qwen_norms(model,*recurrent.values())
        else:
            norm_context=contextlib.nullcontext({})
        with target_options(a.compact or '_last' in name), (lower_right_target(dec,max_batch=getattr(settings.get(name),'branch_width',1)) if '_lower' in name else contextlib.nullcontext({})) as audit, norm_context as norms:
            if name.startswith('greedy'):
                if a.gpu_greedy and '_legacy' not in name:
                    from compact_decode import greedy_decode
                else:
                    greedy_decode=dec.greedy_decode
                value=greedy_decode(model,prompt,cap,tokenizer.eos_token_id,metadata['after']['recurrent_hidden_state_index'],device,omit_target_attention_mask=True)
            else:
                branch=name.split('_')[0];cfg=argparse.Namespace(**vars(settings[name]));cfg.max_new_tokens=cap
                if '_lookup' in name:
                    from lookup_decode import speculative_decode
                elif '_branch' in name:
                    from branched_decode import speculative_decode
                elif '_compact' in name:
                    from compact_decode import speculative_decode
                else:
                    speculative_decode=dec.speculative_decode
                value=speculative_decode(model,recurrent[branch],metadata[branch],prompt,cfg,tokenizer.eos_token_id,device,tokenizer=tokenizer)
            value['lower_right_audit']=dict(audit)
            value['fused_norm_audit']=dict(norms)
            value['baseline']=baseline_for(name)
            return value
    if a.profile_tokens:
        prompt=dec.make_prompt(tokenizer,warmup['question'],cfg0['question_prefix'],cfg0['question_suffix'],True).cuda()[:,-1024:]
        invoke('after_b3',prompt,16)
        orig_fwd=recurrent['after'].forward_with_cache;orig_boundary=recurrent['after'].boundary_logits
        def rfwd(*args,**kw):
            with torch.profiler.record_function('recurrent_q'+str(args[0].shape[1])):
                return orig_fwd(*args,**kw)
        def boundary(*args,**kw):
            with torch.profiler.record_function('boundary_q'+str(args[0].shape[1])):
                return orig_boundary(*args,**kw)
        recurrent['after'].forward_with_cache=rfwd;recurrent['after'].boundary_logits=boundary
        cfg=argparse.Namespace(**vars(settings['after_b3']));cfg.max_new_tokens=a.profile_tokens
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            with target_options(profiling=True):
                dec.speculative_decode(model,recurrent['after'],metadata['after'],prompt,cfg,tokenizer.eos_token_id,device,tokenizer=tokenizer)
        recurrent['after'].forward_with_cache=orig_fwd;recurrent['after'].boundary_logits=orig_boundary
        prof.export_chrome_trace(str(a.output/'diagnostic_trace.json'))
        events=[]
        for e in prof.key_averages():
            events.append(dict(name=e.key,device_type=str(e.device_type),count=e.count,cpu_total_us=e.cpu_time_total,self_cpu_us=e.self_cpu_time_total,
                device_total_us=e.device_time_total,self_device_us=e.self_device_time_total))
        dump(a.output/'diagnostic_events.json',events)
        (a.output/'diagnostic_table.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
    rows=[]
    for rep in range(a.repeats):
        for qi,row in enumerate([warmup]+chosen):
            prompt=dec.make_prompt(tokenizer,row['question'],cfg0['question_prefix'],cfg0['question_suffix'],True).cuda()[:,-1024:]
            order=methods.copy();shift=(rep+qi)%len(order);order=order[shift:]+order[:shift]
            values={};answer=dec.normalize_answer(dec.extract_answer(str(row['answer'])))
            for name in order:
                value,seconds=dec.wall_timed(lambda:invoke(name,prompt,min(64,a.max_new_tokens) if qi==0 else a.max_new_tokens),device)
                text=tokenizer.decode(value['token_ids'],skip_special_tokens=True)
                value.update(wall_seconds=seconds,text=text,answer_correct=answer is not None and dec.normalize_answer(dec.extract_answer(text))==answer,
                    hit_token_cap=len(value['token_ids'])==(min(64,a.max_new_tokens) if qi==0 else a.max_new_tokens))
                if not name.startswith('greedy'):
                    assert value.get('accepted_unchecked_tokens',0)==0 and value.get('accepted_mismatch_tokens',0)==0
                    assert value['fast_strict_blocks']==len(value['block_records'])
                values[name]=value
            record=dict(sample=row['sample'],repeat=rep,phase='warmup' if qi==0 else 'measurement',question=row['question'],answer=answer,
                order=order,prompt_tokens=prompt.shape[1],values=values,time=time.time())
            rows.append(record)
            with (a.output/'results.jsonl').open('a') as f:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
            dump(a.output/'summary.json',summarize(rows))
            dump(a.output/'status.json',dict(status='running',repeat=rep+1,completed_in_repeat=qi,samples=a.samples,time=time.time()))
            print(json.dumps(dict(repeat=rep+1,sample=row['sample'],phase=record['phase'],seconds={k:round(v['wall_seconds'],3) for k,v in values.items()})),flush=True)
    assert all(sha(Path(path))==digest for path,digest in assets.items()), 'Assets changed during run'
    dump(a.output/'status.json',dict(status='completed',measurements=a.samples*a.repeats,time=time.time()))
    stop.set()


if __name__=='__main__':
    sys.modules['benchmark_inference_matrix']=sys.modules[__name__]
    try:
        main()
    except Exception as e:
        if '--output' in sys.argv and not isinstance(e,FileExistsError):
            output=Path(sys.argv[sys.argv.index('--output')+1])
            if output.is_dir():
                dump(output/'status.json',{'status':'failed','error':repr(e),'time':time.time()})
        raise
