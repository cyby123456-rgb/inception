"""Two-stage adaptive drafting, paired against fixed drafting and greedy. See TWO_STAGE_ADAPTIVE.md."""
import os
os.environ.update(DISABLE_VERSION_CHECK='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1')
os.environ.pop('CUDA_MPS_PIPE_DIRECTORY',None)
from pathlib import Path
import sys,json,argparse,time,subprocess,threading,fcntl,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'LLaMA-Factory/src'),str(ROOT/'LLaMA-Factory/experiments/recurft_math')]
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
import adaptive_two_stage_runtime as dec
from recurft_rollout_eval import build_recurrent_module



def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)

def monitor_gpu(output,stop):
    with (output/'gpu_telemetry.jsonl').open('w') as f:
        while not stop.is_set():
            try:
                row={'time':time.time(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.sm','--format=csv,noheader,nounits'],text=True,timeout=5),'processes':gpu_snapshot()}
            except Exception as e:row={'time':time.time(),'error':str(e)}
            f.write(json.dumps(row)+'\n');f.flush();stop.wait(5)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=16);p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--variants',default='fixed,adaptive');p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--compare-checkpoint',type=Path,help='Use this recurrent checkpoint for old_ prefixed variants; target adapters must be identical.')
    p.add_argument('--dtype',choices=['fp32','bf16','fp16'],default='fp32')
    p.add_argument('--omit-baseline-mask',action='store_true',default=True,help='All variants omit the all-ones target attention mask.')
    p.add_argument('--data',type=Path,default=ROOT.parent/'datasets/GSM8K/test_official.jsonl')
    p.add_argument('--model',type=Path,default=ROOT.parent/'models/Qwen--Qwen3-8B/snapshots/master')
    p.add_argument('--max-prompt-tokens',type=int,default=1024)
    p.add_argument('--max-drafts',type=int,default=4,help='Draft tokens, excluding the already known target anchor.')
    p.add_argument('--draft-start-token',type=int,default=0,help='Allow T only after this many response tokens have been committed; applies to fixed and adaptive variants.')
    p.add_argument('--draft-start-tokens',help='Comma-separated start positions; compare all positions within each question using one greedy baseline.')
    p.add_argument('--target-high-margin',type=float,default=4.0)
    p.add_argument('--draft-margin',type=float,default=1.0,help='Raw top1-top2 logit gap; NOT a softmax probability.')
    p.add_argument('--acceptance-window',type=int,default=8)
    p.add_argument('--explore-drafts',type=int,default=1)
    p.add_argument('--good-acceptance',type=float,default=.25)
    p.add_argument('--strong-acceptance',type=float,default=.60)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--route',choices=['auto','boundary','tail'],default='auto')
    p.add_argument('--adaptive-margin',type=float,default=2.0)
    p.add_argument('--adaptive-failures',type=int,default=2)
    p.add_argument('--adaptive-cooldown',type=int,default=8)
    args=p.parse_args()
    if not (args.samples > 0 and args.repeats > 0 and args.start_index >= 0 and args.max_new_tokens > 0 and args.max_prompt_tokens > 0):
        p.error('Require positive samples/repeats/token limits and nonnegative start index')
    if args.draft_start_token < 0:
        p.error('draft-start-token must be nonnegative')
    if not (1 <= args.explore_drafts <= args.max_drafts and args.acceptance_window > 0):
        p.error('Require 1 <= explore-drafts <= max-drafts and positive acceptance-window')
    if not (0 <= args.adaptive_margin <= args.target_high_margin and args.draft_margin >= 0 and 0 <= args.good_acceptance <= args.strong_acceptance <= 1):
        p.error('Invalid margin or acceptance thresholds')
    if not (args.adaptive_failures > 0 and args.adaptive_cooldown > 0):
        p.error('Cooldown settings must be positive')
    variants=args.variants.split(',')
    if not set(variants) <= {'fixed','adaptive','old_fixed','old_adaptive'} or len(set(variants)) != len(variants):
        p.error('Variants: fixed,adaptive,old_fixed,old_adaptive (unique)')
    if any(n.startswith('old_') for n in variants) and not args.compare_checkpoint:
        p.error('old_ variants require --compare-checkpoint')
    if args.route == 'tail':
        p.error('This controlled runner requires a trained boundary head; use --route boundary or auto')
    starts=None
    if args.draft_start_tokens is not None:
        try:
            starts=[int(x) for x in args.draft_start_tokens.split(',')]
        except ValueError:
            p.error('draft-start-tokens must be comma-separated integers')
        if not starts or min(starts)<0 or len(starts)!=len(set(starts)):
            p.error('draft-start-tokens must be unique, nonnegative positions')
        args.variants=','.join(f'{name}_start{start}' for name in variants for start in starts)
    if args.dry_run:
        print(json.dumps(vars(args),default=str,indent=2));return
    args.output.mkdir(parents=True,exist_ok=False)
    import shutil
    source_hashes={}
    for filename in [Path(__file__), Path(dec.__file__), Path(__file__).with_name('adaptive_two_stage_policy.py')]:
        shutil.copy2(filename,args.output/filename.name)
        source_hashes[str(filename.resolve())]=hashlib.sha256(filename.read_bytes()).hexdigest()
    (args.output/'run_manifest.json').write_text(json.dumps({'args':vars(args),'sources':source_hashes,'started_unix':time.time(),'shared_gpu':True},default=str,indent=2))
    # All our milestone/final evaluators share this lock; training and unrelated
    # users' processes remain untouched. Acquire before allocating the model.
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    evaluation_lock=open('/tmp/inception-eval-'+hashlib.sha256(visible.encode()).hexdigest()[:16]+'.lock','w')
    print(json.dumps({'evaluation_gpu':visible,'status':'waiting_for_evaluation_lock'}),flush=True)
    fcntl.flock(evaluation_lock,fcntl.LOCK_EX)
    print(json.dumps({'evaluation_gpu':visible,'status':'evaluation_lock_acquired'}),flush=True)
    stop=threading.Event();monitor=threading.Thread(target=monitor_gpu,args=(args.output,stop),daemon=True);monitor.start()
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.60);torch.backends.cuda.matmul.allow_tf32=False
    saved_argv=sys.argv
    try:
        sys.argv=[str(dec.__file__),'--model-name-or-path',str(args.model),'--checkpoint',str(args.checkpoint),'--data-file',str(args.data),'--output-json',str(args.output/'result.json')]
        cfg0=vars(dec.parse_args())
    finally:
        sys.argv=saved_argv
    cfg0.update(max_prompt_tokens=args.max_prompt_tokens, mode='fixed',
        draft_commit_policy='target_match', target_match_lambda=1.0,
        ngram_draft_mode='off', draft_logit_source='boundary',
        max_block_tokens=args.max_drafts+1, min_block_tokens=1,
        short_block=args.max_drafts+1, medium_block=args.max_drafts+1,
        long_block=args.max_drafts+1, production_async_timing=True,
        compact_runtime_stats=True, fast_strict_diagnostics=True,
        anchor_hook_hidden_states=True, omit_target_attention_mask=True,
        batched_draft_boundary=False, batched_draft_tail=False,
        reuse_verify_cache_for_correction=True, lazy_t_sync=True,
        defer_t_init_until_latent=True, auto_route_after_latent_drafts=0,
        latent_draft_min_position=args.draft_start_token,
        auto_route_early_zero_after=0, auto_route_reevaluate_every=0,
        merge_target_lora=True, merge_recurrent_lora=True,
        attn_implementation='sdpa')
    meta=json.loads((args.checkpoint/'recurft_config.json').read_text())
    assert meta.get('boundary_head_rank',0)>0, 'Checkpoint requires trained boundary head'
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    dtype={'fp32':torch.float32,'bf16':torch.bfloat16,'fp16':torch.float16}[args.dtype]
    base=AutoModelForCausalLM.from_pretrained(cfg0['model_name_or_path'],torch_dtype=dtype,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    recurrent=build_recurrent_module(base,args.checkpoint,meta).cuda().eval()
    old_recurrent=None;old_meta=None
    if args.compare_checkpoint:
        from safetensors.torch import load_file
        a=load_file(str(args.checkpoint/'adapter_model.safetensors'))
        b=load_file(str(args.compare_checkpoint/'adapter_model.safetensors'))
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a),'Target adapters differ'
        del a,b
        old_meta=json.loads((args.compare_checkpoint/'recurft_config.json').read_text())
        assert old_meta['recurrent_hidden_state_index']==meta['recurrent_hidden_state_index']
        assert old_meta.get('boundary_head_rank',0)>0
        other_tokenizer=AutoTokenizer.from_pretrained(args.compare_checkpoint,local_files_only=True)
        assert tokenizer.get_vocab()==other_tokenizer.get_vocab() and tokenizer.chat_template==other_tokenizer.chat_template, 'Tokenizer/template differs'
        old_recurrent=build_recurrent_module(base,args.compare_checkpoint,old_meta).cuda().eval()
        old_recurrent.merge_lora_for_inference()
    model=PeftModel.from_pretrained(base,args.checkpoint).merge_and_unload(safe_merge=True).cuda().eval()
    recurrent.merge_lora_for_inference();dec._SYNC_COMPONENT_TIMING=False
    original=[dict(json.loads(line),sample=i) for i,line in enumerate(args.data.read_text().splitlines()) if line.strip()]
    chosen=original[args.start_index:args.start_index+args.samples]
    assert len(chosen)==args.samples
    # Unscored warmup may reuse a selected question when the dataset is tiny.
    chosen_ids={r['sample'] for r in chosen}
    warmup=next((r for r in original if r['sample'] not in chosen_ids),chosen[0])
    questions=[warmup]+chosen
    results=[];settings={};variants=args.variants.split(',');started=time.time()
    for vi,name in enumerate(variants):
        base_name=name.rsplit('_start',1)[0] if starts is not None else name
        start_position=int(name.rsplit('_start',1)[1]) if starts is not None else args.draft_start_token
        mode=base_name.removeprefix('old_')
        if name.startswith('old_'):assert old_recurrent is not None
        cfg=argparse.Namespace(**cfg0);cfg.checkpoint=str(args.compare_checkpoint if name.startswith('old_') else args.checkpoint);cfg.dtype=args.dtype
        cfg.max_new_tokens=args.max_new_tokens
        cfg.latent_draft_min_position=start_position
        cfg.two_stage_adaptive=mode=='adaptive'
        if mode=='adaptive':
            cfg.mode='heuristic'
            cfg.cheap_verifier_policy='logit_margin'
            cfg.cheap_verifier_margin_skip_below=args.adaptive_margin
            cfg.cheap_verifier_margin_block2_below=args.target_high_margin
            cfg.min_draft_margin=args.draft_margin
            cfg.draft_cooldown_after_failures=args.adaptive_failures
            cfg.draft_cooldown_cycles=args.adaptive_cooldown
            for key in ('max_drafts','explore_drafts','acceptance_window','good_acceptance','strong_acceptance'):
                setattr(cfg,key,getattr(args,key))
        settings[name]=cfg
    for repeat in range(args.repeats):
      for qi,row in enumerate(questions):
        if repeat>0 and qi==0:continue
        prompt=dec.make_prompt(tokenizer,row['question'],cfg0['question_prefix'],cfg0['question_suffix'],True).cuda()
        original_prompt_tokens=prompt.shape[1]
        prompt=prompt[:,-cfg0['max_prompt_tokens']:]
        values={};order=['baseline']+variants
        shift=(qi+repeat)%len(order);order=order[shift:]+order[:shift]
        before=gpu_snapshot()
        intervals={}
        for name in order:
            if name=='baseline':
                fn=lambda:dec.greedy_decode(model,prompt,args.max_new_tokens,tokenizer.eos_token_id,meta['recurrent_hidden_state_index'],torch.device('cuda'),omit_target_attention_mask=args.omit_baseline_mask)
            else:
                cfg=settings[name]
                t=old_recurrent if name.startswith('old_') else recurrent
                metadata=old_meta if name.startswith('old_') else meta
                fn=lambda cfg=cfg,t=t,metadata=metadata:dec.speculative_decode(model,t,metadata,prompt,cfg,tokenizer.eos_token_id,torch.device('cuda'),tokenizer=tokenizer)
            wall_start=time.time()
            value,seconds=dec.wall_timed(fn,torch.device('cuda'));values[name]=dict(value,wall_time_s=seconds)
            intervals[name]={'start':wall_start,'end':time.time()}
        result={'sample':row['sample'],'question':row['question'],'repeat':repeat,'phase':'warmup' if qi==0 else 'measurement',
            'order':order,'intervals':intervals,'prompt_tokens':prompt.shape[1],'original_prompt_tokens':original_prompt_tokens,'gpu_before':before,'gpu_after':gpu_snapshot(),'baseline':values.pop('baseline'),'variants':values}
        for name,value in values.items():value['exact']=value['token_ids']==result['baseline']['token_ids']
        if 'answer' in row:
            reference=dec.normalize_answer(dec.extract_answer(str(row['answer'])))
            result['reference_answer']=reference
            for value in [result['baseline'],*values.values()]:
                value['text']=tokenizer.decode(value['token_ids'],skip_special_tokens=True)
                value['predicted_answer']=dec.normalize_answer(dec.extract_answer(value['text']))
                value['answer_correct']=reference is not None and value['predicted_answer']==reference
        results.append(result)
        output={'checkpoint':str(args.checkpoint),'compare_checkpoint':str(args.compare_checkpoint) if args.compare_checkpoint else None,'target_dtype':str(dtype),'draft_dtype':str(next(recurrent.parameters()).dtype),'baseline_omit_target_attention_mask':args.omit_baseline_mask,'shared_gpu':True,
            'data':str(args.data.resolve()) if args.data else None,'start_index':args.start_index,'samples':args.samples,
            'timing_status':'shared exploratory; no exclusive performance claim','settings':{n:vars(c) for n,c in settings.items()},'results':results}
        (args.output/'result.json').write_text(json.dumps(output))
        print(json.dumps({'completed':len(results),'sample':row['sample'],'repeat':repeat,'exact':{k:v['exact'] for k,v in values.items()},
            'seconds':{k:round(v['wall_time_s'],3) for k,v in dict(values,baseline=result['baseline']).items()}}),flush=True)
    rows=[r for r in results if r['phase']=='measurement'];summary={}
    for name in variants:
        b=sum(r['baseline']['wall_time_s'] for r in rows);a=sum(r['variants'][name]['wall_time_s'] for r in rows)
        accepted=sum(r['variants'][name]['accepted_draft_tokens'] for r in rows)
        cycles=sum(r['variants'][name]['cycles'] for r in rows)
        summary[name]={'pairs':len(rows),'exact':sum(r['variants'][name]['exact'] for r in rows),
            'baseline_seconds':b,'speculative_seconds':a,'speedup':b/a,
            'accepted_drafts':accepted,'cycles':cycles,'mean_accepted_drafts':accepted/max(1,cycles),
            'draft_acceptance':accepted/max(1,sum(r['variants'][name]['draft_tokens'] for r in rows)),
            'shared_gpu':True}
        blocks=[b for r in rows for b in r['variants'][name]['block_records']]
        attempted=[b for b in blocks if b['proposed_len']>1]
        summary[name].update(
            recorded_cycles=len(blocks), draft_cycles=len(attempted),
            skipped_draft_cycles=sum(b['proposed_len']==1 for b in blocks),
            target_margin_skips=sum(b['gate_margin_skip'] for b in blocks),
            confidence_early_stops=sum(b['draft_confidence_stopped'] for b in blocks),
            cooldown_skips=sum(r['variants'][name]['draft_cooldown_skips'] for r in rows),
            position_gate_skips=sum(r['variants'][name]['latent_position_gate_skips'] for r in rows),
            mean_proposed_per_draft_cycle=sum(b['proposed_len']-1 for b in attempted)/max(1,len(attempted)),
            mean_accepted_per_draft_cycle=sum(max(0,b['accepted_len']-1) for b in attempted)/max(1,len(attempted)),
            answer_correct=sum(r['variants'][name].get('answer_correct',False) for r in rows),
        )
    import csv
    with (args.output/'routing_cycles.csv').open('w',newline='') as f:
        columns=['sample','repeat','variant','cycle','generated_start','block_limit','proposed_len','accepted_len','gate_target_margin','gate_margin_skip','recent_acceptance_before_cycle','history_draft_budget','draft_confidence_stopped','draft_step_margins','draft_cooldown','t_cache_clone_skipped']
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for row in rows:
            for name,value in row['variants'].items():
                for block in value['block_records']:
                    w.writerow({key:({'sample':row['sample'],'repeat':row['repeat'],'variant':name}.get(key,block.get(key))) for key in columns})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    assert all(v['exact'] and v.get('accepted_unchecked_tokens',0)==0 and v.get('accepted_mismatch_tokens',0)==0
               for r in rows for v in r['variants'].values()), 'Output mismatch or non-strict acceptance: inspect saved result.json'
    (args.output/'completion.json').write_text(json.dumps({'status':'completed','wall_seconds_including_loading_excluded':time.time()-started}))
    stop.set();monitor.join(timeout=6)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
