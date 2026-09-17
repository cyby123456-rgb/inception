"""Fixed input and fixed KV-prefix target-forward ablation (not generation speedup)."""
from benchmark_inference_matrix import *
from lower_right_target import lower_right_target
from fused_rmsnorm import fused_qwen_norms
from validate_fused_rms import validate


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=12)
    p.add_argument('--merge-target',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    lock=open('/tmp/inception-eval-'+hashlib.sha256(gpu.encode()).hexdigest()[:16]+'.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
    stop=threading.Event();threading.Thread(target=monitor,args=(a.output,stop),daemon=True).start()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.55)
    model_path=ROOT.parent/'models/Qwen--Qwen3-8B/snapshots/master'
    ck=ROOT.parent/'inception-joint/runs/qwen3_joint_core36000_delayed_20260916/joint/checkpoints'
    data_path=ROOT.parent/'datasets/GSM8K/test_official.jsonl'
    base=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    model=PeftModel.from_pretrained(base,ck).cuda().eval()
    if a.merge_target:model=model.merge_and_unload(safe_merge=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(ck,local_files_only=True)
    source=[json.loads(line) for line in data_path.read_text().splitlines()[:32]]
    ids=tokenizer('\n\n'.join(r['question']+'\n'+r['answer'] for r in source),return_tensors='pt')['input_ids'].cuda()
    assert ids.shape[1]>=1028
    dump(a.output/'norm_validation.json',validate())
    paths=[Path(__file__),Path(dec.__file__),Path(__file__).with_name('lower_right_target.py'),Path(__file__).with_name('fused_rmsnorm.py')]
    dump(a.output/'manifest.json',dict(args=vars(a),model=str(model_path),checkpoint=str(ck),adapter_sha256=sha(ck/'adapter_model.safetensors'),data_sha256=sha(data_path),
        input='Tokenize concatenated question + answer from first 32 GSM8K rows; use fixed prefix 128/512/1024, next 1/2/3/4 tokens.',
        method='All variants use the SAME baseline-produced prefix KV. Crop to that prefix outside timing before every call. No rollout, no answer-quality metric.',
        timing='Outer CUDA synchronized wall time; context installation and cache crop excluded. Not an end-to-end speculative speedup.',
        dtype='BF16',shared_gpu=True,sources={str(x):sha(x) for x in paths}))
    for x in paths:shutil.copy2(x,a.output/x.name)
    records=[];device=torch.device('cuda');dec._SYNC_COMPONENT_TIMING=False
    variants=['reference','lower','fused','lower_fused']
    for length in [128,512,1024]:
        prefix=ids[:,:length]
        pre=dec.target_forward(model,prefix,start_position=0,output_hidden_states=False,omit_attention_mask=True,logits_to_keep=1)
        cache=pre.past_key_values
        for width in [1,2,3,4]:
            block=ids[:,length:length+width]
            cache.crop(length)
            ref=dec.target_forward(model,block,past_key_values=cache,start_position=length,output_hidden_states=False,omit_attention_mask=True).logits.clone()
            for iteration in range(-2,a.iterations):
                order=variants.copy();offset=max(0,iteration)%len(order);order=order[offset:]+order[:offset]
                for name in order:
                    cache.crop(length)
                    with (lower_right_target(dec) if 'lower' in name else contextlib.nullcontext({})) as audit, \
                         (fused_qwen_norms(model) if 'fused' in name else contextlib.nullcontext({})):
                        output,seconds=dec.wall_timed(lambda:dec.target_forward(model,block,past_key_values=cache,start_position=length,
                            output_hidden_states=False,omit_attention_mask=True),device)
                    assert output.past_key_values.get_seq_length()==length+width
                    if iteration>=0:
                        record=dict(prefix_tokens=length,query_tokens=width,iteration=iteration,variant=name,seconds=seconds,
                            top1_ids=output.logits[0].argmax(-1).tolist(),reference_top1_ids=ref[0].argmax(-1).tolist(),
                            logit_max_abs=float((output.logits.float()-ref.float()).abs().max()),lower_right_audit=dict(audit))
                        records.append(record)
                        with (a.output/'results.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            summary=[]
            for size in [128,512,1024]:
                for q in [1,2,3,4]:
                    refrows=[r for r in records if r['prefix_tokens']==size and r['query_tokens']==q and r['variant']=='reference']
                    if not refrows:continue
                    for v in variants:
                        sel=[r for r in records if r['prefix_tokens']==size and r['query_tokens']==q and r['variant']==v]
                        seconds=sum(r['seconds'] for r in sel)
                        summary.append(dict(prefix_tokens=size,query_tokens=q,variant=v,iterations=len(sel),mean_ms=1000*seconds/len(sel),
                            forward_speedup=sum(r['seconds'] for r in refrows)/seconds,
                            all_top1_equal=all(r['top1_ids']==r['reference_top1_ids'] for r in sel),
                            logit_max_abs=max(r['logit_max_abs'] for r in sel)))
            dump(a.output/'summary.json',summary)
            print(json.dumps(summary[-4:]),flush=True)
    stop.set();dump(a.output/'status.json',dict(status='completed',forward_measurements=len(records),time=time.time()))


if __name__=='__main__':main()
