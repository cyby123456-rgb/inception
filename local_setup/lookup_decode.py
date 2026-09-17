"""Strict prompt/history lookup with optional neural T fallback and lazy T sync.

This is a hybrid route, NOT evidence that T alone reaches the same speedup.
"""
import torch
from transformers import DynamicCache
import recurft_speculative_generate as dec
from compact_decode import gpu_prefix_decision


class PromptLookup:
    def __init__(self,tokens,min_n=2,max_n=5):
        self.history=[];self.min_n=min_n;self.max_n=max_n;self.indices={n:{} for n in range(min_n,max_n+1)}
        self.append(tokens)
    def append(self,tokens):
        for token in tokens:
            self.history.append(token);end=len(self.history)
            for n,index in self.indices.items():
                if end>=n:index.setdefault(tuple(self.history[-n:]),[]).append(end)
    def propose(self,current,limit):
        if limit<1:return []
        for n in range(self.max_n,self.min_n-1,-1):
            if len(self.history)+1<n:continue
            key=tuple(self.history[-(n-1):])+ (current,)
            for end in reversed(self.indices[n].get(key,[])):
                if end<len(self.history):return self.history[end:end+limit]
        return []


@torch.inference_mode()
def speculative_decode(model,recurrent,metadata,input_ids,args,eos_token_id,device,tokenizer=None):
    if args.mode!='fixed' or args.target_match_lambda!=1. or args.draft_commit_policy!='target_match' or recurrent.has_token_conditioning():
        raise ValueError('Strict unconditioned fixed decoding only')
    eos_ids={eos_token_id} if isinstance(eos_token_id,int) else set(eos_token_id or [])
    base=dec.get_base_causal_lm(model);layers,_=dec.find_decoder_layers(base);capture={}
    def hook(_m,_a,out):capture['h']=out[0] if isinstance(out,tuple) else out
    handle=layers[metadata['recurrent_hidden_state_index']-1].register_forward_hook(hook)
    generated=[];records=[];calls=0;proposed=0;accepted_total=0;t_driver=None;t_cache=None;next_anchor=None
    pending=[];lookup=PromptLookup(input_ids[0].cpu().tolist(),min_n=getattr(args,'lookup_min_n',2))
    neural_blocks=0;lookup_blocks=0;lookup_accepted=0;lookup_proposed=0;sync_calls=0;draft_steps=0
    try:
        prefix=input_ids.shape[1]
        out=dec.target_forward(model,input_ids,start_position=0,output_hidden_states=False,omit_attention_mask=True,logits_to_keep=1)
        calls+=1;cache=out.past_key_values;pending.append(capture.pop('h'))
        current_gpu=out.logits[:,-1].argmax(-1,keepdim=True);current=int(current_gpu.item())
        while len(generated)<args.max_new_tokens:
            if current in eos_ids:generated.append(current);break
            start=prefix+len(generated);remaining=args.max_new_tokens-len(generated)
            candidate=lookup.propose(current,min(args.lookup_max_drafts,remaining-1))
            kind='lookup' if candidate else ('target_only' if args.lookup_only or remaining==1 else 'neural')
            if candidate:
                block=torch.tensor([[current,*candidate]],dtype=torch.long,device=device);lookup_blocks+=1
            elif kind=='target_only':block=current_gpu
            else:
                neural_blocks+=1
                if t_driver is None:
                    h=torch.cat(pending,dim=1);pending=[]
                    assert h.shape[1]==start
                    if getattr(args,'t_graph',False):
                        from recurrent_graph import RecurrentGraphs
                        key='_compact_t_graph_'+str(args.max_block_tokens)
                        if not hasattr(recurrent,key):object.__setattr__(recurrent,key,RecurrentGraphs(recurrent,base,capacity=2304,max_query=args.max_block_tokens))
                        t_driver=getattr(recurrent,key);t_out,t_cache=t_driver.prefill(h,None)
                    else:
                        t_driver=recurrent;t_out,t_cache=recurrent.forward_with_cache(h,model=base,past_key_value=DynamicCache(),token_ids=None)
                    next_anchor=t_out[:,-1:]
                elif pending:
                    h=torch.cat(pending,dim=1);pending=[]
                    for chunk in h.split(args.max_block_tokens,dim=1):
                        t_out,t_cache=t_driver.forward_with_cache(chunk,model=base,past_key_value=t_cache,token_ids=None)
                        sync_calls+=1
                    next_anchor=t_out[:,-1:]
                length=dec.cache_length(t_cache,metadata['loop_start_layer']);assert length==start
                count=min(args.max_block_tokens,remaining);anchors=[next_anchor]
                for step in range(2,count):
                    h,t_cache=t_driver.forward_with_cache(anchors[-1],model=base,past_key_value=t_cache,token_ids=None,rollout_step=step)
                    anchors.append(h[:,-1:]);draft_steps+=1
                ids=recurrent.boundary_logits(torch.cat(anchors,dim=1),base).argmax(-1)
                block=torch.cat((current_gpu,ids),dim=1);t_cache.crop(length)
            assert cache.get_seq_length()==start
            out=dec.target_forward(model,block,past_key_values=cache,start_position=start,output_hidden_states=False,omit_attention_mask=True)
            calls+=1;packet,next_gpu=gpu_prefix_decision(out.logits,block,eos_ids)
            accepted,current,matched,*tokens=packet;committed=tokens[:accepted]
            records.append(dict(generated_start=len(generated),proposed_len=block.shape[1],accepted_len=accepted,matched_drafts=matched,committed_tokens=committed,source=kind))
            generated.extend(committed);lookup.append(committed);proposed+=block.shape[1]-1;accepted_total+=accepted-1
            if kind=='lookup':lookup_proposed+=block.shape[1]-1;lookup_accepted+=accepted-1
            cache=out.past_key_values;cache.crop(start+accepted)
            if any(t in eos_ids for t in committed) or len(generated)>=args.max_new_tokens:break
            if current in eos_ids:generated.append(current);break
            if not args.lookup_only:pending.append(capture.pop('h')[:,:accepted,:])
            else:capture.pop('h',None)
            current_gpu=next_gpu
        return dict(token_ids=generated,target_calls=calls,draft_tokens=proposed,accepted_draft_tokens=accepted_total,
            cycles=len(records),block_records=records,fast_strict_blocks=len(records),accepted_unchecked_tokens=0,accepted_mismatch_tokens=0,
            timings={},neural_blocks=neural_blocks,lookup_blocks=lookup_blocks,lookup_accepted=lookup_accepted,lookup_proposed=lookup_proposed,
            t_sync_calls=sync_calls,t_draft_steps=draft_steps,t_graph_replays=t_driver.calls if getattr(args,'t_graph',False) and t_driver is not None else 0,
            implementation='lookup-only' if args.lookup_only else 'hybrid prompt/history lookup + T fallback, lazy real-hidden T sync')
    finally:handle.remove()
