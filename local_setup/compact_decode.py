"""Single-sequence neural boundary drafting with GPU-resident proposed tokens.

Only strict argmax verification, fixed block cap, no token conditioning, no
unchecked commits. Target and T caches always contain the committed real prefix
at cycle boundaries. Predictions used for draft rollout are cropped before sync.
"""
import torch
from transformers import DynamicCache
import recurft_speculative_generate as dec


@torch.inference_mode()
def greedy_decode(model,input_ids,max_new_tokens,eos_token_id,anchor_idx,device,omit_target_attention_mask=True):
    """Keep the argmax tensor as next input; CPU only reads the ID for stopping/output."""
    eos_ids={eos_token_id} if isinstance(eos_token_id,int) else set(eos_token_id or [])
    out=dec.target_forward(model,input_ids,start_position=0,output_hidden_states=False,
        omit_attention_mask=omit_target_attention_mask,logits_to_keep=1)
    calls=1;cache=out.past_key_values;generated=[]
    while len(generated)<max_new_tokens:
        token_gpu=out.logits[:,-1,:].argmax(-1,keepdim=True)
        token=int(token_gpu.item());generated.append(token)
        if token in eos_ids:break
        out=dec.target_forward(model,token_gpu,past_key_values=cache,
            start_position=input_ids.shape[1]+len(generated)-1,
            output_hidden_states=False,omit_attention_mask=omit_target_attention_mask)
        calls+=1;cache=out.past_key_values
    return dict(token_ids=generated,target_calls=calls,timings={},gpu_input_reuse=True)


def gpu_prefix_decision(logits,block,eos_ids):
    """One D2H packet: accepted length, next target token, matches, block token IDs."""
    targets=logits[0].argmax(-1)
    matches=targets[:-1].eq(block[0,1:])
    accepted=1+matches.long().cumprod(0).sum()
    # A draft EOS ends output only if it lies inside the strictly accepted prefix.
    positions=torch.arange(block.shape[1],device=block.device)
    eos=torch.zeros_like(block[0],dtype=torch.bool)
    for token in eos_ids:eos |= block[0].eq(token)
    eos_end=torch.where(eos,positions+1,block.shape[1]).min()
    accepted=torch.minimum(accepted,eos_end)
    following=targets.gather(0,(accepted-1).reshape(1))[0]
    packet=torch.cat((torch.stack((accepted,following,matches.sum())),block[0])).cpu().tolist()
    return packet,following.reshape(1,1)


@torch.inference_mode()
def speculative_decode(model,recurrent,metadata,input_ids,args,eos_token_id,device,tokenizer=None):
    if (args.mode!='fixed' or args.draft_logit_source!='boundary' or args.draft_commit_policy!='target_match'
        or args.target_match_lambda!=1. or args.ngram_draft_mode!='off' or recurrent.has_token_conditioning()
        or metadata.get('multistep_residual_rank',0) or metadata.get('multistep_step1_residual_rank',0)):
        raise ValueError('Compact decoder supports fixed strict boundary drafting without conditioning/residuals only')
    eos_ids={eos_token_id} if isinstance(eos_token_id,int) else set(eos_token_id or [])
    eos_tensor=torch.tensor(sorted(eos_ids),device=input_ids.device,dtype=torch.long) if getattr(args,'fused_decision',False) else None
    target_base=dec.get_base_causal_lm(model)
    layers,_=dec.find_decoder_layers(target_base)
    anchor_idx=metadata['recurrent_hidden_state_index'];capture={}
    def hook(_module,_inputs,output):capture['hidden']=output[0] if isinstance(output,tuple) else output
    handle=layers[anchor_idx-1].register_forward_hook(hook)
    generated=[];records=[];calls=0;drafts=0;accepted_total=0;sync_calls=0;draft_steps=0
    try:
        prefix=input_ids.shape[1]
        outputs=dec.target_forward(model,input_ids,start_position=0,output_hidden_states=False,
            omit_attention_mask=True,logits_to_keep=1)
        calls+=1;target_cache=outputs.past_key_values
        current_gpu=outputs.logits[:,-1,:].argmax(-1,keepdim=True)
        current=int(current_gpu.item())
        if current in eos_ids:
            return dict(token_ids=[current],target_calls=calls,draft_tokens=0,accepted_draft_tokens=0,
                cycles=1,block_records=[],fast_strict_blocks=0,accepted_unchecked_tokens=0,accepted_mismatch_tokens=0,timings={})
        t_driver=recurrent
        if getattr(args,'t_graph',False):
            from recurrent_graph import RecurrentGraphs
            capacity=2304
            cycle_graph=getattr(args,'cycle_graph',False)
            key=('_compact_cycle_graph_' if cycle_graph else '_compact_t_graph_')+str(args.max_block_tokens)
            if not hasattr(recurrent,key):
                if cycle_graph:
                    from draft_cycle_graph import DraftCycleGraphs
                    graph_class=DraftCycleGraphs
                else:graph_class=RecurrentGraphs
                object.__setattr__(recurrent,key,graph_class(recurrent,target_base,capacity=capacity,max_query=args.max_block_tokens))
            t_driver=getattr(recurrent,key)
            t_out,t_cache=t_driver.prefill(capture.pop('hidden'),input_ids)
        else:
            t_cache=DynamicCache()
            t_out,t_cache=recurrent.forward_with_cache(capture.pop('hidden'),model=target_base,past_key_value=t_cache,token_ids=input_ids)
        next_anchor=t_out[:,-1:,:];prepared_block=None
        while len(generated)<args.max_new_tokens:
            if current in eos_ids:
                generated.append(current);break
            start=prefix+len(generated)
            count=min(args.max_block_tokens,args.max_new_tokens-len(generated))
            t_length=dec.cache_length(t_cache,metadata['loop_start_layer'])
            if t_length!=start or target_cache.get_seq_length()!=start:
                raise RuntimeError(f'Cache alignment failure: target={target_cache.get_seq_length()}, T={t_length}, expected={start}')
            if prepared_block is not None:
                block=prepared_block;prepared_block=None
                assert block.shape[1]==count
            elif count>1:
                anchors=[next_anchor]
                for step in range(2,count):
                    h,t_cache=t_driver.forward_with_cache(anchors[-1],model=target_base,past_key_value=t_cache,
                        token_ids=current_gpu,rollout_step=step)
                    anchors.append(h[:,-1:,:]);draft_steps+=1
                if getattr(args,'shortlist_size',0):
                    from shortlist_boundary import shortlist_ids
                    draft_ids=shortlist_ids(recurrent,torch.cat(anchors,dim=1),target_base,args.shortlist_vocab,args.shortlist_size)[...,0]
                else:
                    draft_logits=recurrent.boundary_logits(torch.cat(anchors,dim=1),target_base)
                    draft_ids=draft_logits.argmax(-1)
                block=torch.cat((current_gpu,draft_ids),dim=1)
            else:
                block=current_gpu
            t_cache.crop(t_length)
            outputs=dec.target_forward(model,block,past_key_values=target_cache,start_position=start,
                output_hidden_states=False,omit_attention_mask=True)
            calls+=1
            if getattr(args,'fused_decision',False):
                from strict_prefix_kernel import prefix_decision
                packet,next_gpu=prefix_decision(outputs.logits,block,eos_tensor)
            else:packet,next_gpu=gpu_prefix_decision(outputs.logits,block,eos_ids)
            accepted,current,matched,*tokens=packet
            committed=tokens[:accepted]
            records.append(dict(generated_start=len(generated),proposed_len=count,accepted_len=accepted,
                matched_drafts=matched,committed_tokens=committed))
            generated.extend(committed);drafts+=count-1;accepted_total+=accepted-1
            target_cache=outputs.past_key_values;target_cache.crop(start+accepted)
            if any(t in eos_ids for t in committed) or len(generated)>=args.max_new_tokens:
                break
            if current in eos_ids:
                generated.append(current);break
            real_hidden=capture.pop('hidden')[:,:accepted,:]
            if getattr(args,'cycle_graph',False) and args.max_new_tokens-len(generated)>=args.max_block_tokens:
                prepared_block=t_driver.sync_and_draft(real_hidden,next_gpu)
                draft_steps+=args.max_block_tokens-2
            else:
                t_out,t_cache=t_driver.forward_with_cache(real_hidden,model=target_base,
                    past_key_value=t_cache,token_ids=block[:,:accepted])
                next_anchor=t_out[:,-1:,:]
            sync_calls+=1;current_gpu=next_gpu
        return dict(token_ids=generated,target_calls=calls,draft_tokens=drafts,accepted_draft_tokens=accepted_total,
            cycles=len(records),block_records=records,fast_strict_blocks=len(records),accepted_unchecked_tokens=0,
            accepted_mismatch_tokens=0,t_sync_calls=sync_calls,t_draft_steps=draft_steps,
            compact_gpu_packets=len(records)+1,timings={},
            t_graph_replays=t_driver.calls if getattr(args,'t_graph',False) else 0,
            implementation='compact GPU proposal path; EOS draft can leave extra verified suffix which is never committed')
    finally:
        handle.remove()
