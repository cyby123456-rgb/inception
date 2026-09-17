"""Parallel candidate paths with one shared committed prefix (strict target verification).

Every branch has an identical real prefix. After selecting a valid path, only
its newly committed KV is copied to the other branches before cropping.
"""
import torch
from transformers import DynamicCache
import recurft_speculative_generate as dec


def expand_cache(cache,width):
    for layer in cache.layers:
        if layer.keys is not None:
            layer.keys=layer.keys.expand(width,-1,-1,-1).clone()
            layer.values=layer.values.expand(width,-1,-1,-1).clone()


def commit_branch(cache,start,accepted,branch):
    for layer in cache.layers:
        for tensor in [layer.keys,layer.values]:
            chosen=tensor[branch:branch+1,:,start:start+accepted,:].clone()
            tensor[:,:,start:start+accepted,:].copy_(chosen.expand(tensor.shape[0],-1,-1,-1))
    cache.crop(start+accepted)


def branch_decision(logits,block,eos_ids):
    targets=logits.argmax(-1)
    # The first next-token decision is common to every path: use canonical branch 0.
    targets[:,0]=targets[0,0].clone()
    matches=targets[:,:-1].eq(block[:,1:])
    lengths=1+matches.long().cumprod(-1).sum(-1)
    chosen=lengths.argmax()
    ids=block.index_select(0,chosen.reshape(1))[0]
    pos=torch.arange(block.shape[1],device=block.device)
    eos=torch.zeros_like(ids,dtype=torch.bool)
    for token in eos_ids:eos|=ids.eq(token)
    accepted=torch.minimum(lengths.gather(0,chosen.reshape(1))[0],torch.where(eos,pos+1,block.shape[1]).min())
    selected=targets.index_select(0,chosen.reshape(1))[0]
    following=selected.gather(0,(accepted-1).reshape(1))[0]
    matched=matches.index_select(0,chosen.reshape(1)).sum()
    packet=torch.cat((torch.stack((chosen,accepted,following,matched)),ids)).cpu().tolist()
    return packet,following.reshape(1,1)


@torch.inference_mode()
def speculative_decode(model,recurrent,metadata,input_ids,args,eos_token_id,device,tokenizer=None):
    if (args.mode!='fixed' or args.draft_logit_source!='boundary' or args.draft_commit_policy!='target_match'
        or args.target_match_lambda!=1. or args.ngram_draft_mode!='off' or recurrent.has_token_conditioning()
        or metadata.get('multistep_residual_rank',0) or metadata.get('multistep_step1_residual_rank',0)):
        raise ValueError('Compact decoder supports fixed strict boundary drafting without conditioning/residuals only')
    branches=args.branch_width
    if branches not in (2,3) or getattr(args,'cycle_graph',False) or getattr(args,'fused_decision',False):
        raise ValueError('Branch path requires width 2/3 and ordinary T graph')
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
        expand_cache(target_cache,branches)
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
                draft_logits=recurrent.boundary_logits(torch.cat(anchors,dim=1),target_base)
                draft_ids=draft_logits.argmax(-1).expand(branches,-1).clone()
                draft_ids[:,0]=draft_logits[0,0].topk(branches).indices
                block=torch.cat((current_gpu.expand(branches,-1),draft_ids),dim=1)
            else:
                block=current_gpu.expand(branches,-1)
            t_cache.crop(t_length)
            outputs=dec.target_forward(model,block,past_key_values=target_cache,start_position=start,
                output_hidden_states=False,omit_attention_mask=True)
            calls+=1
            packet,next_gpu=branch_decision(outputs.logits,block,eos_ids)
            selected_branch,accepted,current,matched,*tokens=packet
            committed=tokens[:accepted]
            records.append(dict(generated_start=len(generated),proposed_len=count,accepted_len=accepted,
                matched_drafts=matched,committed_tokens=committed,candidate_branches=branches,selected_branch=selected_branch))
            generated.extend(committed);drafts+=branches*(count-1);accepted_total+=accepted-1
            target_cache=outputs.past_key_values;commit_branch(target_cache,start,accepted,selected_branch)
            if any(t in eos_ids for t in committed) or len(generated)>=args.max_new_tokens:
                break
            if current in eos_ids:
                generated.append(current);break
            real_hidden=capture.pop('hidden')[selected_branch:selected_branch+1,:accepted,:]
            if getattr(args,'cycle_graph',False) and args.max_new_tokens-len(generated)>=args.max_block_tokens:
                prepared_block=t_driver.sync_and_draft(real_hidden,next_gpu)
                draft_steps+=args.max_block_tokens-2
            else:
                t_out,t_cache=t_driver.forward_with_cache(real_hidden,model=target_base,
                    past_key_value=t_cache,token_ids=block[selected_branch:selected_branch+1,:accepted])
                next_anchor=t_out[:,-1:,:]
            sync_calls+=1;current_gpu=next_gpu
        return dict(token_ids=generated,target_calls=calls,draft_tokens=drafts,accepted_draft_tokens=accepted_total,
            cycles=len(records),block_records=records,fast_strict_blocks=len(records),accepted_unchecked_tokens=0,
            accepted_mismatch_tokens=0,t_sync_calls=sync_calls,t_draft_steps=draft_steps,
            compact_gpu_packets=len(records)+1,timings={},
            t_graph_replays=t_driver.calls if getattr(args,'t_graph',False) else 0,
            branch_width=branches,implementation='parallel top-k first draft paths, canonical root decision, strict prefix, copy selected new KV to other branches')
    finally:
        handle.remove()
