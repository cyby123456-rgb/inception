"""Test fused sync/draft against eager T, including reject/crop/reset/bucket crossing."""
from validate_recurrent_graph import *
from draft_cycle_graph import DraftCycleGraphs


@torch.inference_mode()
def validate_cycle():
    torch.manual_seed(1913)
    config=Qwen3Config(vocab_size=97,hidden_size=64,intermediate_size=96,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16)
    config._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(config).to(device='cuda',dtype=torch.bfloat16).eval()
    meta={'loop_layer_ids':[1,2],'loop_start_layer':1,'loop_end_layer':2,'projection_layer_ids':[1,2,3],
        'recurrent_hidden_state_index':1,'anchor_layer':0,'last_layer':3}
    recurrent=RecurFTRecurrentModule([model.model.layers[1],model.model.layers[2]],['all'],2,4,0.,meta,
        2,4,0.,.02,boundary_head_rank=8).cuda().eval();recurrent.merge_lora_for_inference()
    rows=[]
    with fused_qwen_norms(model,recurrent):
        for block in [2,3,4]:
            runner=DraftCycleGraphs(recurrent,model,capacity=300,max_query=block)
            for trial,prefix in enumerate([250,7]):
                h=torch.randn(1,prefix,64,device='cuda',dtype=torch.bfloat16)
                _,cache=recurrent.forward_with_cache(h,model=model,past_key_value=DynamicCache())
                runner.prefill(h,None);length=prefix
                for width in [1,block,2,1,block,2,block]:
                    h=torch.randn(1,width,64,device='cuda',dtype=torch.bfloat16)
                    real,cache=recurrent.forward_with_cache(h,model=model,past_key_value=cache)
                    length+=width;anchors=[real[:,-1:,:]]
                    for _ in range(block-2):
                        out,cache=recurrent.forward_with_cache(anchors[-1],model=model,past_key_value=cache)
                        anchors.append(out)
                    token=torch.tensor([[13]],device='cuda')
                    expected=torch.cat((token,recurrent.boundary_logits(torch.cat(anchors,dim=1),model).argmax(-1)),dim=1)
                    got=runner.sync_and_draft(h,token)
                    assert torch.equal(got,expected),(block,width,got,expected)
                    cache.crop(length)
                    assert runner.cache.length==cache.get_seq_length(1)==length
                    max_abs=0.
                    for idx in runner.cache.keys:
                        for ref,new in [(cache.layers[idx].keys,runner.cache.keys[idx][...,:length,:]),
                                        (cache.layers[idx].values,runner.cache.values[idx][...,:length,:])]:
                            torch.testing.assert_close(ref,new,rtol=.04,atol=.02)
                            max_abs=max(max_abs,float((ref.float()-new.float()).abs().max()))
                    rows.append(dict(block=block,trial=trial,width=width,length=length,token_exact=True,kv_max_abs=max_abs))
    return rows


if __name__=='__main__':print(json.dumps(validate_cycle(),indent=2))
