"""Small real Qwen T: compare graph updates/crops with dynamic eager KV."""
from benchmark_inference_matrix import *
from transformers import Qwen3Config,Qwen3ForCausalLM,DynamicCache
from llamafactory.model.model_utils.recurft import RecurFTRecurrentModule
from recurrent_graph import RecurrentGraphs
from fused_rmsnorm import fused_qwen_norms


@torch.inference_mode()
def validate():
    torch.manual_seed(913)
    config=Qwen3Config(vocab_size=97,hidden_size=64,intermediate_size=96,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16)
    config._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(config).to(device='cuda',dtype=torch.bfloat16).eval()
    meta={'loop_layer_ids':[1,2],'loop_start_layer':1,'loop_end_layer':2,'projection_layer_ids':[1,2,3],
        'recurrent_hidden_state_index':1,'anchor_layer':0,'last_layer':3}
    recurrent=RecurFTRecurrentModule([model.model.layers[1],model.model.layers[2]],['all'],2,4,0.,meta,
        2,4,0.,.02,boundary_head_rank=8).cuda().eval()
    recurrent.merge_lora_for_inference()
    rows=[]
    with fused_qwen_norms(model,recurrent):
        runner=RecurrentGraphs(recurrent,model,capacity=32,max_query=3)
        for trial in range(2):
            h=torch.randn(1,5,64,device='cuda',dtype=torch.bfloat16)
            ref,cache=recurrent.forward_with_cache(h,model=model,past_key_value=DynamicCache())
            got,gcache=runner.prefill(h,None)
            torch.testing.assert_close(ref,got,rtol=0,atol=0)
            length=5
            for width,crop in [(1,0),(3,2),(2,0),(3,3),(1,0)]:
                h=torch.randn(1,width,64,device='cuda',dtype=torch.bfloat16)
                ref,cache=recurrent.forward_with_cache(h,model=model,past_key_value=cache)
                got,gcache=runner.forward_with_cache(h,model,gcache)
                torch.testing.assert_close(ref,got,rtol=.04,atol=.02)
                length+=width
                rows.append(dict(trial=trial,width=width,length=length,max_abs=float((ref.float()-got.float()).abs().max())))
                length-=crop;cache.crop(length);gcache.crop(length)
                assert gcache.get_seq_length()==cache.get_seq_length(1)==length
    return rows


if __name__=='__main__':print(json.dumps(validate(),indent=2))
