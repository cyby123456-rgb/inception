import sys,json,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_setup'))
from benchmark_inference_matrix import dec
from transformers import Qwen3Config,Qwen3ForCausalLM
from llamafactory.model.model_utils.recurft import RecurFTRecurrentModule
import torch
from shortlist_boundary import adapted_hidden,shortlist_ids,_CACHE


class ShortlistTests(unittest.TestCase):
    def test_adapter_projection_matches_full_readout_selected_rows(self):
        torch.manual_seed(419)
        cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=48,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        cfg._attn_implementation='sdpa';m=Qwen3ForCausalLM(cfg).eval()
        meta={'loop_layer_ids':[1,2],'loop_start_layer':1,'loop_end_layer':2,'projection_layer_ids':[1,2],
              'recurrent_hidden_state_index':1,'anchor_layer':0,'last_layer':2}
        t=RecurFTRecurrentModule([m.model.layers[1],m.model.layers[2]],['all'],2,4,0.,meta,2,4,0.,.02,boundary_head_rank=8).eval()
        h=torch.randn(1,3,32);ids=[3,5,7,11,19]
        with torch.inference_mode(),tempfile.TemporaryDirectory() as td:
            p=Path(td)/'vocab.json';p.write_text(json.dumps({'ids':ids}))
            expected=t.boundary_logits(h,m)[...,ids]
            x,head=adapted_hidden(t,h,m)
            torch.testing.assert_close(torch.nn.functional.linear(x,head.weight[ids]),expected)
            got=shortlist_ids(t,h,m,p,5,topk=2)
            self.assertTrue(torch.equal(got,torch.tensor(ids)[expected.topk(2,dim=-1).indices]))
        _CACHE.clear()
