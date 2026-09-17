import sys,itertools,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_setup'))
from benchmark_inference_matrix import dec
import branched_decode as branch
from lower_right_target import lower_right_target
from test_strict_decode import ToyCache,ToyT,logits_for
import torch


class BranchTests(unittest.TestCase):
    def test_alternative_root_and_strict_later_prefix(self):
        block=torch.tensor([[2,4,5],[2,3,5]])
        logits=logits_for(torch.tensor([[3,5,6],[3,4,6]]))
        packet,_=branch.branch_decision(logits,block,set())
        self.assertEqual(packet[:4],[1,2,4,1])
        packet,_=branch.branch_decision(logits,block,{3})
        self.assertEqual(packet[:3],[1,2,4])
        # A noncanonical branch cannot change the shared root decision.
        logits[1,0]=logits_for(torch.tensor(4))
        packet,_=branch.branch_decision(logits,block,set())
        self.assertEqual(packet[:3],[1,2,4])

    def test_oracle_cache_prefix_identical_after_rejection(self):
        model=torch.nn.Module();model.layers=torch.nn.ModuleList([torch.nn.Identity()])
        def forward(_model,ids,past_key_values=None,start_position=0,**kwargs):
            cache=past_key_values if past_key_values is not None else ToyCache()
            self.assertEqual(cache.get_seq_length(),start_position)
            if start_position:
                keys=cache.layers[0].keys
                expected=(torch.arange(start_position)%17).reshape(1,1,-1,1).expand_as(keys)
                self.assertTrue(torch.equal(keys,expected))
            hidden=model.layers[0](ids.float().unsqueeze(-1));cache.update(hidden.unsqueeze(1),hidden.unsqueeze(1),0)
            return SimpleNamespace(logits=logits_for(ids+1),past_key_values=cache)
        with patch.object(branch,'DynamicCache',ToyCache),patch.object(dec,'get_base_causal_lm',lambda m:m), \
             patch.object(dec,'find_decoder_layers',lambda m:(m.layers,'layers')),patch.object(dec,'target_forward',forward):
            for error,b,k,cap,eos in itertools.product(range(3),[2,3],[2,3,4],[1,2,13,31],[None,2,3,6,16]):
                args=SimpleNamespace(mode='fixed',draft_logit_source='boundary',draft_commit_policy='target_match',
                    target_match_lambda=1.,ngram_draft_mode='off',max_block_tokens=k,max_new_tokens=cap,branch_width=b)
                out=branch.speculative_decode(model,ToyT(error),{'recurrent_hidden_state_index':1,'loop_start_layer':0},
                    torch.tensor([[0,1]]),args,eos,torch.device('cpu'))
                expected=[]
                for i in range(cap):
                    token=(2+i)%17;expected.append(token)
                    if token==eos:break
                self.assertEqual(out['token_ids'],expected,(error,b,k,cap,eos))
                self.assertFalse(model.layers[0]._forward_hooks)

    def test_real_qwen_parallel_paths_match_individual_forwards(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        torch.manual_seed(319)
        cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        cfg._attn_implementation='sdpa';model=Qwen3ForCausalLM(cfg).eval()
        prompt=torch.tensor([[3,5,7]]);ids=torch.tensor([[9,11,13],[9,15,17]])
        with torch.inference_mode(),lower_right_target(dec,max_batch=2):
            pre=dec.target_forward(model,prompt,start_position=0,omit_attention_mask=True)
            branch.expand_cache(pre.past_key_values,2)
            got=dec.target_forward(model,ids,start_position=3,past_key_values=pre.past_key_values,omit_attention_mask=True)
            expected=[]
            for row in ids:
                pre=dec.target_forward(model,prompt,start_position=0,omit_attention_mask=True)
                expected.append(dec.target_forward(model,row[None],start_position=3,past_key_values=pre.past_key_values,omit_attention_mask=True).logits)
            torch.testing.assert_close(got.logits,torch.cat(expected),atol=1e-6,rtol=1e-5)
