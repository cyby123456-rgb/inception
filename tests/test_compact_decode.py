import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_setup'))
from benchmark_inference_matrix import dec
import compact_decode as compact
import torch
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import itertools
from test_strict_decode import ToyCache,ToyT,logits_for


class CompactTests(unittest.TestCase):
    def test_prefix_decision_and_eos_never_commit_after_eos(self):
        for pattern in itertools.product([False,True],repeat=3):
            target=torch.tensor([[2,4,6,8]])
            logits=logits_for(target)
            drafts=[int(target[0,i]) if m else 1 for i,m in enumerate(pattern)]
            block=torch.tensor([[0,*drafts]])
            packet,_=compact.gpu_prefix_decision(logits,block,set())
            count=1+next((i for i,m in enumerate(pattern) if not m),len(pattern))
            self.assertEqual(packet[:3],[count,int(target[0,count-1]),sum(pattern)])
        packet,_=compact.gpu_prefix_decision(logits_for(torch.tensor([[2,4,6,8]])),torch.tensor([[0,2,4,6]]),{4})
        self.assertEqual(packet[0],3)

    def test_gpu_input_greedy_matches_legacy_real_small_model(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from benchmark_inference_matrix import target_options
        torch.manual_seed(719)
        cfg=Qwen3Config(vocab_size=37,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16)
        cfg._attn_implementation='sdpa'
        model=Qwen3ForCausalLM(cfg).eval()
        prompt=torch.tensor([[3,7,11]])
        with torch.inference_mode(),target_options(True):
            for cap,eos in itertools.product([1,2,9],[None,0,[1,2,3]]):
                old=dec.greedy_decode(model,prompt,cap,eos,1,torch.device('cpu'),True)
                new=compact.greedy_decode(model,prompt,cap,eos,1,torch.device('cpu'),True)
                self.assertEqual(old['token_ids'],new['token_ids'])
                self.assertEqual(old['target_calls'],new['target_calls'])

    def test_decoder_oracle_rejection_cache_eos_and_budget(self):
        model=torch.nn.Module();model.layers=torch.nn.ModuleList([torch.nn.Identity()])
        def target_forward(_model,ids,past_key_values=None,start_position=0,**kw):
            cache=past_key_values if past_key_values is not None else ToyCache()
            self.assertEqual(cache.get_seq_length(),start_position)
            hidden=model.layers[0](ids.float().unsqueeze(-1))
            cache.update(hidden.unsqueeze(1),hidden.unsqueeze(1),0)
            return SimpleNamespace(logits=logits_for(ids+1),past_key_values=cache)
        with patch.object(compact,'DynamicCache',ToyCache),patch.object(dec,'get_base_causal_lm',lambda m:m), \
             patch.object(dec,'find_decoder_layers',lambda m:(m.layers,'layers')),patch.object(dec,'target_forward',target_forward):
            for error,block,cap,eos in itertools.product(range(3),[1,2,3,5],[1,2,13,31],[None,2,3,6,16]):
                args=SimpleNamespace(mode='fixed',draft_logit_source='boundary',draft_commit_policy='target_match',
                    target_match_lambda=1.,ngram_draft_mode='off',max_block_tokens=block,max_new_tokens=cap)
                out=compact.speculative_decode(model,ToyT(error),{'recurrent_hidden_state_index':1,'loop_start_layer':0},
                    torch.tensor([[0,1]]),args,eos,torch.device('cpu'))
                expected=[]
                for i in range(cap):
                    token=(2+i)%17;expected.append(token)
                    if token==eos:break
                self.assertEqual(out['token_ids'],expected,(error,block,cap,eos))
                self.assertEqual(len(model.layers[0]._forward_hooks),0)


if __name__=='__main__':unittest.main()
