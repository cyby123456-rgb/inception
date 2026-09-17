import sys,itertools,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_setup'))
from benchmark_inference_matrix import dec
import lookup_decode as lookup
from test_strict_decode import ToyCache,ToyT,logits_for
import torch


class LookupTests(unittest.TestCase):
    def test_lookup_only_uses_already_seen_continuation(self):
        index=lookup.PromptLookup([1,2,3,4,8,1],min_n=2)
        self.assertEqual(index.propose(2,8),[3,4,8,1])
        self.assertEqual(index.propose(9,8),[])
        index.append([2,3]);self.assertEqual(index.propose(4,2),[8,1])
    def test_oracle_lazy_t_and_lookup_rejection(self):
        model=torch.nn.Module();model.layers=torch.nn.ModuleList([torch.nn.Identity()]);counts={'lookup':0,'neural':0}
        def forward(_m,ids,past_key_values=None,start_position=0,**kw):
            cache=past_key_values if past_key_values is not None else ToyCache()
            self.assertEqual(cache.get_seq_length(),start_position)
            h=model.layers[0](ids.float().unsqueeze(-1));cache.update(h.unsqueeze(1),h.unsqueeze(1),0)
            return SimpleNamespace(logits=logits_for(ids+1),past_key_values=cache)
        with patch.object(lookup,'DynamicCache',ToyCache),patch.object(dec,'get_base_causal_lm',lambda m:m), \
             patch.object(dec,'find_decoder_layers',lambda m:(m.layers,'layers')),patch.object(dec,'target_forward',forward):
            for error,only,cap,eos in itertools.product(range(3),[False,True],[1,2,13,45],[None,2,3,6,16]):
                args=SimpleNamespace(mode='fixed',draft_logit_source='boundary',draft_commit_policy='target_match',
                    target_match_lambda=1.,ngram_draft_mode='off',max_block_tokens=3,max_new_tokens=cap,lookup_only=only,lookup_max_drafts=8)
                out=lookup.speculative_decode(model,ToyT(error),{'recurrent_hidden_state_index':1,'loop_start_layer':0},
                    torch.tensor([[0,1]]),args,eos,torch.device('cpu'))
                expected=[]
                for i in range(cap):
                    token=(2+i)%17;expected.append(token)
                    if token==eos:break
                self.assertEqual(out['token_ids'],expected,(error,only,cap,eos))
                if only:self.assertEqual(out['neural_blocks'],0)
                counts['lookup']+=out['lookup_blocks'];counts['neural']+=out['neural_blocks']
                self.assertFalse(model.layers[0]._forward_hooks)
        self.assertGreater(counts['lookup'],0);self.assertGreater(counts['neural'],0)
