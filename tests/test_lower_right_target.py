import os
os.environ['DISABLE_VERSION_CHECK']='1'
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_setup'))
import torch
from torch.nn import functional as F
from transformers import Qwen3Config,Qwen3ForCausalLM
from benchmark_inference_matrix import dec
from lower_right_target import lower_right_attention,lower_right_target


class LowerRightTest(unittest.TestCase):
    def test_rectangular_mask_and_future_isolation(self):
        torch.manual_seed(27)
        for length in [2,3,4]:
            q=torch.randn(1,4,length,8);k=torch.randn(1,2,11,8);v=torch.randn_like(k)
            mod=SimpleNamespace(num_key_value_groups=2)
            mask=torch.arange(11)[None,:] <= (11-length+torch.arange(length))[:,None]
            expected=F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),attn_mask=mask)
            actual,_=lower_right_attention(mod,q,k,v)
            torch.testing.assert_close(actual.transpose(1,2),expected)
            # Last new key/value is in the future of the first query row.
            changed=v.clone();changed[:,:,-1,:]+=1000
            other,_=lower_right_attention(mod,q,k,changed)
            torch.testing.assert_close(actual[:,0],other[:,0])

    def test_real_qwen_kv_forward(self):
        torch.manual_seed(29)
        config=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        config._attn_implementation='sdpa';model=Qwen3ForCausalLM(config).eval()
        prefix=torch.tensor([[3,5,7,9,11]]);block=torch.tensor([[13,15,17]])
        with torch.inference_mode():
            x=dec.target_forward(model,prefix,start_position=0,omit_attention_mask=True)
            y=dec.target_forward(model,prefix,start_position=0,omit_attention_mask=True)
            ref=dec.target_forward(model,block,start_position=5,past_key_values=x.past_key_values,omit_attention_mask=True)
            with lower_right_target(dec) as audit:
                out=dec.target_forward(model,block,start_position=5,past_key_values=y.past_key_values,omit_attention_mask=True)
            torch.testing.assert_close(ref.logits,out.logits)
            self.assertEqual(out.past_key_values.get_seq_length(),8)
            self.assertEqual(audit['attention_calls'],2)
            self.assertEqual(audit['eligible_forwards'],1)


if __name__=='__main__':unittest.main()
