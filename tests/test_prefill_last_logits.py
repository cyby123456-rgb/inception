"""Ensure limited prefill readout preserves full KV and verification positions."""
import os
os.environ['DISABLE_VERSION_CHECK']='1'
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_setup'))
from benchmark_inference_matrix import target_options,dec
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM


class PrefillReadoutTest(unittest.TestCase):
    def test_real_model_preserves_hidden_cache_and_verification(self):
        torch.manual_seed(18)
        cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        model=Qwen3ForCausalLM(cfg).eval()
        ids=torch.tensor([[3,8,7,4,9]])
        with torch.inference_mode():
            full=dec.target_forward(model,ids,start_position=0)
            with target_options(last_prefill=True):
                small=dec.target_forward(model,ids,start_position=0)
                self.assertEqual(small.logits.shape,(1,1,97))
                self.assertEqual(small.past_key_values.get_seq_length(),5)
                self.assertEqual(small.hidden_states[-1].shape,(1,5,32))
                torch.testing.assert_close(full.logits[:,-1:],small.logits)
                torch.testing.assert_close(full.hidden_states[-1],small.hidden_states[-1])
                block=torch.tensor([[10,11,12]])
                verify=dec.target_forward(model,block,past_key_values=small.past_key_values,start_position=5)
                self.assertEqual(verify.logits.shape,(1,3,97))
                self.assertEqual(verify.past_key_values.get_seq_length(),8)

    def test_wrapper_restores_after_exception(self):
        original=dec.target_forward
        with self.assertRaises(RuntimeError):
            with target_options(last_prefill=True):
                raise RuntimeError('test')
        self.assertIs(dec.target_forward,original)


if __name__=='__main__':
    unittest.main()
