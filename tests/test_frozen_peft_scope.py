import torch
from peft import LoraConfig,get_peft_model
from transformers import LlamaConfig,LlamaForCausalLM
from llamafactory.model.model_utils.recurft import disable_adapter

def test_frozen_real_peft_adapters_stay_frozen_after_reference():
    model=get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)),LoraConfig(r=2,target_modules=['q_proj'],task_type='CAUSAL_LM'))
    for p in model.parameters():p.requires_grad_(False)
    for _ in range(2):
        with torch.no_grad(),disable_adapter(model):model(torch.tensor([[1,2,3]]))
        assert not any(p.requires_grad for p in model.parameters())

def test_trainable_scope_survives_exception():
    model=get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)),LoraConfig(r=2,target_modules=['q_proj'],task_type='CAUSAL_LM'))
    before={n:p.requires_grad for n,p in model.named_parameters()}
    try:
        with disable_adapter(model):raise RuntimeError('reference failed')
    except RuntimeError:pass
    assert before=={n:p.requires_grad for n,p in model.named_parameters()}
