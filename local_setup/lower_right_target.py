"""Opt-in causal SDPA for contiguous, unpadded batch-1 Qwen3 target verification.

Uses PyTorch's lower-right causal bias, NOT is_causal=True on a rectangular
public SDPA call (which has upper-left alignment and would corrupt KV decoding).
Does not patch recurrent T attention or alter model weights. Context installation
is intended for the single-threaded benchmark, not concurrent model serving.
"""
import contextlib
from contextvars import ContextVar
import torch
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import repeat_kv


def lower_right_attention(module, query, key, value, *, scaling=None):
    groups=getattr(module,'num_key_value_groups',1)
    if groups>1:
        key=repeat_kv(key,groups);value=repeat_kv(value,groups)
    result=F.scaled_dot_product_attention(query,key,value,
        attn_mask=causal_lower_right(query.shape[-2],key.shape[-2]),dropout_p=0.,scale=scaling)
    return result.transpose(1,2).contiguous(),None


@torch.inference_mode()
def validate_cuda_lower_right():
    from types import SimpleNamespace
    from torch.nn.attention import sdpa_kernel,SDPBackend
    generator=torch.Generator(device='cuda').manual_seed(801)
    results=[]
    for length in [2,3,4]:
        for prefix in [17,128,512]:
            klen=prefix+length
            q=torch.randn(1,8,length,128,device='cuda',dtype=torch.bfloat16,generator=generator)
            k=torch.randn(1,2,klen,128,device='cuda',dtype=torch.bfloat16,generator=generator)
            v=torch.randn_like(k)
            mask=torch.arange(klen,device='cuda')[None,:] <= (prefix+torch.arange(length,device='cuda'))[:,None]
            with sdpa_kernel(SDPBackend.MATH):
                expected=F.scaled_dot_product_attention(q.float(),k.float().repeat_interleave(4,1),v.float().repeat_interleave(4,1),attn_mask=mask)
            actual,_=lower_right_attention(SimpleNamespace(num_key_value_groups=4),q,k,v)
            torch.testing.assert_close(actual.transpose(1,2).float(),expected,atol=.008,rtol=.025)
            altered=v.clone();altered[:,:,-1,:]+=100
            future,_=lower_right_attention(SimpleNamespace(num_key_value_groups=4),q,k,altered)
            torch.testing.assert_close(actual[:,0],future[:,0],rtol=0,atol=0)
            results.append(dict(prefix=prefix,query=length,max_abs=float((actual.transpose(1,2).float()-expected).abs().max()),future_isolated=True))
    return results


@contextlib.contextmanager
def lower_right_target(decoder,max_batch=1):
    original_forward=decoder.target_forward
    original_attention=ALL_ATTENTION_FUNCTIONS['sdpa']
    active=ContextVar('lower_right_target_active',default=False)
    counters={'eligible_forwards':0,'attention_calls':0,'fallback_forwards':0}
    def attention(module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
        if active.get():
            if dropout or getattr(module,'sliding_window',None) is not None:
                raise ValueError('Lower-right target path requires full attention without dropout')
            counters['attention_calls']+=1
            return lower_right_attention(module,query,key,value,scaling=scaling)
        return original_attention(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)
    def forward(model,ids,**kwargs):
        cache=kwargs.get('past_key_values');position=kwargs.get('start_position')
        eligible=(1<=ids.shape[0]<=max_batch and ids.shape[1]>1 and cache is not None and
            kwargs.get('omit_attention_mask',False) and position is not None and position>0 and
            getattr(model.config,'model_type',None)=='qwen3' and
            getattr(model.config,'_attn_implementation',None)=='sdpa' and
            not getattr(model.config,'use_sliding_window',False))
        if eligible and cache.get_seq_length()!=position:
            raise ValueError('Cache length and contiguous target position differ')
        counters['eligible_forwards' if eligible else 'fallback_forwards']+=1
        token=active.set(eligible)
        # Transformers' torch<2.5 mask fallback writes in-place into an expanded
        # batch mask when no padding mask exists. Supplying all-ones materializes
        # its temporary mask safely; our attention still uses lower-right bias.
        if ids.shape[0]>1 and kwargs.get('omit_attention_mask',False):
            kwargs=dict(kwargs,omit_attention_mask=False)
        try:
            return original_forward(model,ids,**kwargs)
        finally:
            active.reset(token)
    ALL_ATTENTION_FUNCTIONS['sdpa']=attention;decoder.target_forward=forward
    try:
        yield counters
    finally:
        decoder.target_forward=original_forward
        ALL_ATTENTION_FUNCTIONS['sdpa']=original_attention
