"""Experimental inference-only Qwen3 RMSNorm fusion, preserving BF16 rounding order.

The normalized activation is rounded to the input dtype BEFORE multiplying by
the weight, as in Transformers Qwen3RMSNorm. Reduction order may still differ;
real model token equivalence must be measured rather than assumed.
"""
import contextlib
import types
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_kernel(X,W,Y,N:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    idx=tl.arange(0,BLOCK)
    x=tl.load(X+row*N+idx,idx<N,other=0).to(tl.float32)
    variance=tl.sum(x*x,0)/N
    normalized=(x*tl.rsqrt(variance+EPS)).to(Y.dtype.element_ty).to(tl.float32)
    weight=tl.load(W+idx,idx<N,other=0).to(tl.float32)
    tl.store(Y+row*N+idx,normalized*weight,idx<N)


def fused_qwen_rms(x,weight,epsilon):
    if torch.is_grad_enabled():
        raise ValueError('Fused RMSNorm is inference-only; use torch.inference_mode() or torch.no_grad()')
    if not x.is_cuda or x.dtype not in (torch.bfloat16,torch.float16,torch.float32) or weight.dtype!=x.dtype:
        raise ValueError('Fused RMSNorm requires CUDA activations and weights of the same floating dtype')
    contiguous=x.contiguous();out=torch.empty_like(contiguous)
    width=x.shape[-1]
    _rms_kernel[(x.numel()//width,)](contiguous,weight,out,width,epsilon,
        triton.next_power_of_2(width),num_warps=4 if width<=4096 else 8,enable_fp_fusion=False)
    return out


@contextlib.contextmanager
def fused_qwen_norms(*models):
    changed=[];seen=set()
    def forward(module,x):
        return fused_qwen_rms(x,module.weight,module.variance_epsilon)
    for model in models:
        for module in model.modules():
            if id(module) not in seen and type(module).__name__=='Qwen3RMSNorm':
                seen.add(id(module));changed.append((module,module.forward))
                module.forward=types.MethodType(forward,module)
    try:
        yield {'fused_norm_modules':len(changed)}
    finally:
        for module,original in changed:
            module.forward=original
