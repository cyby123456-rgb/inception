"""GPU numerical check; run outside timed inference measurements."""
import os
from pathlib import Path
os.environ.setdefault('TRITON_CACHE_DIR',str(Path(__file__).resolve().parents[1]/'runs/triton_cache'))
import json
import torch
from fused_rmsnorm import fused_qwen_rms

@torch.inference_mode()
def validate():
    results=[]
    torch.manual_seed(83)
    for dtype in [torch.bfloat16,torch.float16,torch.float32]:
        for width in [128,4096]:
            for rows in [1,3,12,96]:
                x=torch.randn(rows,width,device='cuda',dtype=dtype)
                w=(1+.1*torch.randn(width,device='cuda')).to(dtype)
                ref=w*(x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)).to(dtype)
                got=fused_qwen_rms(x,w,1e-6)
                torch.testing.assert_close(got,ref,rtol=0.016 if dtype==torch.bfloat16 else 0.002,atol=1e-5)
                results.append(dict(dtype=str(dtype),width=width,rows=rows,max_abs=float((ref.float()-got.float()).abs().max()),
                    exact_fraction=float((ref==got).float().mean())))
    return results

if __name__=='__main__':print(json.dumps(validate(),indent=2))
