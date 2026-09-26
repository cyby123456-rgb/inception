"""Exact argmax prefix decision in two CUDA kernels; full target logits retained."""
import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_chunks(X,S,I,V:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);chunk=tl.program_id(1)
    idx=chunk*B+tl.arange(0,B)
    val=tl.load(X+row*V+idx,idx<V,other=-float('inf')).to(tl.float32)
    # torch.argmax chooses the first NaN, or first maximum on ties.
    nan=val!=val
    has_nan=tl.sum(nan.to(tl.int32),0)>0
    maximum=tl.max(val,0)
    choose=tl.where(has_nan,nan,val==maximum)&(idx<V)
    winner=tl.min(tl.where(choose,idx,2147483647),0)
    tl.store(S+row*C+chunk,tl.where(has_nan,float('nan'),maximum))
    tl.store(I+row*C+chunk,winner)


@triton.jit
def _packet(S,I,TOKENS,EOS,PACKET,NEXT,R:tl.constexpr,C:tl.constexpr,E:tl.constexpr,BC:tl.constexpr,BR:tl.constexpr,BE:tl.constexpr):
    rows=tl.arange(0,BR);cols=tl.arange(0,BC)
    scores=tl.load(S+rows[:,None]*C+cols[None,:],(rows[:,None]<R)&(cols[None,:]<C),other=-float('inf'))
    ids=tl.load(I+rows[:,None]*C+cols[None,:],(rows[:,None]<R)&(cols[None,:]<C),other=2147483647)
    has_nan=tl.sum((scores!=scores).to(tl.int32),1)>0
    maxima=tl.max(scores,1)
    choose=tl.where(has_nan[:,None],scores!=scores,scores==maxima[:,None])
    targets=tl.min(tl.where(choose,ids,2147483647),1)
    block=tl.load(TOKENS+rows,rows<R,other=-1)
    draft=tl.load(TOKENS+rows+1,rows<R-1,other=-1)
    matches=(targets==draft)&(rows<R-1)
    first_failure=tl.min(tl.where((rows<R-1)&~matches,rows+1,R),0)
    eidx=tl.arange(0,BE)
    eos=tl.load(EOS+eidx,eidx<E,other=-2)
    is_eos=tl.sum(((block[:,None]==eos[None,:])&(eidx[None,:]<E)).to(tl.int32),1)>0
    eos_end=tl.min(tl.where(is_eos&(rows<R),rows+1,R),0)
    accepted=tl.minimum(first_failure,eos_end)
    following=tl.sum(tl.where(rows==accepted-1,targets,0),0)
    tl.store(PACKET,accepted);tl.store(PACKET+1,following);tl.store(PACKET+2,tl.sum(matches.to(tl.int32),0))
    tl.store(PACKET+3+rows,block,rows<R);tl.store(NEXT,following)


def prefix_decision(logits,block,eos_tensor):
    if not logits.is_contiguous():logits=logits.contiguous()
    rows,vocab=logits.shape[-2:];chunks=triton.cdiv(vocab,1024)
    scores=torch.empty((rows,chunks),device=logits.device,dtype=torch.float32)
    indices=torch.empty((rows,chunks),device=logits.device,dtype=torch.int32)
    packet=torch.empty((rows+3,),device=logits.device,dtype=torch.long)
    following=torch.empty((1,1),device=logits.device,dtype=torch.long)
    _argmax_chunks[(rows,chunks)](logits,scores,indices,vocab,chunks,1024)
    _packet[(1,)](scores,indices,block,eos_tensor,packet,following,rows,chunks,eos_tensor.numel(),
        triton.next_power_of_2(chunks),triton.next_power_of_2(rows),triton.next_power_of_2(max(1,eos_tensor.numel())))
    return packet.cpu().tolist(),following


@torch.inference_mode()
def validate(vocab=151936):
    from compact_decode import gpu_prefix_decision
    torch.manual_seed(9171);rows=[]
    for dtype in [torch.bfloat16,torch.float32]:
        for width in [1,2,3,4]:
            for mode in ['random','all_match','ties','nan','eos','eos_draft']+[f'fail_{i}' for i in range(width-1)]:
                logits=torch.randn((1,width,vocab),device='cuda',dtype=dtype)
                if mode=='ties':logits.zero_()
                if mode=='nan':logits[:,:,7]=float('nan');logits[:,:,-1]=float('nan')
                block=torch.zeros((1,width),device='cuda',dtype=torch.long)
                if mode!='random':block[:,1:]=logits[:,:-1].argmax(-1)
                if mode.startswith('fail_'):
                    idx=int(mode.split('_')[1]);block[:,idx+1]=(block[:,idx+1]+1)%vocab
                eos={int(block[0,-1])} if mode=='eos_draft' else ({0} if mode=='eos' else {vocab-1})
                expected,next_expected=gpu_prefix_decision(logits,block,eos)
                actual,next_actual=prefix_decision(logits,block,torch.tensor(sorted(eos),device='cuda',dtype=torch.long))
                assert expected==actual,(width,mode,expected,actual)
                assert torch.equal(next_expected,next_actual)
                rows.append(dict(dtype=str(dtype),width=width,vocab=vocab,mode=mode,exact=True))
    return rows
