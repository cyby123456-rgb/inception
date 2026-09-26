"""Approximate DRAFTER readout using a training-derived vocabulary subset.

Target verification still projects to the complete vocabulary. Reducing draft
vocabulary can lose candidates; no claim of identical draft argmax is made.
"""
import json
import torch
from torch.nn import functional as F
_CACHE={}


def adapted_hidden(recurrent,hidden,model):
    x=recurrent.boundary_norm(hidden.to(recurrent.boundary_norm.weight.dtype))
    x=recurrent.boundary_A(x.to(recurrent.boundary_A.weight.dtype))
    residual=recurrent.boundary_B(x.to(recurrent.boundary_B.weight.dtype))
    x=hidden+residual.to(hidden.dtype)
    lm=recurrent._resolve_causal_lm(model);decoder=getattr(lm,'model',lm)
    norm=getattr(decoder,'norm',None) or getattr(decoder,'ln_f',None)
    if norm is not None:x=norm(x.to(norm.weight.dtype))
    return x,lm.get_output_embeddings()


def shortlist_ids(recurrent,hidden,model,path,size,topk=1):
    key=(id(recurrent),id(model),str(path),size)
    if key not in _CACHE:
        head=recurrent._resolve_causal_lm(model).get_output_embeddings()
        with open(path) as file:ids=json.load(file)['ids'][:size]
        if not ids or min(ids)<0 or max(ids)>=head.weight.shape[0]:raise ValueError('Invalid draft vocabulary IDs')
        indices=torch.tensor(ids,device=head.weight.device,dtype=torch.long)
        _CACHE[key]=(indices,head.weight.index_select(0,indices).contiguous(),head.bias.index_select(0,indices) if head.bias is not None else None)
    indices,weight,bias=_CACHE[key]
    x,head=adapted_hidden(recurrent,hidden,model)
    logits=F.linear(x.to(weight.dtype),weight,bias)
    selected=logits.argmax(-1,keepdim=True) if topk==1 else logits.topk(topk,dim=-1).indices
    return indices[selected]
