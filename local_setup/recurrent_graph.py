"""CUDA Graphs for T only, with preallocated KV and an explicit logical cursor.

Never substitute predicted KV for real prefix KV: draft rollout is cropped,
then actual target boundary states overwrite those positions during sync.
"""
import torch
from transformers import DynamicCache


class GraphKV:
    def __init__(self,recurrent,capacity):
        first=next(iter(recurrent._iter_layers())).self_attn
        dtype=next(recurrent.parameters()).dtype;device=next(recurrent.parameters()).device
        heads=first.k_proj.out_features//first.head_dim if hasattr(first.k_proj,'out_features') else first.k_proj.base_layer.out_features//first.head_dim
        self.keys={};self.values={};self.capacity=capacity;self.bucket=capacity;self.length=0
        for layer in recurrent._iter_layers():
            idx=layer.self_attn.layer_idx
            self.keys[idx]=torch.zeros((1,heads,capacity,first.head_dim),dtype=dtype,device=device)
            self.values[idx]=torch.zeros_like(self.keys[idx])
    def update(self,k,v,layer_idx,cache_kwargs):
        pos=cache_kwargs['cache_position']
        self.keys[layer_idx].index_copy_(2,pos,k)
        self.values[layer_idx].index_copy_(2,pos,v)
        return self.keys[layer_idx][...,:self.bucket,:],self.values[layer_idx][...,:self.bucket,:]
    def get_seq_length(self,layer_idx=0):return self.length
    def crop(self,length):
        if length<0:length=self.length+length
        self.length=min(self.length,length)


class RecurrentGraphs:
    @torch.inference_mode()
    def __init__(self,recurrent,model,capacity=2304,max_query=4):
        if recurrent.has_token_conditioning() or recurrent.multistep_residual_rank or recurrent.multistep_step1_residual_rank:
            raise ValueError('Graph T currently requires unconditioned T without multistep residuals')
        self.recurrent=recurrent;self.model=model;self.max_query=max_query
        self.cache=GraphKV(recurrent,capacity)
        self.graphs={};self.calls=0
        self.buckets=[x for x in [256,512,1024,2048,capacity] if x<=capacity]
        self.buckets=sorted(set(self.buckets))
        self.capacity=capacity
        parameter=next(recurrent.parameters());self.device=parameter.device;self.dtype=parameter.dtype
        # Compile/capture before prefill so capture writes cannot corrupt a real prefix.
        for width in range(1,max_query+1):
            for bucket in self.buckets:
                self._capture(width,bucket)

    def _run(self,h,position,bucket):
        pos=position+torch.arange(h.shape[1],device=h.device)
        allowed=torch.arange(bucket,device=h.device)[None,:]<=pos[:,None]
        mask=torch.zeros((1,1,h.shape[1],bucket),dtype=h.dtype,device=h.device).masked_fill(~allowed[None,None],torch.finfo(h.dtype).min)
        embeddings=self.recurrent._maybe_make_position_embeddings(self.model,h,pos[None])
        for layer in self.recurrent._iter_layers():
            h=layer(h,attention_mask=mask,position_ids=pos[None],past_key_values=self.cache,
                cache_position=pos,position_embeddings=embeddings,use_cache=True)
            if isinstance(h,tuple):h=h[0]
        return h

    def _capture(self,width,bucket):
        self.cache.bucket=bucket
        h=torch.zeros((1,width,self.recurrent.hidden_size),device=self.device,dtype=self.dtype)
        position=torch.zeros((),device=self.device,dtype=torch.long)
        stream=torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):self._run(h,position,bucket)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            out=self._run(h,position,bucket)
        self.graphs[width,bucket]=(graph,h,position,out)

    @torch.inference_mode()
    def prefill(self,hidden,token_ids):
        out,dynamic=self.recurrent.forward_with_cache(hidden,model=self.model,past_key_value=DynamicCache(),token_ids=token_ids)
        length=hidden.shape[1]
        if length>self.capacity:raise ValueError('T graph capacity exceeded')
        for idx in self.cache.keys:
            self.cache.keys[idx][...,:length,:].copy_(dynamic.layers[idx].keys)
            self.cache.values[idx][...,:length,:].copy_(dynamic.layers[idx].values)
        self.cache.length=length;self.calls=0
        return out,self.cache

    @torch.inference_mode()
    def forward_with_cache(self,hidden,model,past_key_value,token_ids=None,rollout_step=1):
        if past_key_value is not self.cache:raise ValueError('Graph T received a different cache')
        width=hidden.shape[1];end=self.cache.length+width
        if end>self.capacity or width>self.max_query:raise ValueError('T graph shape/capacity exceeded')
        bucket=next(x for x in self.buckets if x>=end)
        graph,buffer,position,out=self.graphs[width,bucket]
        buffer.copy_(hidden);position.fill_(self.cache.length)
        graph.replay();self.cache.length=end;self.calls+=1
        # A later replay can overwrite out. Draft anchor lists must own their data.
        return out.clone(),self.cache
