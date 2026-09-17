"""Capture real-prefix T sync, latent rollout and boundary readout in one graph.

The logical cursor commits only real hidden states. Speculative KV beyond that
cursor is overwritten on the next cycle. Target verification remains unchanged.
"""
import torch
from recurrent_graph import RecurrentGraphs,GraphKV


class DraftCycleGraphs(RecurrentGraphs):
    @torch.inference_mode()
    def __init__(self,recurrent,model,capacity=2304,max_query=3):
        if recurrent.has_token_conditioning() or recurrent.multistep_residual_rank or recurrent.multistep_step1_residual_rank:
            raise ValueError('Unconditioned T without multistep residuals required')
        self.recurrent=recurrent;self.model=model;self.max_query=max_query;self.capacity=capacity
        self.cache=GraphKV(recurrent,capacity);self.calls=0;self.graphs={};self.cycles={}
        self.buckets=sorted(set(x for x in [256,512,1024,2048,capacity] if x<=capacity))
        p=next(recurrent.parameters());self.device=p.device;self.dtype=p.dtype
        # Ordinary graphs handle first block and shortened final blocks.
        for width in range(1,max_query+1):
            for bucket in self.buckets:self._capture(width,bucket)
        for width in range(1,max_query+1):
            for bucket in self.buckets:self._capture_cycle(width,bucket)

    def _cycle(self,h,position,token,bucket):
        real=self._run(h,position,bucket)
        anchors=[real[:,-1:,:]]
        for i in range(self.max_query-2):
            anchors.append(self._run(anchors[-1],position+h.shape[1]+i,bucket))
        logits=self.recurrent.boundary_logits(torch.cat(anchors,dim=1),self.model)
        return torch.cat((token,logits.argmax(-1)),dim=1)

    def _capture_cycle(self,width,bucket):
        self.cache.bucket=bucket
        h=torch.zeros((1,width,self.recurrent.hidden_size),device=self.device,dtype=self.dtype)
        position=torch.zeros((),device=self.device,dtype=torch.long)
        token=torch.zeros((1,1),device=self.device,dtype=torch.long)
        stream=torch.cuda.Stream(device=self.device);stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):self._cycle(h,position,token,bucket)
        torch.cuda.current_stream(self.device).wait_stream(stream);torch.cuda.synchronize(self.device)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):out=self._cycle(h,position,token,bucket)
        self.cycles[width,bucket]=(graph,h,position,token,out)

    @torch.inference_mode()
    def sync_and_draft(self,hidden,token):
        end=self.cache.length+hidden.shape[1]
        physical_end=end+self.max_query-2
        if physical_end>self.capacity:raise ValueError('Draft cycle graph capacity exceeded')
        bucket=next(x for x in self.buckets if x>=physical_end)
        graph,buffer,position,token_buffer,out=self.cycles[hidden.shape[1],bucket]
        buffer.copy_(hidden);position.fill_(self.cache.length);token_buffer.copy_(token)
        graph.replay();self.cache.length=end;self.calls+=1
        return out
