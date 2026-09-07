"""Sparse paired, shuffled and pooled ranking; binary same-class neighbor control."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from retinal_release.data import masked_loss
from retinal_m1.loss import m1_loss


def tail(value, largest):
    return value.topk(max(1,math.ceil(value.numel()*.2)),largest=largest).values.mean()


def auxiliary(logits, batch_pairs, mode, seeds):
    terms=[]; margins=[]
    for z,pairs,seed in zip(logits.float()[:,0],batch_pairs,seeds):
        if len(pairs)<2:continue
        positive=[z[tuple(torch.as_tensor(p['pos'],device=z.device,dtype=torch.long).T)] for p in pairs]
        negative=[z[tuple(torch.as_tensor(p['bg'],device=z.device,dtype=torch.long).T)] for p in pairs]
        if mode=='POOL':
            gap=(1+tail(torch.cat(negative),True)-tail(torch.cat(positive),False)).reshape(1)
        else:
            a=torch.stack([tail(v,False) for v in positive]);b=torch.stack([tail(v,True) for v in negative])
            if mode=='SHUFFLE':
                shift=int(np.random.default_rng(seed).integers(1,len(pairs)))
                b=torch.roll(b,shift)
            elif mode!='PAIR':raise ValueError('Unknown ranking control')
            gap=1+b-a
        terms.append(F.softplus(gap).mean());margins.extend(gap.detach().cpu().tolist())
    zero=logits.float().sum()*0
    return (torch.stack(terms).mean() if terms else zero), dict(valid_images=len(terms),gaps=margins)


def scnp_logits(logits, target, fov):
    z=logits.float();y=target.bool();valid=fov.bool()
    pos=z.masked_fill(~(y&valid),float('inf'))
    neg=z.masked_fill(~(~y&valid),float('-inf'))
    worst_pos=-F.max_pool2d(-pos,3,stride=1,padding=1)
    worst_neg=F.max_pool2d(neg,3,stride=1,padding=1)
    return torch.where(valid,torch.where(y,worst_pos,worst_neg),torch.zeros_like(z))


def objective(logits, target, fov, tube, batch_pairs, method, seeds):
    if method=='M1':
        loss,parts=m1_loss(logits,target,fov,tube,coefficient=.1)
        return loss,dict(base=float(parts['pixel']),aux=float(parts['skeleton']),valid_images=int((tube.sum((1,2,3))>0).sum()))
    base=masked_loss(scnp_logits(logits,target,fov) if method=='SCNP' else logits,target,fov)
    if method in ('M0','SCNP'):return base,dict(base=float(base.detach()),aux=0.,valid_images=0)
    aux,info=auxiliary(logits,batch_pairs,method,seeds)
    return base+.1*aux,dict(base=float(base.detach()),aux=float(aux.detach()),valid_images=info['valid_images'])
