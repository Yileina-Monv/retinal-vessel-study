"""Bounded optimization acceptance, not a scientific development run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from retinal_prefetch.resources import Budget
import argparse
import cProfile
import gc
import math
from pathlib import Path
import pstats
import random
import time
import numpy as np
import torch
from retinal_release.common import ROOT,write,read,digest
from retinal_m1.verify import verify
from retinal_m1.data import SkeletonDataset
from retinal_data.fives import EpochDrawSampler
from retinal_m1.loss import m1_loss
from retinal_m0.model import UNet
from retinal_m0.state import atomic_save,state_digest,rng_state,restore_rng
from retinal_prefetch.loader import batch,OrderedBatches

OUT=ROOT/'.runtime/prefetch_round1'

def plan(steps):
    ds=SkeletonDataset('train',seed=2026,cache_size=0)
    values=[];epoch=0
    while len(values)<steps*4:
        values.extend(EpochDrawSampler(ds,epoch=epoch,shuffle=True));epoch+=1
    return [values[i:i+4] for i in range(0,steps*4,4)]

def cpu(guard):
    torch.set_num_threads(4);schedule=plan(32);results=[];reference=None
    for repetition,modes in enumerate(((0,2,4),(4,2,0))):
        for workers in modes:
            gc.collect();signatures=[];timings=[];start=time.perf_counter()
            if workers==0:
                ds=SkeletonDataset('train',seed=2026,cache_size=8)
                for indices in schedule:
                    guard.check();tensors,sig,timing=batch(ds,indices);signatures.extend(sig);timings.append(timing)
                del ds
            else:
                with OrderedBatches(schedule,workers=workers,depth=workers*2,guard=guard) as batches:
                    for tensors,sig,timing in batches:signatures.extend(sig);timings.append(timing)
            elapsed=time.perf_counter()-start
            if reference is None:reference=signatures
            if reference!=signatures:raise ValueError('Prefetch changed actual inputs')
            record=dict(repetition=repetition,workers=workers,batches=32,seconds=elapsed,seconds_per_batch=elapsed/32,
                per_batch_stage_worker_seconds={k:float(np.mean([t[k] for t in timings])) for k in timings[0]})
            results.append(record);print(record,flush=True)
    profiler=cProfile.Profile();ds=SkeletonDataset('train',cache_size=1)
    profiler.enable()
    for indices in schedule[:4]:batch(ds,indices)
    profiler.disable();stats=pstats.Stats(profiler)
    profile=[dict(file=Path(k[0]).name,line=k[1],function=k[2],calls=v[1],self_seconds=v[2],cumulative_seconds=v[3])
        for k,v in sorted(stats.stats.items(),key=lambda x:x[1][3],reverse=True)[:30]]
    write(OUT/'cpu.json',dict(results=results,profile=profile,matched_samples=128,signature_digest=state_digest(reference),resources=guard.finish()))

def gpu(guard,workers,updates,tag,resume=None,stop=None):
    torch.set_num_threads(4);guard.cuda();torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    c=read(ROOT/'configs/development_v1.json');model=UNet(c['widths'],c['group_norm_groups']).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'],foreach=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024);cursor=0;history=[];signatures=[]
    if resume:
        saved=torch.load(OUT/(resume+'.pt'),map_location='cpu',weights_only=True)
        if saved['budget']!=updates:raise ValueError('Resume budget changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng']);cursor=saved['step'];history=saved['history'];signatures=saved['signatures'];del saved
    schedule=plan(updates)[cursor:stop or updates];durations=[];waits=[]
    start=time.perf_counter();last=start
    def consume(tensors,sig):
        nonlocal cursor,last
        ready=time.perf_counter();waits.append(ready-last)
        x,y,f,t=(v.cuda() for v in tensors)
        lr=c['minimum_learning_rate']+.5*(c['learning_rate']-c['minimum_learning_rate'])*(1+math.cos(math.pi*cursor/updates))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss,parts=m1_loss(model(x),y,f,t,coefficient=.1)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True)
        scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<scale:raise ValueError('Skipped update')
        torch.cuda.synchronize();cursor+=1;signatures.extend(sig)
        history.append(dict(step=cursor,loss=float(loss.detach()),lr=lr,gradient_norm=float(norm)))
        now=time.perf_counter();durations.append(now-last);last=now;guard.cuda_sample()
        if cursor%16==0:print(f'{tag} step={cursor} batch_seconds={np.mean(durations[-16:]):.4f}',flush=True)
    if workers:
        with OrderedBatches(schedule,workers=workers,depth=workers*2,guard=guard) as batches:
            for tensors,sig,timing in batches:consume(tensors,sig)
    else:
        ds=SkeletonDataset('train',seed=2026,cache_size=8)
        for indices in schedule:
            guard.check();tensors,sig,timing=batch(ds,indices);consume(tensors,sig)
    elapsed=time.perf_counter()-start
    saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),
        step=cursor,budget=updates,history=history,signatures=signatures)
    hashes={k:state_digest(v) for k,v in saved.items()}
    atomic_save(saved,OUT/(tag+'.pt'))
    record=dict(tag=tag,workers=workers,resume=resume,step=cursor,segment_updates=len(durations),seconds=elapsed,
        seconds_per_update=elapsed/len(durations),warm_seconds_per_update=float(np.mean(durations[8:])) if len(durations)>8 else None,
        mean_input_wait_seconds=float(np.mean(waits)),state_sha256=hashes,checkpoint_sha256=digest(OUT/(tag+'.pt')),
        resources=guard.finish(),scope='engineering_optimizer_and_resume_acceptance_only')
    write(OUT/(tag+'.json'),record);print(record,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu']);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--updates',type=int,default=64);p.add_argument('--tag',default='gpu');p.add_argument('--resume');p.add_argument('--stop',type=int);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);guard=Budget();verify()
    if a.mode=='cpu':cpu(guard)
    else:gpu(guard,a.workers,a.updates,a.tag,a.resume,a.stop)

if __name__=='__main__':main()
