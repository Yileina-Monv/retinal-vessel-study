"""Matched bounded timings and exact state acceptance; never starts a research run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import gc
import time
import numpy as np
import torch
from retinal_release.common import ROOT,read,write
from retinal_prefetch import benchmark
from retinal_prefetch.loader import OrderedBatches
from retinal_prefetch.resources import Budget
from retinal_m1.data import SkeletonDataset,fingerprint
from retinal_m0.state import state_digest
from local_deployment.cached_data import OUT,prepare,Snapshot,CachedDataset,CachedBatches


def snapshot(schedule):
    ds=SkeletonDataset('train',cache_size=0)
    return Snapshot([ds.rows[d//4]['key'] for b in schedule for e,d in b])


def cpu():
    torch.set_num_threads(4);schedule=benchmark.plan(32);snap=snapshot(schedule);CachedBatches.snapshot=snap
    results=[];reference=None
    for mode,workers in [('original',2),('cached',2),('cached',4),('cached',4),('cached',2),('original',2)]:
        gc.collect();start=time.perf_counter();signatures=[]
        cls=OrderedBatches if mode=='original' else CachedBatches
        with cls(schedule,workers=workers,depth=workers*2) as batches:
            for tensors,sig,timing in batches:signatures.extend(sig)
        elapsed=time.perf_counter()-start
        if reference is None:reference=signatures
        if signatures!=reference:raise ValueError('Cached samples differ from frozen input stream')
        result=dict(mode=mode,workers=workers,seconds=elapsed,seconds_per_batch=elapsed/32,signature_sha256=state_digest(signatures))
        results.append(result);print(result,flush=True)
    write(OUT/'cpu.json',dict(results=results,cache_verification_seconds=snap.verification_seconds,matched_draws=128))


def gpu(args):
    schedule=benchmark.plan(args.steps);snap=None
    if args.loader=='cached':
        snap=snapshot(schedule);CachedBatches.snapshot=snap;benchmark.OrderedBatches=CachedBatches
    benchmark.OUT=OUT/'training';benchmark.OUT.mkdir(parents=True,exist_ok=True)
    if (benchmark.OUT/(args.tag+'.pt')).exists():raise ValueError('Do not overwrite a performance checkpoint')
    benchmark.gpu(Budget(),args.workers,args.steps,args.tag,resume=args.resume,stop=args.stop)
    path=benchmark.OUT/(args.tag+'.json');r=read(path)
    r.update(loader=args.loader,cache_verification_seconds=snap.verification_seconds if snap else 0,cache_id=snap.cache_id if snap else None)
    write(path,r)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','cpu','gpu'))
    p.add_argument('--loader',choices=('original','cached'),default='original');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--tag',default='probe');p.add_argument('--stop',type=int);p.add_argument('--resume')
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if not 1<=a.steps<=256:raise ValueError('Engineering probe limited to 256 updates')
    if a.mode=='prepare':prepare()
    elif a.mode=='cpu':cpu()
    else:gpu(a)


if __name__=='__main__':main()
