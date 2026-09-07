"""Ordered bounded prefetch. Each worker owns its reader; augmentation is unchanged."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import torch
from retinal_m1.data import SkeletonDataset,fingerprint

def batch(dataset,indices):
    started=time.perf_counter();items=[dataset[i] for i in indices]
    loaded=time.perf_counter()
    tensors=tuple(torch.stack([r[k] for r in items]) for k in ('image','mask','fov','tube'))
    stacked=time.perf_counter();signatures=[fingerprint(r) for r in items]
    return tensors,signatures,dict(read_seconds=loaded-started,stack_seconds=stacked-loaded,
        fingerprint_seconds=time.perf_counter()-stacked)

class OrderedBatches:
    def __init__(self,plan,*,seed=2026,workers=2,depth=4,guard=None):
        if not 1<=workers<=4 or not workers<=depth<=8:raise ValueError('Bounded profile: 1-4 workers, depth <=8')
        self.plan=iter(plan);self.seed=seed;self.depth=depth;self.guard=guard;self.local=threading.local()
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='retinal-input')
        self.pending=deque();self.closed=False
    def _make(self,indices):
        if self.guard:self.guard.check()
        if not hasattr(self.local,'dataset'):
            self.local.dataset=SkeletonDataset('train',seed=self.seed,cache_size=1)
        return batch(self.local.dataset,indices)
    def __enter__(self):
        for _ in range(self.depth):self._submit()
        return self
    def _submit(self):
        try:indices=next(self.plan)
        except StopIteration:return
        self.pending.append(self.pool.submit(self._make,indices))
    def __iter__(self):return self
    def __next__(self):
        if not self.pending:raise StopIteration
        if self.guard:self.guard.check()
        value=self.pending.popleft().result()
        self._submit()
        return value
    def __exit__(self,*args):
        self.close()
    def close(self):
        if not self.closed:
            self.closed=True
            for future in self.pending:future.cancel()
            self.pool.shutdown(wait=True,cancel_futures=True);self.pending.clear()
