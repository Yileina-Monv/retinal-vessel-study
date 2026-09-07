"""PAIR engineering timing only, isolated from the held research pilot."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from retinal_release.common import read,write
from retinal_prefetch import benchmark
from retinal_prefetch.resources import Budget
from retinal_c1.pilot import Batches
from retinal_c1.objectives import objective
from retinal_m1.data import SkeletonDataset
from local_deployment.cached_data import OUT
from local_deployment.cached_c1 import CachedC1Batches


def main():
    p=argparse.ArgumentParser();p.add_argument('--loader',choices=('original','cached'),required=True);p.add_argument('--tag',required=True);a=p.parse_args()
    steps=64;schedule=benchmark.plan(steps);ds=SkeletonDataset('train',cache_size=0)
    setup=CachedC1Batches.configure([ds.rows[d//4]['key'] for b in schedule for e,d in b]) if a.loader=='cached' else None
    current={};Base=CachedC1Batches if a.loader=='cached' else Batches
    class Adapter(Base):
        def __next__(self):
            tensors,pairs,seeds,signatures=super().__next__()
            current['pairs']=pairs;current['seeds']=seeds
            return tensors,signatures,{}
    def loss(logits,target,fov,tube,*,coefficient):
        return objective(logits,target,fov,tube,current['pairs'],'PAIR',current['seeds'])
    benchmark.OrderedBatches=Adapter;benchmark.m1_loss=loss
    benchmark.OUT=OUT/'c1_training';benchmark.OUT.mkdir(parents=True,exist_ok=True)
    if (benchmark.OUT/(a.tag+'.pt')).exists():raise ValueError('Do not overwrite previous engineering run')
    benchmark.gpu(Budget(),workers=2,updates=steps,tag=a.tag)
    path=benchmark.OUT/(a.tag+'.json');r=read(path)
    r.update(loader=a.loader,cache_setup=setup,scope='64_update_PAIR_engineering_equivalence_and_timing_not_candidate_efficacy')
    write(path,r)


if __name__=='__main__':main()
