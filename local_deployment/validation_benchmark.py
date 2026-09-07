"""Paired whole-image inference timings with exact input/probability checks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time
import numpy as np
import torch
from retinal_release.common import ROOT,read,write,digest
from retinal_m0.model import UNet
from retinal_m1.data import SkeletonDataset,fingerprint
from retinal_m1.inference import probability
from retinal_m1.metrics import score
from local_deployment.cached_data import OUT,Snapshot,CachedDataset


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    c=read(ROOT/'configs/development_v1.json');model=UNet(c['widths'],c['group_norm_groups'])
    ckpt=ROOT/'runs/prefetch_round1/m0_seed2026_dev10k_w1/last.pt'
    if digest(ckpt)!='8bce5b8f47c7fe43d553d4467adc46f5387eade014265b7a6a6f4b5211e67d79':raise ValueError('Checkpoint mismatch')
    saved=torch.load(ckpt,map_location='cpu',weights_only=True);model.load_state_dict(saved['model']);del saved;model.cuda().eval()
    old=SkeletonDataset('validation',full=True,cache_size=0)
    keys=read(ROOT/'.runtime/c1_local_v1/protocol.json')['validation_24'][::6]
    snap=Snapshot(keys);new=CachedDataset('validation',snapshot=snap,full=True,cache_size=0)
    indices={r['key']:i for i,r in enumerate(old.rows)};records=[];references={}
    # Warm convolution kernels before timing, same input for every candidate.
    item=old[indices[keys[0]]];probability(model,item['image'],item['fov'][0],batch_size=2)
    for repeat,modes in enumerate(([('original',2),('cached',2),('cached',4)],[('cached',4),('cached',2),('original',2)])):
        for mode,bs in modes:
            ds=old if mode=='original' else new;start=time.perf_counter();differences=[];threshold_differences=[]
            inputs_exact=True
            for key in keys:
                item=ds[indices[key]];p,_=probability(model,item['image'],item['fov'][0],batch_size=bs);p=p.numpy()
                sig=fingerprint(item)
                if key not in references:references[key]=(p.copy(),sig)
                ref,ref_sig=references[key];inputs_exact &= sig==ref_sig
                differences.append(float(np.max(np.abs(p-ref))));threshold_differences.append(int(((p>=.5)!=(ref>=.5)).sum()))
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            r=dict(repeat=repeat,mode=mode,batch_size=bs,images=4,seconds=elapsed,seconds_per_image=elapsed/4,
                   inputs_exact=inputs_exact,max_probability_difference=max(differences),threshold_changed_pixels=sum(threshold_differences),
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            records.append(r);print(r,flush=True)
    write(OUT/'validation.json',dict(results=records,cache_verification_seconds=snap.verification_seconds,
          acceptance='Only accept candidates with exactly matching input and fused probabilities; timings include fingerprints'))


if __name__=='__main__':main()
