"""Frozen follow-up: inference batching and fresh 8192-draw paired pilot."""
import argparse
import gc
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from retinal_batchstudy import run as base
from retinal_data.fives import trial_dataset
from retinal_data.prepare import ROOT, sha256
from retinal_m0.model import UNet
from retinal_m0.inference import tiled_probability
from retinal_m0.objectives import binary_metrics

OUT = ROOT / 'runs/batch_supplement_v1/rtx5080'

def config():
    c = base.read(base.CONFIG)
    c.update(protocol_id='batch_supplement_v1', seeds=[2026,2027,2028], training_draws=8192,
             evaluation_draws=[0,1024,2048,4096,6144,8192])
    c['arms'] = [a for a in c['arms'] if a['id'] in ('m2_a2_e4','m4_a1_e4','m2_a8_e16')]
    return c

def datasets():
    train, val = trial_dataset('smoke_train'), trial_dataset('smoke_val')
    for ds in (train,val):
        ds.cache_size = len(ds.rows)
        for row in ds.rows: ds._arrays(row)
    return train,val

def inference(c, val):
    path = OUT/'inference.json'
    if path.exists(): return
    rows=[]
    checkpoints={}
    for seed in c['seeds']:
        cp=ROOT/f'runs/batch_study_v1/rtx5080/seed_{seed}/m2_a2_e4/last.pt'
        checkpoints[str(seed)]=sha256(cp)
        saved=torch.load(cp,map_location='cpu',weights_only=True)
        model=UNet(c['widths'],c['group_norm_groups']).cuda()
        model.load_state_dict(saved['model']); del saved
        model.eval()
        # Warm each distinct layout, excluded from timings.
        for b in (1,2,4,8,16):
            tiled_probability(model,val[0]['image'],batch_size=b)
        for index in range(len(val)):
            item=val[index]
            predictions={}; timings={}; peaks={}
            order=np.random.default_rng(seed+index).permutation([1,2,4,8,16])
            for b in order:
                b=int(b)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                predictions[b],_=tiled_probability(model,item['image'],batch_size=b)
                torch.cuda.synchronize()
                timings[b]=time.perf_counter()-start
                peaks[b]=torch.cuda.max_memory_allocated()/2**30
            ref=predictions[4]
            ref_metrics=binary_metrics(ref,item['mask'][0])
            for b in (1,2,4,8,16):
                prob=predictions[b]; delta=prob-ref
                metrics=binary_metrics(prob,item['mask'][0])
                rows.append(dict(seed=seed,key=item['key'],batch=b,seconds=timings[b],peak_gib=peaks[b],
                    max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()),
                    flipped_pixels=int(((prob>=.5)!=(ref>=.5)).sum()),pixels=ref.numel(),
                    dice_delta=metrics['dice']-ref_metrics['dice'],**metrics))
            print(f'inference seed={seed} image={index+1}/{len(val)}',flush=True)
        del model,predictions; gc.collect(); torch.cuda.empty_cache()
    summary=[]
    for b in (1,2,4,8,16):
        r=[x for x in rows if x['batch']==b]
        summary.append(dict(batch=b,seconds=sum(x['seconds'] for x in r),
            macro_dice=float(np.mean([x['dice'] for x in r])),
            mean_dice_delta=float(np.mean([x['dice_delta'] for x in r])),
            max_abs_image_dice_delta=max(abs(x['dice_delta']) for x in r),
            max_probability_delta=max(x['max_abs'] for x in r),
            flipped_fraction=sum(x['flipped_pixels'] for x in r)/sum(x['pixels'] for x in r),
            peak_gib=max(x['peak_gib'] for x in r)))
    base.dump(path,dict(reference_batch=4,checkpoints=checkpoints,summary=summary,rows=rows,
        scope='Three fixed baseline checkpoints; same eight development images. Paired deterministic output check, single timed pass after warmup; timing descriptive, not a significance test.'))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['inference','train','all'],default='all')
    p.add_argument('--max-seconds',type=int,default=5400); args=p.parse_args()
    c=config(); base.setup(c,c['seeds'][0]); OUT.mkdir(parents=True,exist_ok=True)
    fingerprints=base.binding() | {'batch_study/supplement.py':sha256(Path(__file__)),
        'supplement_config':hashlib.sha256(__import__('json').dumps(c,sort_keys=True).encode()).hexdigest()}
    for name,value in [('binding',fingerprints),('config',c),('hardware',base.hardware())]:
        path=OUT/f'{name}.json'
        if path.exists() and base.read(path)!=value: raise ValueError(f'{name} changed')
        base.dump(path,value)
    train,val=datasets()
    if args.mode in ('inference','all'): inference(c,val)
    if args.mode=='inference': return
    deadline=time.perf_counter()+args.max_seconds
    plan=[]
    for seed in c['seeds']:
        for i in np.random.default_rng(c['run_order_seed']+seed).permutation(len(c['arms'])):
            plan.append((seed,c['arms'][int(i)]))
    base.dump(OUT/'run_order.json',[dict(seed=s,arm=a) for s,a in plan])
    for seed,arm in plan:
        if not base.train_arm(c,arm,seed,train,val,OUT,fingerprints,deadline):
            print('Paused safely; repeat command to resume.',flush=True); return
    print('Supplement training completed.',flush=True)

if __name__=='__main__': main()
