"""Build value-blind splits, deterministic FOV maps, and training-only normalization."""
import csv
from collections import Counter
import math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from retinal_data.fives import read_pair
from retinal_release.common import ROOT,read,write,digest,identity

DEST=ROOT/'releases/development_v1'

def make_fov(image,threshold=10):
    foreground=image.max(axis=2)>threshold
    labels,count=ndimage.label(foreground,structure=np.ones((3,3),dtype=np.uint8))
    if not count: raise ValueError('No FOV component')
    sizes=np.bincount(labels.ravel());sizes[0]=0
    # FOV describes acquisition field, not visibility: retain dark sectors inside it.
    # FIVES originals are centered square fundus photographs. This assumption is
    # dataset-specific and must not be silently reused for external datasets.
    yy,xx=np.nonzero(labels==int(sizes.argmax()))
    cy,cx=(np.array(foreground.shape)-1)/2
    radius=min(float(np.sqrt(np.quantile((yy-cy)**2+(xx-cx)**2,.995)))+4,min(foreground.shape)/2)
    y,x=np.ogrid[:foreground.shape[0],:foreground.shape[1]]
    return (y-cy)**2+(x-cx)**2<=radius**2

def split(rows,config):
    eligible=[r for r in rows if int(r['group_size'])==1 and not int(r['label_is_empty'])]
    counts=Counter(r['disease'] for r in eligible)
    raw={d:config['validation_count']*n/len(eligible) for d,n in counts.items()}
    quotas={d:math.floor(v) for d,v in raw.items()}
    for d in sorted(raw,key=lambda d:(-(raw[d]-quotas[d]),d))[:config['validation_count']-sum(quotas.values())]: quotas[d]+=1
    chosen=set()
    for disease,quota in sorted(quotas.items()):
        forced=[r for r in eligible if r['disease']==disease and r['trial_split']=='smoke_val']
        candidates=[r for r in eligible if r['disease']==disease and r['trial_split'] not in ('smoke_train','smoke_val')]
        candidates.sort(key=lambda r:identity([config['split_seed'],r['key']]))
        if not 0<=quota-len(forced)<=len(candidates): raise ValueError('Impossible validation allocation')
        chosen.update(r['key'] for r in forced+candidates[:quota-len(forced)])
    output=[]
    for old in rows:
        r=dict(old)
        r['prior_trial_split']=r['trial_split']
        r['trial_split']='edge_only' if int(r['group_size'])>1 or int(r['label_is_empty']) else ('validation' if r['key'] in chosen else 'train')
        output.append(r)
    assert sum(r['trial_split']=='validation' for r in output)==config['validation_count']
    assert all(r['trial_split']=='train' for r in output if r['prior_trial_split']=='smoke_train')
    return output

def main():
    if DEST.exists(): raise ValueError('Release directory already exists; do not overwrite a freeze')
    c=read(ROOT/'configs/development_v1.json')
    with (ROOT/'data_preparation/manifests/all_train.csv').open(encoding='utf-8-sig',newline='') as f: rows=split(list(csv.DictReader(f)),c)
    DEST.mkdir(parents=True)
    fields=list(rows[0])+['fov_path','fov_sha256']
    total=np.zeros(3,dtype=np.float64);squares=total.copy();pixels=0;qa=[]
    for i,r in enumerate(rows):
        image,label=read_pair({**r,'foreground_pixels':int(r['foreground_pixels'])},ROOT)
        fov=make_fov(image,c['fov']['threshold_max_rgb_gt'])
        path=DEST/'fov'/r['filename'];path.parent.mkdir(exist_ok=True)
        Image.fromarray(fov.astype(np.uint8)*255).save(path)
        r['fov_path']=path.relative_to(ROOT).as_posix();r['fov_sha256']=digest(path)
        qa.append(dict(key=r['key'],split=r['trial_split'],disease=r['disease'],quality=r['quality_class'],
            area_fraction=float(fov.mean()),label_pixels_outside=int((label.astype(bool)&~fov).sum()),foreground_pixels=int(label.sum())))
        if r['trial_split']=='train':
            values=image[fov].astype(np.float64)/255
            total+=values.sum(0);squares+=np.square(values).sum(0);pixels+=len(values)
        if (i+1)%25==0: print(f'FOV and source hashes {i+1}/{len(rows)}',flush=True)
    mean=total/pixels;std=np.sqrt(squares/pixels-mean**2)
    for name in ('all','train','validation','edge_only'):
        with (DEST/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(r for r in rows if name=='all' or r['trial_split']==name)
    write(DEST/'normalization.json',dict(mean=mean.tolist(),std=std.tolist(),pixel_count=pixels,source_split='train_only',accumulator='float64',input_scale=255))
    write(DEST/'fov_qa.json',dict(rows=qa,algorithm=c['fov'],label_used_to_generate_fov=False))
    write(DEST/'data_summary.json',dict(counts=dict(Counter(r['trial_split'] for r in rows)),
        disease_counts={s:dict(Counter(r['disease'] for r in rows if r['trial_split']==s)) for s in ('train','validation','edge_only')},
        prior_train_preserved=32,prior_validation_preserved=8,
        group_overlap=len({r['group_id'] for r in rows if r['trial_split']=='train'}&{r['group_id'] for r in rows if r['trial_split']=='validation'}),
        raw_train_pairs_verified=len(rows),official_test_images_opened=0,external_images_opened=0,
        fov_area_min=min(r['area_fraction'] for r in qa),fov_area_max=max(r['area_fraction'] for r in qa),
        maximum_labeled_pixels_outside_fov=max(r['label_pixels_outside'] for r in qa),
        status='awaiting_QA_and_release_manifest'))
    print('Data build complete; seal only after QA.',flush=True)

if __name__=='__main__': main()
