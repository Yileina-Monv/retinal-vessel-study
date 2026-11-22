"""Paired native-scale continuation from each trained nnU-Net endpoint."""
import argparse
import hashlib
import json
import time

from prepare import ROOT
import numpy as np
from scipy import ndimage as ndi
import torch
from torch.nn import functional as F

import paired
from evaluate import evaluate,make_predictor,probabilities
from train import trainer

METHODS=("continue","branch","branch_aux")


def mine(items,names,fold,seed):
    model=make_predictor(fold,seed,"standard")
    summaries=[]
    for name in names:
        d=items[name]
        d["vcoords"]=np.argwhere(d["v"])
        pr=probabilities(model,d["rgb"],native_only=True)["native"]
        hit=ndi.binary_dilation((pr>=.5)&d["f"],structure=paired.cross)
        ids=d["i"]
        keys,den=np.unique(ids[ids>0],return_counts=True)
        numer=np.bincount(ids[hit],minlength=int(ids.max())+1)[keys]
        eligible=np.where(numer<den)[0]
        selected=sorted(eligible,key=lambda j:(numer[j]/den[j],keys[j]))[:(len(eligible)+1)//2]
        d["hard_ids"]=np.where(np.isin(ids,keys[selected]),ids,0)
        d["miss_coords"]=[np.argwhere((ids==keys[j])&~hit) for j in selected]
        summaries.append(dict(image=name,paths=len(keys),failed=len(eligible),selected=keys[selected].tolist(),
                              hits=numer[selected].tolist(),den=den[selected].tolist()))
    return summaries


def crop(d,center,shape,flip,gain):
    start=np.clip(center-np.array(shape)//2,0,np.array(d["v"].shape)-shape)
    window=tuple(slice(int(a),int(a+b)) for a,b in zip(start,shape))
    rgb=d["rgb"][window]
    mask=np.where(d["f"][window],d["v"][window],2).astype(np.int16)
    near=d["n"][window]
    ids=d["hard_ids"][window]
    arr=[]
    for a in (rgb,mask,near,ids):
        if flip[0]: a=a[::-1]
        if flip[1]: a=a[:,::-1]
        arr.append(torch.from_numpy(a.copy()))
    x=(arr[0].permute(2,0,1).float()*(gain/255)).clamp_(0,1)
    return x,arr[1][None],arr[2],arr[3]


def get_batch(items,names,seed,step,shape):
    rng=np.random.default_rng(seed*10000+step)
    ordinary=[];focused=[]
    active=0
    for slot in range(2):
        name=names[rng.integers(len(names))]
        d=items[name]
        flip=rng.integers(2,size=2)
        gain=rng.uniform(.9,1.1)
        center=rng.integers(np.array(shape)//2,np.array(d["f"].shape)-np.array(shape)//2+1)
        if slot==1:
            center=d["vcoords"][rng.integers(len(d["vcoords"]))]
        a=crop(d,center,shape,flip,gain)
        ordinary.append(a)
        if slot==1 and d["miss_coords"]:
            points=d["miss_coords"][rng.integers(len(d["miss_coords"]))]
            centered=points[rng.integers(len(points))]
            focused.append(crop(d,centered,shape,flip,gain))
            active+=1
        else: focused.append(a)
    return [[torch.stack(parts) for parts in zip(*block)] for block in (ordinary,focused)],active


class BranchLoss(torch.nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base=base

    def forward(self,outputs,targets):
        base=self.base(outputs,targets)
        logit=outputs[0][:,1].float()-outputs[0][:,0].float()
        terms=[]
        for z,ids,near,y in zip(logit,self.ids,self.near,targets[0][:,0]):
            sel=ids>0
            if not sel.any(): continue
            _,idx,counts=torch.unique(ids[sel],return_inverse=True,return_counts=True)
            pos=(F.softplus(-z[sel])/counts[idx]).sum()/len(counts)
            bg=near&(y==0)
            terms.append((pos+F.softplus(z[bg]).mean())/2 if bg.any() else pos)
        return base+.10*torch.stack(terms).mean() if terms else base


def focus_train(fold,seed):
    dest=ROOT/"models_focus"/f"fold{fold}"/str(seed)
    dest.mkdir(parents=True,exist_ok=True)
    if all((dest/m/"train.json").is_file() for m in METHODS): return
    items,names,_=paired.dataset(fold)
    mining=mine(items,names,fold,seed)
    (dest/"mining.json").write_text(json.dumps(mining))
    arms={m:trainer(fold,seed) for m in METHODS}
    source=ROOT/"models_paired"/f"fold{fold}"/str(seed)/"standard"/"weights.pt"
    state=torch.load(source,map_location="cpu",weights_only=True)
    for t in arms.values(): t.network.load_state_dict(state);t.network.train()
    del state
    arms["branch_aux"].loss=BranchLoss(arms["branch_aux"].loss)
    shape=arms["continue"].configuration_manager.patch_size
    scales=arms["continue"]._get_deep_supervision_scales()
    losses={m:[] for m in METHODS}
    skipped={m:0 for m in METHODS}
    active=0
    started=time.monotonic()
    for step in range(300):
        batch,n=get_batch(items,names,seed,step,shape)
        active+=n
        for m,t in arms.items():
            x,y,near,ids=batch[int(m!="continue")]
            targets=[y if np.allclose(scale,1) else F.interpolate(y.float(),size=tuple(
                     int(round(dim*f)) for dim,f in zip(shape,scale)),mode="nearest-exact").to(y.dtype) for scale in scales]
            for group in t.optimizer.param_groups: group["lr"]=.002*(1-(step//30)/10)**.9
            if m=="branch_aux":
                t.loss.ids=ids.cuda();t.loss.near=near.cuda()
            before=t.grad_scaler.get_scale()
            loss=float(t.train_step(dict(data=x,target=targets))["loss"])
            skipped[m]+=int(t.grad_scaler.get_scale()<before)
            if not np.isfinite(loss): raise ValueError("Nonfinite continuation loss")
            t.optimizer.zero_grad(set_to_none=True)
            losses[m].append(loss)
        if (step+1)%100==0: print(fold,seed,step+1,"focus",active,round(time.monotonic()-started),flush=True)
    with source.open("rb") as f: origin=hashlib.file_digest(f,"sha256").hexdigest()
    for m,t in arms.items():
        folder=dest/m;folder.mkdir(exist_ok=True)
        torch.save(t.network.state_dict(),folder/"weights.tmp")
        (folder/"weights.tmp").replace(folder/"weights.pt")
        info=dict(method=m,fold=fold,seed=seed,steps=300,amp_skips=skipped[m],shared_source_sha=origin,
                  focused_crops=active if m!="continue" else 0,loss=losses[m])
        (folder/"train.tmp").write_text(json.dumps(info));(folder/"train.tmp").replace(folder/"train.json")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--fold",type=int,required=True)
    ap.add_argument("--seed",type=int,required=True)
    args=ap.parse_args()
    torch.set_num_threads(2)
    focus_train(args.fold,args.seed)
    evaluate(args.fold,args.seed,focus=True)
