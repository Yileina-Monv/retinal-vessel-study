"""Paired, fixed-endpoint Refined IDRiD experiment; only the author Train.tar is opened."""
import argparse
import csv
import hashlib
import io
import json
import tarfile
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from retinal_m0.model import UNet
from retinal_m0.inference import tiled_probability

CHECKPOINT = ROOT / "runs/prefetch_round1/m0_seed2026_dev10k_w1/last.pt"
SHA = "8bce5b8f47c7fe43d553d4467adc46f5387eade014265b7a6a6f4b5211e67d79"
STEPS = 240
norm = json.loads((ROOT / "releases/development_v1/normalization.json").read_text())
mean = torch.tensor(norm["mean"])[:,None,None]
std = torch.tensor(norm["std"])[:,None,None]
cross = ndi.generate_binary_structure(2, 1)


def normalize(rgb):
    # The deployed model receives only RGB. GT FOV is reserved for scoring and loss.
    support = rgb.amax(dim=-3,keepdim=True)>10
    return ((rgb/255-mean)/std)*support


def features(mask, include_tracks=False):
    vessel, he, fov = mask == 24, mask == 127, mask != 0
    radius = int(np.ceil(mask.shape[1]/100))
    near = (ndi.distance_transform_edt(~he) <= radius) & ~he & fov if he.any() else np.zeros_like(he)
    sk = skeletonize(vessel)
    width = 2 * ndi.distance_transform_edt(vessel)
    thin = sk & (width <= 3)
    # Use unbranched interiors for a new, explicit reference on this cohort.
    neighbours = ndi.convolve(sk.astype(np.uint8), np.ones((3,3),np.uint8), mode="constant") - sk
    tracks, n = ndi.label(sk & (neighbours <= 2), structure=np.ones((3,3),bool))
    keep = np.zeros(n+1,bool)
    for i, slices in enumerate(ndi.find_objects(tracks),1):
        if slices is None: continue
        body = tracks[slices] == i
        if body.sum() >= 20 and np.median(width[slices][body]) <= 6:
            keep[i] = True
    near_ids = np.where(near & keep[tracks], tracks, 0).astype(np.int32)
    result = (vessel, fov, near, thin, near_ids)
    if include_tracks: return (*result, np.where(keep[tracks],tracks,0).astype(np.int32))
    return result


def dataset(fold=0, include_tracks=False):
    archive = HERE / "Train.tar"
    if hashlib.md5(archive.read_bytes()).hexdigest() != "e1d19eb415bea20c9d812276499e1f2b":
        raise ValueError("Unexpected author archive")
    items = {}
    with tarfile.open(archive) as t:
        for i in range(1,55):
            name = f"IDRiD_{i:02d}"
            def get_image(path):
                return np.asarray(Image.open(io.BytesIO(t.extractfile(path).read()))).copy()
            inputs = [m.name for m in t.getmembers() if m.isfile() and str(Path(m.name).parent.as_posix())=="Train/Images" and Path(m.name).stem==name]
            if len(inputs)!=1: raise ValueError(f"Nonunique RGB source for {name}")
            rgb = get_image(inputs[0])
            y = get_image(f"Train/Labels/{name}_vessel.png")
            if rgb.shape != (1024,1024,3) or y.shape != rgb.shape[:2]: raise ValueError(name)
            vessel, fov, near, thin, ids, *tracks = features(y, include_tracks)
            items[name] = dict(rgb=rgb,v=vessel,f=fov,h=y==127,n=near,s=thin,i=ids,
                               centers=np.argwhere(near & vessel))
            if tracks: items[name]["all_tracks"] = tracks[0]
    rgb_hash = [hashlib.sha256(x["rgb"].tobytes()).hexdigest() for x in items.values()]
    if len(set(rgb_hash)) != 54: raise ValueError("Duplicate image")
    order = sorted(items,key=lambda k:hashlib.sha256(("refined-holdout-v1/"+k).encode()).hexdigest())
    dev = sorted(order[18*fold:18*(fold+1)])
    return items, sorted(set(order)-set(dev)), dev


def draw(items, names, rng):
    batches = []
    for _ in range(4):
        d = items[names[int(rng.integers(len(names)))]]
        if rng.random() < .5 and len(d["centers"]):
            center = d["centers"][rng.integers(len(d["centers"]))]
            top,left = np.clip(center-rng.integers(32,224,size=2),0,768)
        else: top,left = rng.integers(769,size=2)
        crop = np.s_[top:top+256,left:left+256]
        arr = [d[k][crop] for k in ("rgb","v","f","n","i","h")]
        if rng.random()<.5: arr = [a[:,::-1] for a in arr]
        batches.append([torch.from_numpy(a.copy()) for a in arr])
    x,v,f,n,idx,he = [torch.stack(t) for t in zip(*batches)]
    x = normalize(x.permute(0,3,1,2))
    return [t.cuda() for t in (x,v,f,n,idx,he)]


def loss_fn(logit,v,f,n,ids,special):
    z = logit[:,0].float()
    axes=(1,2)
    bce=(F.binary_cross_entropy_with_logits(z,v.float(),reduction="none")*f).sum(axes)/f.sum(axes).clamp_min(1)
    p=z.sigmoid()*f
    base=(bce + 1-(2*(p*v).sum(axes)+1)/(p.sum(axes)+v.sum(axes)+1)).mean()
    if special=="base": return base
    terms=[]
    for pred,path,near,truth in zip(z,ids,n,v):
        selected=path>0
        if not selected.any(): continue
        _, inverse, counts = torch.unique(path[selected],return_inverse=True,return_counts=True)
        background=near & ~truth
        if special=="rank":
            if not background.any(): continue
            negative=pred[background]
            hardest=torch.topk(negative,max(1,int(np.ceil(.03*negative.numel())))).values.mean()
            # Pair a path position with the challenging-background mean in the same crop.
            positive=F.softplus(1+hardest-pred[selected])
            terms.append((positive/counts[inverse]).sum()/len(counts))
        else:
            positive=(F.softplus(-pred[selected])/counts[inverse]).sum()/len(counts)
            terms.append((positive+F.softplus(pred[background]).mean())/2 if background.any() else positive)
    return base + .12*torch.stack(terms).mean() if terms else base


def evaluate(model,items,names,dest):
    rows=[]
    for name in names:
        d=items[name]
        x=normalize(torch.from_numpy(d["rgb"]).permute(2,0,1))
        prob,_=tiled_probability(model,x,patch=512,stride=256,batch_size=4)
        pred=(prob.numpy()>=.5)&d["f"]
        hit=ndi.binary_dilation(pred,structure=cross)
        ids=d["i"]; sel=ids>0
        keys,den=np.unique(ids[sel],return_counts=True)
        num=np.bincount(ids[sel & hit],minlength=int(ids.max())+1)[keys]
        thin=d["s"]&d["n"]; far=d["f"]&~(d["n"]|d["h"])
        tp=(pred&d["v"]).sum(); fp=(pred&~d["v"]).sum()
        def div(a,b): return float(a/b) if b else None
        rows.append(dict(image=name,paths=len(keys),path_points=int(sel.sum()),path_d1=float((num/den).mean()) if len(keys) else None,
                         thin_den=int(thin.sum()),thin_hits=int((thin&hit).sum()),thin_d1=div((thin&hit).sum(),thin.sum()),
                         near_fp=int((pred & d["n"] & ~d["v"]).sum()),near_area=int(d["n"].sum()),
                         near_fp_density=div((pred&d["n"]&~d["v"]).sum(),d["n"].sum()),
                         far_recall=div((pred&d["v"]&far).sum(),(d["v"]&far).sum()),dice=div(2*tp,pred.sum()+d["v"].sum()),
                         precision=div(tp,tp+fp),fp_fov=div(fp,d["f"].sum())))
        np.savez_compressed(dest/(name+".npz"),prediction=np.packbits(pred),branch_ids=keys,hits=num,denominators=den)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--rank",action="store_true")
    parser.add_argument("--fold",type=int,choices=(0,1,2),default=0)
    args=parser.parse_args()
    torch.set_num_threads(2)
    items,train,dev=dataset(args.fold)
    stage=HERE if args.fold==0 else HERE/f"fold_{args.fold}"
    stage.mkdir(exist_ok=True)
    with (stage/"split.json").open("w") as f: json.dump(dict(train=train,dev=dev),f,indent=2)
    if args.prepare_only:
        print("Train/dev",len(train),len(dev),"effective near-path images",*[sum(bool(items[k]["i"].any()) for k in group) for group in (train,dev)])
        return
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()!=SHA: raise ValueError("M0 mismatch")
    for seed in (2027,2028,2029):
        for method in (("rank",) if args.rank else ("base","local")):
            dest=stage/"rgb_only"/f"{method}_{seed}"
            dest.mkdir(parents=True,exist_ok=True)
            result=dest/"dev.json"
            if result.exists(): continue
            rng=np.random.default_rng(seed)
            torch.manual_seed(seed)
            model=UNet().cuda()
            model.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",weights_only=True)["model"])
            opt=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.0001)
            for step in range(1,STEPS+1):
                x,v,f,n,idx,_=draw(items,train,rng)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    loss=loss_fn(model(x),v,f,n,idx,method)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                if step%60==0: print(seed,method,step,float(loss.detach()),flush=True)
            torch.save(model.state_dict(),dest/"weights.pt")
            rows=evaluate(model,items,dev,dest)
            tmp=result.with_suffix(".tmp")
            tmp.write_text(json.dumps(rows,indent=2,allow_nan=False))
            tmp.replace(result)
            print("Saved",dest.name,flush=True)


if __name__=="__main__": main()
