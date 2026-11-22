"""Train-only calibration of a single global threshold to baseline near FP/image-area."""
import csv
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi

import paired as p
from analyze import SEEDS, summarize
from retinal_m0.model import UNet
from retinal_m0.inference import tiled_probability

RUN=p.HERE/"rgb_only"
GRID=np.arange(1,1000,dtype=float)/1000


def predict(model,d):
    x=p.normalize(torch.from_numpy(d["rgb"]).permute(2,0,1))
    prob,_=tiled_probability(model,x,patch=512,stride=256,batch_size=4)
    return prob.numpy()


def calibration(model,items,names):
    densities=[]
    for name in names:
        d=items[name]
        if not d["n"].any(): continue
        pr=predict(model,d)
        bg=np.sort(pr[d["n"] & ~d["v"]])
        densities.append((len(bg)-np.searchsorted(bg,GRID))/d["n"].sum())
    return np.mean(densities,axis=0)


def scoring(model,items,names,threshold,run,method,seed):
    rows=[]
    for i,name in enumerate(names):
        d=items[name]
        prob=predict(model,d)
        with np.load(run/f"{method}_{seed}"/(name+".npz")) as saved:
            fixed=np.unpackbits(saved["prediction"]).reshape(d["f"].shape)
        assert np.array_equal((prob>=.5)&d["f"],fixed), (name,"prediction drift")
        pred=(prob>=threshold)&d["f"]
        hit=ndi.binary_dilation(pred,structure=p.cross)
        ids=d["i"];sel=ids>0
        keys,den=np.unique(ids[sel],return_counts=True)
        hits=np.bincount(ids[sel & hit],minlength=int(ids.max())+1)[keys]
        thin=d["s"]&d["n"];far=d["f"]&~(d["n"]|d["h"])
        tp=(pred&d["v"]).sum();fp=(pred&~d["v"]).sum()
        def div(a,b): return float(a/b) if b else None
        rows.append(dict(image=name,path_d1=float((hits/den).mean()) if len(den) else None,
          thin_d1=div((thin&hit).sum(),thin.sum()),near_fp_density=div((pred&d["n"]&~d["v"]).sum(),d["n"].sum()),
          far_recall=div((pred&d["v"]&far).sum(),(d["v"]&far).sum()),dice=div(2*tp,pred.sum()+d["v"].sum()),
          precision=div(tp,tp+fp),fp_fov=div(fp,d["f"].sum())))
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--candidate",choices=["local","rank","joint"],default="local")
    parser.add_argument("--fold",type=int,choices=[0,1,2],default=0)
    parser.add_argument("--reverse",action="store_true",help="Match baseline to candidate's original FP level on Train")
    args=parser.parse_args()
    candidate=args.candidate
    from joint import make_model
    stage=p.HERE if args.fold==0 else p.HERE/f"fold_{args.fold}"
    run=stage/("joint_720" if candidate=="joint" else "rgb_only")
    target_dir=run/("rank_calibrated" if candidate=="rank" else "calibrated")
    first,second=("base",candidate)
    if args.reverse:
        target_dir=run/("reverse_"+candidate)
        first,second=second,first
    torch.set_num_threads(2)
    items,train,dev=p.dataset(args.fold)
    records=[]
    for seed in SEEDS:
        for name in (first,second):
            model=make_model(name).cuda()
            model.load_state_dict(torch.load(run/f"{name}_{seed}"/"weights.pt",map_location="cpu",weights_only=True))
            curve=calibration(model,items,train)
            if name==first:
                target=float(curve[499])
                t=.5
            else:
                eligible=np.flatnonzero(curve<=target)
                if not len(eligible): raise ValueError("No train-only FP match")
                t=float(GRID[eligible[0]])
            records.append(dict(seed=seed,method=name,threshold=t,train_target=target,train_fp=float(curve[round(t*1000)-1])))
            old=json.loads((run/f"{name}_{seed}"/"dev.json").read_text())
            rows=scoring(model,items,dev,t,run,name,seed)
            if name==first:
                for a,b in zip(rows,old):
                    assert all(b[k]==v for k,v in a.items())
            out=target_dir/f"{name}_{seed}"
            out.mkdir(parents=True,exist_ok=True)
            (out/"dev.json").write_text(json.dumps(rows,indent=2,allow_nan=False))
    (target_dir/"thresholds.json").write_text(json.dumps(records,indent=2))
    summarize(target_dir,candidate=candidate,images=dev)


if __name__=="__main__": main()
