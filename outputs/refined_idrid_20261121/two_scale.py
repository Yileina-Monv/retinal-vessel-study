"""Post-training two-scale RGB inference with thresholds fixed on each training fold."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import torch
from torch.nn import functional as F
import paired as p
from matched_fp import GRID
from analyze import SEEDS, summarize

def score(d,prob,t):
    pred=(prob>=t)&d["f"]
    hit=ndi.binary_dilation(pred,structure=p.cross)
    ids=d["i"];sel=ids>0
    keys,den=np.unique(ids[sel],return_counts=True)
    hits=np.bincount(ids[sel&hit],minlength=int(ids.max())+1)[keys]
    thin=d["s"]&d["n"];far=d["f"]&~(d["n"]|d["h"])
    tp=int((pred&d["v"]).sum());fp=int((pred&~d["v"]).sum())
    def ratio(a,b):return float(a/b) if b else None
    return dict(path_d1=float((hits/den).mean()) if len(den) else None,thin_d1=ratio((hit&thin).sum(),thin.sum()),
                near_fp_density=ratio((pred&d["n"]&~d["v"]).sum(),d["n"].sum()),far_recall=ratio((pred&far&d["v"]).sum(),(far&d["v"]).sum()),
                dice=ratio(2*tp,pred.sum()+d["v"].sum()),precision=ratio(tp,tp+fp),fp_fov=ratio(fp,d["f"].sum()))


@torch.inference_mode()
def predict(model,d):
    x=p.normalize(torch.from_numpy(d["rgb"]).permute(2,0,1))
    native,_=p.tiled_probability(model,x,patch=512,stride=256,batch_size=4)
    large=F.interpolate(x[None],scale_factor=2,mode="bilinear",align_corners=False)[0]
    enlarged,_=p.tiled_probability(model,large,patch=512,stride=256,batch_size=4)
    down=F.avg_pool2d(enlarged[None,None],2)[0,0]
    return native.numpy(),((native+down)*.5).numpy()


def evaluate_fold(fold):
    stage=p.HERE if fold==0 else p.HERE/f"fold_{fold}"
    run=stage/"rgb_only"
    out=stage/"two_scale"
    items,train,dev=p.dataset(fold)
    for seed in SEEDS:
        for method in ("base","local"):
            dest=out/f"{method}_{seed}"; dest.mkdir(exist_ok=True,parents=True)
            if (dest/"dev.json").is_file(): continue
            model=p.UNet().cuda()
            model.load_state_dict(torch.load(run/f"{method}_{seed}"/"weights.pt",map_location="cpu",weights_only=True))
            target=json.loads((run/"calibrated"/"thresholds.json").read_text())
            target=next(r["train_target"] for r in target if r["seed"]==seed and r["method"]=="base")
            curves=[]
            for name in train:
                d=items[name]
                if not d["n"].any():continue
                _,prob=predict(model,d)
                bg=np.sort(prob[d["n"]&~d["v"]])
                curves.append((len(bg)-np.searchsorted(bg,GRID))/d["n"].sum())
            curve=np.mean(curves,axis=0)
            idx=np.flatnonzero(curve<=target)[0];t=float(GRID[idx])
            rows=[]
            for name in dev:
                native,prob=predict(model,items[name])
                with np.load(run/f"{method}_{seed}"/(name+".npz")) as z:
                    assert np.array_equal(np.packbits((native>=.5)&items[name]["f"]),z["prediction"])
                rows.append(dict(image=name,**score(items[name],prob,t)))
            (dest/"threshold.json").write_text(json.dumps(dict(target=target,threshold=t,achieved=curve[idx])))
            tmp=dest/"dev.tmp";tmp.write_text(json.dumps(rows,indent=2,allow_nan=False));tmp.replace(dest/"dev.json")
            print(fold,seed,method,"threshold",t,flush=True)
    summarize(out,images=dev)
    # Compare the complete two-scale specialist to the original single-scale baseline.
    compare=out/"vs_native"
    for seed in SEEDS:
        for m in ("base","local"):
            src=(run if m=="base" else out)/f"{m}_{seed}"/"dev.json"
            dest=compare/f"{m}_{seed}";dest.mkdir(parents=True,exist_ok=True)
            (dest/"dev.json").write_bytes(src.read_bytes())
    summarize(compare,images=dev)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--fold",choices=[0,1,2],type=int,default=0)
    torch.set_num_threads(2)
    evaluate_fold(parser.parse_args().fold)
