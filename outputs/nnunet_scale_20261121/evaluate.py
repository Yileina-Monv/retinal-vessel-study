"""Evaluate fixed weights at three scales with train-fold-only threshold matching."""
import argparse
import json
import time
from pathlib import Path
from prepare import ROOT,OLD,split
import numpy as np
import torch
from scipy import ndimage as ndi
from torch.nn import functional as F
import paired
from two_scale import score
from matched_fp import GRID
from train import trainer
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

VIEWS=("native","large","dual")


def make_predictor(fold,seed,method):
    t=trainer(fold,seed)
    group=("models_paired" if method in ("standard","mix") else
           "models_teacher" if method in ("extra","correction") else "models_focus")
    source=ROOT/group/f"fold{fold}"/str(seed)/method/"weights.pt"
    t.network.load_state_dict(torch.load(source,map_location="cpu",weights_only=True))
    t.set_deep_supervision_enabled(False)
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,
                              device=torch.device("cuda"),allow_tqdm=False)
    predictor.manual_initialization(t.network.eval(),t.plans_manager,t.configuration_manager,None,
                                    t.dataset_json,type(t).__name__,None)
    return predictor


@torch.inference_mode()
def probabilities(pred,rgb,native_only=False,*,large_only=False):
    # The official predictor enables cuDNN benchmarking; pin inference algorithm choice across fresh processes.
    with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=True,allow_tf32=torch.backends.cudnn.allow_tf32):
        if native_only and large_only: raise ValueError("Select one scale")
        return _probabilities(pred,rgb,native_only,large_only)


def _probabilities(pred,rgb,native_only,large_only=False):
    # Whole-image RGB preprocessing follows the same nonzero crop and normalization as training.
    props={"spacing":(999,1,1)}
    cfg=pred.configuration_manager
    a,_,_=cfg.preprocessor_class(verbose=False).run_case_npy(
        rgb.transpose(2,0,1)[:,None],None,props,pred.plans_manager,cfg,pred.dataset_json)
    assert list(pred.plans_manager.transpose_forward)==[0,1,2]
    x=torch.from_numpy(np.ascontiguousarray(a))
    def network(data):
        logits=pred.predict_sliding_window_return_logits(data)
        result=logits.float().softmax(0)[1,0].cpu()
        if not result.isfinite().all(): raise ValueError("Invalid probability")
        return result
    native=None if large_only else network(x)
    sl=tuple(slice(*bounds) for bounds in props["bbox_used_for_cropping"][1:])
    def restore(prob):
        out=np.zeros(rgb.shape[:2],np.float32)
        out[sl]=prob.numpy()
        return out
    if native_only: return {"native":restore(native)}
    big=F.interpolate(x[:,0][None],scale_factor=2,mode="bilinear",align_corners=False)[0,:,None]
    large=F.avg_pool2d(network(big)[None,None],2)[0,0]
    if large_only: return {"large":restore(large)}
    return dict(zip(VIEWS,(restore(native),restore(large),restore((native+large)*.5))))


def describe(d,prob,threshold):
    metrics=score(d,prob,threshold)
    mask=(prob>=threshold)&d["f"]
    hit=ndi.binary_dilation(mask,structure=paired.cross)
    ids=d["i"]
    keys,den=np.unique(ids[ids>0],return_counts=True)
    counts=np.bincount(ids[hit],minlength=int(ids.max())+1)[keys]
    assert metrics["path_d1"]==(float(np.mean(counts/den)) if len(den) else None)
    return dict(**metrics,paths=keys.tolist(),hits=counts.tolist(),den=den.tolist(),
                near_fp=int((mask&d["n"]&~d["v"]).sum()),near_area=int(d["n"].sum()),
                thin_hits=int((hit&d["n"]&d["s"]).sum()),thin_den=int((d["n"]&d["s"]).sum()))


def evaluate(fold,seed,focus=False,teacher=False):
    data,tr,dev=paired.dataset(fold)
    assert split(fold)==dict(train=tr,dev=dev)
    original=ROOT/"eval_paired"/f"fold{fold}"/str(seed)
    run=ROOT/("eval_teacher" if teacher else "eval_focus")/f"fold{fold}"/str(seed) if focus or teacher else original
    run.mkdir(exist_ok=True,parents=True)
    methods=("extra","correction") if teacher else ("continue","branch","branch_aux") if focus else ("standard","mix")
    for method in methods:
        folder=run/method
        folder.mkdir(exist_ok=True)
        if (folder/"dev.json").is_file(): continue
        pred=make_predictor(fold,seed,method)
        curves={v:[] for v in VIEWS}
        for name in tr:
            d=data[name]
            area=int(d["n"].sum())
            if area==0: continue
            for view,pr in probabilities(pred,d["rgb"]).items():
                bg=np.sort(pr[d["n"]&~d["v"]])
                curves[view].append((len(bg)-np.searchsorted(bg,GRID))/area)
        curves={v:np.mean(rows,axis=0) for v,rows in curves.items()}
        target=float(curves["native"][499]) if method=="standard" else json.loads((original/"standard"/"calibration.json").read_text())["target"]
        th={}
        for view,curve in curves.items():
            valid=np.flatnonzero(curve<=target)
            if not len(valid): raise ValueError("FP target is outside the grid")
            th[view]=.5 if method=="standard" and view=="native" else float(GRID[valid[0]])
        oldfold=OLD if fold==0 else OLD/f"fold_{fold}"
        prior=json.loads((oldfold/"rgb_only/calibrated/thresholds.json").read_text())
        m0target=next(r["train_target"] for r in prior if r["method"]=="base" and r["seed"]==seed)
        m0th={v:float(GRID[np.flatnonzero(curve<=m0target)[0]]) for v,curve in curves.items()}
        (folder/"calibration.json").write_text(json.dumps(dict(target=target,thresholds=th,train=tr,
                                                               m0_target=m0target,m0_thresholds=m0th)))
        rows=[]
        for name in dev:
            d=data[name]
            masks={}
            for view,pr in probabilities(pred,d["rgb"]).items():
                for point,t in (("fixed",.5),("matched",th[view]),("m0matched",m0th[view])):
                    metrics=describe(d,pr,t)
                    rows.append(dict(image=name,view=view,point=point,**metrics))
                    masks[view+"_"+point]=np.packbits((pr>=t)&d["f"])
            np.savez_compressed(folder/(name+".npz"),**masks)
        tmp=folder/"dev.tmp"
        tmp.write_text(json.dumps(rows,allow_nan=False))
        tmp.replace(folder/"dev.json")
        print(fold,seed,method,th,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--fold",type=int,required=True)
    ap.add_argument("--seed",type=int,required=True)
    args=ap.parse_args()
    torch.set_num_threads(2)
    from train import train_pair
    train_pair(args.fold,args.seed)
    evaluate(args.fold,args.seed)
