"""Run a fitted local scale selector on one decoded RGB image without annotations."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import torch
from PIL import Image
from predict_view_fusion import models
from switch_selector import OUT,ROOT,load,enhance,views,fuse,METHODS
from union_budget_cycle import probabilities_pair

BUDGET_MODES=("mask_union","lower_reference")

def settings(fold,seed,mode,point):
    if mode in BUDGET_MODES:
        params=load(OUT/"union_budget/union_vs_lower"/f"fold{fold}_{seed}.json")
        return tuple(params[k][point] for k in ("base_cutoffs","zoom_cutoffs","lower_cutoffs"))
    trees=joblib.load(OUT/f"selector{fold}"/"models.joblib") if mode.startswith(("learned","shuffled")) else []
    fname=f"fold{fold}_{seed}.json"
    anchor=load(ROOT/"cross_family_cycle/m0"/fname)["thresholds"]["m0matched"]
    cutoff=load(OUT/mode/fname)["thresholds"][point]
    return trees,anchor,cutoff

def predict(m,n,rgb,mode,settings_):
    if rgb.dtype!=np.uint8 or rgb.shape!=(1024,1024,3):
        raise ValueError("Expected a 1024 by 1024 RGB image.")
    if mode in BUDGET_MODES:
        b,z=probabilities_pair(m,n,enhance(rgb));tb,tz,tl=settings_
        mask=((b>=tb)|(z>=tz)) if mode=="mask_union" else b>=tl
        return mask.astype(np.uint8)*255
    trees,anchor,cutoff=settings_
    maps=views(m,n,enhance(rgb))
    probs=fuse(rgb,maps,anchor,trees,(mode,))[2][mode]
    return (probs>=cutoff).astype(np.uint8)*255

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--fold",type=int,required=True,choices=range(3))
    parser.add_argument("--seed",type=int,choices=(2027,2028,2029),default=2027)
    parser.add_argument("--mode",choices=(*METHODS,*BUDGET_MODES),default="learned_add")
    parser.add_argument("--point",choices=("m0matched","matched"),default="m0matched")
    a=parser.parse_args()
    with Image.open(a.input) as image:rgb=np.array(image.convert("RGB"))
    torch.set_num_threads(2);m,n=models(a.fold,a.seed)
    mask=predict(m,n,rgb,a.mode,settings(a.fold,a.seed,a.mode,a.point))
    with a.output.open("xb") as stream:Image.fromarray(mask).save(stream,format="PNG")
