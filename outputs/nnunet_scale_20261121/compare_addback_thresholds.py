"""Frozen inference. Calibrate three fixed scale weights against addback on train only."""
import json
import numpy as np
import torch
from scipy import ndimage as ndi
from prepare import ROOT, split
from evaluate import probabilities
from predict_image import PlansManager, nnUNetPredictor, nnUNetTrainer, stage
from matched_fp import GRID
from two_scale import score
import paired

OUT = ROOT / "addback_calibration"
WEIGHTS = {"equal": .5, "native75": .75, "large75": .25}

def load(path):
    return json.loads(path.read_text())

def make_model(fold, seed):
    plans = PlansManager(str(ROOT/"preprocessed"/stage(fold)/"nnUNetPlans.json"))
    cfg = plans.get_configuration("2d")
    meta = load(ROOT/"raw"/stage(fold)/"dataset.json")
    net = nnUNetTrainer.build_network_architecture(cfg.network_arch_class_name,cfg.network_arch_init_kwargs,
            cfg.network_arch_init_kwargs_req_import,3,2,enable_deep_supervision=True)
    net.load_state_dict(torch.load(ROOT/"models_teacher"/f"fold{fold}"/str(seed)/"correction/weights.pt",
                                   map_location="cpu",weights_only=True))
    net.decoder.deep_supervision=False
    pred=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,device=torch.device("cuda"),allow_tqdm=False)
    pred.manual_initialization(net.eval().cuda(),plans,cfg,None,meta,"nnUNetTrainer",None)
    return pred

def addback(p,t):
    dual=p["dual"]>=t["dual"]
    return dual | ((p["native"]>=t["native"]) & ndi.binary_dilation(dual,structure=paired.cross))

def blend(p,k):
    w=WEIGHTS[k]
    return p["dual"] if k=="equal" else w*p["native"]+(1-w)*p["large"]

def run(cases,fold,seed):
    group=ROOT/"eval_teacher"/f"fold{fold}"/str(seed)/"correction"
    t=load(group/"calibration.json")["m0_thresholds"]
    pred=make_model(fold,seed)
    ids=split(fold)
    curves={k:[] for k in WEIGHTS}
    targets=[]
    for name in ids["train"]:
        d=cases[name]
        area=d["n"].sum()
        if not area: continue
        p=probabilities(pred,d["rgb"])
        bg=d["n"] & ~d["v"]
        targets.append((addback(p,t)&bg).sum()/area)
        for k in curves:
            dist=np.sort(blend(p,k)[bg])
            curves[k].append((dist.size-np.searchsorted(dist,GRID))/area)
    target=float(np.mean(targets))
    cutoffs={k:float(GRID[np.flatnonzero(np.mean(c,axis=0)<=target)[0]]) for k,c in curves.items()}
    rows=[]
    for name in ids["dev"]:
        d=cases[name]
        p=probabilities(pred,d["rgb"])
        with np.load(group/(name+".npz")) as z:
            for v in ("native","dual"):
                assert np.array_equal(np.packbits((p[v]>=t[v]) & d["f"]),z[v+"_m0matched"])
        masks={k:blend(p,k)>=cutoffs[k] for k in curves}
        masks["addback"]=addback(p,t)
        for k,mask in masks.items():
            mask=mask&d["f"]
            s=score(d,mask,.5)
            thin=d["n"]&d["s"]
            s["thin_d0"]=float(mask[thin].mean()) if thin.any() else None
            rows.append(dict(image=name,method=k,**s))
    output=OUT/f"fold{fold}_{seed}.json"
    tmp=output.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(seed=seed,fold=fold,train_addback_fp=target,thresholds=cutoffs,rows=rows),allow_nan=False))
    tmp.replace(output)
    print(fold,seed,cutoffs,flush=True)

if __name__=="__main__":
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cases=paired.dataset(0)[0]
    for fold in range(3):
        for seed in (2027,2028,2029):
            if not (OUT/f"fold{fold}_{seed}.json").exists():
                run(cases,fold,seed)
