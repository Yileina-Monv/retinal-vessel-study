"""Reusable, scale-selective RGB inference for the appearance and scale ablations."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from prepare import ROOT,OLD
from compare_addback_thresholds import make_model,load
from contrast_cycle import enhance
from evaluate import probabilities
from two_scale import predict as m0_dual
import paired

MODES=("full","raw","eight","m0_supported","m0_two_thirds","regional")
SCALES=("native","large","dual")

def catalog(mode):
    if mode=="full": return ROOT/"cross_family_cycle/nnunet"
    if mode=="regional": return ROOT/"local_scale_gate/regional"
    if mode in MODES: return ROOT/"appearance_fusion_cycle"/mode
    m,n=mode.split("_")
    if m not in SCALES or n not in SCALES or m==n=="dual": raise ValueError(mode)
    return ROOT/"family_scale_ablation"/mode

def models(fold,seed):
    stage=OLD if not fold else OLD/f"fold_{fold}"
    m=paired.UNet().cuda().eval()
    m.load_state_dict(torch.load(stage/"rgb_only"/f"local_{seed}"/"weights.pt",map_location="cpu",weights_only=True))
    n=make_model(fold,seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    return m,n

def m0_single(model,rgb,view):
    x=paired.normalize(torch.from_numpy(rgb).permute(2,0,1))
    if view=="large":
        x=F.interpolate(x[None],scale_factor=2,mode="bilinear",align_corners=False)[0]
    p=paired.tiled_probability(model,x,patch=512,stride=256,batch_size=4)[0]
    if view=="large": p=F.avg_pool2d(p[None,None],2)[0,0]
    return p.numpy()

def region_parameters(fold):
    import joblib
    return joblib.load(ROOT/"local_scale_gate"/f"gate{fold}"/"region.joblib")

@torch.inference_mode()
def probability(m,n,rgb,mode,region=None):
    if rgb.shape!=(1024,1024,3) or rgb.dtype!=np.uint8: raise ValueError("Expected 1024x1024 RGB uint8")
    if mode in MODES:
        image=rgb if mode=="raw" else enhance(rgb)
        native,mp=m0_dual(m,{"rgb":image})
        nn=probabilities(n,image)["dual"]
        base=(mp+nn)*.5
        if mode=="regional":
            from local_scale_gate import features,gate
            g=gate(*features(rgb),*region)
            return base+g*((np.clip(2*mp-native,0,1)-mp)*.5)
        if mode=="eight":
            raw=(m0_dual(m,{"rgb":rgb})[1]+probabilities(n,rgb)["dual"])*.5
            return (base+raw)*.5
        if mode=="m0_supported":
            return base+.5*np.maximum(np.minimum(native,np.clip(2*mp-native,0,1))-nn,0)
        if mode=="m0_two_thirds": return (2*mp+nn)/3
        return base
    mv,nv=mode.split("_");x=enhance(rgb)
    mp=m0_dual(m,{"rgb":x})[1] if mv=="dual" else m0_single(m,x,mv)
    nn=probabilities(n,x,native_only=nv=="native",large_only=nv=="large")[nv]
    return (mp+nn)*.5

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input",type=Path);ap.add_argument("output",type=Path)
    ap.add_argument("--mode",required=True);ap.add_argument("--fold",type=int,choices=range(3),required=True)
    ap.add_argument("--seed",type=int,choices=(2027,2028,2029),required=True)
    ap.add_argument("--point",choices=("m0matched","matched"),default="m0matched")
    a=ap.parse_args();torch.set_num_threads(2)
    threshold=load(catalog(a.mode)/f"fold{a.fold}_{a.seed}.json")["thresholds"][a.point]
    with Image.open(a.input) as im: rgb=np.array(im.convert("RGB"))
    m,n=models(a.fold,a.seed)
    region=region_parameters(a.fold) if a.mode=="regional" else None
    mask=(probability(m,n,rgb,a.mode,region)>=threshold).astype(np.uint8)*255
    with a.output.open("xb") as stream: Image.fromarray(mask).save(stream,format="PNG")
