"""Predict one RGB photo using the evaluated contrast-enhanced fold model."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from prepare import ROOT,OLD
from contrast_cycle import enhance
from compare_addback_thresholds import make_model,load
from evaluate import probabilities

def predict(rgb,fold,seed,point,family="nnunet"):
    if rgb.shape!=(1024,1024,3) or rgb.dtype!=np.uint8:
        raise ValueError("Expected one 1024x1024 uint8 RGB fundus photo")
    if family in ("m0","pair"):
        import paired
        from two_scale import predict as two_scale
        stage=OLD if fold==0 else OLD/f"fold_{fold}"
        model=paired.UNet().cuda().eval()
        model.load_state_dict(torch.load(stage/"rgb_only"/f"local_{seed}"/"weights.pt",map_location="cpu",weights_only=True))
        enhanced=enhance(rgb)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        prob=two_scale(model,{"rgb":enhanced})[1]
        if family=="pair":
            del model
            predictor=make_model(fold,seed)
            prob=(prob+probabilities(predictor,enhanced)["dual"])*.5
            run=ROOT/"cross_family_cycle/nnunet"
        else:
            run=ROOT/"m0_contrast_transfer"
        cutoff=load(run/f"fold{fold}_{seed}.json")["thresholds"][point]
        return (prob>=cutoff).astype(np.uint8)*255
    cutoff=load(ROOT/"contrast_cycle"/f"fold{fold}_{seed}.json")["thresholds"][point]
    model=make_model(fold,seed)
    prob=probabilities(model,enhance(rgb))["dual"]
    return (prob>=cutoff).astype(np.uint8)*255

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image",type=Path)
    ap.add_argument("output",type=Path)
    ap.add_argument("--fold",type=int,choices=range(3),required=True)
    ap.add_argument("--seed",type=int,choices=(2027,2028,2029),required=True)
    ap.add_argument("--point",choices=("m0matched","matched"),default="m0matched")
    ap.add_argument("--family",choices=("m0","nnunet","pair"),default="nnunet")
    a=ap.parse_args()
    torch.set_num_threads(2)
    with Image.open(a.image) as im: rgb=np.array(im.convert("RGB"))
    mask=predict(rgb,a.fold,a.seed,a.point,a.family)
    with a.output.open("xb") as stream: Image.fromarray(mask).save(stream,format="PNG")
