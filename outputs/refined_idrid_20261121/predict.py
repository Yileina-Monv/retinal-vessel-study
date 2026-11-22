"""Predict a 1024x1024 Refined-IDRiD-format RGB photograph with a saved model."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import paired
from joint import make_model

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights",type=Path)
    parser.add_argument("image",type=Path)
    parser.add_argument("output_png",type=Path)
    options=parser.add_mutually_exclusive_group()
    options.add_argument("--threshold",type=float,default=.5)
    options.add_argument("--two-scale-calibration",type=Path,help="Use both scales and the training-only calibration for these weights")
    args=parser.parse_args()
    if args.two_scale_calibration:
        args.threshold=float(json.loads(args.two_scale_calibration.read_text())["threshold"])
    if not 0<args.threshold<1 or args.output_png.suffix.lower()!=".png":
        raise ValueError("Use a probability threshold in (0,1) and PNG output")
    with Image.open(args.image) as source:
        if source.size!=(1024,1024): raise ValueError("Expected a cropped 1024x1024 RGB image")
        rgb=np.array(source.convert("RGB"))
    state=torch.load(args.weights,map_location="cpu",weights_only=True)
    kind="joint" if state["head.weight"].shape[0]==2 else "base"
    model=make_model(kind)
    model.load_state_dict(state)
    model.cuda().eval()
    if args.two_scale_calibration:
        from two_scale import predict
        _,prob=predict(model,{"rgb":rgb})
    else:
        x=paired.normalize(torch.from_numpy(rgb).permute(2,0,1))
        prob,_=paired.tiled_probability(model,x,patch=512,stride=256,batch_size=4)
        prob=prob.numpy()
    # GT FOV is used only when scoring. Do not require labels at prediction time.
    mask=(prob>=args.threshold).astype(np.uint8)*255
    with args.output_png.open("xb") as f: Image.fromarray(mask).save(f,format="PNG")

if __name__=="__main__": main()
