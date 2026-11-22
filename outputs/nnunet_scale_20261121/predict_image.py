"""Predict from a 1024-square RGB fundus photograph with a research fold endpoint."""
import argparse
import json
from pathlib import Path

from prepare import ROOT,stage
import numpy as np
from PIL import Image
import torch
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from evaluate import probabilities


def infer(image,fold,seed,method,view,point):
    family="paired" if method in ("standard","mix") else "teacher" if method in ("extra","correction") else "focus"
    weights=ROOT/("models_"+family)/f"fold{fold}"/str(seed)/method/"weights.pt"
    cal=json.loads((ROOT/("eval_"+family)/f"fold{fold}"/str(seed)/method/"calibration.json").read_text())
    t=.5 if point=="fixed" else cal["thresholds" if point=="matched" else "m0_thresholds"][view]
    pm=PlansManager(str(ROOT/"preprocessed"/stage(fold)/"nnUNetPlans.json"))
    cfg=pm.get_configuration("2d")
    meta=json.loads((ROOT/"raw"/stage(fold)/"dataset.json").read_text())
    net=nnUNetTrainer.build_network_architecture(cfg.network_arch_class_name,cfg.network_arch_init_kwargs,
             cfg.network_arch_init_kwargs_req_import,3,2,enable_deep_supervision=True)
    net.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True))
    net.decoder.deep_supervision=False
    pred=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,device=torch.device("cuda"),allow_tqdm=False)
    pred.manual_initialization(net.eval().cuda(),pm,cfg,None,meta,"nnUNetTrainer",None)
    pr=probabilities(pred,image,native_only=view=="native")[view]
    return (pr>=t).astype(np.uint8)*255


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image",type=Path)
    ap.add_argument("output",type=Path)
    ap.add_argument("--fold",type=int,choices=(0,1,2),required=True)
    ap.add_argument("--seed",type=int,choices=(2027,2028,2029),required=True)
    ap.add_argument("--method",choices=("standard","mix","continue","branch","branch_aux","extra","correction"),required=True)
    ap.add_argument("--view",choices=("native","large","dual"),required=True)
    ap.add_argument("--point",choices=("fixed","matched","m0matched"),default="matched")
    args=ap.parse_args()
    torch.set_num_threads(2)
    with Image.open(args.image) as file:
        if file.size!=(1024,1024): raise ValueError("Expected a 1024x1024 Refined-IDRiD-format RGB.")
        rgb=np.asarray(file.convert("RGB"))
    mask=infer(rgb,args.fold,args.seed,args.method,args.view,args.point)
    with args.output.open("xb") as f: Image.fromarray(mask).save(f,format="PNG")
