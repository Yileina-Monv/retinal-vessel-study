"""Isolated inference time after warm-up; same 18 photos and resident weights."""
import json
import random
import time
import hashlib
import numpy as np
import torch
from PIL import Image
from prepare import ROOT,OLD,split
from contrast_cycle import enhance
from compare_addback_thresholds import make_model,load
from two_scale import predict as m0_predict,score
from evaluate import probabilities
import paired

def main():
    torch.set_num_threads(2)
    m=paired.UNet().cuda().eval()
    m.load_state_dict(torch.load(OLD/"rgb_only/local_2027/weights.pt",map_location="cpu",weights_only=True))
    n=make_model(0,2027)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    paths={"m0":"m0_contrast_transfer","nnunet":"contrast_cycle","pair":"cross_family_cycle/nnunet"}
    c={k:load(ROOT/v/"fold0_2027.json") for k,v in paths.items()}
    cases=paired.dataset()[0]; names=split(0)["dev"]
    def infer(rgb,key):
        rgb=enhance(rgb)
        if key!="nnunet": a=m0_predict(m,{"rgb":rgb})[1]
        if key!="m0": b=probabilities(n,rgb)["dual"]
        p=(a+b)*.5 if key=="pair" else a if key=="m0" else b
        return (p>=c[key]["thresholds"]["m0matched"]).astype(np.uint8)*255
    for key in paths:
        for _ in range(2): infer(cases[names[0]]["rgb"],key)
    rows=[];rng=random.Random(20261122)
    for i in range(2):
        shuffled=names.copy();rng.shuffle(shuffled)
        for name in shuffled:
            keys=list(paths);rng.shuffle(keys)
            for k in keys:
                d=cases[name];torch.cuda.synchronize();start=time.perf_counter()
                mask=infer(d["rgb"],k)
                torch.cuda.synchronize();ms=1000*(time.perf_counter()-start)
                expected=next(r["candidate"] for r in c[k]["rows"] if r["point"]=="m0matched" and r["image"]==name)
                assert all(expected[j]==v for j,v in score(d,mask>0,.5).items())
                rows.append(dict(image=name,model=k,repeat=i,ms=ms,sha=hashlib.sha256(mask.tobytes()).hexdigest()))
    summary={k:float(np.median([r["ms"] for r in rows if r["model"]==k])) for k in paths}
    (ROOT/"cross_family_cycle/timing.json").write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),median_ms=summary,rows=rows),indent=2))
    print(summary)

if __name__=="__main__": main()
