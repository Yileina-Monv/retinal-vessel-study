"""Time only the forwards required by each recipe; verify thresholded mask scores."""
import argparse
import hashlib
import json
import random
import time
import numpy as np
import torch
from predict_view_fusion import models,probability,catalog,region_parameters
from compare_addback_thresholds import load
from prepare import ROOT,split
from two_scale import score
import paired

def main(modes):
    torch.set_num_threads(2);fold=0;seed=2027
    m,n=models(fold,seed)
    cases=paired.dataset()[0]; names=split(fold)["dev"]
    ar={k:load(catalog(k)/"fold0_2027.json") for k in modes}
    region=region_parameters(fold) if "regional" in modes else None
    rng=random.Random(20261122)
    def infer(rgb,k):
        return (probability(m,n,rgb,k,region)>=ar[k]["thresholds"]["m0matched"]).astype(np.uint8)*255
    for k in modes:
        for _ in range(2): infer(cases[names[0]]["rgb"],k)
    rows=[]
    for rep in range(2):
        order=names.copy();rng.shuffle(order)
        for name in order:
            modes1=modes.copy();rng.shuffle(modes1)
            d=cases[name]
            for k in modes1:
                torch.cuda.synchronize();start=time.perf_counter()
                mask=infer(d["rgb"],k)
                torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
                expected=next(r["candidate"] for r in ar[k]["rows"] if r["point"]=="m0matched" and r["image"]==name)
                vals=score(d,mask>0,.5)
                delta={j:v-expected[j] for j,v in vals.items() if v is not None and v!=expected[j]}
                rows.append(dict(image=name,mode=k,repeat=rep,ms=ms,metric_delta=delta,sha=hashlib.sha256(mask.tobytes()).hexdigest()))
    med={k:float(np.median([r["ms"] for r in rows if r["mode"]==k])) for k in modes}
    dif=[r for r in rows if r["metric_delta"]]
    (ROOT/"appearance_fusion_cycle/timing.json").write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),median_ms=med,
                        comparisons_with_difference=len(dif),rows=rows),indent=2))
    print(med,"metric mismatches",len(dif))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",nargs="+")
    main(parser.parse_args().mode)
