"""Interleaved standalone replay and latency for base, additive rules and RGB routing."""
import json
import random
import time
import numpy as np
import torch
from prepare import ROOT,split
from predict_switch import OUT,load,models,settings,predict
from predict_view_fusion import probability,region_parameters
from two_scale import score
import paired

if __name__=="__main__":
    torch.set_num_threads(2);m,n=models(0,2027)
    cases=paired.dataset()[0];names=split(0)["dev"]
    modes=["base","regional","learned_add","all_add"]
    config={k:settings(0,2027,k,"m0matched") for k in modes[2:]}
    regional=load(ROOT/"local_scale_gate/regional/fold0_2027.json")
    reg_model=region_parameters(0)
    base_cut=load(ROOT/"cross_family_cycle/m0/fold0_2027.json")["thresholds"]["m0matched"]
    def infer(rgb,mode):
        if mode=="regional":return probability(m,n,rgb,mode,reg_model)>=regional["thresholds"]["m0matched"]
        if mode=="base":return probability(m,n,rgb,"full")>=base_cut
        return predict(m,n,rgb,mode,config[mode])>0
    for mode in modes:
        for _ in range(2):infer(cases[names[0]]["rgb"],mode)
    rows=[];rng=random.Random(20261122)
    for repeat in range(2):
        order=names.copy();rng.shuffle(order)
        for k in order:
            d=cases[k];mix=modes.copy();rng.shuffle(mix)
            with np.load(OUT/"heldout_masks/fold0_2027"/(k+".npz")) as z:
                for mode in mix:
                    torch.cuda.synchronize();start=time.perf_counter()
                    mask=infer(d["rgb"],mode)
                    torch.cuda.synchronize();ms=1000*(time.perf_counter()-start)
                    if mode=="regional":
                        expected=next(r["candidate"] for r in regional["rows"] if r["image"]==k and r["point"]=="m0matched")
                        assert all(expected[a]==b for a,b in score(d,mask,.5).items())
                    else:assert np.array_equal(np.packbits(mask&d["f"]),z["m0matched_"+mode])
                    rows.append(dict(image=k,mode=mode,repeat=repeat,ms=ms))
    med={mode:float(np.median([x["ms"] for x in rows if x["mode"]==mode])) for mode in modes}
    (OUT/"timing.json").write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),median_ms=med,rows=rows),indent=2))
    print("verified",len(rows),med)
