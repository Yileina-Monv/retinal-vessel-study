"""Three disjoint 18-image miners. Each outer training image is mined by the other inner half."""
import hashlib
import json
import numpy as np
import torch
from scipy import ndimage as ndi
from prepare import ROOT, split
import paired
from two_scale import predict
from matched_fp import GRID

OUT=ROOT/"thin_training"

def run():
    torch.set_num_threads(2)
    cases=paired.dataset()[0]
    init=paired.CHECKPOINT
    assert hashlib.sha256(init.read_bytes()).hexdigest()==paired.SHA
    halves=[split(i)["dev"] for i in range(3)]
    assert len(set.union(*map(set,halves)))==54
    for group,names in enumerate(halves):
        dest=OUT/f"miner{group}"
        dest.mkdir(parents=True,exist_ok=True)
        if (dest/"mining.json").is_file(): continue
        torch.manual_seed(2027); rng=np.random.default_rng(2027)
        model=paired.UNet().cuda()
        model.load_state_dict(torch.load(init,map_location="cpu",weights_only=True)["model"])
        opt=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.0001)
        for step in range(240):
            x,v,f,n,ids,_=paired.draw(cases,names,rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss=paired.loss_fn(model(x),v,f,n,ids,"base")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
        model.eval()
        curves=[]; targets=[]
        for name in names:
            d=cases[name]; area=d["n"].sum()
            if not area: continue
            native,dual=predict(model,d)
            bg=d["n"]&~d["v"]
            targets.append(((native>=.5)&bg).sum()/area)
            vals=np.sort(dual[bg]); curves.append((vals.size-np.searchsorted(vals,GRID))/area)
        threshold=float(GRID[np.flatnonzero(np.mean(curves,axis=0)<=np.mean(targets))[0]])
        mined={}
        for name in sorted(set(cases)-set(names)):
            d=cases[name]
            _,pr=predict(model,d)
            mask=(pr>=threshold)&d["f"]
            hard=d["n"]&d["s"]&(ndi.distance_transform_edt(~mask)>5)
            segments=[]
            for parent in np.unique(d["i"][hard]):
                lab,_=ndi.label(hard&(d["i"]==parent),np.ones((3,3)))
                for i,sl in enumerate(ndi.find_objects(lab),1):
                    segments.append((np.argwhere(lab[sl]==i)+[s.start for s in sl]).tolist())
            assert sum(map(len,segments))==int(hard.sum())
            mined[name]=segments
        torch.save(model.state_dict(),dest/"weights.pt")
        (dest/"mining.json").write_text(json.dumps(dict(train=names,threshold=threshold,segments=mined)))
        print("inner miner",group,"done",flush=True)

if __name__=="__main__":
    run()
