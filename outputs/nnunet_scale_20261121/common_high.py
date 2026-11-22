"""Compare the existing M0 specialist at the higher nnU-Net FP target, on identical holdouts."""
import json
from prepare import ROOT,OLD,split
import numpy as np
import torch
import paired
from two_scale import predict
from evaluate import describe
from matched_fp import GRID
from analyze import SEEDS,METRICS
from compare import aggregate


def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    cases,_,_=paired.dataset()
    out=ROOT/"common_high"
    out.mkdir(exist_ok=True)
    pairs={}
    for fold in range(3):
        stage=OLD if fold==0 else OLD/f"fold_{fold}"
        tr,dev=[split(fold)[k] for k in ("train","dev")]
        for seed in SEEDS:
            here=out/f"fold{fold}_{seed}.json"
            if not here.exists():
                model=paired.UNet().cuda().eval()
                model.load_state_dict(torch.load(stage/"rgb_only"/f"local_{seed}"/"weights.pt",map_location="cpu",weights_only=True))
                q=ROOT/"eval_paired"/f"fold{fold}"/str(seed)/"standard/calibration.json"
                target=json.loads(q.read_text())["target"]
                curves=[]
                for name in tr:
                    d=cases[name]
                    if not d["n"].any(): continue
                    _,pr=predict(model,d)
                    bg=np.sort(pr[d["n"]&~d["v"]])
                    curves.append((len(bg)-np.searchsorted(bg,GRID))/d["n"].sum())
                curve=np.mean(curves,axis=0)
                idx=np.flatnonzero(curve<=target)[0]
                t=float(GRID[idx])
                rows=[]
                for name in dev:
                    d=cases[name]
                    _,pr=predict(model,d)
                    rows.append(dict(image=name,**describe(d,pr,t)))
                tmp=here.with_suffix(".tmp")
                tmp.write_text(json.dumps(dict(threshold=t,target=target,achieved=curve[idx],train=tr,dev=rows)))
                tmp.replace(here)
            baseline=json.loads(here.read_text())["dev"]
            nn=json.loads((ROOT/"eval_teacher"/f"fold{fold}"/str(seed)/"extra/dev.json").read_text())
            candidate=[r for r in nn if r["point"]=="matched" and r["view"]=="dual"]
            assert [r["image"] for r in baseline]==dev==[r["image"] for r in candidate]
            for a,b in zip(baseline,candidate): pairs[a["image"],seed]=(a,b)
            print("Common higher FP",fold,seed,flush=True)
    measures={k:aggregate(pairs,k,SEEDS) for k in METRICS}
    (out/"comparison.json").write_text(json.dumps(measures,indent=2))


if __name__=="__main__": main()
