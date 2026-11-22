"""Paired 400-step continuation from the same ordinary 800-step endpoint."""
import hashlib
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from prepare import ROOT, split
from train import trainer
from focus import get_batch
from mine_thin_nested import OUT
import paired

ARMS=("A","B","C")

class NarrowLoss(torch.nn.Module):
    def __init__(self,base,positive_only=False):
        super().__init__()
        self.base=base
        self.positive_only=positive_only

    def forward(self,outs,targets):
        value=self.base(outs,targets)
        z=outs[0][:,1].float()-outs[0][:,0].float()
        terms=[]
        for pred,hard,ring,truth in zip(z,self.hard,self.ring,targets[0][:,0]):
            pos=hard>0
            if not pos.any(): continue
            assert (truth[pos]==1).all()
            positive=F.softplus(-pred[pos]).mean()
            bg=ring&(truth==0)
            if bg.any():
                negative=0 if self.positive_only else F.softplus(pred[bg]).mean()
                terms.append((positive+negative)/2)
            else:
                terms.append(positive)
        return value+.05*torch.stack(terms).mean() if terms else value

def attach(cases,fold):
    train=set(split(fold)["train"])
    for g in range(3):
        if g==fold: continue
        meta=json.loads((OUT/f"miner{g}"/"mining.json").read_text())
        inner=set(meta["train"])
        assert inner < train and len(inner)==18 and set(meta["segments"]).isdisjoint(inner)
        for name in sorted(train-inner):
            d=cases[name]
            coords=[np.asarray(x,dtype=int) for x in meta["segments"][name]]
            d["vcoords"]=np.argwhere(d["v"])
            d["miss_coords"]=coords
            d["hard_ids"]=np.zeros(d["v"].shape,np.int32)
            for i,points in enumerate(coords,1):
                d["hard_ids"][tuple(points.T)]=i
            assert not np.any((d["hard_ids"]>0)&~(d["s"]&d["n"]))

def fit(cases,fold,seed,arms=ARMS):
    dest=OUT/f"fold{fold}"/str(seed)
    if all((dest/m/"train.json").exists() for m in arms): return
    start=ROOT/"models_focus"/f"fold{fold}"/str(seed)/"continue/weights.pt"
    nets={m:trainer(fold,seed) for m in arms}
    with start.open("rb") as f: sha=hashlib.file_digest(f,"sha256").hexdigest()
    state=torch.load(start,map_location="cpu",weights_only=True)
    for t in nets.values():
        t.network.load_state_dict(state); t.network.train()
    del state
    for m in ("C","D"):
        if m in nets: nets[m].loss=NarrowLoss(nets[m].loss,positive_only=m=="D")
    first=next(iter(nets.values()))
    shape=first.configuration_manager.patch_size
    scales=first._get_deep_supervision_scales()
    names=split(fold)["train"]
    traces=[hashlib.sha256(),hashlib.sha256()]
    losses={m:[] for m in arms}; skips={m:0 for m in arms}; targeted=0
    for step in range(400):
        batch,active=get_batch(cases,names,seed+3031,step,shape)
        targeted+=active
        for i,(x,y,near,hard) in enumerate(batch):
            traces[i].update(x[:,:,::32,::32].numpy().tobytes())
            targets=[y if np.allclose(sc,1) else F.interpolate(y.float(),size=tuple(int(round(dim*f))
                         for dim,f in zip(shape,sc)),mode="nearest-exact").to(y.dtype) for sc in scales]
            for m in (("A",) if i==0 else ("B","C","D")):
                if m not in nets: continue
                t=nets[m]
                if m in ("C","D"): t.loss.hard=hard.cuda(); t.loss.ring=near.cuda()
                for group in t.optimizer.param_groups: group["lr"]=.0015*(1-(step//40)/10)**.9
                before=t.grad_scaler.get_scale()
                loss=float(t.train_step(dict(data=x,target=targets))["loss"])
                if not np.isfinite(loss): raise ValueError("nonfinite loss")
                skips[m]+=int(t.grad_scaler.get_scale()<before)
                losses[m].append(loss); t.optimizer.zero_grad(set_to_none=True)
        if (step+1)%100==0: print("training",fold,seed,step+1,flush=True)
    for m,t in nets.items():
        path=dest/m;path.mkdir(parents=True,exist_ok=True)
        torch.save(t.network.state_dict(),path/"weights.tmp");(path/"weights.tmp").replace(path/"weights.pt")
        meta=dict(method=m,steps=400,parent_sha=sha,trace=traces[int(m!="A")].hexdigest(),loss=losses[m],
                  targeted=targeted if m!="A" else 0,amp_skips=skips[m])
        (path/"train.tmp").write_text(json.dumps(meta));(path/"train.tmp").replace(path/"train.json")

if __name__=="__main__":
    from eval_thin_abc import evaluate
    torch.set_num_threads(2)
    cases=paired.dataset()[0]
    for seed in (2027,2028,2029):
        for fold in range(3):
            attach(cases,fold)
            fit(cases,fold,seed)
            evaluate(cases,fold,seed)
