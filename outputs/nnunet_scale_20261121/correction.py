"""Compare identical extra training batches with and without teacher-selected GT correction."""
import argparse
import hashlib
import json
import time
from prepare import ROOT,OLD
import numpy as np
import torch
from torch.nn import functional as F
import paired
from two_scale import predict as m0_predict
from train import trainer
from evaluate import evaluate,make_predictor,probabilities
from focus import get_batch


def select_positions(cases,names,fold,seed):
    old=OLD if fold==0 else OLD/f"fold_{fold}"
    m0=paired.UNet().cuda().eval()
    m0_file=old/"rgb_only"/f"local_{seed}"/"weights.pt"
    m0.load_state_dict(torch.load(m0_file,map_location="cpu",weights_only=True))
    m0_cal=json.loads((old/"two_scale"/f"local_{seed}"/"threshold.json").read_text())["threshold"]
    prior=ROOT/"eval_focus"/f"fold{fold}"/str(seed)/"continue"/"calibration.json"
    student_cal=json.loads(prior.read_text())["m0_thresholds"]["native"]
    student=make_predictor(fold,seed,"continue")
    rows=[]
    for name in names:
        d=cases[name]
        d["vcoords"]=np.argwhere(d["v"])
        _,m0_prob=m0_predict(m0,d)
        nn_prob=probabilities(student,d["rgb"],native_only=True)["native"]
        pos=d["n"]&d["v"]&(m0_prob>=m0_cal)&(nn_prob<student_cal)
        neg=d["n"]&~d["v"]&(m0_prob<m0_cal)&(nn_prob>=student_cal)
        d["hard_ids"]=pos.astype(np.int32)+2*neg.astype(np.int32)
        d["miss_coords"]=[np.argwhere(x) for x in (pos,neg) if x.any()]
        rows.append(dict(image=name,positive=int(pos.sum()),negative=int(neg.sum())))
    with m0_file.open("rb") as f: sha=hashlib.file_digest(f,"sha256").hexdigest()
    return dict(m0_sha256=sha,m0_threshold=m0_cal,student_threshold=student_cal,train=rows)


class CorrectionLoss(torch.nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base=base

    def forward(self,outputs,targets):
        loss=self.base(outputs,targets)
        logits=outputs[0][:,1].float()-outputs[0][:,0].float()
        terms=[]
        for z,indices in zip(logits,self.indices):
            groups=[]
            if (indices==1).any(): groups.append(F.softplus(-z[indices==1]).mean())
            if (indices==2).any(): groups.append(F.softplus(z[indices==2]).mean())
            if groups: terms.append(torch.stack(groups).mean())
        return loss+.05*torch.stack(terms).mean() if terms else loss


def train(fold,seed):
    dest=ROOT/"models_teacher"/f"fold{fold}"/str(seed)
    methods=("extra","correction")
    if all((dest/m/"train.json").is_file() for m in methods): return
    dest.mkdir(parents=True,exist_ok=True)
    cases,names,_=paired.dataset(fold)
    info=select_positions(cases,names,fold,seed)
    (dest/"selection.json").write_text(json.dumps(info))
    source=ROOT/"models_focus"/f"fold{fold}"/str(seed)/"continue"/"weights.pt"
    arms={m:trainer(fold,seed) for m in methods}
    state=torch.load(source,map_location="cpu",weights_only=True)
    for t in arms.values(): t.network.load_state_dict(state);t.network.train()
    del state
    arms["correction"].loss=CorrectionLoss(arms["correction"].loss)
    shape=arms["extra"].configuration_manager.patch_size
    scales=arms["extra"]._get_deep_supervision_scales()
    losses={m:[] for m in methods}; skips={m:0 for m in methods}
    started=time.monotonic()
    targeted=0
    trace=hashlib.sha256()
    for step in range(400):
        batch,n=get_batch(cases,names,seed+3031,step,shape)
        targeted+=n
        x,y,_,ids=batch[1]
        trace.update(x[:,:,::32,::32].numpy().tobytes())
        targets=[y if np.allclose(sc,1) else F.interpolate(y.float(),size=tuple(
                 int(round(dim*f)) for dim,f in zip(shape,sc)),mode="nearest-exact").to(y.dtype) for sc in scales]
        arms["correction"].loss.indices=ids.cuda()
        for m,t in arms.items():
            for group in t.optimizer.param_groups: group["lr"]=.0015*(1-(step//40)/10)**.9
            before=t.grad_scaler.get_scale()
            loss=float(t.train_step(dict(data=x,target=targets))["loss"])
            skips[m]+=int(t.grad_scaler.get_scale()<before)
            if not np.isfinite(loss): raise ValueError("Nonfinite correction loss")
            t.optimizer.zero_grad(set_to_none=True)
            losses[m].append(loss)
        if (step+1)%100==0: print(fold,seed,step+1,"targeted",targeted,round(time.monotonic()-started),flush=True)
    with source.open("rb") as f: parent=hashlib.file_digest(f,"sha256").hexdigest()
    for m,t in arms.items():
        folder=dest/m;folder.mkdir(exist_ok=True)
        torch.save(t.network.state_dict(),folder/"weights.tmp")
        (folder/"weights.tmp").replace(folder/"weights.pt")
        meta=dict(method=m,fold=fold,seed=seed,steps=400,amp_skips=skips[m],focused_crops=targeted,
                  shared_source_sha=parent,batch_trace=trace.hexdigest(),loss=losses[m])
        (folder/"train.tmp").write_text(json.dumps(meta));(folder/"train.tmp").replace(folder/"train.json")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--fold",type=int,required=True)
    ap.add_argument("--seed",type=int,required=True)
    args=ap.parse_args()
    torch.set_num_threads(2)
    train(args.fold,args.seed)
    evaluate(args.fold,args.seed,teacher=True)
