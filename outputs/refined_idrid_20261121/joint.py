"""New cohort: real-RGB paired long-budget follow-up for an explicit haemorrhage head."""
import hashlib
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import paired as p

RUN=p.HERE/"joint_720"
STEPS=720


class JointUNet(p.UNet):
    def __init__(self):
        super().__init__()
        self.head=nn.Conv2d(self.widths[0],2,1)

    def components(self,x):
        two=super().forward(x).float()
        vessel,hem=two.split(1,dim=1)
        # Three mutually exclusive classes: ordinary background, vessel, haemorrhage.
        return vessel-F.softplus(hem), hem-F.softplus(vessel)

    def forward(self,x):
        return self.components(x)[0]


def make_model(method):
    return JointUNet() if method=="joint" else p.UNet()


def initialize(method,state):
    model=make_model(method).cuda()
    state=state.copy()
    if method=="joint":
        w=state["head.weight"];b=state["head.bias"]
        state["head.weight"]=torch.cat([w,torch.zeros_like(w)])
        state["head.bias"]=torch.cat([b,torch.full_like(b,-4)])
    model.load_state_dict(state)
    return model


def main():
    torch.set_num_threads(2)
    items,train,dev=p.dataset()
    if hashlib.sha256(p.CHECKPOINT.read_bytes()).hexdigest()!=p.SHA: raise ValueError("M0 identity")
    state=torch.load(p.CHECKPOINT,map_location="cpu",weights_only=True)["model"]
    for seed in (2027,2028,2029):
        for method in ("base","joint"):
            dest=RUN/f"{method}_{seed}"; dest.mkdir(parents=True,exist_ok=True)
            result=dest/"dev.json"
            if result.exists(): continue
            torch.manual_seed(seed)
            rng=np.random.default_rng(seed)
            model=initialize(method,state)
            opt=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.0001)
            start=0
            last=dest/"resume.pt"
            if last.exists():
                ck=torch.load(last,map_location="cpu",weights_only=True)
                model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
                rng.bit_generator.state=ck["rng"];start=ck["step"]
            for step in range(start+1,STEPS+1):
                x,v,f,n,idx,he=p.draw(items,train,rng)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    if method=="joint":
                        z,h=model.components(x)
                        loss=p.loss_fn(z,v,f,n,idx,"local")
                        prob=h[:,0].sigmoid()*f
                        dice=1-(2*(prob*he).sum((1,2))+1)/(prob.sum((1,2))+he.sum((1,2))+1)
                        ce=F.binary_cross_entropy_with_logits(h[:,0],he.float(),reduction="none")
                        loss=loss+.2*((ce*f).sum((1,2))/f.sum((1,2)).clamp_min(1)+dice).mean()
                    else: loss=p.loss_fn(model(x),v,f,n,idx,"base")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                if step%240==0:
                    tmp=last.with_suffix(".tmp")
                    torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),rng=rng.bit_generator.state,step=step),tmp)
                    tmp.replace(last);print(seed,method,step,flush=True)
            torch.save(model.state_dict(),dest/"weights.pt")
            rows=p.evaluate(model,items,dev,dest)
            tmp=result.with_suffix(".tmp");tmp.write_text(json.dumps(rows,indent=2,allow_nan=False));tmp.replace(result)


if __name__=="__main__": main()
