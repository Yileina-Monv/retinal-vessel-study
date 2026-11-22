"""Inner specialists trained on one 18-image group to create unseen switch labels."""
import json
import random
import numpy as np
import torch
from torch.nn import functional as F
from prepare import ROOT, split, stage
from train import trainer, loader
from focus import get_batch
from correction import CorrectionLoss
from appearance_fusion_cycle import paired, predict, probabilities, GRID, load
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

OUT=ROOT/"switch_supervision"
SEED=2027

def cutoff(probs,cases,names,target):
    curves=[]
    for p,k in zip(probs,names):
        d=cases[k]
        if d["n"].any():
            bg=np.sort(p[d["n"]&~d["v"]])
            curves.append((len(bg)-np.searchsorted(bg,GRID))/d["n"].sum())
    return float(GRID[np.flatnonzero(np.mean(curves,axis=0)<=target)[0]])

def predictor(t):
    t.set_deep_supervision_enabled(False)
    p=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,device=torch.device("cuda"),allow_tqdm=False)
    p.manual_initialization(t.network.eval(),t.plans_manager,t.configuration_manager,None,t.dataset_json,"nnUNetTrainer",None)
    return p

def persist(net,path):
    tmp=path.with_suffix(".tmp")
    torch.save(net.state_dict(),tmp)
    tmp.replace(path)

def m0_model(cases,names,path):
    torch.manual_seed(SEED)
    m=paired.UNet().cuda()
    if path.exists():
        m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        return m.eval()
    m.load_state_dict(torch.load(paired.CHECKPOINT,map_location="cpu",weights_only=True)["model"])
    opt=torch.optim.AdamW(m.parameters(),lr=.00015,weight_decay=.0001)
    rng=np.random.default_rng(SEED)
    for _ in range(240):
        x,v,f,n,ids,_=paired.draw(cases,names,rng)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss=paired.loss_fn(m(x),v,f,n,ids,"local")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        opt.step()
    persist(m,path)
    return m.eval()

def fit_group(cases,group):
    folder=OUT/f"inner{group}";folder.mkdir(parents=True,exist_ok=True)
    if (folder/"training.json").is_file(): return
    names=split(group)["dev"]
    plan=(group+1)%3
    assert set(names)<=set(split(plan)["train"]) and len(names)==18
    m=m0_model(cases,names,folder/"m0.pt")
    ordinary=paired.UNet().cuda().eval()
    miner=ROOT/"thin_training"/f"miner{group}"
    assert set(load(miner/"mining.json")["train"])==set(names)
    ordinary.load_state_dict(torch.load(miner/"weights.pt",map_location="cpu",weights_only=True))
    def density(k):
        d=cases[k];pr=predict(ordinary,d)[0]
        return ((pr>=.5)&d["n"]&~d["v"]).sum()/d["n"].sum()
    target=float(np.mean([density(k) for k in names if cases[k]["n"].any()]))
    del ordinary
    t=trainer(plan,SEED)
    dl=loader(t,plan,names)
    losses=[]
    for step in range(500):
        key=SEED*10000+step
        np.random.seed(key);torch.manual_seed(key);random.seed(key)
        if step%50==0:
            t.current_epoch=step//50;t.lr_scheduler.step(t.current_epoch)
        val=float(t.train_step(next(dl))["loss"])
        assert np.isfinite(val)
        losses.append(val)
    print("inner",group,"standard ready",flush=True)
    for k in names:
        d=cases[k];d["vcoords"]=np.argwhere(d["v"]);d["hard_ids"]=np.zeros_like(d["i"]);d["miss_coords"]=[]
    shape=t.configuration_manager.patch_size
    for mode,count,offset,rate in (("ordinary",300,0,.002),("correction",400,3031,.0015)):
        # The archived outer continuation starts each stage with a fresh optimizer.
        t.optimizer,t.lr_scheduler=t.configure_optimizers()
        if mode=="correction":
            p=predictor(t)
            m_maps=[predict(m,cases[k])[1] for k in names]
            s_maps=[probabilities(p,cases[k]["rgb"],native_only=True)["native"] for k in names]
            tm=cutoff(m_maps,cases,names,target);ts=cutoff(s_maps,cases,names,target)
            for k,a,b in zip(names,m_maps,s_maps):
                d=cases[k]
                pos=d["n"]&d["v"]&(a>=tm)&(b<ts)
                neg=d["n"]&~d["v"]&(a<tm)&(b>=ts)
                d["hard_ids"]=pos.astype(np.int32)+2*neg.astype(np.int32)
                d["miss_coords"]=[np.argwhere(q) for q in (pos,neg) if q.any()]
            del p,m_maps,s_maps
            t.loss=CorrectionLoss(t.loss)
        t.set_deep_supervision_enabled(True);t.network.train()
        for step in range(count):
            batch,_=get_batch(cases,names,SEED+offset,step,shape)
            x,y,_,ids=batch[int(mode=="correction")]
            targets=[y if np.allclose(sc,1) else F.interpolate(y.float(),size=tuple(int(round(dim*f)) for dim,f in zip(shape,sc)),mode="nearest-exact").to(y.dtype) for sc in t._get_deep_supervision_scales()]
            for gp in t.optimizer.param_groups:gp["lr"]=rate*(1-(step//(count//10))/10)**.9
            if mode=="correction":t.loss.indices=ids.cuda()
            val=float(t.train_step(dict(data=x,target=targets))["loss"])
            assert np.isfinite(val)
            losses.append(val)
        print("inner",group,mode,"ready",flush=True)
    persist(t.network,folder/"nn.pt")
    (folder/"training.json").write_text(json.dumps(dict(group=group,train=names,seed=SEED,steps=[240,500,300,400],
            m0_fp_target=target,selection_thresholds=[tm,ts],last_loss=losses[-1])))

if __name__=="__main__":
    torch.set_num_threads(2)
    configs=[load(ROOT/"preprocessed"/stage(i)/"nnUNetPlans.json")["configurations"]["2d"] for i in range(3)]
    assert configs[0]==configs[1]==configs[2]
    cases=paired.dataset()[0]
    for group in range(3): fit_group(cases,group)
