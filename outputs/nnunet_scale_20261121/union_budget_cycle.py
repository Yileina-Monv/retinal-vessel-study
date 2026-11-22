"""Compare preserving both masks with simply lowering the reference threshold at like FP."""
import json
import numpy as np
import torch
from scipy import ndimage as ndi
from predict_view_fusion import models
from appearance_fusion_cycle import ROOT,split,paired,predict,probabilities,enhance,load,GRID,score,summarize

OUT=ROOT/"switch_supervision/union_budget"

def probabilities_pair(m,n,img):
    native,mp=predict(m,{"rgb":img})
    nn=probabilities(n,img)["dual"]
    # Match the archived scale ablation's precise FP32 arithmetic.
    return (mp+nn)*.5,(np.clip(2*mp-native,0,1)+nn)*.5

def calibrate(cases,train,m,n,tb,tz):
    curves=[];fps={k:[] for k in tb}
    for name in train:
        d=cases[name];near=d["n"]
        if not near.any():continue
        b,z=probabilities_pair(m,n,d["enhanced"])
        bg=near&~d["v"];vb,vz=b[bg],z[bg];vals=np.sort(vb)
        curves.append((len(vals)-np.searchsorted(vals,GRID))/near.sum())
        for k in tb:fps[k].append(float(((vb>=tb[k])|(vz>=tz[k])).sum()/near.sum()))
    avg=np.mean(curves,axis=0);targets={k:float(np.mean(v)) for k,v in fps.items()}
    cuts={k:float(GRID[np.flatnonzero(avg<=v)[0]]) for k,v in targets.items()}
    assert all(cuts[k]<=tb[k]+1e-8 for k in cuts)
    return cuts,targets

def record(base,cand,d,hard,name,key):
    t=d["s"]&d["n"];outside=d["s"]&d["f"]&~(d["n"]|d["h"])
    old,new=[ndi.binary_dilation(x,structure=paired.cross) for x in (base,cand)]
    data=[]
    for mask,h in ((base,old),(cand,new)):
        sc=score(d,mask,.5);sc["thin_d0"]=float(mask[t].mean()) if t.any() else None
        sc["far_thin_d1"]=float(h[outside].mean()) if t.any() and outside.any() else None
        data.append(sc)
    return dict(image=name,point=key,base=data[0],candidate=data[1],
                gains=int((t&new&~old).sum()),losses=int((t&old&~new).sum()),hard_base=int((hard&old).sum()),hard_cand=int((hard&new).sum()))

def run(cases,hard,fold,seed):
    filename=f"fold{fold}_{seed}.json";arms=("union_vs_previous","lower_vs_previous","union_vs_lower")
    if all((OUT/a/filename).exists() for a in arms):return
    m,n=models(fold,seed)
    archive=load(ROOT/"cross_family_cycle/m0"/filename)
    zoom=load(ROOT/"family_scale_ablation/large_dual"/filename)
    tb=archive["thresholds"];tz=zoom["thresholds"]
    cuts,targets=calibrate(cases,split(fold)["train"],m,n,tb,tz);rows={a:[] for a in arms}
    for name in split(fold)["dev"]:
        d=cases[name];b,z=probabilities_pair(m,n,d["enhanced"])
        for point in tb:
            ref=(b>=tb[point])&d["f"];other=(z>=tz[point])&d["f"]
            for mask,store in ((ref,archive),(other,zoom)):
                prior=next(r["candidate"] for r in store["rows"] if r["image"]==name and r["point"]==point)
                assert all(prior[k]==v for k,v in score(d,mask,.5).items())
            add=ref|other;lower=(b>=cuts[point])&d["f"]
            for a,base,cand in zip(arms,(ref,ref,lower),(add,lower,add)):
                r=record(base,cand,d,hard[name],name,point)
                if a!="union_vs_lower":assert r["losses"]==0
                rows[a].append(r)
    for a in arms:
        folder=OUT/a;folder.mkdir(parents=True,exist_ok=True)
        tmp=folder/(filename+".tmp")
        tmp.write_text(json.dumps(dict(fold=fold,seed=seed,base_cutoffs=tb,zoom_cutoffs=tz,lower_cutoffs=cuts,
                                      training_union_fp_density=targets,rows=rows[a])))
        tmp.replace(folder/filename)
    print("completed union budget",fold,seed,flush=True)

if __name__=="__main__":
    torch.set_num_threads(2);cases=paired.dataset()[0]
    hard={}
    for k,d in cases.items():
        d["enhanced"]=enhance(d["rgb"]);hard[k]=np.zeros_like(d["n"])
    for r in load(ROOT/"far_fragments/segments.json"):hard[r["image"]][tuple(np.array(r["yx"]).T)]=True
    for fold in range(3):
        for seed in (2027,2028,2029):run(cases,hard,fold,seed)
    for a in ("union_vs_previous","lower_vs_previous","union_vs_lower"):summarize(OUT/a)
