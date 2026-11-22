"""Fold-local RGB region proposal to route scale fusion. No lesion label at inference."""
import json
import joblib
import numpy as np
import torch
from scipy import ndimage as ndi
from sklearn.ensemble import ExtraTreesClassifier
from appearance_fusion_cycle import ROOT,split,run,predict,probabilities,enhance,paired,load,summarize

OUT=ROOT/"local_scale_gate"
METHODS=("regional","rotated","annotation_diagnostic")

def features(rgb):
    # Local color and dark-blob/line structure on a 4px grid.
    x=rgb.astype(np.float32).reshape(256,4,256,4,3).mean((1,3))
    f=[x[:,:,i] for i in range(3)]
    g=x[:,:,1]
    for sigma in (1.5,4,10):
        for i in (0,1):
            src=x[:,:,i];f.append(src-ndi.gaussian_filter(src,sigma))
        gx=ndi.gaussian_filter(g,sigma,order=(1,0))
        gy=ndi.gaussian_filter(g,sigma,order=(0,1))
        f.append(np.hypot(gx,gy)*sigma)
        xx=ndi.gaussian_filter(g,sigma,order=(2,0));yy=ndi.gaussian_filter(g,sigma,order=(0,2))
        xy=ndi.gaussian_filter(g,sigma,order=(1,1))
        f.extend([(xx+yy)*sigma**2,np.sqrt((xx-yy)**2+4*xy**2)*sigma**2])
    return np.stack(f,axis=-1),x.max(2)>10

def gate(features_,valid,model,cut):
    shape=valid.shape
    p=model.predict_proba(features_.reshape(-1,features_.shape[-1]))[:,1].reshape(shape)
    p=ndi.gaussian_filter(p,1)
    small=(p>=cut)&valid
    return ndi.gaussian_filter(small.astype(np.float32).repeat(4,0).repeat(4,1),3)

def build_gates(cases,fold):
    dest=OUT/f"gate{fold}";dest.mkdir(exist_ok=True,parents=True)
    names=split(fold)["train"]; rng=np.random.default_rng(20261122)
    data={k:features(d["rgb"]) for k,d in cases.items()}
    file=dest/"region.joblib"
    if file.exists():
        model,cut=joblib.load(file)
    else:
        xs=[];ys=[]
        for k in names:
            d=cases[k];feat,valid=data[k]
            target=(d["n"]|d["h"]).reshape(256,4,256,4).mean((1,3))>=.5
            for cls,cap in ((True,1400),(False,3500)):
                coords=np.argwhere(valid & (target==cls))
                if len(coords)>cap:coords=coords[rng.choice(len(coords),cap,replace=False)]
                xs.append(feat[tuple(coords.T)]);ys.append(np.full(len(coords),cls))
        model=ExtraTreesClassifier(n_estimators=48,max_depth=10,min_samples_leaf=15,
                                   class_weight="balanced",n_jobs=2,random_state=20261122)
        model.fit(np.concatenate(xs),np.concatenate(ys))
        vals=[]
        for k in names:
            f,v=data[k]
            pr=model.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(v.shape)
            vals.append(ndi.gaussian_filter(pr,1)[v])
        cut=float(np.quantile(np.concatenate(vals),.90))
        joblib.dump((model,cut),file)
    masks={}; audit=[]
    for k,d in cases.items():
        g=gate(*data[k],model,cut)
        shifted=np.rot90(g,2)
        oracle=ndi.gaussian_filter((d["n"]|d["h"]).astype(np.float32),3)
        masks[id(d["rgb"])]=(g,shifted,oracle)
        if k in split(fold)["dev"]:
            thin=d["s"]&d["n"]
            audit.append(dict(image=k,fov=float(g[d["f"]].mean()),
                near=float(g[d["n"]].mean()) if d["n"].any() else None,
                thin=float(g[thin].mean()) if thin.any() else None,
                rotated_thin=float(shifted[thin].mean()) if thin.any() else None))
    (dest/"audit.json").write_text(json.dumps(dict(train=names,cut=cut,rows=audit)))
    return masks

def mapper(gates):
    def variants(rgb,enhanced,m,n):
        native,mp=predict(m,{"rgb":enhanced})
        nn=probabilities(n,enhanced)["dual"]
        base=(mp+nn)*.5
        delta=(np.clip(2*mp-native,0,1)-mp)*.5
        return base,mp,{k:base+g*delta for k,g in zip(METHODS,gates[id(rgb)])}
    return variants

if __name__=="__main__":
    torch.set_num_threads(2)
    cases=paired.dataset()[0];hard={}
    for name,d in cases.items():
        d["enhanced"]=enhance(d["rgb"]);hard[name]=np.zeros(d["n"].shape,bool)
    for r in load(ROOT/"far_fragments/segments.json"):
        hard[r["image"]][tuple(np.array(r["yx"]).T)]=True
    for fold in range(3):
        produce=mapper(build_gates(cases,fold))
        for seed in (2027,2028,2029):
            if not all((OUT/k/f"fold{fold}_{seed}.json").exists() for k in METHODS):
                run(cases,hard,fold,seed,METHODS,OUT,produce)
    for k in METHODS:summarize(OUT/k)
