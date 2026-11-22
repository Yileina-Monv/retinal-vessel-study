"""Cross-fitted choice between the frozen four-view map and the M0 zoom-only alternative."""
import json
import joblib
import numpy as np
import torch
from scipy import ndimage as ndi
from sklearn.ensemble import ExtraTreesClassifier
from appearance_fusion_cycle import ROOT,split,paired,predict,probabilities,enhance,load,run,summarize
from train_nested_experts import OUT,SEED,cutoff,predictor
from train import trainer

METHODS=("learned_signed","learned_add","all_add","shuffled_signed","shuffled_add","zoom_control")
CAP=160

def views(m,n,img):
    native,mp=predict(m,{"rgb":img})
    zoom=np.clip(2*mp-native,0,1)
    nn=probabilities(n,img)["dual"]
    base=(mp+nn)*.5
    delta=(zoom-mp)*.5
    return native,zoom,nn,base,delta,mp

def eligible(base,delta):
    alt=base+delta
    return (np.abs(delta)>.01)&(np.maximum(alt,base)>.08)&(np.minimum(alt,base)<.85)

def feature_rows(rgb,maps,t,ij):
    a,z,n,b,d,_=maps
    g=rgb[:,:,1].astype(np.float32)/255
    channel=[a,z,n,b-t,d,np.minimum(a,z)-n]
    for arr in (d,b):
        for width in (3,9):channel.append(ndi.uniform_filter(arr,width))
    channel.extend((g,rgb[:,:,0],g-ndi.gaussian_filter(g,4),ndi.maximum_filter(b,3)-b))
    return np.column_stack([v[ij] for v in channel]).astype(np.float32)

def training_events(case,maps,t,rng):
    b,delta=maps[3:5];alt=b+delta
    before=(b>=t)&case["f"];after=(alt>=t)&case["f"]
    near=case["n"];thin=near&case["s"];v=case["v"]
    dilate=lambda x:ndi.binary_dilation(x,structure=paired.cross)
    gain=thin&dilate(after)&~dilate(before)
    loss=thin&dilate(before)&~dilate(after)
    new=after&~before;removed=before&~after
    away=near&~v&~dilate(thin)
    # Equal image weight, then fixed weights by meaning. The first two groups carry
    # most weight. Background and wider-vessel groups discourage overpainting and erosion.
    groups=((dilate(gain)&new,1,1.),(dilate(loss)&removed,0,1.),
            (new&away,0,.5),(removed&away,1,.5),
            (removed&v&~dilate(thin),0,.25),(new&v&~dilate(thin),1,.25))
    coords=[];labels=[];weights=[];kinds=[];count=[]
    ok=eligible(b,delta)
    for idx,(mask,label,priority) in enumerate(groups):
        ij=np.argwhere(mask&ok);count.append(len(ij))
        if not len(ij):continue
        if len(ij)>CAP:ij=ij[rng.choice(len(ij),CAP,replace=False)]
        coords.extend(ij);labels.extend([label]*len(ij));kinds.extend([idx]*len(ij))
        weights.extend([priority/len(ij)]*len(ij))
    ij=np.array(coords,dtype=int).reshape(-1,2)
    x=feature_rows(case["rgb"],maps,t,tuple(ij.T))
    w=np.array(weights);w=w/w.sum() if len(w) else w
    return dict(x=x,y=np.array(labels,np.uint8),w=w,kind=np.array(kinds,np.uint8),
                counts=np.array(count),thin_gained=gain.sum(),thin_lost=loss.sum())

def inner_examples(cases,group):
    folder=OUT/f"inner{group}"
    info=load(folder/"training.json");seen=info["train"]
    others=sorted(set(cases)-set(seen))
    dest=folder/"examples";dest.mkdir(exist_ok=True)
    if all((dest/(k+".npz")).exists() for k in others):return
    m=paired.UNet().cuda().eval()
    m.load_state_dict(torch.load(folder/"m0.pt",map_location="cpu",weights_only=True))
    t=trainer((group+1)%3,SEED)
    t.network.load_state_dict(torch.load(folder/"nn.pt",map_location="cpu",weights_only=True))
    n=predictor(t);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    base=[views(m,n,cases[k]["enhanced"])[3] for k in seen]
    threshold=cutoff(base,cases,seen,info["m0_fp_target"]);del base
    for idx,k in enumerate(others):
        assert k not in seen
        out=dest/(k+".npz")
        if out.exists():continue
        data=training_events(cases[k],views(m,n,cases[k]["enhanced"]),threshold,np.random.default_rng(SEED+idx))
        tmp=out.with_suffix(".tmp")
        with tmp.open("wb") as f:np.savez_compressed(f,**data)
        tmp.replace(out)
    (folder/"events.json").write_text(json.dumps(dict(train=seen,scored=others,threshold=threshold)))
    print("inner unseen example groups ready",group,flush=True)

def fit_selector(fold):
    fit=split(fold)["train"];held=set(split(fold)["dev"])
    groups=[i for i in range(3) if i!=fold]
    arrays=[];files=[]
    for a,b in (groups,groups[::-1]):
        folder=OUT/f"inner{a}";source=set(load(folder/"training.json")["train"]);target=split(b)["dev"]
        assert not(source & held or set(target)&held or source&set(target))
        assert source|set(target)==set(fit)
        for k in target:
            with np.load(folder/"examples"/(k+".npz")) as z:arrays.append({i:z[i] for i in z})
            files.append(dict(image=k,expert_group=a))
    assert len({x["image"] for x in files})==len(files)==36
    x,y,w=[np.concatenate([z[key] for z in arrays]) for key in ("x","y","w")]
    rng=np.random.default_rng(SEED); labels=y.copy()
    # Shuffle inside each image and direction; move the weights together with the labels.
    control_weights=w.copy()
    start=0
    for z in arrays:
        end=start+len(z["y"])
        for pos in (True,False):
            inds=np.flatnonzero((x[start:end,4]>0)==pos)+start
            perm=rng.permutation(inds);labels[inds]=y[perm];control_weights[inds]=w[perm]
        start=end
    models=[ExtraTreesClassifier(n_estimators=48,min_samples_leaf=8,max_depth=7,random_state=SEED,n_jobs=2).fit(x,target,sample_weight=weights)
            for target,weights in ((y,w),(labels,control_weights))]
    dest=OUT/f"selector{fold}";dest.mkdir(exist_ok=True)
    joblib.dump(models,dest/"models.joblib")
    info=dict(examples=files,scored=sorted(held),event_candidates=np.sum([z["counts"] for z in arrays],axis=0).tolist(),
              points_gained=int(sum(z["thin_gained"] for z in arrays)),points_lost=int(sum(z["thin_lost"] for z in arrays)),fitted_examples=len(y))
    (dest/"training.json").write_text(json.dumps(info,indent=2))
    return models

def fuse(rgb,maps,t,models,methods=METHODS):
    _,_,_,base,delta,mp=maps
    out={}
    if "all_add" in methods:out["all_add"]=base+np.maximum(delta,0)
    if "zoom_control" in methods:out["zoom_control"]=base+delta
    if len(out)==len(methods):return base,mp,out
    ij=np.nonzero(eligible(base,delta))
    x=feature_rows(rgb,maps,t,ij)
    for prefix,model in zip(("learned","shuffled"),models):
        signed=prefix+"_signed";add=prefix+"_add"
        if signed not in methods and add not in methods:continue
        g=np.zeros_like(base)
        if len(x):g[ij]=model.predict_proba(x)[:,1]
        change=g*delta
        if signed in methods:out[signed]=base+change
        if add in methods:out[add]=base+np.maximum(change,0)
    return base,mp,out

if __name__=="__main__":
    torch.set_num_threads(2)
    cases=paired.dataset()[0];hard={}
    for k,d in cases.items():
        d["enhanced"]=enhance(d["rgb"]);hard[k]=np.zeros_like(d["n"])
    for r in load(ROOT/"far_fragments/segments.json"):hard[r["image"]][tuple(np.array(r["yx"]).T)]=True
    for group in range(3):inner_examples(cases,group)
    for fold in range(3):
        models=fit_selector(fold)
        for seed in (2027,2028,2029):
            if all((OUT/k/f"fold{fold}_{seed}.json").exists() for k in METHODS):continue
            t=load(ROOT/"cross_family_cycle/m0"/f"fold{fold}_{seed}.json")["thresholds"]["m0matched"]
            def mapper(rgb,img,m,n):return fuse(rgb,views(m,n,img),t,models)
            run(cases,hard,fold,seed,METHODS,OUT,mapper,mask_archive=OUT/"heldout_masks")
    for method in METHODS:summarize(OUT/method)
