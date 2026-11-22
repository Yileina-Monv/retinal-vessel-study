"""Trace the pre-defined 897-point cohort using frozen held-out predictions."""
import base64
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi
import torch
from prepare import ROOT,split
from compare_addback_thresholds import make_model,load
from evaluate import probabilities
import paired

out=ROOT/"far_fragments"
out.mkdir(exist_ok=True)
seeds=(2027,2028,2029)
views=("native","large","dual")

def panel(d,xy,center,probs,reference=False):
    cy,cx=center
    y=max(0,min(cy-32,960)); x=max(0,min(cx-32,960))
    sl=np.s_[y:y+64,x:x+64]
    rgb=Image.fromarray(d["rgb"][sl])
    green=ImageOps.autocontrast(rgb.getchannel("G"),cutoff=1).convert("RGB")
    layer=np.asarray(rgb).copy()
    layer[d["v"][sl]]=[60,170,220]
    layer[d["h"][sl]]=[170,60,75]
    yy,xx=xy.T
    inside=(yy>=y)&(yy<y+64)&(xx>=x)&(xx<x+64)
    layer[yy[inside]-y,xx[inside]-x]=[255,235,30]
    tiles=[rgb,green,Image.fromarray(layer)]
    for pr in probs:
        tiles.append(Image.fromarray(np.rint(255*pr[sl]).astype("uint8")).convert("RGB"))
    canvas=Image.new("RGB",(128*len(tiles),128))
    for j,tile in enumerate(tiles):
        canvas.paste(tile.resize((128,128),Image.Resampling.NEAREST),(128*j,0))
    return canvas

def main():
    torch.set_num_threads(2)
    cases=paired.dataset()[0]
    old={r["image"]:r for r in load(ROOT/"persistent_miss_distances.json")}
    records=[]
    for fold in range(3):
        nets=[make_model(fold,s) for s in seeds]
        folders=[ROOT/"eval_teacher"/f"fold{fold}"/str(s)/"correction" for s in seeds]
        cals=[load(f/"calibration.json") for f in folders]
        middle=[load(ROOT/"addback_calibration"/f"fold{fold}_{s}.json")["thresholds"]["equal"] for s in seeds]
        for name in split(fold)["dev"]:
            if not old.get(name,{}).get("far_all",0): continue
            d=cases[name]
            target=d["s"]&d["n"]
            margin=[]; anylow=[]; hi=[]; med=[]; bothlow=[]
            maps=[[] for _ in views]
            for j,(net,folder,c) in enumerate(zip(nets,folders,cals)):
                p=probabilities(net,d["rgb"])
                lo=(p["dual"]>=c["m0_thresholds"]["dual"])&d["f"]
                high=(p["dual"]>=c["thresholds"]["dual"])&d["f"]
                with np.load(folder/(name+".npz")) as ar:
                    for mask,key in ((lo,"dual_m0matched"),(high,"dual_matched")):
                        assert np.array_equal(np.packbits(mask),ar[key])
                anylow.append(ndi.distance_transform_edt(~lo)>5)
                bothlow.append(ndi.binary_dilation(lo,structure=paired.cross))
                med.append(ndi.binary_dilation((p["dual"]>=middle[j])&d["f"],structure=paired.cross))
                hviews=[]
                for k,v in enumerate(views):
                    maps[k].append(p[v])
                    hviews.append(ndi.binary_dilation((p[v]>=c["thresholds"][v])&d["f"],structure=paired.cross))
                hi.append(hviews)
                margin.append(ndi.maximum_filter(np.where(d["f"],p["dual"],0),footprint=paired.cross)-c["thresholds"]["dual"])
            far=target & np.logical_and.reduce(anylow)
            assert far.sum()==old[name]["far_all"]
            controls=np.argwhere(target & np.logical_and.reduce(bothlow))
            maps=[np.median(np.stack(group),axis=0) for group in maps]
            hs=np.asarray(hi); ms=np.asarray(med)
            for path in np.unique(d["i"][far]):
                labels,_=ndi.label(far&(d["i"]==path),np.ones((3,3)))
                for idx,slices in enumerate(ndi.find_objects(labels),1):
                    xy=np.argwhere(labels[slices]==idx)+[s.start for s in slices]
                    yy,xx=xy.T
                    center=xy[np.argmin(np.linalg.norm(xy-np.mean(xy,axis=0),axis=1))]
                    key=f"{name}_p{path}_{idx}"
                    hits=hs[:,:,yy,xx]; m=ms[:,yy,xx]
                    rows=panel(d,xy,center,maps)
                    ctl=None
                    if len(controls):
                        ctl=controls[np.argmin(np.linalg.norm(controls-center,axis=1))]
                        ctlrow=panel(d,ctl[None,:],ctl,maps)
                        stack=Image.new("RGB",(768,256));stack.paste(rows,(0,0));stack.paste(ctlrow,(0,128));rows=stack
                    rows.save(out/(key+".png"))
                    records.append(dict(id=key,image=name,path=int(path),points=len(xy),yx=xy.tolist(),
                        center=center.tolist(),control=None if ctl is None else ctl.tolist(),
                        mid_seeds_per_point=m.sum(0).tolist(),high_seeds_per_point=hits[:,2].sum(0).tolist(),
                        high_native_seeds_per_point=hits[:,0].sum(0).tolist(),high_large_seeds_per_point=hits[:,1].sum(0).tolist(),
                        high_any_view_any_seed=np.any(hits,axis=(0,1)).tolist(),
                        high_dual_delta_median=float(np.median(np.asarray(margin)[:,yy,xx]))))
        del nets
        print("completed fold",fold,flush=True)
    assert sum(r["points"] for r in records)==897
    (out/"segments.json").write_text(json.dumps(records,indent=1),encoding="utf-8")
    ordered=sorted(records,key=lambda r:-r["points"])
    head='<meta charset=utf-8><title>出血邻域细骨架漏点图册</title><style>body{max-width:920px;margin:auto;font:16px system-ui}img{max-width:100%}summary{cursor:pointer;padding:5px}header{position:sticky;top:0;background:white;padding:6px}</style><h1>出血邻域细骨架漏点图册</h1><header>各列从左到右为 RGB、绿通道局部对比增强、标注、原图概率、两倍图概率、等权融合概率。每幅原生裁块 64×64，目标点在裁块中央附近。蓝为血管，红为出血，黄为目标。上排漏点，下排为同图最近的稳定命中窄点（若有）。概率取三种子中位数，各档命中数单独按种子计算。</header>'
    html=[head]
    for r in ordered:
        html.append(f'<details><summary>{r["id"]} · {r["points"]} 点 · 高误报档至少一个种子命中 {sum(v>0 for v in r["high_seeds_per_point"])} 点</summary><img loading="lazy" src="{r["id"]}.png"></details>')
    (out/"index.html").write_text("\n".join(html),encoding="utf-8")

if __name__=="__main__":
    main()
