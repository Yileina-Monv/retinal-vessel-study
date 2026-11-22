"""Paired native RGB crops with point transitions shared by at least two seeds."""
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from predict_view_fusion import models
from family_scale_ablation import variants
from contrast_cycle import enhance
from prepare import ROOT,split
from two_scale import score
import paired

GROUP=ROOT/"family_scale_ablation/large_dual"
NAMES=("IDRiD_18","IDRiD_05","IDRiD_54")

def render():
    torch.set_num_threads(2)
    cases=paired.dataset()[0]
    changes={name:[np.zeros((1024,1024),np.uint8) for _ in range(2)] for name in NAMES}
    for fold in range(3):
        names=[k for k in NAMES if k in split(fold)["dev"]]
        if not names: continue
        for seed in (2027,2028,2029):
            m,n=models(fold,seed)
            fname=f"fold{fold}_{seed}.json"
            ref=json.loads((ROOT/"cross_family_cycle/nnunet"/fname).read_text())
            alt=json.loads((GROUP/fname).read_text())
            for k in names:
                d=cases[k]
                ref_p,_,probs=variants(d["rgb"],enhance(d["rgb"]),m,n)
                before,after=[(prob>=rec["thresholds"]["m0matched"])&d["f"] for prob,rec in ((ref_p,ref),(probs["large_dual"],alt))]
                exp=next(q for q in alt["rows"] if q["point"]=="m0matched" and q["image"]==k)
                for mask,side in ((before,"base"),(after,"candidate")):
                    assert all(exp[side][j]==v for j,v in score(d,mask,.5).items())
                a,b=[ndi.binary_dilation(mask,structure=paired.cross) for mask in (before,after)]
                target=d["n"]&d["s"]
                changes[k][0]+=target&b&~a
                changes[k][1]+=target&a&~b
            del m,n
    canvas=Image.new("RGB",(630,876),"white");draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype("msyh.ttc",16)
    draw.text((12,4),"M0 仅两倍图 vs 双尺度融合｜相同位置至少 2/3 种子出现同向变化",font=font,fill="black")
    draw.text((12,28),"左：原始 RGB 局部。右：青圈 = 新命中，橙圈 = 新漏掉。",font=font,fill="black")
    audit=[]
    size=96
    for i,k in enumerate(NAMES):
        g,l=[v>=2 for v in changes[k]]
        focus=l if k=="IDRiD_54" else g
        # Choose the 96px square with most reproducible changes in that direction.
        integral=np.pad(focus.astype(np.int32),((1,0),(1,0))).cumsum(0).cumsum(1)
        density=integral[size:,size:]-integral[:-size,size:]-integral[size:,:-size]+integral[:-size,:-size]
        peak=np.argwhere(density==density.max());y,x=map(int,peak[len(peak)//2])
        points=np.argwhere(focus[y:y+size,x:x+size])+[y,x]
        lower=np.maximum(points.max(0)-size+1,0)
        upper=np.minimum(points.min(0),np.array(focus.shape)-size)
        y,x=map(int,np.clip(np.rint(points.mean(0)-(size-1)/2),lower,upper))
        crop=Image.fromarray(cases[k]["rgb"][y:y+size,x:x+size]).resize((240,240),Image.Resampling.NEAREST)
        annotated=crop.copy();overlay=ImageDraw.Draw(annotated)
        for mask,color in ((l,"#ff9f22"),(g,"#31ffff")):
            for yy,xx in np.argwhere(mask[y:y+size,x:x+size]):
                u,v=(xx+.5)*2.5,(yy+.5)*2.5
                overlay.ellipse((u-2,v-2,u+2,v+2),outline=color,width=1)
        y0=52+272*i
        plus,minus=[int(v[y:y+size,x:x+size].sum()) for v in (g,l)]
        draw.text((12,y0),f"{k}   本框新命中 {plus} 点，新漏掉 {minus} 点",font=font,fill="black")
        canvas.paste(crop,(12,y0+26));canvas.paste(annotated,(266,y0+26))
        audit.append(dict(image=k,x=x,y=y,size=size,whole_gains=int(g.sum()),whole_losses=int(l.sum()),crop_gains=plus,crop_losses=minus))
    canvas=canvas.crop((0,0,628,870))
    canvas.save(GROUP/"paired_crops.png",optimize=True)
    (GROUP/"paired_crops.json").write_text(json.dumps(audit,indent=2))

if __name__=="__main__":render()
