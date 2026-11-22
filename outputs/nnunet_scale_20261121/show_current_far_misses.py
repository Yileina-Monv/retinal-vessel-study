"""Select reproducible illustrations of current four-view failures from saved RGB/masks."""
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
from prepare import ROOT,split
import paired

OUT=ROOT/"switch_supervision"

def make():
    cases=paired.dataset()[0];best={}
    for fold in range(3):
        for k in split(fold)["dev"]:
            d=cases[k];target=d["n"]&d["s"]
            if not target.any():continue
            far=target.copy();votes=np.zeros_like(target,np.uint8);rescue=votes.copy();paths=set()
            for seed in (2027,2028,2029):
                with np.load(OUT/"heldout_masks"/f"fold{fold}_{seed}"/(k+".npz")) as z:
                    orig,zoom=[np.unpackbits(z["m0matched_"+m]).reshape(target.shape).astype(bool) for m in ("base","zoom_control")]
                votes+=orig;far&=ndi.distance_transform_edt(~orig)>5
                hit=ndi.binary_dilation(orig,structure=paired.cross)
                paths.update(np.unique(d["i"][hit&(d["i"]>0)]).tolist())
                rescue+=ndi.binary_dilation(orig|(zoom&d["v"]),structure=paired.cross)&target
            iso=ndi.convolve(far.astype(np.uint8),np.ones((3,3),np.uint8))==1
            choices=(far&(rescue<2)&(d["i"]>0)&~np.isin(d["i"],list(paths)),far&(rescue>=2),far&iso&(rescue<2))
            for mode,sel in enumerate(choices):
                if not sel.any():continue
                labels,_=ndi.label(sel,np.ones((3,3),bool));counts=np.bincount(labels.ravel())[1:]
                idx=int(np.argmax(counts))+1;size=int(counts[idx-1])
                if mode in best and best[mode][0]>=size:continue
                pts=np.argwhere(labels==idx)
                y,x=np.clip(np.rint(pts.mean(0))-48,0,928).astype(int)
                box=np.s_[y:y+96,x:x+96]
                best[mode]=(size,k,int(x),int(y),d["rgb"][box].copy(),(votes[box]>=2),d["h"][box].copy(),(labels[box]==idx))
    canvas=Image.new("RGB",(632,934),"white");draw=ImageDraw.Draw(canvas);font=ImageFont.truetype("msyh.ttc",16)
    titles=("未救回且整条合格环内分支均无命中的连续远漏点","增加两倍图的真实血管像素后得到命中的远漏点","孤立的远漏点，增加真实血管像素后仍未命中")
    draw.text((10,8),"左为原图。右图黄圈为目标点，青线为至少两种子预测的血管轮廓，",font=font,fill="black")
    draw.text((10,32),"紫线为人工出血边界。每格为 96 × 96 原生像素的局部。",font=font,fill="black")
    info=[]
    for mode,(size,k,x,y,rgb,mask,he,pts) in sorted(best.items()):
        raw=Image.fromarray(rgb).resize((240,240),Image.Resampling.NEAREST)
        annotation=rgb.copy()
        for layer,color in ((mask,(25,230,220)),(he,(222,65,186))):
            annotation[layer&~ndi.binary_erosion(layer,border_value=1)]=color
        overlay=Image.fromarray(annotation).resize(raw.size,Image.Resampling.NEAREST)
        pen=ImageDraw.Draw(overlay)
        for yy,xx in np.argwhere(pts):
            u,v=(xx+.5)*2.5,(yy+.5)*2.5
            pen.ellipse((u-2,v-2,u+2,v+2),outline="#ffe125",width=1)
        yy=61+290*mode
        draw.text((10,yy),f"{k}　{titles[mode]}（片段 {size} 点）",font=font,fill="black")
        canvas.paste(raw,(10,yy+27));canvas.paste(overlay,(266,yy+27))
        info.append(dict(image=k,group=mode,x=x,y=y,points=size,shown=int(pts.sum())))
    canvas.save(OUT/"far_examples.png")
    (OUT/"far_examples.json").write_text(json.dumps(info,indent=2))
    print(info)

if __name__=="__main__":make()
