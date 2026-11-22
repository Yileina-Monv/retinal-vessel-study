"""Render two explicitly selected held-out examples at the preregistered threshold."""
import csv,json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import paired as p

def main():
    items,_,_=p.dataset()
    r=list(csv.DictReader((p.HERE/"out_of_fold/all/paired_per_image.csv").open()))
    path={x["image"]:float(x["delta_pp"]) for x in r if x["metric"]=="path_d1"}
    fp={x["image"]:float(x["delta_pp"]) for x in r if x["metric"]=="near_fp_density"}
    example=max(path,key=path.get)
    adverse=max([k for k in path if abs(path[k])<1e-7],key=fp.get)
    font=ImageFont.truetype("arial.ttf",18);small=ImageFont.truetype("arial.ttf",15)
    sheet=Image.new("RGB",(1000,785),"white");draw=ImageDraw.Draw(sheet)
    draw.text((14,9),"Out-of-fold Refined IDRiD predictions | seed 2027 | threshold 0.5",fill="black",font=font)
    draw.text((14,38),"White = outer haemorrhage ring boundary; cyan = reference vessel covered; blue = missed; orange = FP",fill="black",font=small)
    for i,(name,role) in enumerate([(example,"Largest mean path gain"),(adverse,"Highest FP cost among unchanged path scores")]):
        d=items[name]
        stage=next(s for s in [p.HERE,p.HERE/"fold_1",p.HERE/"fold_2"] if name in json.loads((s/"split.json").read_text())["dev"])
        predictions=[]
        for method in ("base","local"):
            with np.load(stage/"rgb_only"/f"{method}_2027"/(name+".npz")) as z:
                predictions.append(np.unpackbits(z["prediction"]).reshape(d["f"].shape).astype(bool))
        a,b=predictions
        signal=((d["i"]>0)&~ndi.binary_dilation(a,p.cross)&ndi.binary_dilation(b,p.cross)) if i==0 else (d["n"]&~d["v"]&~a&b)
        edge=96
        if not signal.any(): raise ValueError(f"No change to illustrate for {name}")
        density=ndi.gaussian_filter(signal.astype(np.float32),sigma=10)
        center=np.unravel_index(np.argmax(density),density.shape)
        y,x=np.clip(np.array(center)-edge//2,0,np.array(signal.shape)-edge)
        crop=np.s_[y:y+edge,x:x+edge]
        top=71+i*350
        draw.text((14,top),f"{name} | {role} | image-macro delta across seeds: path {path[name]:+.2f} pp, near FP {fp[name]:+.2f} pp",fill="black",font=small)
        for j,title in enumerate(("RGB","Ordinary fine-tuning","Local auxiliary loss")):
            img=d["rgb"].copy()
            if j:
                pred=predictions[j-1]
                for mask,color in [(d["v"]&pred,(0,200,190)),(d["v"]&~pred,(65,110,255)),(~d["v"]&pred,(244,128,36))]:
                    img[mask&d["n"]]=color
                img[ndi.binary_dilation(d["n"]) & ~d["n"]]=(255,255,255)
            sheet.paste(Image.fromarray(img[crop]).resize((285,285),Image.Resampling.NEAREST),(14+330*j,top+50))
            draw.text((14+330*j,top+24),title,fill="black",font=font)
    sheet.save(p.HERE/"heldout_examples.png")

if __name__=="__main__": main()
