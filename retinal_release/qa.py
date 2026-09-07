import csv
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from retinal_release.common import ROOT,read

def main():
    q=read(ROOT/'releases/development_v1/fov_qa.json')['rows']
    with (ROOT/'releases/development_v1/all.csv').open(encoding='utf-8') as f:rows={r['key']:r for r in csv.DictReader(f)}
    selected=[]
    for key in ('area_fraction','label_pixels_outside'):
        for r in sorted(q,key=lambda r:r[key],reverse=key!='area_fraction')[:2]:
            if r['key'] not in selected:selected.append(r['key'])
    for key in ('train/373_G.png','train/353_G.png','train/361_G.png'):
        if key not in selected:selected.append(key)
    for disease in ('A','D','G','N'):
        key=next(r['key'] for r in q if r['disease']==disease and r['quality']=='poor')
        if key not in selected:selected.append(key)
    fig,axes=plt.subplots(3,4,figsize=(14,11))
    for ax,key in zip(axes.flat,selected):
        row=rows[key];image=np.array(Image.open(ROOT/row['image_path']).resize((512,512)))
        fov=np.array(Image.open(ROOT/row['fov_path']).resize((512,512),Image.Resampling.NEAREST))>0
        ax.imshow(image);ax.contour(fov,levels=[.5],colors=['cyan'],linewidths=.7)
        record=next(r for r in q if r['key']==key)
        ax.set_title(f'{key}\nFOV {record["area_fraction"]:.1%}; outside labels {record["label_pixels_outside"]}',fontsize=9)
    for ax in axes.flat:ax.axis('off')
    fig.suptitle('FOV QC: smallest fields, most excluded label pixels, and poor-quality examples')
    fig.tight_layout(h_pad=2.5,rect=(0,0,1,.96));fig.savefig(ROOT/'releases/development_v1/fov_QC.png',dpi=120)

if __name__=='__main__':main()
