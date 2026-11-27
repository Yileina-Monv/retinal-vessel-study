"""Sensitivity to nominal angular diameter at frozen original cutoffs; not physical calibration."""
from check import ev,WORK,FOLD,ROOT,summarize
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
p,_,geo=ev.setup(WORK)
data,tr,dev=p.dataset(0)
rows=[]
for image in dev:
    aa={angle:ev.angular(data[image],angle,geo) for angle in (47.5,50.,52.5)}
    for m in ('unet','nnunet'):
        d=FOLD/m
        t=ev.read_json(d/'calibration.json')['thresholds']
        with np.load(d/'probabilities'/(image+'.npz')) as z:
            one=z['native_1.0']; big=z['native_1.5']
        for angle,a in aa.items():
            one_w=a['warp'](one,ev.cv2.INTER_LINEAR)
            fused=(one_w+a['warp'](big,ev.cv2.INTER_LINEAR))/2
            pts=a['sk'] & (a['width']<=3)
            def hits(pr,tt):
                return np.sum(binary_dilation((pr>=tt)&a['support'],structure=ev.CROSS)&pts)
            change=int(hits(fused,t['1+1.5'])-hits(one_w,t['1']))
            if angle==50.:
                old=ev.read_json(d/'dev'/(image+'.json'))['rows']
                r=next(v for v in old if (v['arm'],v['point'],v['band'])==('1+1.5','matched','W<=3'))
                assert r['n']==pts.sum() and change==r['gained_points']-r['lost_points']
            rows.append(dict(model=m,image=image,angle=angle,n=pts.sum(),net_hits=change,
                             delta=100*change/pts.sum() if pts.any() else np.nan))
df=pd.DataFrame(rows)
df.to_csv(ROOT/'angle_checks.csv',index=False)
for key,s in df.groupby(['model','angle']):
    print(key,s.n.sum(),summarize(s.delta))
