"""Resample source calibration and held-out images independently, with paired indices across arms/models."""
from check import ev, WORK, FOLD, ROOT
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

paired,_,geometry=ev.setup(WORK)
data,tr,dev=paired.dataset(0)
B=20000
rng=np.random.default_rng(2027)
ti=rng.integers(len(tr),size=(B,len(tr)))
di=rng.integers(len(dev),size=(B,len(dev)))
grid=ev.GRID
models=['unet','nnunet']
arms=['1','1+1.5']
metrics={}
rows=[]
for model in models:
    folder=FOLD/model
    curves=[ev.read_json(folder/'train'/(n+'.json'))['curves'] for n in tr]
    for arm in arms:
        c=np.array([x[arm] if x else np.full(len(grid),np.nan) for x in curves])
        idx=np.concatenate([np.argmax(np.nanmean(c[ix],axis=1)<=.015,axis=1) for ix in np.array_split(ti,100)])
        assert (np.nanmean(c,axis=0)<=.015).argmax()/128==ev.read_json(folder/'calibration.json')['thresholds'][arm]
        per=[]
        for name in dev:
            a=ev.angular(data[name],50.,geometry)
            with np.load(folder/'probabilities'/(name+'.npz')) as z: pr=z['angular_'+arm]
            pts=a['sk'] & (a['width']<=3)
            # Do not score invalid pixels in the D1 neighbourhood.
            scores=maximum_filter(np.where(a['support'],pr,-1.),footprint=ev.CROSS,mode='constant',cval=-1)[pts]
            per.append((scores[:,None]>=grid).mean(axis=0) if len(scores) else np.full(len(grid),np.nan))
        table=np.array(per)
        metrics[model,arm]=table[di,idx[:,None]]
        print(model,arm,'source threshold quantiles',np.quantile(idx/128,[.025,.5,.975]),flush=True)
def save(label,arr):
    estimates=100*np.nanmean(arr,axis=1)
    rows.append(dict(contrast=label,median=np.median(estimates),low=np.percentile(estimates,2.5),high=np.percentile(estimates,97.5)))
deltas={m:metrics[m,arms[1]]-metrics[m,arms[0]] for m in models}
for m,v in deltas.items(): save(m,v)
save('unet minus nnunet',deltas['unet']-deltas['nnunet'])
out=pd.DataFrame(rows);out.to_csv(ROOT/'calibration_resampling.csv',index=False);print(out)
