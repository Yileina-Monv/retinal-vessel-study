"""Post-hoc paired checks on frozen local runs. No optimization for deployment."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import binomtest, rankdata
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
FOLD = BASE / "首折双尺度_本地GPU_20261126/idrid_fold0"
TRANSFER = BASE / "双尺度反向迁移_20261126"
WORK = BASE / ".work/colab_stagecheck/work"
sys.path.insert(0,str(BASE / "Colab_训练推理补充_20261126/runner"))
import scale_eval as ev


def summarize(x):
    x=np.asarray(x,dtype=float)
    x=x[np.isfinite(x)]
    boot=np.random.default_rng(2027).choice(x,(20000,len(x))).mean(1)
    low,high=np.quantile(boot,[.025,.975])
    return dict(n=len(x),mean=x.mean(),low=low,high=high)


def sensitivities():
    rows=[]
    interactions=[]
    for ds, folder in [('IDRiD',FOLD),('MAPLES',TRANSFER)]:
        by_model={}
        for model in ['unet','nnunet']:
            df=pd.read_csv(folder/model/'per_image.csv',dtype={'arm':str})
            for (point,band), group in df.groupby(['point','band']):
                a,b=[group[group.arm==s].set_index('image').sort_index() for s in ['1','1+1.5']]
                assert a.index.equals(b.index) and a.n.equals(b.n)
                for metric in ['D0','D1','dice','far_recall']:
                    diff=100*(b[metric]-a[metric])
                    row=dict(dataset=ds,model=model,point=point,band=band,metric=metric,**summarize(diff))
                    if metric in ['D0','D1']:
                        inc=(b[metric.lower()]-a[metric.lower()])
                        wins=int((inc>0).sum())
                        losses=int((inc<0).sum())
                        row.update(pooled=100*inc.sum()/a.n.sum(),wins=wins,losses=losses,
                                   denominator_ge5=diff[a.n>=5].mean(),num_ge5=int((a.n>=5).sum()),
                                   sign_p=binomtest(wins,wins+losses).pvalue if wins+losses else 1.)
                    rows.append(row)
                    if band=='W<=3':
                        by_model[model,point,metric]=diff
        for (model,point,metric),x in by_model.items():
            if model=='unet':
                interactions.append(dict(dataset=ds,point=point,metric=metric,**summarize(x-by_model['nnunet',point,metric])))
    pd.DataFrame(rows).to_csv(ROOT/'sensitivity.csv',index=False)
    pd.DataFrame(interactions).to_csv(ROOT/'interaction.csv',index=False)


def far_checks():
    paired,_,geometry=ev.setup(WORK)
    data,tr,dev=paired.dataset(0)
    rows=[]
    for image in dev:
        a=ev.angular(data[image],50.,geometry)
        roi=a['scored'] & ~a['ring'] & ~a['h']
        pos=roi & a['v']
        neg=roi & ~a['v']
        edge=pos & (distance_transform_edt(a['v'])<=np.sqrt(2))
        for model in ['unet','nnunet']:
            base=FOLD/model
            cal=ev.read_json(base/'calibration.json')
            assert tr==cal['train'] and not set(tr)&set(dev)
            t0,t1=[cal['thresholds'][s] for s in ['1','1+1.5']]
            with np.load(base/'probabilities'/(image+'.npz')) as z:
                p,q=z['angular_1'],z['angular_1+1.5']
            # Exact empirical ROC AUC on all far pixels, with midranks for ties.
            def auc(v):
                vpos,vneg=v[pos],v[neg]
                r=rankdata(np.concatenate((vpos,vneg)))[:len(vpos)]
                return ((r.mean()-(len(vpos)+1)/2)/len(vneg))
            # Match the native false-positive count separately within each image.
            k=int((p[neg]>=t0).sum())
            assert 0<k<neg.sum()
            v=q[neg]
            tmatch=np.partition(v,len(v)-k)[len(v)-k]
            def recall(v,t):
                return (v[pos]>=t).mean()
            base_rec=recall(p,t0)
            def contributions(mask):
                return ((q[mask]>=t1).sum()-(p[mask]>=t0).sum())/pos.sum()
            pc=.5*((recall(q,t0)-base_rec)+(recall(q,t1)-recall(p,t1)))
            total=recall(q,t1)-base_rec
            assert np.isclose(contributions(edge)+contributions(pos & ~edge),total)
            rows.append(dict(image=image,model=model,total=100*total,symmetric_map=100*pc,
                symmetric_threshold=100*(total-pc),edge=100*contributions(edge),
                interior=100*contributions(pos & ~edge),matched_far_fp=100*(recall(q,tmatch)-base_rec),
                fp_mismatch=int((v>=tmatch).sum()-k),auc_delta=100*(auc(q)-auc(p))))
    df=pd.DataFrame(rows)
    df.to_csv(ROOT/'far_checks.csv',index=False)
    print(df.groupby('model').mean(numeric_only=True).round(3))
    print('max FP mismatch',df.fp_mismatch.abs().max())


if __name__=='__main__':
    sensitivities()
    far_checks()
    checks = pd.read_csv(ROOT / "far_checks.csv")
    summaries = []
    for model, group in checks.groupby("model"):
        for metric in ["total", "symmetric_map", "symmetric_threshold", "edge", "interior", "matched_far_fp", "auc_delta"]:
            summaries.append(dict(model=model, metric=metric, **summarize(group[metric])))
    pd.DataFrame(summaries).to_csv(ROOT / "far_summary.csv", index=False)
