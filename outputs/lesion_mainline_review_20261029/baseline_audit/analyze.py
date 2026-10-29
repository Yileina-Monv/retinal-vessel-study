"""Read-only CSV audit; no model, image, GPU, training or Test access.
Run: python analyze.py (numpy and pandas required).
All *_pp columns are percentage points; raw CSV measures keep original units.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'derived'
OUT.mkdir(exist_ok=True)
def read(k,f): return pd.read_csv(ROOT/k/f,float_precision='round_trip')
data={k:read(k,'fixed_endpoint_per_image_metrics.csv') for k in ['old','new']}
checks={}
for k,d in data.items():
    keys=['seed','method','image','metric']
    assert not d.duplicated(keys).any()
    assert len(d.metric.unique())==31 and len(d.image.unique())==28
    assert d.threshold.eq(.5).all()
    assert d.groupby(['seed','method']).size().eq(868).all()
    assert d.groupby(['seed','method']).value.apply(lambda x:x.isna().sum()).eq(48).all()
    na=d.value.isna()
    assert d.loc[na,'na_reason'].notna().all()
    assert d.loc[~na,'na_reason'].isna().all()
    r=d[d.value.notna()&d.numerator.notna()&d.denominator.notna()]
    assert r.denominator.gt(0).all()
    err=float((r.value-r.numerator/r.denominator).abs().max())
    assert err<1e-12
    means=read(k,'absolute_metric_summary.csv')
    if k=='old': means['seed']=2026
    mg=d.groupby(['seed','method','metric']).value.agg(['mean','count']).reset_index().merge(means,on=['seed','method','metric'],suffixes=('_calc','_reported'),validate='one_to_one')
    assert len(mg)==len(means)
    me=float((mg.mean_calc-mg.mean_reported).abs().max())
    assert me<1e-12
    pairs=read(k,'paired_image_deltas.csv')
    if k=='new': pairs['contrast']='NNUNET-A0'
    lookup=d.set_index(keys).value
    errors=[]
    for r in pairs.itertuples():
        arm=r.contrast.removesuffix('-A0')
        a=lookup.loc[(r.seed,arm,r.image,r.metric)]
        b=lookup.loc[(r.seed,'A0',r.image,r.metric)]
        if np.isnan(a) or np.isnan(b): assert np.isnan(r.delta)
        else: errors.append(abs(a-b-r.delta))
    assert max(errors)<1e-12
    indices=json.loads((ROOT/k/'bootstrap_indices_by_n.json').read_text())
    stats=read(k,'paired_comparison_summary.csv')
    if k=='old': stats['seed']=2026
    se=[]
    for r in stats.itertuples():
        sub=pairs[(pairs.seed==r.seed)&(pairs.contrast==r.contrast)&(pairs.metric==r.metric)].dropna(subset=['delta'])
        if k=='new':
            sub=sub.sort_values('pair_index')
            assert sub.pair_index.tolist()==list(range(len(sub)))
        v=sub.delta.to_numpy(); n=len(v)
        assert n==r.n_images
        bs=v[np.asarray(indices[str(n)])].mean(axis=1)
        est=[v.mean(),*np.quantile(bs,[.025,.975],method='linear')]
        se.extend(abs(np.array(est)-[r.estimate,r.lower,r.upper]))
    assert max(se)<1e-12
    checks[k]=dict(rows=len(d),unique_images=28,metrics=31,NA=int(na.sum()),mean_groups=len(means),pair_rows=len(pairs),interval_groups=len(stats),ratio_error=err,mean_error=me,pair_error=float(max(errors)),interval_error=float(max(se)))

a=data['old'].query('method=="A0"').set_index(['seed','image','metric'])
b=data['new'].query('method=="A0" and seed==2026').set_index(['seed','image','metric'])
assert a.index.equals(b.index)
for col in ['value','numerator','denominator']:
    assert np.allclose(a[col],b[col],atol=0,rtol=0,equal_nan=True)
checks['shared_A0_2026']='All 868 values/numerators/denominators identical, including 48 NA.'
assert set(data['old'].image)==set(data['new'].image)
# Eligibility lists and GT-derived denominators must match across seeds/arms.
all_data=pd.concat([data['old'],data['new'].query('not (seed==2026 and method=="A0")')],ignore_index=True)
for m,g in all_data.groupby('metric'):
    eligible=g.groupby(['seed','method']).apply(lambda x:tuple(sorted(x.loc[x.value.notna(),'image'])),include_groups=False)
    assert eligible.nunique()==1
    if m in ['recall','far_recall','fp_density','coarse_vessel_d0','candidate_path_d1']:
        assert g.groupby('image').denominator.nunique(dropna=False).eq(1).all()
checks['eligibility']='Same image IDs and per-metric NA sets across all arms; checked GT/FOV denominator fields match.'
(OUT/'verification.json').write_text(json.dumps(checks,indent=2))

main=['dice','precision','recall','fp_density','candidate_path_d1_macro','hard_cldice','far_recall','coarse_vessel_d0']
absolute=all_data.groupby(['seed','method','metric']).value.mean().unstack('metric')
(100*absolute[main]).to_csv(OUT/'absolute_percent.csv')
olds=read('old','paired_comparison_summary.csv'); olds['seed']=2026; olds['cohort']='old_U-Net_single_seed'
news=read('new','paired_comparison_summary.csv')
stats=pd.concat([olds,news],ignore_index=True)
stats.to_csv(OUT/'all_source_comparisons.csv',index=False)
# Recompute every pair; records here include valid metric values only.
wide=all_data.pivot(index=['seed','image','metric'],columns='method',values='value')
records=[]
for seed in [2026,2027,2028]:
    w=wide.loc[seed]
    for arm in w.columns:
        if arm=='A0': continue
        delta=(w[arm]-w.A0).dropna()
        for (im,m),v in delta.items(): records.append(dict(seed=seed,method=arm,image=im,metric=m,delta=v))
delta=pd.DataFrame(records)
delta.to_csv(OUT/'recomputed_image_deltas.csv',index=False)
tail=[]
for (seed,arm,m),g in delta.groupby(['seed','method','metric'],sort=False):
    x=g.delta.to_numpy(); factor=1 if m in ['longest_hit_mean','longest_miss_mean'] else 100
    q=np.quantile(x,[0,.1,.5,.9,1])*factor
    tail.append(dict(seed=seed,method=arm,metric=m,unit='reference_scale_pixels' if factor==1 else 'pp',n=len(x),positive=int((x>0).sum()),negative=int((x<0).sum()),zero=int((x==0).sum()),mean=x.mean()*factor,min=q[0],p10=q[1],median=q[2],p90=q[3],max=q[4],worst3_mean=np.sort(x)[:3].mean()*factor,best3_mean=np.sort(x)[-3:].mean()*factor))
pd.DataFrame(tail).to_csv(OUT/'tails_all_metrics.csv',index=False)
# Exploratory historical comparisons: same images and same numeric seed, different recipes.
cross=[]
for arm in ['A0','B1','B2','C','D','UFL','SPATIAL','SCSE']:
    for m in main:
        v=(wide.loc[(2026,slice(None),m),'NNUNET']-wide.loc[(2026,slice(None),m),arm]).dropna().to_numpy()
        cross.append(dict(seed=2026,cohort='historical_interrupted_only',contrast='NNUNET-'+arm,metric=m,mean_pp=v.mean()*100,positive=int((v>0).sum()),negative=int((v<0).sum()),min_pp=v.min()*100))
pd.DataFrame(cross).to_csv(OUT/'historical_same_seed_comparisons.csv',index=False)

joint={}
nn=delta.query('method=="NNUNET"')
for seed in [2026,2027,2028]:
    w=nn.query('seed==@seed').pivot(index='image',columns='metric',values='delta')
    joint[str(seed)]={
        'Dice_down_gt_1pp':int((w.dice<-.01).sum()),
        'Precision_down_gt_1pp':int((w.precision<-.01).sum()),
        'FP_up_gt_0_5pp':int((w.fp_density>.005).sum()),
        'all_six_per_image_guards_nonadverse':int(((w.precision>=0)&(w.fp_density<=0)&(w.recall>=0)&(w.far_recall>=0)&(w.coarse_vessel_d0>=0)&(w.hard_cldice>=0)).sum()),
        'Dice_drop_images':w.loc[w.dice<0,main].mul(100).reset_index().to_dict('records'),
        'worst_FP_images':w.nlargest(3,'fp_density')[main].mul(100).reset_index().to_dict('records'),
        'worst_precision_images':w.nsmallest(3,'precision')[main].mul(100).reset_index().to_dict('records')
    }
w27=nn.query('seed==2027').pivot(index='image',columns='metric',values='delta')
w28=nn.query('seed==2028').pivot(index='image',columns='metric',values='delta')
joint['two_seed_same_images']={
    'dice_delta_Pearson':w27.dice.corr(w28.dice),
    'dice_improves_both':int(((w27.dice>0)&(w28.dice>0)).sum()),
    'FP_worse_both':int(((w27.fp_density>0)&(w28.fp_density>0)).sum()),
    'precision_worse_both':int(((w27.precision<0)&(w28.precision<0)).sum()),
    'path_recall_worse_both':int(((w27.candidate_path_d1_macro<0)&(w28.candidate_path_d1_macro<0)).sum()),
}
# Average within image FIRST; this remains 28 images, not 56 independent samples.
avg=(w27+w28)/2
avg_out=avg.copy()
for m in avg_out:
    if m not in ['longest_hit_mean','longest_miss_mean']: avg_out[m]*=100
avg_out=avg_out.rename(columns={m:m+('_reference_px' if m in ['longest_hit_mean','longest_miss_mean'] else '_pp') for m in avg_out})
avg_out.to_csv(OUT/'two_seed_mean_image_deltas.csv')
joint['two_seed_same_images']['average_Dice_mean_pp']=avg.dice.mean()*100
joint['two_seed_same_images']['average_Dice_min_pp']=avg.dice.min()*100
joint['two_seed_same_images']['average_Dice_drop_top3_gains_mean_pp']=avg.dice.sort_values().iloc[:-3].mean()*100
joint['two_seed_same_images']['per_image_average_FP_mean_pp']=avg.fp_density.mean()*100
(OUT/'tail_details.json').write_text(json.dumps(joint,indent=2))

print(json.dumps(checks,indent=2))
print('\nABSOLUTE PERCENT\n'+(100*absolute[main]).to_string(float_format=lambda x:f'{x:.4f}'))
print('\nTAILS (delta units pp except lengths)\n'+pd.DataFrame(tail).query('method=="NNUNET" and seed!=2026 and metric in @main').to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\nHISTORICAL NNUNET VS OLD\n'+pd.DataFrame(cross).pivot(index='contrast',columns='metric',values='mean_pp').to_string(float_format=lambda x:f'{x:.4f}'))
print('\nJOINT\n'+json.dumps(joint,indent=2))
