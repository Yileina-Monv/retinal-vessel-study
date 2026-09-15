from pathlib import Path
import csv, json, hashlib, time, threading, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
START=time.time()
timer=threading.Timer(3300,lambda:os._exit(124));timer.daemon=True;timer.start()
def write(name,x):
    (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(name,rs):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def readmask(p):
    with Image.open(p) as im:a=np.asarray(im)
    return (a[:,:,0] if a.ndim==3 else a)>0
rows=list(csv.DictReader((ROOT/'releases/development_v1/validation.csv').open(encoding='utf-8-sig')))
old=list(csv.DictReader((ROOT/'outputs/morphology_minimal_20260907/per_image_measurements.csv').open(encoding='utf-8-sig')))
old={(r['key'],r['model']):r for r in old}
tau=json.loads((ROOT/'releases/m1_pretrial_v1/calibration.json').read_text(encoding='utf-8'))['tau']
protocol=dict(scope='Exploratory paired error audit, existing 118 development images; no training/inference/test data',threshold=.5,tolerances=[0,1,2],thin_radius_tau=tau,thin_source='existing train-only calibration',primary='paired GT skeleton recovery and regression at tolerance 1',decomposition='delta total predicted skeleton = delta GT-mask-supported predicted skeleton + delta unsupported predicted skeleton; supported is not proof of correct topology',probability_probe='pointwise probabilities on GT skeleton missed by both at zero tolerance; descriptive bins <0.1, 0.1-0.3, 0.3-0.5',limit_seconds=3300,script_sha256=sha(Path(__file__)))
if not (OUT/'protocol.json').exists():write('protocol.json',protocol)
write('resume_protocol.json',dict(**protocol,checkpoint='Atomic per-image JSON; resume completed pairs',original_protocol_sha256=sha(OUT/'protocol.json')))
(OUT/'checkpoints').mkdir(exist_ok=True)
res=[];strata=[];provenance=[]
struct={d:sum(v*v for v in np.indices((2*d+1,2*d+1))-d)<=d*d for d in (1,2)}
def load(r):
    y=readmask(ROOT/r['label_path']);f=readmask(ROOT/r['fov_path']);y &= f
    ps=[np.load(ROOT/'.runtime/c1_validation118_v1'/m/(Path(r['key']).stem+'_probability.npy'),allow_pickle=False) for m in ('M0','M1')]
    return y,f,ps
for i,r in enumerate(rows):
    cp=OUT/'checkpoints'/(Path(r['key']).stem+'.json')
    if cp.exists():
        cached=json.loads(cp.read_text(encoding='utf-8'))
        res.append(cached['record']);strata.extend(cached['strata']);provenance.append(cached['provenance'])
        continue
    if time.time()-START>3000:raise TimeoutError('Data deadline')
    files={k:ROOT/r[k+'_path'] for k in ('label','fov')}
    for k,p in files.items():assert sha(p)==r[k+'_sha256']
    for m in ('M0','M1'):
        p=ROOT/'.runtime/c1_validation118_v1'/m/(Path(r['key']).stem+'_probability.npy')
        meta=json.loads(p.with_name(Path(r['key']).stem+'.json').read_text(encoding='utf-8'))
        assert sha(p)==meta['image']['probability_sha256'];files[m]=p
    provenance.append(dict(key=r['key'],sha256={k:sha(p) for k,p in files.items()}))
    y,f,ps=load(r);pred=[(p>=.5)&f for p in ps]
    sy=skeletonize(y,method='zhang');sp=[skeletonize(p,method='zhang') for p in pred]
    n=int(sy.sum());rad=ndi.distance_transform_edt(y);thin=sy&(rad<=tau)
    rec=dict(key=r['key'],disease=r['disease'],Blur=int(r['Blur']),quality_class=r['quality_class'],gt_skeleton=n)
    for j,m in enumerate(('M0','M1')):
        assert int(sp[j].sum())==int(old[r['key'],m]['pred_skeleton'])
        rec[m+'_mare']=float(old[r['key'],m]['absolute_relative_error'])
    for d in (0,1,2):
        supp=[p if d==0 else ndi.binary_dilation(p,structure=struct[d]) for p in pred]
        yd=y if d==0 else ndi.binary_dilation(y,structure=struct[d])
        recover=sy&~supp[0]&supp[1];regress=sy&supp[0]&~supp[1];bothmiss=sy&~supp[0]&~supp[1]
        for name,mask in [('recovered',recover),('regressed',regress),('both_missed',bothmiss)]:rec[f'{name}_d{d}']=int(mask.sum())/n
        for j,m in enumerate(('M0','M1')):
            extra=int((sp[j]&~yd).sum());supported=int((sp[j]&yd).sum())
            rec[f'{m}_extra_d{d}']=extra/n;rec[f'{m}_supported_d{d}']=supported/n
            assert abs(extra/n-float(old[r['key'],m][f'extra_pred_skeleton_per_gt_d{d}']))<1e-12
        rec[f'delta_extra_d{d}']=rec[f'M1_extra_d{d}']-rec[f'M0_extra_d{d}']
        rec[f'delta_supported_d{d}']=rec[f'M1_supported_d{d}']-rec[f'M0_supported_d{d}']
        assert abs(rec[f'delta_extra_d{d}']+rec[f'delta_supported_d{d}']-(int(sp[1].sum())-int(sp[0].sum()))/n)<1e-12
        if d==1:
            for label,mask in [('thin',thin),('thicker',sy&~thin)]:
                count=int(mask.sum())
                strata.append(dict(key=r['key'],stratum=label,count=count,recovered=int((recover&mask).sum()),regressed=int((regress&mask).sum()),M0_missed=int((mask&~supp[0]).sum()),M1_missed=int((mask&~supp[1]).sum())))
        if d==0:
            for j,m in enumerate(('M0','M1')):
                v=ps[j][bothmiss]
                for name,lo,hi in [('low',0,.1),('mid',.1,.3),('near',.3,.5)]:rec[f'{m}_bothmiss_prob_{name}_per_gt']=int(((v>=lo)&(v<hi)).sum())/n
    rec['delta_mare']=rec['M1_mare']-rec['M0_mare'];res.append(rec)
    tmp=cp.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8') as fh:
        json.dump(dict(record=rec,strata=strata[-2:],provenance=provenance[-1]),fh,ensure_ascii=False)
        fh.flush();os.fsync(fh.fileno())
    tmp.replace(cp)
    write('progress.json',dict(completed_pairs=len(res),total=118,last_key=r['key'],elapsed_seconds=time.time()-START,resumable=True))
    if (i+1)%20==0:print(f'{i+1}/118 pairs, {time.time()-START:.1f}s',flush=True)
table('paired_transitions.csv',res);table('width_strata.csv',strata);write('provenance.json',provenance)
rng=np.random.default_rng(20260908)
def stats(key,rs=res):
    a=np.array([r[key] for r in rs]);boot=a[rng.integers(len(a),size=(3000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
keys=[k for k in res[0] if k not in ('key','disease','Blur','quality_class','gt_skeleton')]
summary=dict(n=len(res),metrics={k:stats(k) for k in keys},subgroups={},width={},improved_with_more_extra=sum(r['delta_mare']<0 and r['delta_extra_d1']>0 for r in res),improved=sum(r['delta_mare']<0 for r in res))
for field in ('disease','Blur'):
    for val in sorted({r[field] for r in res}):
        rr=[r for r in res if r[field]==val]
        summary['subgroups'][f'{field}={val}']=dict(n=len(rr),**{k:stats(k,rr) for k in ('recovered_d1','regressed_d1','delta_extra_d1','delta_mare')})
for label in ('thin','thicker'):
    rr=[r for r in strata if r['stratum']==label and r['count']>0]
    summary['width'][label]={k:float(np.mean([r[k]/r['count'] for r in rr])) for k in ('recovered','regressed','M0_missed','M1_missed')}
write('summary.json',summary)
# Three transparently selected illustrative crops; local annotation comparison, not clinical adjudication.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
selections=[('largest_recovery',max(res,key=lambda r:r['recovered_d1'])),('largest_extra_increase',max(res,key=lambda r:r['delta_extra_d1'])),('largest_density_worsening',max(res,key=lambda r:r['delta_mare']))]
fig,axes=plt.subplots(3,4,figsize=(12,9));case_records=[]
for rowidx,(reason,rr) in enumerate(selections):
    r=next(r for r in rows if r['key']==rr['key']);y,f,ps=load(r);pred=[(p>=.5)&f for p in ps];sy=skeletonize(y,method='zhang')
    recover=sy&~ndi.binary_dilation(pred[0],structure=struct[1])&ndi.binary_dilation(pred[1],structure=struct[1])
    change=(pred[1]&~pred[0]&~y) if reason=='largest_extra_increase' else recover
    tiles=[(int(change[a:a+384,b:b+384].sum()),a,b) for a in range(0,y.shape[0]-383,192) for b in range(0,y.shape[1]-383,192)]
    _,a,b=max(tiles);sl=np.s_[a:a+384,b:b+384]
    overlay=np.zeros((*y.shape,3),dtype=np.uint8);overlay[sy]=[180,180,180];overlay[recover]=[0,255,0];overlay[sy&pred[0]&~pred[1]]=[255,0,255];overlay[pred[1]&~pred[0]&~y]=[255,100,0]
    for col,(img,title) in enumerate([(y[sl],'GT'),(pred[0][sl],'M0'),(pred[1][sl],'M1'),(overlay[sl],'Changes')]):
        axes[rowidx,col].imshow(img,cmap='gray',vmin=0,vmax=1 if img.ndim==2 else 255);axes[rowidx,col].set_title(title);axes[rowidx,col].axis('off')
    axes[rowidx,0].set_title(r['key']+'\nGT',fontsize=9)
    case_records.append(dict(key=r['key'],selection=reason,crop_y=a,crop_x=b,size=384))
fig.suptitle('Illustrative extremes (not representative): green recovered GT at 1px;\nmagenta lost GT at 0px; orange newly added non-GT pixels',fontsize=11)
fig.tight_layout();fig.savefig(OUT/'cases.png',dpi=150);plt.close(fig)
write('cases.json',case_records)
write('verification.json',dict(status='PASS',pairs=len(res),hash_verified_files=len(provenance)*4,old_skeleton_and_extra_metrics_reproduced=True,exact_count_identity=True,elapsed_seconds=time.time()-START,within_60min=time.time()-START<3600,no_training=True))
print(json.dumps(summary,ensure_ascii=False),flush=True)
