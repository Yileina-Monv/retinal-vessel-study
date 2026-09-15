from pathlib import Path
import os,time,csv,json,hashlib
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
P=Path(__file__).resolve().parent;R=P.parents[1];START=time.time()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):
    p=P/p;p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def mask(p):
    with Image.open(p) as im:a=np.asarray(im)
    return (a[:,:,0] if a.ndim==3 else a)>0
rows=list(csv.DictReader((R/'releases/development_v1/validation.csv').open(encoding='utf-8-sig')))
grid=np.round(np.arange(.1,.9001,.005),3);foot=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)
folds={}
for disease in 'ADGN':
    rr=sorted([r for r in rows if r['disease']==disease],key=lambda r:hashlib.sha256(('matched-fp-20260908'+r['key']).encode()).hexdigest())
    folds.update({r['key']:i%2 for i,r in enumerate(rr)})
tau=read(R/'releases/m1_pretrial_v1/calibration.json')['tau']
protocol=dict(thresholds=grid.tolist(),folds=folds,split='two folds stratified by disease, fixed hash ordering; no patient identity available',selection='On opposite fold, match image-mean FP/FOV to anchor model at 0.5; nearest grid value, tie higher threshold',primary='anchor M0@0.5, adjust M1',secondary='anchor M1@0.5, adjust M0',metrics=['FP/FOV','thin reference centerline recall at 1px','skeleton density MARE','unsupported skeleton/GT at 1px','precision'],thin_radius=tau,scope='Exploratory development cross-fit; no training, inference or official test',script_sha256=sha(Path(__file__)),manifest_sha256=sha(R/'releases/development_v1/validation.csv'))
if (P/'protocol.json').exists():assert read(P/'protocol.json')==protocol
else:put('protocol.json',protocol)
def paths(r):return {**{k:R/r[k+'_path'] for k in ('label','fov')},**{m:R/'.runtime/c1_validation118_v1'/m/(Path(r['key']).stem+'_probability.npy') for m in ('M0','M1')}}
def load(r):
    pp=paths(r);hashes={k:sha(p) for k,p in pp.items()}
    for k in ('label','fov'):assert hashes[k]==r[k+'_sha256']
    for m in ('M0','M1'):
        meta=read(pp[m].with_name(Path(r['key']).stem+'.json'));assert hashes[m]==meta['image']['probability_sha256']
    y=mask(pp['label']);f=mask(pp['fov']);y &= f;sy=skeletonize(y,method='zhang');thin=sy&(ndi.distance_transform_edt(y)<=tau)
    probs={m:np.load(pp[m],allow_pickle=False) for m in ('M0','M1')}
    return y,f,sy,thin,probs,hashes
curves=[]
for i,r in enumerate(rows):
    name='curves/'+Path(r['key']).stem+'.json';cp=P/name
    if cp.exists():
        c=read(cp);assert c['hashes']=={k:sha(p) for k,p in paths(r).items()}
    else:
        y,f,sy,thin,probs,h=load(r);c=dict(key=r['key'],hashes=h,fold=folds[r['key']],models={})
        for m,p in probs.items():
            bg=np.sort(p[f&~y]);fp=(len(bg)-np.searchsorted(bg,grid,side='left'))/int(f.sum())
            maximum=ndi.maximum_filter(np.where(f,p,-1),footprint=foot,mode='constant',cval=-1)
            vv=np.sort(maximum[thin]);rec=(len(vv)-np.searchsorted(vv,grid,side='left'))/len(vv)
            c['models'][m]=dict(fp_fov=fp.tolist(),thin_recall=rec.tolist())
        put(name,c)
    curves.append(c);put('progress.json',dict(stage='threshold_curves',completed=i+1,total=118,seconds=time.time()-START))
    if (i+1)%20==0:print('curves',i+1,flush=True)
idx=int(np.flatnonzero(grid==.5)[0]);selected={}
for fold in (0,1):
    train=[c for c in curves if c['fold']!=fold];selected[str(fold)]={}
    for anchor,other in [('M0','M1'),('M1','M0')]:
        target=float(np.mean([c['models'][anchor]['fp_fov'][idx] for c in train]));a=np.mean([c['models'][other]['fp_fov'] for c in train],axis=0)
        j=min(range(len(grid)),key=lambda j:(abs(a[j]-target),-grid[j]))
        selected[str(fold)][anchor]=dict(anchor=anchor,other=other,threshold=float(grid[j]),calibration_target_fp=target,calibration_actual_fp=float(a[j]),at_grid_edge=j in (0,len(grid)-1))
put('selected_thresholds.json',selected)
old=list(csv.DictReader((R/'outputs/morphology_minimal_20260907/per_image_measurements.csv').open(encoding='utf-8-sig')));old={(r['key'],r['model']):r for r in old}
evaluated=[]
for i,r in enumerate(rows):
    name='evaluation/'+Path(r['key']).stem+'.json';cp=P/name
    choices=selected[str(folds[r['key']])]
    if cp.exists():
        e=read(cp);assert e['choices']==choices;assert e['hashes']=={k:sha(p) for k,p in paths(r).items()}
    else:
        y,f,sy,thin,probs,h=load(r);n=int(sy.sum());nf=int(f.sum());yd=ndi.binary_dilation(y,structure=foot)
        settings={'M0_fixed':('M0',.5),'M1_fixed':('M1',.5),'M1_match_M0':('M1',choices['M0']['threshold']),'M0_match_M1':('M0',choices['M1']['threshold'])};measure={}
        for tag,(m,t) in settings.items():
            pred=(probs[m]>=t)&f;sp=skeletonize(pred,method='zhang');support=ndi.binary_dilation(pred,structure=foot);tp=int((pred&y).sum());fp=int((pred&~y).sum())
            z=dict(threshold=t,fp_fov=fp/nf,thin_recall=float(support[thin].mean()),mare=abs(int(sp.sum())-n)/n,extra=float((sp&~yd).sum()/n),precision=tp/int(pred.sum()))
            if t==.5:
                for k,oldk in [('fp_fov','fp_fov'),('mare','absolute_relative_error'),('precision','precision'),('extra','extra_pred_skeleton_per_gt_d1')]:assert abs(z[k]-float(old[r['key'],m][oldk]))<1e-12
            j=int(np.flatnonzero(grid==t)[0]);cc=curves[i]['models'][m]
            assert abs(z['fp_fov']-cc['fp_fov'][j])<1e-12 and abs(z['thin_recall']-cc['thin_recall'][j])<1e-12
            measure[tag]=z
        e=dict(key=r['key'],fold=folds[r['key']],choices=choices,hashes=h,metrics=measure);put(name,e)
    evaluated.append(e);put('progress.json',dict(stage='evaluation',completed=i+1,total=118,seconds=time.time()-START))
    if (i+1)%20==0:print('evaluation',i+1,flush=True)
rng=np.random.default_rng(20260908)
def stat(a):
    a=np.array(a);means=a[rng.integers(len(a),size=(5000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist())
summary=dict(n=118,thresholds=selected,arms={},comparisons={},uncertainty='Paired image bootstrap conditional on selected cross-fit thresholds; does not include selection variability or unknown patient clustering')
for tag in evaluated[0]['metrics']:
    summary['arms'][tag]={k:stat([e['metrics'][tag][k] for e in evaluated]) for k in ('fp_fov','thin_recall','mare','extra','precision')}
for label,a,b in [('fixed','M0_fixed','M1_fixed'),('primary','M0_fixed','M1_match_M0'),('secondary','M0_match_M1','M1_fixed')]:
    summary['comparisons'][label]={k:stat([e['metrics'][b][k]-e['metrics'][a][k] for e in evaluated]) for k in ('fp_fov','thin_recall','mare','extra','precision')}
put('summary.json',summary)
with (P/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as fh:
    writer=csv.DictWriter(fh,fieldnames=['key','fold','arm','threshold','fp_fov','thin_recall','mare','extra','precision']);writer.writeheader()
    for e in evaluated:
        for tag,z in e['metrics'].items():writer.writerow(dict(key=e['key'],fold=e['fold'],arm=tag,**z))
put('verification.json',dict(status='PASS',pairs=118,curve_checkpoints=len(list((P/'curves').glob('*.json'))),evaluation_checkpoints=len(list((P/'evaluation').glob('*.json'))),hashes_checked=True,fixed_metrics_reproduced=True,curve_metrics_crosschecked=True,seconds=time.time()-START))
print(json.dumps(summary,ensure_ascii=False),flush=True)
