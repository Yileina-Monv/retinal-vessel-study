"""Bounded, read-only-input audit of frozen development probability maps."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
USER_START = datetime.fromisoformat('2026-09-07T14:09:56+00:00').timestamp()
HARD_DEADLINE = USER_START + 3480  # reserve 120 seconds before user's 60-min cap
DATA_DEADLINE = USER_START + 3000  # reserve analysis/report time
SOURCE = ROOT / '.runtime/c1_validation118_v1'
MANIFEST = ROOT / 'releases/development_v1/validation.csv'
CHECKPOINTS = {
    'M0': ('m0_seed2026_dev10k_w1', '8bce5b8f47c7fe43d553d4467adc46f5387eade014265b7a6a6f4b5211e67d79'),
    'M1': ('m1_lambda01_seed2026_dev10k_w1', '74b2aab9afc7d14702508914f49deaf63b33b411671a6140a8c78c4dd8eafd56'),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(name, obj):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def table(name, rows):
    if not rows:
        return
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    if time.time() >= HARD_DEADLINE:
        raise TimeoutError('Absolute session deadline reached; no reset permitted')
    watchdog = threading.Timer(HARD_DEADLINE - time.time(), lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    import numpy as np
    import scipy
    from scipy import ndimage as ndi
    import skimage
    from skimage.morphology import skeletonize
    from PIL import Image

    rows = list(csv.DictReader(MANIFEST.open(encoding='utf-8-sig', newline='')))
    train = list(csv.DictReader((ROOT / 'releases/development_v1/train.csv').open(encoding='utf-8-sig', newline='')))
    previous = read(SOURCE / 'protocol.json')
    archived = read(ROOT / 'releases/local_review_2026-09-07/c1_validation118_v1/protocol.json')
    assert previous == archived
    assert len(rows) == 118 and len({r['key'] for r in rows}) == 118
    assert {r['key'] for r in rows} == set(previous['keys'])
    assert not ({r['key'] for r in rows} & {r['key'] for r in train})
    assert sha(MANIFEST) == previous['source_sha256']['releases/development_v1/validation.csv']
    assert all(r['source_split'] == 'train' and r['trial_split'] == 'validation'
               and r['group_size'] == '1' and r['label_is_empty'] == '0' for r in rows)
    assert len({r['group_id'] for r in rows}) == len(rows)
    # Round-robin groups: partial completion cannot be an alphabetical disease prefix.
    groups = {d: sorted([r for r in rows if r['disease'] == d], key=lambda r:r['key']) for d in 'ADGN'}
    ordered = [groups[d][i] for i in range(max(map(len, groups.values()))) for d in 'ADGN' if i < len(groups[d])]
    protocol = dict(
        status='FROZEN_BEFORE_NEW_MEASUREMENTS', created_utc=datetime.now(timezone.utc).isoformat(),
        user_start_utc=datetime.fromtimestamp(USER_START, timezone.utc).isoformat(), user_limit_seconds=3600,
        worker_absolute_stop_utc=datetime.fromtimestamp(HARD_DEADLINE, timezone.utc).isoformat(),
        data_absolute_stop_utc=datetime.fromtimestamp(DATA_DEADLINE, timezone.utc).isoformat(),
        scope='Exploratory minimal audit, not the full proposed morphology_discovery_v1 protocol',
        keys=[r['key'] for r in ordered], source_protocol_id=previous['protocol_id'], threshold=.5,
        skeleton='skimage.morphology.skeletonize(method=zhang) applied after fixed FOV clipping',
        primary='image macro mean abs(pred_skeleton_count - GT_skeleton_count) / GT_skeleton_count',
        secondary=['signed relative skeleton error', 'area density error', 'disease-vs-N difference of signed density errors'],
        spatial='GT skeleton unsupported by prediction mask; predicted skeleton unsupported by GT mask; Euclidean pixel tolerance 0,1,2; not an additive count-error decomposition or false-bridge test',
        high_dice_subset='M0 Dice >= 0.90, same selected images used for M1',
        uncertainty='5000 image bootstrap resamples; seed 20260907; paired by image for M1-M0; unadjusted descriptive 95% subgroup intervals',
        decision='Apply report proposed RQ1 5% point / 3% lower-CI trigger only as exploratory next-step suggestion; no training approval. Otherwise HOLD unless upper CI <3% and subgroup question resolved.',
        no_new_training=True, no_new_inference=True, official_test_access=False,
        omissions=['threshold search', 'weighted graph length', 'ICC', 'path identity/false bridges', 'physical diameter', 'clinical validation', 'new seeds', 'independent human metric acceptance'],
        software=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__),
        script_sha256=sha(Path(__file__)), manifest_sha256=sha(MANIFEST),
    )
    if (OUT / 'protocol.json').exists():
        raise ValueError('Existing run must not be silently restarted or overwritten')
    write('protocol.json', protocol)
    provenance = {'sources':[], 'checkpoints':{}, 'source_protocol_sha256':sha(SOURCE/'protocol.json')}
    for model, (folder, expected) in CHECKPOINTS.items():
        path = ROOT / 'runs/prefetch_round1' / folder / 'last.pt'
        actual = sha(path)
        assert actual == expected
        provenance['checkpoints'][model] = dict(path=str(path.relative_to(ROOT)), sha256=actual)

    def skeleton(a):
        return skeletonize(np.asarray(a, dtype=bool), method='zhang')

    def basics(y, p, f):
        y = y & f
        p = p & f
        sy, sp = skeleton(y), skeleton(p)
        ny, ns, npix, nsp = map(int, (y.sum(), sy.sum(), p.sum(), sp.sum()))
        nf = int(f.sum())
        if not nf:
            raise ValueError('Empty FOV')
        tp = int((p & y).sum())
        fp, fn = npix-tp, ny-tp
        assert npix-ny == fp-fn
        return dict(fov_pixels=nf,gt_area=ny,pred_area=npix,gt_skeleton=ns,pred_skeleton=nsp,
                    gt_density=ns/nf,pred_density=nsp/nf,signed_density_error=(nsp-ns)/nf,
                    signed_relative_error=(nsp-ns)/ns if ns else None,
                    absolute_relative_error=abs(nsp-ns)/ns if ns else None,
                    gt_area_density=ny/nf,pred_area_density=npix/nf,
                    signed_area_density_error=(npix-ny)/nf,absolute_area_density_error=abs(npix-ny)/nf,
                    area_relative_error=(npix-ny)/ny if ny else None,
                    dice=2*tp/(npix+ny) if npix+ny else None,
                    precision=tp/npix if npix else None,recall=tp/ny if ny else None,
                    fp=fp,fn=fn,fp_fov=fp/nf), sy, sp

    f = np.ones((33,33),bool)
    line = np.zeros_like(f); line[16,8:25] = True
    diag = np.zeros_like(f); diag[np.arange(8,25),np.arange(8,25)] = True
    assert skeleton(line).sum() == 17 and skeleton(diag).sum() == 17
    same,_,_ = basics(line,line,f)
    assert same['absolute_relative_error'] == 0 and same['dice'] == 1
    gap=line.copy(); gap[16,16]=False
    assert basics(line,gap,f)[0]['signed_relative_error'] == -1/17
    extra=line.copy(); extra[5,5:10]=True
    assert basics(line,extra,f)[0]['signed_relative_error'] == 5/17
    empty=np.zeros_like(f)
    assert basics(empty,empty,f)[0]['absolute_relative_error'] is None
    assert basics(line,empty,f)[0]['absolute_relative_error'] == 1
    fat=ndi.binary_dilation(line)
    assert basics(line,fat,f)[0]['signed_area_density_error'] > 0
    clipped=f.copy(); clipped[:,:12]=False
    assert basics(line,line,clipped)[0]['gt_area'] == 13
    try:
        basics(line,line,empty)
        raise AssertionError('Empty FOV should fail')
    except ValueError:
        pass
    write('synthetic_checks.json',dict(status='PASS',checks=['horizontal 17','diagonal 17','identity','one-pixel gap','isolated extra branch','empty reference undefined','empty prediction total miss','thickening area increase','FOV clipping','empty FOV rejection'],scope='basic measurement semantics only, not full graph geometry validation'))
    print('Preflight and synthetic checks PASS; starting 118 paired images',flush=True)

    results=[]
    structures={r:(np.indices((2*r+1,2*r+1))-r)**2 for r in (1,2)}
    structures={r:v.sum(axis=0)<=r*r for r,v in structures.items()}
    started=time.time()
    max_metric_diff=0.
    for i,row in enumerate(ordered):
        if time.time() >= DATA_DEADLINE:
            break
        stem=Path(row['key']).stem
        source_record={'key':row['key'],'files':{}}
        arrays={}
        for role in ('label','fov'):
            path=(ROOT/row[role+'_path']).resolve()
            assert path.is_relative_to(ROOT.resolve())
            actual=sha(path)
            assert actual == row[role+'_sha256']
            source_record['files'][role]={'path':str(path.relative_to(ROOT)),'sha256':actual}
            with Image.open(path) as im:
                a=np.asarray(im)
                if role == 'label' and a.ndim == 3:
                    assert np.all(a[:,:,:3] == a[:,:,:1])
                    if a.shape[2] == 4: assert np.all(a[:,:,3] == 255)
                    a=a[:,:,0]
                assert a.ndim == 2 and np.all((a==0)|(a==255))
                arrays[role]=a>0
        y,f=arrays['label'],arrays['fov']
        assert y.shape == f.shape == (int(row['height']),int(row['width']))
        assert int(y.sum()) == int(row['foreground_pixels'])
        y=y&f
        gt_dil={0:y,**{r:ndi.binary_dilation(y,structure=s) for r,s in structures.items()}}
        image_results=[]
        for model,(_,expected_ckpt) in CHECKPOINTS.items():
            path=SOURCE/model/(stem+'_probability.npy')
            meta_path=SOURCE/model/(stem+'.json')
            meta=read(meta_path)
            old=meta['image']
            assert meta['checkpoint_sha256'] == expected_ckpt and meta['protocol_id'] == previous['protocol_id']
            assert old['key'] == row['key']
            actual=sha(path)
            assert actual == old['probability_sha256']
            p=np.load(path,allow_pickle=False)
            assert p.shape == y.shape and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
            pred=(p>=.5)&f
            m,sy,sp=basics(y,pred,f)
            assert m['gt_skeleton']>0
            for k in ('dice','precision','recall'):
                diff=abs(m[k]-old[k]); max_metric_diff=max(max_metric_diff,diff)
                assert diff < 1e-12, (row['key'],model,k,diff)
            assert m['fov_pixels']==old['fov_pixels'] and m['gt_area']==old['label_pixels'] and m['pred_area']==old['predicted_pixels']
            tprec=float(y[sp].mean()) if sp.any() else 0.
            tsens=float(pred[sy].mean())
            cl=2*tprec*tsens/(tprec+tsens) if tprec+tsens else 0.
            assert abs(cl-old['cldice'])<1e-12
            for r in (0,1,2):
                support=pred if r==0 else ndi.binary_dilation(pred,structure=structures[r])
                m[f'missing_gt_skeleton_fraction_d{r}']=float((sy&~support).sum()/sy.sum())
                m[f'extra_pred_skeleton_count_d{r}']=int((sp&~gt_dil[r]).sum())
                m[f'extra_pred_skeleton_per_gt_d{r}']=m[f'extra_pred_skeleton_count_d{r}']/m['gt_skeleton']
            record=dict(key=row['key'],model=model,disease=row['disease'],IC=int(row['IC']),Blur=int(row['Blur']),LC=int(row['LC']),quality_class=row['quality_class'],threshold=.5,**m,average_precision_archived=old['average_precision'])
            image_results.append(record)
            source_record['files'][model]={'path':str(path.relative_to(ROOT)),'sha256':actual,'record_sha256':sha(meta_path)}
        results.extend(image_results)
        provenance['sources'].append(source_record)
        # Flush pairs so interruption never leaves an apparently complete cohort.
        with (OUT/'measurements_partial.jsonl').open('a',encoding='utf-8') as out:
            for r in image_results:out.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        if (i+1)%10==0 or i==0:
            print(f'Paired images {i+1}/118, data seconds={time.time()-started:.1f}',flush=True)
    if not results:
        raise RuntimeError('No complete image pairs within deadline')
    table('per_image_measurements.csv',results)
    write('provenance.json',provenance)
    write('verification.json',dict(status='PASS',paired_images=len(results)//2,probability_hashes_verified=len(results),raw_label_and_fov_hashes_verified=len(results),checkpoints_verified=2,max_dice_precision_recall_difference=max_metric_diff,cldice_reproduced=True,area_FP_minus_FN_identity=True,official_test_access=False))
    rng=np.random.default_rng(20260907)

    def ci(a):
        a=np.asarray(a,dtype=float)
        means=a[rng.integers(0,len(a),(5000,len(a)))].mean(axis=1)
        return [float(v) for v in np.quantile(means,[.025,.975])]

    def describe(rr):
        keys=('absolute_relative_error','signed_relative_error','signed_density_error','absolute_area_density_error','area_relative_error','dice','precision','fp_fov','missing_gt_skeleton_fraction_d1','extra_pred_skeleton_per_gt_d1')
        out={'n':len(rr)}
        for k in keys:
            a=np.array([r[k] for r in rr],float)
            out[k]={'mean':float(a.mean()),'ci95':ci(a),'median':float(np.median(a)),'p90':float(np.quantile(a,.9))}
        return out

    bymodel={m:[r for r in results if r['model']==m] for m in CHECKPOINTS}
    assert [r['key'] for r in bymodel['M0']]==[r['key'] for r in bymodel['M1']]
    paired=[]
    for a,b in zip(bymodel['M0'],bymodel['M1']):
        paired.append(dict(key=a['key'],disease=a['disease'],M0_error=a['absolute_relative_error'],M1_error=b['absolute_relative_error'],delta_error=b['absolute_relative_error']-a['absolute_relative_error'],delta_dice=b['dice']-a['dice'],delta_area_relative=b['area_relative_error']-a['area_relative_error'],delta_missing_d1=b['missing_gt_skeleton_fraction_d1']-a['missing_gt_skeleton_fraction_d1'],delta_extra_d1=b['extra_pred_skeleton_per_gt_d1']-a['extra_pred_skeleton_per_gt_d1']))
    table('paired_errors.csv',paired)
    summary={'status':'COMPLETE_118_PAIRS' if len(paired)==118 else 'PARTIAL_DEADLINE', 'models':{m:describe(rr) for m,rr in bymodel.items()},'paired':{k:{'mean':float(np.mean([r[k] for r in paired])),'ci95':ci([r[k] for r in paired])} for k in ('delta_error','delta_dice','delta_area_relative','delta_missing_d1','delta_extra_d1')},'images_improved':sum(r['delta_error']<0 for r in paired),'images_worsened':sum(r['delta_error']>0 for r in paired),'data_processing_seconds':time.time()-started,'elapsed_since_user_start_seconds':time.time()-USER_START,'new_training':0,'new_inference':0,'GPU_seconds':0}
    high={r['key'] for r in bymodel['M0'] if r['dice']>=.9}
    summary['high_M0_dice_subset']={m:describe([r for r in rr if r['key'] in high]) for m,rr in bymodel.items()} if high else {}
    summary['disease']={m:{d:describe([r for r in rr if r['disease']==d]) for d in 'ADGN' if any(r['disease']==d for r in rr)} for m,rr in bymodel.items()}
    summary['quality']={m:{q:{str(v):describe([r for r in rr if r[q]==v]) for v in (0,1) if any(r[q]==v for r in rr)} for q in ('IC','Blur','LC')} for m,rr in bymodel.items()}
    contrasts=[]
    for model,rr in bymodel.items():
        normal=[r for r in rr if r['disease']=='N']
        if not normal:continue
        for disease in 'ADG':
            other=[r for r in rr if r['disease']==disease]
            if not other:continue
            e1=np.array([r['signed_density_error'] for r in other])
            e0=np.array([r['signed_density_error'] for r in normal])
            samples=e1[rng.integers(0,len(e1),(5000,len(e1)))].mean(axis=1)-e0[rng.integers(0,len(e0),(5000,len(e0)))].mean(axis=1)
            gt=np.mean([r['gt_density'] for r in other])-np.mean([r['gt_density'] for r in normal])
            ai=np.mean([r['pred_density'] for r in other])-np.mean([r['pred_density'] for r in normal])
            assert abs((ai-gt)-(e1.mean()-e0.mean()))<1e-14
            contrasts.append(dict(model=model,disease=disease,normal_n=len(normal),disease_n=len(other),GT_group_difference=float(gt),AI_group_difference=float(ai),bias_difference=float(e1.mean()-e0.mean()),ci95_low=float(np.quantile(samples,.025)),ci95_high=float(np.quantile(samples,.975)),interval='unadjusted descriptive; no causal attribution'))
    table('disease_contrasts.csv',contrasts)
    counts=[]
    for d in 'ADGN':
        for ic in (0,1):
            for blur in (0,1):
                for lc in (0,1):
                    counts.append(dict(disease=d,IC=ic,Blur=blur,LC=lc,n=sum(r['disease']==d and r['IC']==ic and r['Blur']==blur and r['LC']==lc for r in bymodel['M0'])))
    table('subgroup_counts.csv',counts)
    a=np.array([r['absolute_relative_error'] for r in bymodel['M0']])
    summary['M0_leave_one_image_out_MARE_range']=[float((a.sum()-a.max())/(len(a)-1)),float((a.sum()-a.min())/(len(a)-1))] if len(a)>1 else None
    gate=summary['models']['M0']['absolute_relative_error']
    summary['decision']='GO_TO_THRESHOLD_CONTROL_ONLY' if len(paired)==118 and gate['mean']>=.05 and gate['ci95'][0]>.03 else 'HOLD'
    summary['decision_note']='Exploratory application of proposed technical gate; no clinical threshold, no new training approved, no threshold optimized.'
    write('summary.json',summary)

    os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for model,color in (('M0','#3465a4'),('M1','#d95f02')):
        rr=bymodel[model]
        axs[0].scatter([r['gt_density']*100 for r in rr],[r['pred_density']*100 for r in rr],s=17,alpha=.6,label=model,color=color)
        axs[1].scatter([r['dice'] for r in rr],[r['absolute_relative_error']*100 for r in rr],s=17,alpha=.6,label=model,color=color)
    lim=axs[0].get_xlim();axs[0].plot(lim,lim,'k--',lw=1)
    axs[0].set(xlabel='GT skeleton density (%)',ylabel='AI skeleton density (%)',title='Does AI measure the same amount?')
    axs[1].set(xlabel='Dice',ylabel='Absolute relative skeleton error (%)',title='Pixel overlap vs measurement error')
    x=np.arange(4)
    for model,offset,color in (('M0',-.18,'#3465a4'),('M1',.18,'#d95f02')):
        vals=[summary['disease'][model][d]['signed_relative_error']['mean']*100 for d in 'ADGN']
        axs[2].bar(x+offset,vals,.36,label=model,color=color)
    axs[2].axhline(0,color='black',lw=1);axs[2].set(xticks=x,xticklabels=list('ADGN'),ylabel='Mean signed relative error (%)',title='Disease groups (descriptive)')
    for ax in axs:ax.legend();ax.grid(alpha=.15)
    fig.savefig(OUT/'overview.png',dpi=160);plt.close(fig)
    print(json.dumps({'decision':summary['decision'],'M0_MARE':summary['models']['M0']['absolute_relative_error'],'M1_MARE':summary['models']['M1']['absolute_relative_error'],'paired':summary['paired'],'elapsed_minutes':(time.time()-USER_START)/60},ensure_ascii=False),flush=True)
    watchdog.cancel()


if __name__=='__main__':
    if '--worker' in sys.argv:
        try:main()
        except Exception:
            (OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            raise
    else:
        if (OUT/'budget.json').exists():raise RuntimeError('Budget already used; do not restart')
        remaining=HARD_DEADLINE-time.time()
        if remaining<=0:raise TimeoutError('Session deadline expired')
        write('budget.json',dict(status='RUNNING',user_limit_seconds=3600,user_start_epoch=USER_START,absolute_worker_deadline_epoch=HARD_DEADLINE))
        with (OUT/'run.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            timed_out=False
            try:code=child.wait(timeout=max(1,HARD_DEADLINE-time.time()))
            except subprocess.TimeoutExpired:
                timed_out=True;child.kill();code=child.wait(timeout=10)
        write('budget.json',dict(status='COMPLETED' if code==0 else 'TIMEOUT' if timed_out else 'FAILED',returncode=code,hard_timeout=timed_out,user_limit_seconds=3600,elapsed_since_user_start_seconds=time.time()-USER_START,official_test_access=False,new_training=0,new_inference=0))
        print((OUT/'budget.json').read_text(encoding='utf-8'))
        sys.exit(code)
