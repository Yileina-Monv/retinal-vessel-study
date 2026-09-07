"""Full frozen development-set review, with a single 120-minute supervised GPU budget."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np
from scipy import ndimage as ndi
import torch
from retinal_release.common import ROOT,read,write,digest,identity
from retinal_data.fives import read_manifest
from retinal_c1.audit import OUT as PREVIOUS,one
from retinal_c1.preflight import CHECKPOINTS
from retinal_c1.evaluation import geometry,with_chain_ownership,local,summary
from retinal_m1.metrics import score
from retinal_m1.inference import probability
from retinal_m0.model import UNet
from retinal_prefetch.resources import Budget
from local_deployment.cached_data import CACHE,prepare_one,Snapshot,CachedDataset

OUT=ROOT/'.runtime/c1_validation118_v1'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=read_manifest(ROOT/'releases/development_v1/validation.csv')
    train=read_manifest(ROOT/'releases/development_v1/train.csv')
    assert len(rows)==118 and not ({r['key'] for r in rows}&{r['key'] for r in train})
    sources=[ROOT/'releases/development_v1/validation.csv',PREVIOUS/'protocol.json',PREVIOUS/'diagnostics/definitions.json']
    for folder in ('retinal_c1','retinal_m0','retinal_m1','retinal_release','retinal_data','local_deployment'):
        sources.extend(sorted((ROOT/folder).glob('*.py')))
    protocol=dict(scope='exploratory expansion of prior 24-image review to the complete frozen 118-image development cohort',
                  keys=[r['key'] for r in rows],previous_24=read(PREVIOUS/'protocol.json')['validation_24'],
                  per_path_conflict_threshold=.05,primary_macro_image_conflict_threshold=.10,primary_model='M0',
                  unchanged_geometry=True,inference_batch=2,seed=2026,threshold=.5,thin_radius=4,
                  outlier_policy='retain all images; report worst-image and leave-one-image-out sensitivity descriptively',
                  gpu_budget_seconds=7200,soft_stop_seconds=6900,worker_emergency_exit_seconds=7100,
                  CPU_preparation_outside_GPU_budget=True,optimizer_updates=0,source_test_access=False,
                  source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sources})
    protocol=dict(protocol_id=identity(protocol),**protocol)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:write(OUT/'protocol.json',protocol)
    start=time.perf_counter();maps={r['key']:r for r in read(ROOT/'releases/m1_pretrial_v1/maps.json')['records']}
    old_manifest=read(CACHE/'manifest.json')
    if not (OUT/'cache_manifest_before.json').exists():write(OUT/'cache_manifest_before.json',old_manifest)
    records={r['key']:r for r in old_manifest['records']};geometry_records=[]
    def work(row):return prepare_one(row,maps),one(row,maps)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(r,g) in enumerate(pool.map(work,rows)):
            if r['key'] in records:assert records[r['key']]==r
            records[r['key']]=r;geometry_records.append(g)
            if (i+1)%20==0:print(f'CPU preparation {i+1}/118 seconds={time.perf_counter()-start:.1f}',flush=True)
    payload={k:v for k,v in old_manifest.items() if k not in ('records','cache_id')}
    payload['records']=list(records.values());new_manifest=dict(cache_id=identity(payload),**payload)
    write(CACHE/'manifest.json',new_manifest)
    write(OUT/'preparation.json',dict(seconds=time.perf_counter()-start,geometry_records=geometry_records,
          cache_id_before=old_manifest['cache_id'],cache_id_after=new_manifest['cache_id'],
          existing_cache_records_unchanged=True,cached_images=len(records),
          cache_bytes=sum((ROOT/v['path']).stat().st_size for r in records.values() for v in r['files'].values())))
    print('CPU preparation complete',flush=True)


def worker():
    start=time.perf_counter();watchdog=threading.Timer(7100,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
    guard=Budget();protocol=read(OUT/'protocol.json')
    for path,sha in protocol['source_sha256'].items():
        if digest(ROOT/path)!=sha:raise ValueError('Review source version changed')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    snap=Snapshot(protocol['keys']);ds=CachedDataset('validation',snapshot=snap,full=True,cache_size=0)
    guard.cuda();config=read(ROOT/'configs/development_v1.json')
    write(OUT/'runtime.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
          fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
          cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cache_verification_seconds=snap.verification_seconds,cache_id=snap.cache_id))
    for name,(folder,sha) in CHECKPOINTS.items():
        destination=OUT/name;destination.mkdir(exist_ok=True)
        checkpoint=ROOT/'runs/prefetch_round1'/folder/'last.pt'
        if digest(checkpoint)!=sha:raise ValueError('Frozen checkpoint changed')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True);assert saved['step']==10000
        model=UNet(config['widths'],config['group_norm_groups']);model.load_state_dict(saved['model']);del saved
        model.cuda().eval();results=[]
        for i,row in enumerate(ds.rows):
            if time.perf_counter()-start>6900:raise TimeoutError('Soft deadline: preserve already-written image results')
            guard.check();key=row['key'];stem=Path(key).stem;path=destination/(stem+'.json')
            if path.exists():raise ValueError('Existing image result must not be overwritten')
            item=ds[i];g=with_chain_ownership(geometry(key),item['skeleton'][0].numpy(),item['mask'][0].numpy())
            p,tiles=probability(model,item['image'],item['fov'][0],batch_size=2);p=p.numpy()
            old_match=None
            if key in protocol['previous_24']:
                oldpath=PREVIOUS/'diagnostics'/(name+'_'+stem+'_probability.npy')
                oldrecord=read(PREVIOUS/'diagnostics'/(name+'_'+stem+'.json'))['image']
                if digest(oldpath)!=oldrecord['probability_sha256']:raise ValueError('Previous probability artifact changed')
                old=np.load(oldpath,allow_pickle=False);old_match=bool(np.array_equal(p,old))
                if not old_match:raise ValueError('Overlapping 24-image probabilities changed')
            loc,paths=local(p,g,key)
            thin=item['thin'][0].numpy().astype(bool);fov=item['fov'][0].numpy().astype(bool);label=item['mask'][0].numpy().astype(bool)
            ordinary=score(p,label,fov,thin)
            pred=p>=.5;distance=ndi.distance_transform_edt(~pred) if pred.any() else np.full(p.shape,np.inf)
            missed=thin&(distance>1);covered=np.zeros(p.shape,bool)
            q=g['pos'][np.diff(g['bg_offsets'])>=16].reshape(-1,2);covered[tuple(q.T)]=True
            record=dict(key=key,disease=row['disease'],quality=row['quality_class'],tiles=tiles,**ordinary,local=loc,
                        thin_missed=int(missed.sum()),thin_missed_on_eligible_paths=int((missed&covered).sum()),
                        missed_thin_coverage=float((missed&covered).sum()/missed.sum()) if missed.any() else None,
                        previous_24_probability_exact=old_match,conflict_path_count=sum(x['rank_conflict'] for x in paths))
            probpath=destination/(stem+'_probability.npy')
            with probpath.open('xb') as f:np.save(f,p,allow_pickle=False)
            record['probability_sha256']=digest(probpath)
            write(path,dict(protocol_id=protocol['protocol_id'],checkpoint_sha256=sha,image=record,paths=paths));results.append(record)
            guard.cuda_sample()
            if (i+1)%10==0:print(f'{name} {i+1}/118 elapsed_minutes={(time.perf_counter()-start)/60:.2f}',flush=True)
        write(destination/'summary.json',dict(rows=results,summary=summary(results),checkpoint_sha256=sha))
        del model;torch.cuda.empty_cache()
    write(OUT/'worker_complete.json',dict(status='COMPLETED_236_IMAGE_INFERENCES',seconds=time.perf_counter()-start,
          resources=guard.finish(),optimizer_updates=0,overlapping_48_probability_arrays_exact=True))
    watchdog.cancel()


def supervise():
    if (OUT/'budget.json').exists():raise ValueError('One budget per review; do not silently restart it')
    started=time.perf_counter();write(OUT/'budget.json',dict(status='RUNNING',limit_seconds=7200))
    process=subprocess.Popen([sys.executable,'-m','retinal_c1.review118','worker'],cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
    timed_out=False
    try:code=process.wait(timeout=7200)
    except subprocess.TimeoutExpired:
        timed_out=True;process.kill();code=process.wait(timeout=10)
    elapsed=time.perf_counter()-started
    write(OUT/'budget.json',dict(status='COMPLETED' if code==0 else 'TIMEOUT' if timed_out else 'FAILED',
          limit_seconds=7200,elapsed_seconds=elapsed,returncode=code,hard_timeout=timed_out,
          accounting='single worker entire lifetime, includes startup, inference, CPU metrics and saving; CPU preparation excluded'))
    if code:raise RuntimeError(f'Review worker stopped with code {code}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','worker'));args=p.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':supervise()
    else:worker()
