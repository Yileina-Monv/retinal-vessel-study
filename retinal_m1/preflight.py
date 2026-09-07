"""No optimizer.step, no run ticket, no validation model score selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import csv
import subprocess
import sys
import time
import numpy as np
import torch
from PIL import Image
from retinal_release.common import ROOT,read,write,digest,safe
from retinal_release.verify import verify as verify_parent
from retinal_m0.model import UNet
from retinal_m0.state import state_digest
from retinal_m1.maps import DEST,interior
from retinal_m1.data import SkeletonDataset,fingerprint,transform
from retinal_m1.loss import m1_loss

def main():
    started=time.perf_counter();parent=verify_parent()
    tests=[]
    for name in ('test_release.py','test_batch_study.py','test_m1_pretrial.py'):
        done=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',name,'-v'],cwd=ROOT,capture_output=True,text=True)
        tests.append(dict(suite=name,returncode=done.returncode,output=done.stdout+done.stderr))
        print(f'{name}: {done.returncode}',flush=True)
        if done.returncode:raise ValueError(tests[-1]['output'])
    maps=read(ROOT/DEST/'maps.json');calibration=read(ROOT/DEST/'calibration.json')
    if maps['parent_release_id']!=parent['release_id']:raise ValueError('Wrong map parent')
    with (ROOT/'releases/development_v1/all.csv').open(encoding='utf-8',newline='') as handle:
        rows={r['key']:r for r in csv.DictReader(handle)}
    if len(maps['records'])!=600 or {r['key'] for r in maps['records']}!=set(rows):raise ValueError('Map cohort mismatch')
    counts={};radii=[];bytes_total=0
    for r in maps['records']:
        source=rows[r['key']];path=safe(ROOT,r['path'])
        if digest(path)!=r['sha256'] or any(r[k]!=source[k] for k in ('label_sha256','fov_sha256')):raise ValueError('Map binding mismatch')
        with np.load(path,allow_pickle=False) as a:
            skeleton=a['skeleton'];tube=a['tube'];radius=a['skeleton_radius'];tr=a['tube_radius']
            if not skeleton.shape==tube.shape==radius.shape==tr.shape==tuple(r['shape']):raise ValueError('Map shape mismatch')
            if skeleton.dtype!=bool or tube.dtype!=bool or radius.dtype!=np.float32 or tr.dtype!=np.float32:raise ValueError('Map dtype mismatch')
            if np.any(skeleton & ~tube) or np.any(radius[~skeleton]) or np.any(tr[~tube]) or np.any(radius[skeleton]<=0) or np.any(tr[tube]<=0):raise ValueError('Invalid auxiliary support')
            if not np.isfinite(radius).all() or not np.isfinite(tr).all():raise ValueError('Nonfinite radius')
            if source['trial_split']=='train':radii.append(radius[skeleton])
        counts[r['split']]=counts.get(r['split'],0)+1;bytes_total+=path.stat().st_size
    actual=np.quantile(np.concatenate(radii),[.25,.5],method='linear')
    np.testing.assert_array_equal(actual,[calibration['tau'],calibration['r_ref']])
    ds=SkeletonDataset('train',cache_size=4)
    crop_records=[]
    for index in ((0,0),(0,7),(0,128),(1,0),(1,127),(1,len(ds)-1)):
        item=ds[index]
        if torch.any(item['tube']>item['mask']*item['fov']) or torch.any(item['thin']>item['skeleton']):raise ValueError('Real crop transform mismatch')
        crop_records.append(fingerprint(item))
    # Exercise exactly the planned 4 x 512 x 512 profile, without updating weights.
    torch.set_num_threads(4);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    c=read(ROOT/'configs/development_v1.json');model=UNet(c['widths'],c['group_norm_groups']).cuda().train()
    initial=state_digest(model.state_dict());items=[ds[(0,i)] for i in (0,7,128,129)]
    x,y,f,t=[torch.stack([i[k] for i in items]).cuda() for k in ('image','mask','fov','tube')]
    gpu=[]
    for coefficient in (0.,.1,1.):
        model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.float16):loss,parts=m1_loss(model(x),y,f,t,coefficient=coefficient)
        (loss*1024).backward();torch.cuda.synchronize()
        if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite AMP loss/gradients')
        gpu.append(dict(coefficient=coefficient,loss=float(loss.detach()),pixel=float(parts['pixel']),skeleton=float(parts['skeleton']),
            empty_skeleton_crops=parts['empty_skeleton_crops'],allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            reserved_gib=torch.cuda.max_memory_reserved()/2**30,seconds=time.perf_counter()-start))
    if state_digest(model.state_dict())!=initial:raise ValueError('Preflight modified weights')
    paths=list((ROOT/'retinal_m1').glob('*.py'))+[ROOT/'configs/m1_pretrial_v1.json',ROOT/'tests/test_m1_pretrial.py']
    acceptance=dict(status='passed',parent_release_id=parent['release_id'],optimizer_steps=0,new_training_tickets=0,
        tests=tests,maps_by_split=counts,map_bytes=bytes_total,calibration=calibration,real_crop_checks=crop_records,
        gpu_name=torch.cuda.get_device_name(),gpu_profiles=gpu,model_weights_unchanged=True,
        boundary_skeleton_pixels=sum(r['boundary_skeleton_pixels'] for r in maps['records']),
        total_skeleton_pixels=sum(r['skeleton_pixels'] for r in maps['records']),
        implementation_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        elapsed_seconds=time.perf_counter()-started,cross_gpu_equivalence='not_established',
        runner_optimizer_and_resume_execution='deferred_to_first_development_trial')
    write(ROOT/DEST/'acceptance.json',acceptance);print({k:acceptance[k] for k in ('status','optimizer_steps','maps_by_split','gpu_profiles')},flush=True)

if __name__=='__main__':main()
