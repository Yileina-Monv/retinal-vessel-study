"""Isolated, matched 32-image / 256-update C1 development pilot."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from retinal_release.common import ROOT,read,write,digest,identity,machine
from retinal_m1.data import SkeletonDataset,fingerprint
from retinal_data.fives import EpochDrawSampler
from retinal_data.prepare import stable_seed
from retinal_m0.model import UNet
from retinal_m0.state import atomic_save,state_digest,rng_state,restore_rng
from retinal_prefetch.resources import Budget
from retinal_prefetch.loader import OrderedBatches
from retinal_c1.audit import OUT
from retinal_c1.evaluation import geometry,evaluate_item,summary
from retinal_c1.geometry import select
from retinal_c1.objectives import objective


def plan():
    ds=SkeletonDataset('train',cache_size=0);keys=set(read(OUT/'protocol.json')['training_32'])
    allowed={i for i,r in enumerate(ds.rows) if r['key'] in keys};assert len(allowed)==32
    values=[];epoch=0
    while len(values)<256*4:
        values.extend((e,d) for e,d in EpochDrawSampler(ds,epoch=epoch,shuffle=True) if d//4 in allowed);epoch+=1
    return [values[i:i+4] for i in range(0,1024,4)]


class Batches(OrderedBatches):
    def _dataset(self):
        return SkeletonDataset('train',seed=2026,cache_size=1)
    def _geometry(self,key):
        return geometry(key)
    def _make(self,indices):
        if self.guard:self.guard.check()
        if not hasattr(self.local,'dataset'):
            self.local.dataset=self._dataset()
        items=[self.local.dataset[i] for i in indices];pairs=[];seeds=[];signatures=[]
        for item in items:
            seed=stable_seed('C1',2026,item['epoch'],item['draw'],item['key'])
            selected=select(self._geometry(item['key']),item['geometry'].tolist(),seed)
            h=hashlib.sha256()
            for pair in selected:
                assert item['mask'][0][tuple(pair['pos'].T)].bool().all()
                assert not item['mask'][0][tuple(pair['bg'].T)].bool().any()
                assert item['fov'][0][tuple(pair['bg'].T)].bool().all()
                h.update(str(pair['id']).encode());h.update(pair['pos'].tobytes());h.update(pair['bg'].tobytes())
            sig=fingerprint(item);sig['pairs_sha256']=h.hexdigest();sig['pairs']=len(selected)
            signatures.append(sig);pairs.append(selected);seeds.append(seed)
        tensors=tuple(torch.stack([r[k] for r in items]) for k in ('image','mask','fov','tube'))
        return tensors,pairs,seeds,signatures


def binding(schedule,loader_profile=None):
    paths=[OUT/'protocol.json',OUT/'preflight/definitions.json',ROOT/'configs/development_v1.json']
    for folder in ('retinal_c1','retinal_m0','retinal_m1','retinal_release','retinal_data','retinal_prefetch'):
        paths.extend(sorted((ROOT/folder).glob('*.py')))
    if loader_profile and loader_profile.get('loader')=='cached':
        paths.extend(ROOT/'local_deployment'/name for name in ('cached_data.py','cached_c1.py'))
    protocol=read(OUT/'protocol.json')
    for key in protocol['training_32']+protocol['validation_24']:
        p=OUT/'geometry'/(Path(key).stem+'.npz');paths.extend((p,p.with_suffix('.json')))
    payload=dict(protocol_id=protocol['protocol_id'],machine_id=machine(),gpu=torch.cuda.get_device_name(0),
                 torch=torch.__version__,cuda=torch.version.cuda,schedule_id=identity(schedule),
                 loader_profile=loader_profile,
                 files={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
                 optimization='same seed 2026 initialization; original full-index crops filtered to fixed 32; 8 epochs / 256 updates; cosine .001 to .00001 over this pilot budget',
                 authority='isolated local pilot; no migration authority or formal-study activation')
    record=dict(binding_id=identity(payload),**payload);path=OUT/'pilot/binding.json'
    if path.exists():assert read(path)==record,'Pilot binding changed; do not resume or mix results'
    else:write(path,record)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('method',choices=('M0','M1','PAIR','SHUFFLE','POOL','SCNP'))
    p.add_argument('--loader',choices=('original','cached'),default='cached')
    p.add_argument('--stop',type=int,default=256);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not 1<=a.stop<=256:raise ValueError('Bounded 256-update pilot only')
    if read(OUT/'preflight/decision.json')['status']!='PASS_PREFLIGHT':
        raise RuntimeError('C1 preflight is HOLD: training is blocked before optimizer or output creation')
    if (ROOT/'.runtime/coordination').exists():
        raise RuntimeError('Historical coordination must remain inactive for this isolated pilot')
    dest=OUT/'pilot'/a.method;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():raise ValueError('Completed run exists; do not overwrite')
    guard=Budget();torch.set_num_threads(4);guard.cuda();torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    schedule=plan();batch_class=Batches;loader_profile=dict(loader=a.loader,workers=2)
    if a.loader=='cached':
        from local_deployment.cached_c1 import CachedC1Batches
        accepted=read(ROOT/'.runtime/performance_4070_v1/acceptance.json')
        if accepted['status']!='PASS_EXACT_TRAINING_ACCELERATION':raise RuntimeError('Cached loader has not passed performance acceptance')
        for name in ('cached_data.py','cached_c1.py'):
            path=ROOT/'local_deployment'/name
            if digest(path)!=accepted['code_sha256'][path.relative_to(ROOT).as_posix()]:raise RuntimeError('Accepted cached loader source changed')
        setup=CachedC1Batches.configure(read(OUT/'protocol.json')['training_32'])
        loader_profile['cache_id']=setup['cache_id'];batch_class=CachedC1Batches
    bound=binding(schedule,loader_profile)
    # Lock exclusively for this new local experiment; historical registry remains inactive.
    lock=OUT/'pilot/active.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid())+' '+a.method)
    try:
        random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
        c=read(ROOT/'configs/development_v1.json');model=UNet(c['widths'],c['group_norm_groups']).cuda().train()
        initial=state_digest(model.state_dict())
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'],foreach=False)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024);cursor=0;history=[];signatures=[];elapsed=0.
        checkpoint=dest/'last.pt'
        if a.resume:
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert saved['binding_id']==bound['binding_id'] and saved['method']==a.method and saved['initial_state_sha256']==initial
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
            restore_rng(saved['rng']);cursor=saved['step'];history=saved['history'];signatures=saved['signatures'];elapsed=saved['training_seconds'];del saved
        elif checkpoint.exists():raise ValueError('Existing checkpoint requires explicit --resume')
        start=time.perf_counter();saved_elapsed=elapsed
        with batch_class(schedule[cursor:a.stop],workers=2,depth=4,guard=guard) as batches:
            for tensors,pairs,seeds,sig in batches:
                x,y,f,t=(v.cuda() for v in tensors)
                lr=c['minimum_learning_rate']+.5*(c['learning_rate']-c['minimum_learning_rate'])*(1+math.cos(math.pi*cursor/256))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss,parts=objective(model(x),y,f,t,pairs,a.method,seeds)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True)
                scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()<scale:raise ValueError('Skipped optimizer update')
                torch.cuda.synchronize();cursor+=1;signatures.extend(sig);guard.cuda_sample()
                history.append(dict(step=cursor,loss=float(loss.detach()),lr=lr,gradient_norm=float(norm),**parts))
                elapsed=saved_elapsed+time.perf_counter()-start
                if cursor%16==0:print(f'{a.method} step={cursor}/256 seconds={elapsed:.1f} loss={history[-1]["loss"]:.5f}',flush=True)
                if cursor%64==0 or cursor==a.stop:
                    saved=dict(binding_id=bound['binding_id'],method=a.method,initial_state_sha256=initial,
                               model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),
                               step=cursor,budget=256,history=history,signatures=signatures,training_seconds=elapsed)
                    atomic_save(saved,checkpoint);del saved
        if cursor<256:
            write(dest/'paused.json',dict(step=cursor,binding_id=bound['binding_id']));return
        final_model=state_digest(model.state_dict());model.eval();rows=[]
        if a.loader=='cached':
            from local_deployment.cached_data import Snapshot,CachedDataset
            validation_snapshot=Snapshot(read(OUT/'protocol.json')['validation_24'])
            ds=CachedDataset('validation',snapshot=validation_snapshot,full=True,cache_size=0)
        else:ds=SkeletonDataset('validation',full=True,cache_size=1)
        indices={r['key']:i for i,r in enumerate(ds.rows)}
        val_start=time.perf_counter()
        for i,row in enumerate(read(OUT/'selected_rows.json')['validation']):
            record,paths,_=evaluate_item(model,ds[indices[row['key']]])
            record['disease']=row['disease'];rows.append(record)
            write(dest/'validation'/(Path(row['key']).stem+'.json'),dict(image=record,paths=paths))
            guard.cuda_sample();print(f'{a.method} validation={i+1}/24',flush=True)
        assert state_digest(model.state_dict())==final_model,'Evaluation changed model state'
        result=dict(status='COMPLETED_BOUNDED_PILOT',method=a.method,step=cursor,binding_id=bound['binding_id'],
                    initial_state_sha256=initial,input_and_pairs_sha256=state_digest(signatures),final_model_sha256=final_model,
                    checkpoint_sha256=digest(checkpoint),training_seconds=elapsed,validation_seconds=time.perf_counter()-val_start,
                    history=history,rows=rows,summary=summary(rows),resources=guard.finish(),source_test_images_opened=0,
                    training_draws=len(signatures),training_unique_images=len({s['key'] for s in signatures}),
                    scope='single-seed small-sample directional pilot; no formal efficacy or novelty claim')
        write(dest/'result.json',result);print(f'{a.method} COMPLETED '+str(result['summary']['local']),flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
