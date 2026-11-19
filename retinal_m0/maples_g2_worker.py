"""Fixed G2 worker. Internal CPU fixtures exercise this same update loop.

No scientific training, evaluation, alternate dataset, or arbitrary GPU CLI.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time

from retinal_m0.maples_g2_engineering import (
    AUDIT_STEPS, Budget, GateError, atomic_json, canonical_sha, file_sha, read_json,
    lock, optimizer_update, require_finite_tensors, same_forward_gradient_audit)

SEGMENTS = (("R0_LAMBDA0",0,200),("B1_CONT",0,50),("B1_SPLIT",0,25),
            ("B1_SPLIT",25,50),("B2_CONT",0,50),("B2_SPLIT",0,25),("B2_SPLIT",25,50))

def exact_equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact_equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b

def durable_checkpoint(path,payload):
    import torch
    p=Path(path)
    if p.exists():raise GateError("checkpoint already exists; no overwrite")
    tmp=p.with_name(p.name+'.pending')
    with tmp.open('xb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def append_record(stream,record):
    stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())

def pair_record(batch,step):
    return {'step':step,'draws':batch['draw'].tolist(),'filenames':list(batch['filename']),
            'geometry':batch['geometry'].tolist(),'input_sha256':list(batch['input_sha256'])}

class BranchDataset:
    def __init__(self,base,method,source,manifest):
        self.base=base;self.method=method;self.source=Path(source);self.manifest=manifest
    def __len__(self):return len(self.base)
    def __getitem__(self,draw):
        import numpy as np
        import torch
        from retinal_m0.maples_branch_bce import transform_native_target
        item=self.base[draw]
        if self.method=='R0':return item
        rel=f"candidate_masks_NOT_FOR_TRAINING/{item['filename']}.npz"
        if file_sha(self.source/rel)!=self.manifest[rel]:raise GateError('native target source drift')
        g=item['geometry'].tolist()
        with np.load(self.source/rel,allow_pickle=False) as masks:
            selected=transform_native_target(masks['B1_reference' if self.method=='B1' else 'B2_candidate'],
                top=g[0],left=g[1],size=512,hflip=bool(g[2]),vflip=bool(g[3]),quarter_turns=g[4])
        if np.any(selected.astype(bool)&~(item['mask'][0].numpy().astype(bool)&item['fov'][0].numpy().astype(bool))):raise GateError('target outside GT/FOV')
        item['selected']=torch.from_numpy(selected.copy()[None]).float()
        item['target_sha256']=hashlib.sha256(selected.tobytes()).hexdigest()
        item['target_count']=int(selected.sum())
        return item

def run_segment(*,model,optimizer,scaler,dataset,track,start,end,out,budget,binding_sha,
                expected_pairs,device,authorize,clip_norm=1.0,seed=2026):
    """Production update algorithm; only fixtures supply a CPU model/dataset."""
    import torch
    from torch.utils.data import DataLoader
    from retinal_m0.maples_core import weighted_masked_loss
    from retinal_m0.maples_branch_bce import normalized_positive_bce
    from retinal_m0.state import rng_state,restore_rng
    if (track,start,end) not in SEGMENTS:raise GateError('unapproved segment')
    out=Path(out);is_base=track=='R0_LAMBDA0';cuda=str(device).startswith('cuda')
    if start==0:out.mkdir(parents=True,exist_ok=False)
    elif not out.is_dir():raise GateError('resume output missing')
    loader_rng=torch.Generator().manual_seed(seed)
    initial={};resume_loader=None
    if start:
        saved=torch.load(out/f'step_{start:06d}.pt',map_location='cpu',weights_only=True)
        if saved['binding_sha256']!=binding_sha or saved['track']!=track or saved['step']!=start or saved['next_global_draw']!=start*4:raise GateError('resume checkpoint identity drift')
        if budget.snapshot()['tracks'][track]['verified_successes']!=start:raise GateError('resume ledger/checkpoint mismatch')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng']);resume_loader=saved['loader_generator_state'];loader_rng.set_state(resume_loader)
        for key,actual in [('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scaler',scaler.state_dict()),('rng',rng_state())]:
            if not exact_equal(actual,saved[key]):raise GateError('restore mismatch: '+key)
        initial=saved['initial']
        old_rows=[json.loads(x) for x in (out/'train.jsonl').read_text().splitlines()]
        if len(old_rows)!=start or old_rows[-1]['step']!=start:raise GateError('resume log/checkpoint mismatch; no truncation')
        atomic_json(out/f'restore_{start:06d}.json',{'passed':True,'pid':os.getpid(),'previous_pid':saved['pid'],'checkpoint_sha256':file_sha(out/f'step_{start:06d}.pt')})
        if saved['pid']==os.getpid():raise GateError('split resume must use a new process')
        del saved
    else:
        from retinal_m0.state import state_digest
        initial={k:state_digest(v) for k,v in [('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scaler',scaler.state_dict()),('rng',rng_state())]}
    loader=DataLoader(dataset,batch_size=4,sampler=range(start*4,end*4),num_workers=0,
        pin_memory=cuda,drop_last=True,generator=loader_rng)
    iterator=iter(loader)
    # num_workers=0 iterator recreation consumes only the private base-seed draw.
    # Restore that private state to the checkpoint AFTER constructing the iterator.
    if resume_loader is not None:loader_rng.set_state(resume_loader)
    step=start;attempt=0;started=time.perf_counter()
    active={'step':None,'attempt':None}
    def pre_forward(*_):
        authorize();budget.before_forward(track,active['step'],active['attempt'])
    hook=model.register_forward_pre_hook(pre_forward)
    if cuda:torch.cuda.reset_peak_memory_stats(0)
    try:
        with (out/'train.jsonl').open('a' if start else 'x') as log,(out/'attempts.jsonl').open('a' if start else 'x') as attempts:
            for batch in iterator:
                next_step=step+1;pair=pair_record(batch,next_step)
                if pair!=expected_pairs[next_step-1]:raise GateError('actual T0 input/draw/geometry fingerprint mismatch')
                x,y,f,w=[batch[k].to(device,non_blocking=cuda) for k in ('image','mask','fov','weight')]
                selected=None if is_base else batch['selected'].to(device,non_blocking=cuda)
                if cuda and (x.shape!=(4,3,512,512) or y.shape!=(4,1,512,512) or f.shape!=y.shape or w.shape!=y.shape or (selected is not None and selected.shape!=y.shape)):raise GateError('production batch shape mismatch')
                attempt=0
                while True:
                    attempt+=1;active.update(step=next_step,attempt=attempt)
                    optimizer.zero_grad(set_to_none=True)
                    scale_before=float(scaler.get_scale())
                    with torch.autocast('cuda' if cuda else 'cpu',dtype=torch.float16 if cuda else torch.bfloat16,enabled=cuda):z=model(x)
                    base=weighted_masked_loss(z,y,f,w,epsilon=1e-6)
                    aux=None if is_base else .1*normalized_positive_bce(z,y,f,selected)
                    loss=base if is_base else base+aux
                    require_finite_tensors(loss)
                    diagnostic=None;audit_error=None
                    count_grad=lambda:budget.before_gradient(track,next_step,attempt)
                    if not is_base and next_step in AUDIT_STEPS:
                        try:diagnostic=same_forward_gradient_audit(model,z,base,aux,scaler=scaler,audit_step=next_step,before_gradient=count_grad)
                        except GateError as e:
                            if 'nonfinite' not in str(e):raise
                            audit_error=str(e)
                    count_grad();scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    gn=torch.nn.utils.clip_grad_norm_(model.parameters(),clip_norm,error_if_nonfinite=False)
                    finite=bool(torch.isfinite(gn))
                    record={'step':next_step,'attempt':attempt,'input':pair,'base_loss':float(base.detach()),
                        'weighted_aux_loss':None if aux is None else float(aux.detach()),'loss':float(loss.detach()),
                        'scale_before':scale_before,'gradient_norm':float(gn) if finite else None,
                        'combined_gradient_finite':finite,'audit':diagnostic,'audit_error':audit_error}
                    if not is_base:record.update(target_sha256=list(batch['target_sha256']),target_count=batch['target_count'].tolist())
                    require_finite_tensors(model.state_dict());require_finite_tensors(optimizer.state_dict())
                    if not finite:
                        # Exactly the original U-Net backoff: no scaler.step and no draw advance.
                        budget.reserve(track,next_step,attempt);scaler.update();budget.resolve(track,'amp_overflow',amp_retry=True)
                        record.update(outcome='amp_overflow',scale_after=float(scaler.get_scale()))
                        append_record(attempts,record)
                        if budget.snapshot()['status']!='READY':raise GateError('AMP overflow exceeds eight retries')
                        continue
                    if audit_error or (diagnostic and diagnostic['pause_ratio_gt_2']):
                        record['outcome']='PAUSE_BEFORE_UPDATE';append_record(attempts,record)
                        raise GateError('auxiliary gradient audit requires Sol review')
                    def validate_after():
                        if cuda:torch.cuda.synchronize(0)
                        require_finite_tensors(model.state_dict());require_finite_tensors(scaler.state_dict())
                        if head_before is not None and torch.equal(head_before,model.head.weight):raise GateError('original U-Net head did not update')
                    authorize()
                    head_before=model.head.weight.detach().clone() if hasattr(model,'head') else None
                    result=optimizer_update(budget,track,next_step,attempt,optimizer,scaler,amp_enabled=True,validate_after=validate_after)
                    if result['outcome']!='success':raise GateError('unexpected AMP skip after finite combined gradient')
                    record.update(outcome='success',scale_after=float(scaler.get_scale()))
                    append_record(attempts,record);break
                step=next_step
                row={'step':step,'pair':pair,'loss':float(loss.detach()),'amp_retries':attempt-1,'amp_scale':float(scaler.get_scale()),
                     'gradient_norm':float(gn),'audit':diagnostic}
                if not is_base:row.update(target_sha256=list(batch['target_sha256']),target_count=batch['target_count'].tolist())
                append_record(log,row)
                save_steps={1,10,25,50,100,200} if is_base else AUDIT_STEPS
                if step in save_steps:
                    payload={'schema':2,'track':track,'step':step,'next_global_draw':step*4,'binding_sha256':binding_sha,
                        'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                        'rng':rng_state(),'loader_generator_state':loader_rng.get_state(),'initial':initial,'pid':os.getpid()}
                    durable_checkpoint(out/f'step_{step:06d}.pt',payload)
                if step%25==0:print(f'{track}: {step}/{end}',flush=True)
        if step!=end:raise GateError('incomplete segment')
        result={'status':'SEGMENT_COMPLETE_FOR_REVIEW','track':track,'start':start,'end':end,'pid':os.getpid(),
                'elapsed_seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(0) if cuda else 0,
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(0) if cuda else 0,'checkpoint_sha256':file_sha(out/f'step_{end:06d}.pt')}
        atomic_json(out/f'segment_{start:03d}_{end:03d}.json',result)
        return result
    except BaseException as e:
        budget.stop(str(e))
        atomic_json(out/f'failure_{start:03d}_{end:03d}.json',{'error':str(e),'step':step,'attempt':attempt,'ledger':budget.snapshot()})
        raise
    finally:hook.remove()

def compare_checkpoint(a,b,*,loader=True,initial=True):
    keys=['model','optimizer','scaler','rng','step','next_global_draw']
    if loader:keys.append('loader_generator_state')
    if initial:keys.append('initial')
    for key in keys:
        if not exact_equal(a[key],b[key]):raise GateError('exact checkpoint comparison failed: '+key)

def validate_segment(root,index,binding_sha):
    """CPU comparison after the GPU subprocess exits. No inference."""
    import torch
    from retinal_m0 import maples_g2_runtime as rt
    root=Path(root);track,start,end=SEGMENTS[index];out=root/track
    current=torch.load(out/f'step_{end:06d}.pt',map_location='cpu',weights_only=True)
    if current['binding_sha256']!=binding_sha:raise GateError('completed checkpoint binding mismatch')
    if index==0:
        original=torch.load(rt.BASE/'step_000200.pt',map_location='cpu',weights_only=True)
        compare_checkpoint(current,original,loader=False,initial=False)
        from retinal_m0.state import state_digest
        for key in ('model','optimizer','scaler','rng'):
            if hashlib.sha256(current['initial'][key].encode()).hexdigest()!=original['initial_'+key]:raise GateError('original T0 initialization differs: '+key)
        actual_rows=[json.loads(s) for s in (out/'train.jsonl').read_text().splitlines()]
        original_rows=[json.loads(s) for s in (rt.BASE/'train.jsonl').read_text().splitlines()][:200]
        if len(actual_rows)!=200 or len(original_rows)!=200:raise GateError('R0 replay log length mismatch')
        for a,b in zip(actual_rows,original_rows):
            for key in ('step','loss','gradient_norm','amp_retries','amp_scale'):
                if a[key]!=b[key]:raise GateError('R0 original trajectory differs: '+key)
    elif track.endswith('SPLIT'):
        reference=root/track.replace('SPLIT','CONT')
        continuous=torch.load(reference/f'step_{end:06d}.pt',map_location='cpu',weights_only=True)
        compare_checkpoint(current,continuous)
        for name in ('train.jsonl','attempts.jsonl'):
            a=[json.loads(s) for s in (out/name).read_text().splitlines()]
            b=[json.loads(s) for s in (reference/name).read_text().splitlines()]
            b=[r for r in b if r['step']<=end]
            if a!=b:raise GateError('continuous/split '+name+' mismatch')
    else:
        r0=torch.load(root/'R0_LAMBDA0/step_000001.pt',map_location='cpu',weights_only=True)
        if current['initial']!=r0['initial']:raise GateError('paired initialization mismatch')
    receipt={'passed':True,'segment_index':index,'checkpoint_sha256':file_sha(out/f'step_{end:06d}.pt'),'binding_sha256':binding_sha}
    atomic_json(root/f'accept_segment_{index:02d}.json',receipt)
    return receipt

def gpu_worker(index):
    from retinal_m0 import maples_g2_runtime as rt
    # Authorization and masking precede importing torch or constructing any model.
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=rt.GPU_UUID:raise GateError('only physical GPU0 UUID may be visible')
    binding=rt.verify_release(full=True)
    launch=read_json(rt.G2OUT/'matrix_launch.json')
    token=os.environ.pop('RETINA_G2_SEGMENT_TOKEN','')
    if index not in range(len(SEGMENTS)) or launch['segment_index']!=index or not token or hashlib.sha256(token.encode()).hexdigest()!=launch['token_sha256']:raise GateError('invalid parent-bound segment request')
    if launch['parent_pid']!=os.getppid():raise GateError('segment parent identity mismatch')
    from retinal_m0.maples_g1_runtime import probe_resources
    with lock(rt.GPU_LOCK):
        rt.verify_release(full=False);resources=probe_resources(require_idle=True)
        if resources['free_disk_bytes']<32*(rt.BASE/'step_000200.pt').stat().st_size+1024**3:raise GateError('insufficient space for fixed checkpoint matrix')
        import torch
        import numpy as np
        import random
        from retinal_m0.maples_core import CorePatches
        from retinal_m0.model import UNet
        if torch.cuda.device_count()!=1:raise GateError('exactly one permitted visible device required')
        torch.cuda.set_device(0);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed(2026)
        cfg=read_json(rt.BASE/'config.json');base=CorePatches(cfg,'T0')
        arch=read_json(rt.CODE/'configs/development_v1.json')
        model=UNet(arch['widths'],arch['group_norm_groups']).cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],betas=tuple(cfg['betas']),eps=cfg['eps'],foreach=False)
        scaler=torch.amp.GradScaler('cuda',init_scale=cfg['amp_initial_scale'])
        track,start,end=SEGMENTS[index]
        if start==0:
            m0=torch.load(cfg['source_m0'],map_location='cpu',weights_only=True)
            if m0['step']!=10000:raise GateError('M0 step mismatch')
            model.load_state_dict(m0['model'],strict=True);del m0
        method='R0' if track=='R0_LAMBDA0' else track[:2]
        data=BranchDataset(base,method,rt.G1_SOURCE,read_json(rt.G1_SOURCE/'artifact_sha256.json'))
        expected=[json.loads(line) for line in (rt.BASE/'pairing.jsonl').read_text().splitlines()][:200]
        atomic_json(rt.G2OUT/f'resources_segment_{index:02d}.json',resources)
        return run_segment(model=model,optimizer=optimizer,scaler=scaler,dataset=data,track=track,start=start,end=end,
            out=rt.G2OUT/'runs'/track,budget=Budget(rt.G2_LEDGER,canonical_sha(binding)),binding_sha=canonical_sha(binding),
            expected_pairs=expected,device='cuda:0',authorize=lambda:rt.verify_release(full=False),clip_norm=cfg['clip_norm'])
