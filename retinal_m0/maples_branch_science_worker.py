"""Scientific execution uses the same update loop as CPU tests. No import-time GPU work."""
from __future__ import annotations
import hashlib,json,os,time,random
from pathlib import Path
import numpy as np
import torch
from retinal_m0 import maples_g2_engineering as old
from retinal_m0.maples_g2_engineering import (GateError,atomic_json,canonical_sha,file_sha,read_json,lock,
 optimizer_update,require_finite_tensors)
GLOBAL_CAP=4000
TRACK_CAPS={'B1':2000,'B2':2000}
AUDIT_STEPS={1,5,10,25,45,100,*range(200,2001,200)}
LAMBDA=.03
class CleanPause(RuntimeError):
    """Stop only between committed updates; no accounting reset is permitted."""
    pass

def empty_ledger(binding_sha):
    return {"schema":5,"policy":"P1_B1_B2_SCIENTIFIC_2000_V1_20261022",
      "binding_sha256":binding_sha,"global_cap":GLOBAL_CAP,"track_caps":TRACK_CAPS,
      "tracks":{},"charged_events":0,"verified_successes":0,"attempts":0,
      "amp_retries":0,"failed_events":0,"uncertain_events":0,"optimizer_events":0,
      "status":"READY"}

class BudgetLedger:
    """Write-ahead charging: unknown/failed optimizer calls are never refunded."""
    def __init__(self,path,binding_sha):
        self.path=Path(path);self.lock_path=Path(str(path)+".lock");self.binding_sha=binding_sha
    def _read(self):
        if not self.path.is_file():raise GateError("science ledger missing; refusing reset")
        s=read_json(self.path)
        if s.get("schema")!=5 or s.get("binding_sha256")!=self.binding_sha or s.get("global_cap")!=4000 or s.get("track_caps")!=TRACK_CAPS:
            raise GateError("science ledger schema/binding/budget mismatch")
        totals={k:0 for k in ("charged_events","verified_successes","attempts","amp_retries","failed_events","uncertain_events","optimizer_events")}
        if any(type(s.get(k))is not int or s[k]<0 for k in totals):raise GateError('invalid aggregate counter type/value')
        for name,t in s["tracks"].items():
            if name not in TRACK_CAPS:raise GateError("unknown science track")
            if any(type(t.get(k))is not int or t[k]<0 for k in totals):raise GateError('invalid track counter type/value')
            if t['charged_events']!=t['verified_successes']+t['failed_events']+t['uncertain_events'] or t['optimizer_events']!=t['verified_successes']+t['failed_events']:raise GateError('inconsistent charged/event totals')
            for k in totals:totals[k]+=int(t.get(k,0))
            if t["charged_events"]>TRACK_CAPS[name]:raise GateError("per-track charged cap exceeded")
        if any(s[k]!=v for k,v in totals.items()):raise GateError("science aggregate counter mismatch")
        if s["charged_events"]>4000:raise GateError("global charged cap exceeded")
        if sum(t.get('pending') is not None for t in s['tracks'].values())>1:raise GateError('multiple unresolved optimizer invocations')
        return s
    def initialize_isolated(self):
        with lock(self.lock_path):
            if self.path.exists():raise GateError("ledger already exists; no implicit reset")
            atomic_json(self.path,empty_ledger(self.binding_sha))
    def snapshot(self):
        with lock(self.lock_path):return self._read()
    def _mutate(self,fn):
        with lock(self.lock_path):
            s=self._read();fn(s);atomic_json(self.path,s);self._read()
    def reserve(self,track,planned_step,attempt):
        def f(s):
            if s["status"]!="READY":raise GateError("ledger is not runnable")
            if track not in TRACK_CAPS:raise GateError("unknown track")
            t=s["tracks"].setdefault(track,{"charged_events":0,"verified_successes":0,"attempts":0,"amp_retries":0,"failed_events":0,"uncertain_events":0,"optimizer_events":0,"pending":None,"retry_cursor":None})
            if any(v.get("pending") is not None for v in s["tracks"].values()):raise GateError("unresolved prior optimizer invocation")
            if s.get("require_compute") and s.get("compute_pending")!={"track":track,"step":planned_step,"attempt":attempt}:raise GateError("optimizer requires its counted forward")
            if t["charged_events"]>=TRACK_CAPS[track] or s["charged_events"]>=4000:raise GateError("science budget exhausted")
            if type(planned_step)is not int or type(attempt)is not int or attempt<1 or attempt>9:raise GateError("invalid planned step/attempt")
            cursor=t.get("retry_cursor")
            if cursor is not None and (planned_step!=cursor["planned_step"] or attempt!=cursor["next_attempt"]):raise GateError("AMP retry must repeat the identical planned draw with next attempt number")
            if cursor is None and attempt!=1:raise GateError("new planned draw must start at attempt 1")
            if cursor is None and planned_step!=t["verified_successes"]+1:raise GateError("planned updates must be unique and strictly ordered")
            t["attempts"]+=1;s["attempts"]+=1
            t["pending"]={"planned_step":planned_step,"attempt":attempt,"reserved_at":time.time(),"outcome":"UNKNOWN"}
            atomic_json(self.path,s) # durable write-ahead before optimizer.step
        self._mutate(f)
    def resolve(self,track,outcome,*,amp_retry=False):
        if outcome not in ("success","failed","uncertain","amp_overflow"):raise GateError("bad optimizer outcome")
        def f(s):
            t=s["tracks"].get(track)
            if not t or t["pending"] is None:raise GateError("no write-ahead optimizer reservation")
            p=t["pending"];t["pending"]=None
            s["compute_pending"]=None
            if outcome=="success":
                t["retry_cursor"]=None
                t["charged_events"]+=1;t["verified_successes"]+=1;t["optimizer_events"]+=1
                for k in ("charged_events","verified_successes","optimizer_events"):s[k]+=1
            elif outcome=="failed":
                t["retry_cursor"]=None
                t["charged_events"]+=1;t["failed_events"]+=1;t["optimizer_events"]+=1
                for k in ("charged_events","failed_events","optimizer_events"):s[k]+=1
                s["status"]="FAILED_STOP"
            elif outcome=="uncertain":
                t["retry_cursor"]=None
                t["charged_events"]+=1;t["uncertain_events"]+=1
                s["charged_events"]+=1;s["uncertain_events"]+=1;s["status"]="UNCERTAIN_STOP"
            elif outcome=="amp_overflow":
                if not amp_retry:raise GateError("AMP overflow must be explicitly recorded as retry")
                # No optimizer invocation and no charged update. Enforce same draw on next attempt.
                if p["attempt"]>=9:
                    t["retry_cursor"]=None;s["status"]="FAILED_STOP"
                else:
                    t["retry_cursor"]={"planned_step":p["planned_step"],"next_attempt":p["attempt"]+1}
                t["amp_retries"]+=1;s["amp_retries"]+=1
            if t["charged_events"]>TRACK_CAPS[track] or s["charged_events"]>4000:raise GateError("budget exceeded")
        self._mutate(f)
    def mark_crash_uncertain(self,track):
        self.resolve(track,"uncertain")

    def stop(self,reason):
        def f(s):
            s["status"]="FAILED_STOP";s["stop_reason"]=str(reason)
        self._mutate(f)

    def before_forward(self,track,step,attempt):
        def f(s):
            if s["status"]!="READY" or s.get("compute_pending") is not None or any(t.get("pending") for t in s["tracks"].values()):raise GateError("unresolved compute/update or stopped ledger")
            t=s["tracks"].get(track,{"verified_successes":0,"charged_events":0,"retry_cursor":None})
            if track not in TRACK_CAPS or type(step)is not int or type(attempt)is not int or not 1<=attempt<=9:raise GateError("invalid compute identity")
            if step!=t["verified_successes"]+1 or t["charged_events"]>=TRACK_CAPS[track] or s["charged_events"]>=4000:raise GateError("compute exceeds remaining budget")
            cursor=t.get("retry_cursor")
            if (cursor is None and attempt!=1) or (cursor is not None and cursor!={"planned_step":step,"next_attempt":attempt}):raise GateError("compute retry drift")
            if s.get("forward_calls",0)>=36000:raise GateError("forward budget exhausted")
            s["require_compute"]=True;s["compute_pending"]={"track":track,"step":step,"attempt":attempt}
            s["forward_calls"]=s.get("forward_calls",0)+1;s["current_gradient_calls"]=0
        self._mutate(f)

    def before_gradient(self,track,step,attempt):
        def f(s):
            if s["status"]!="READY" or s.get("compute_pending")!={"track":track,"step":step,"attempt":attempt}:raise GateError("gradient without counted forward")
            limit=3 if track!="R0_LAMBDA0" and step in AUDIT_STEPS else 1
            if s.get("current_gradient_calls",0)>=limit or s.get("gradient_calls",0)>=36576:raise GateError("gradient budget exhausted")
            s["current_gradient_calls"]=s.get("current_gradient_calls",0)+1;s["gradient_calls"]=s.get("gradient_calls",0)+1
        self._mutate(f)

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

def save_last(path,payload):
    import tempfile
    import torch
    p=Path(path);fd,tmp=tempfile.mkstemp(prefix=p.name+'.',dir=p.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
        fd=os.open(p.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

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

def gradient_audit(model,logits,base_loss,weighted_aux_loss,*,scaler,audit_step,before_gradient=lambda:None):
    import math
    import torch
    if audit_step not in AUDIT_STEPS:raise GateError('unscheduled V4 audit')
    ps=tuple(p for p in model.parameters() if p.requires_grad)
    rng=old._rng_snapshot(logits.device);model_hash=old._tensor_digest(model);scale_state=scaler.state_dict()
    grads=[None if p.grad is None else p.grad.clone() for p in ps];scale=float(scaler.get_scale())
    def get(loss):
        before_gradient()
        return tuple(g.detach().float()/scale for g in torch.autograd.grad(scaler.scale(loss),ps,retain_graph=True,allow_unused=False))
    a,b=get(base_loss),get(weighted_aux_loss)
    n0=math.sqrt(sum(float(g.double().square().sum()) for g in a));n1=math.sqrt(sum(float(g.double().square().sum()) for g in b))
    dot=sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b))
    ratio=n1/n0 if n0 else None;c=dot/(n0*n1) if n0 and n1 else None
    result={'audit_step':audit_step,'base_grad_l2':n0,'weighted_aux_grad_l2':n1,'aux_to_base_l2':ratio,'cosine':c,
        'pause_ratio_gt_2':bool((ratio is not None and ratio>2) or (n0==0 and n1>0)),
        'zero_base_positive_aux':bool(n0==0 and n1>0),'negative_cosine':bool(c is not None and c<0),'unscaled':True,'autograd_calls':2}
    old._tree_finite(result)
    if old._tensor_digest(model)!=model_hash or scaler.state_dict()!=scale_state or not old._rng_equal(rng,old._rng_snapshot(logits.device)):raise GateError('V4 audit state side effect')
    if any((g is None)!=(p.grad is None) or (g is not None and not torch.equal(g,p.grad)) for g,p in zip(grads,ps)):raise GateError('V4 audit changed p.grad')
    return result

def load_committed_checkpoint(out,track,start,budget,binding_sha,*,allow_later_arm=False):
    """One CPU-only identity/state/log gate, including completed-arm/evaluation resume."""
    from retinal_m0.state import state_digest
    out=Path(out);saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
    if saved.get('schema')!=5 or saved.get('binding_sha256')!=binding_sha or saved.get('track')!=track or saved.get('step')!=start or saved.get('next_global_draw')!=start*4 or saved.get('lambda')!=LAMBDA:raise GateError('resume checkpoint identity drift')
    live=budget.snapshot();prior=saved.get('budget',{})
    if live['tracks'].get(track,{}).get('verified_successes')!=start:raise GateError('resume ledger/checkpoint mismatch')
    if allow_later_arm:
        # A completed B1 checkpoint predates B2. Never weaken B1 state/log checks.
        if prior.get('binding_sha256')!=binding_sha or prior.get('status')!='READY' or prior.get('compute_pending') or any(t.get('pending') or t.get('retry_cursor') for t in prior.get('tracks',{}).values()):raise GateError('unsafe historical committed budget')
        if prior.get('tracks',{}).get(track)!=live['tracks'].get(track):raise GateError('completed arm counters drift')
        for name,t in prior['tracks'].items():
            if any(t.get(k,0)>live['tracks'].get(name,{}).get(k,0) for k in ('charged_events','verified_successes','attempts','amp_retries','failed_events','uncertain_events','optimizer_events')):raise GateError('historical budget ahead of live budget')
    elif prior!=live:raise GateError('resume full durable ledger/checkpoint mismatch')
    for name in ('train','attempts'):
        if saved.get(name+'_log_sha256')!=file_sha(out/(name+'.jsonl')):raise GateError('resume complete log prefix mismatch:'+name)
    rows=[json.loads(x) for x in (out/'train.jsonl').read_text().splitlines()]
    if len(rows)!=start or [r['step'] for r in rows]!=list(range(1,start+1)):raise GateError('resume log/checkpoint mismatch; no truncation')
    for key in ('model','optimizer','scaler','rng'):
        if state_digest(saved[key])!=rows[-1]['state_digests'][key]:raise GateError('resume checkpoint/log digest mismatch: '+key)
    for key in ('model','optimizer','scaler'):require_finite_tensors(saved[key])
    gen=torch.Generator();gen.set_state(saved['loader_generator_state'])
    if not exact_equal(gen.get_state(),saved['loader_generator_state']):raise GateError('invalid private loader state')
    milestone=out/f'step_{start:06d}.pt'
    if start%200==0 and not milestone.exists():raise GateError('committed milestone missing; no automatic reconstruction')
    if milestone.exists() and not exact_equal(saved,torch.load(milestone,map_location='cpu',weights_only=True)):raise GateError('last/milestone full states differ')
    return saved


def run_segment(*,model,optimizer,scaler,dataset,track,start,end,out,budget,binding_sha,
                expected_pairs,device,authorize,clip_norm=1.0,seed=2026,expected_targets=None):
    """Production update algorithm; only fixtures supply a CPU model/dataset."""
    import torch
    from torch.utils.data import DataLoader
    from retinal_m0.maples_core import weighted_masked_loss
    from retinal_m0.maples_branch_bce import normalized_positive_bce
    from retinal_m0.state import rng_state,restore_rng
    if track not in TRACK_CAPS or not (0<=start<end<=2000):raise GateError('unapproved science segment')
    out=Path(out);is_base=track=='R0_LAMBDA0';cuda=str(device).startswith('cuda')
    if start==0:out.mkdir(parents=True,exist_ok=False)
    elif not out.is_dir():raise GateError('resume output missing')
    loader_rng=torch.Generator().manual_seed(seed)
    initial={};resume_loader=None
    if start:
        saved=load_committed_checkpoint(out,track,start,budget,binding_sha)
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng']);resume_loader=saved['loader_generator_state'];loader_rng.set_state(resume_loader)
        for key,actual in [('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scaler',scaler.state_dict()),('rng',rng_state())]:
            if not exact_equal(actual,saved[key]):raise GateError('restore mismatch: '+key)
        initial=saved['initial']
        atomic_json(out/f'restore_{start:06d}.json',{'passed':True,'pid':os.getpid(),'previous_pid':saved['pid'],'checkpoint_sha256':file_sha(out/'last.pt')})
        # Live same-process milestone segments and external exact resumes share restoration checks.
        del saved
    else:
        from retinal_m0.state import state_digest
        initial={k:state_digest(v) for k,v in [('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scaler',scaler.state_dict()),('rng',rng_state())]}
    model.train()
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
                if expected_targets is not None:
                    rr=expected_targets[(next_step-1)*4:next_step*4]
                    if list(batch['target_sha256'])!=[r['target_sha256'][track] for r in rr] or batch['target_count'].tolist()!=[r[track+'_count'] for r in rr]:raise GateError('actual selected target fingerprint drift')
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
                    aux=None if is_base else LAMBDA*normalized_positive_bce(z,y,f,selected)
                    loss=base if is_base else base+aux
                    require_finite_tensors(loss)
                    bce=torch.nn.functional.binary_cross_entropy_with_logits(z.float(),y.float(),reduction='none')
                    bce=float(((bce*f).sum((1,2,3))/f.sum((1,2,3))).mean().detach())
                    diagnostic=None;audit_error=None
                    count_grad=lambda:budget.before_gradient(track,next_step,attempt)
                    if not is_base and next_step in AUDIT_STEPS:
                        try:diagnostic=gradient_audit(model,z,base,aux,scaler=scaler,audit_step=next_step,before_gradient=count_grad)
                        except GateError as e:
                            if 'nonfinite' not in str(e):raise
                            audit_error=str(e)
                    count_grad();scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    gn=torch.nn.utils.clip_grad_norm_(model.parameters(),clip_norm,error_if_nonfinite=False)
                    finite=bool(torch.isfinite(gn))
                    record={'step':next_step,'attempt':attempt,'input':pair,'base_loss':float(base.detach()),'base_bce':bce,'base_dice':float(base.detach())-bce,
                        'weighted_aux_loss':None if aux is None else float(aux.detach()),'unweighted_aux_loss':None if aux is None else float(aux.detach())/LAMBDA,'loss':float(loss.detach()),
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
                from retinal_m0.state import state_digest
                states={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state()}
                row['state_digests']={k:state_digest(v) for k,v in states.items()}
                payload={'schema':5,'track':track,'step':step,'next_global_draw':step*4,'binding_sha256':binding_sha,
                    **states,'loader_generator_state':loader_rng.get_state(),'initial':initial,'pid':os.getpid(),'lambda':LAMBDA,'budget':budget.snapshot()}
                append_record(log,row)
                payload['train_log_sha256']=file_sha(out/'train.jsonl')
                payload['attempts_log_sha256']=file_sha(out/'attempts.jsonl')
                save_last(out/'last.pt',payload)
                save_steps=set(range(200,2001,200)) if cuda else (AUDIT_STEPS|{end})
                if step in save_steps:
                    durable_checkpoint(out/f'step_{step:06d}.pt',payload)
                if step%25==0:print(f'{track}: {step}/{end}',flush=True)
        if step!=end:raise GateError('incomplete segment')
        result={'status':'SEGMENT_COMPLETE_FOR_REVIEW','track':track,'start':start,'end':end,'pid':os.getpid(),
                'elapsed_seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(0) if cuda else 0,
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(0) if cuda else 0,'checkpoint_sha256':file_sha(out/f'step_{end:06d}.pt')}
        atomic_json(out/f'segment_{start:03d}_{end:03d}.json',result)
        return result
    except BaseException as e:
        if not (isinstance(e,CleanPause) and budget.snapshot().get('compute_pending') is None):budget.stop(str(e))
        atomic_json(out/f'failure_{start:03d}_{end:03d}.json',{'error':str(e),'step':step,'attempt':attempt,'ledger':budget.snapshot(),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(0) if cuda else 0,'peak_reserved_bytes':torch.cuda.max_memory_reserved(0) if cuda else 0})
        raise
    finally:hook.remove()


atomic_checkpoint=save_last

def _fixture_components(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    model=torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.Tanh(),torch.nn.Conv2d(4,1,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False)
    scaler=torch.amp.GradScaler('cpu',init_scale=1024)
    return model,optimizer,scaler

class FixtureDataset:
    def __init__(self,method='B1',seed=2026):self.method=method;self.seed=seed
    def __len__(self):return 8000
    def __getitem__(self,draw):
        gen=torch.Generator().manual_seed(self.seed*10000+draw)
        x=torch.randn(3,8,8,generator=gen);y=(torch.rand(1,8,8,generator=gen)<.3).float();f=torch.ones_like(y)
        s=y.clone()
        if self.method=='B1':s[:,:,1::2]=0
        if draw%7==0:s.zero_()
        geom=torch.tensor([0,0,0,0,0],dtype=torch.int64)
        h=hashlib.sha256()
        for k,v in [('image',x),('mask',y),('fov',f),('geometry',geom)]:
            a=v.numpy();h.update(f'{k}:{a.dtype}:{a.shape}'.encode());h.update(a.tobytes())
        return dict(image=x,mask=y,fov=f,weight=f,selected=s,geometry=geom,filename='fixture_'+str(draw%5),
          draw=draw,input_sha256=h.hexdigest(),target_sha256=hashlib.sha256(s.numpy().astype(np.uint8).tobytes()).hexdigest(),target_count=int(s.sum()))

def fixture_pairs(data,end):
    from torch.utils.data import DataLoader
    gen=torch.Generator().manual_seed(999)
    return [pair_record(b,i+1) for i,b in enumerate(DataLoader(data,batch_size=4,sampler=range(end*4),num_workers=0,generator=gen))]

def run_fixture_segment(out,*,start,end,seed=2026,method='B1'):
    if not 0<=start<end<=45:raise ValueError('fixture only1..45')
    out=Path(out);model,opt,scaler=_fixture_components(seed)
    budget=BudgetLedger(out.parent/(out.name+'.ledger.json'),'CPU_FIXTURE_ONLY')
    if start==0:budget.initialize_isolated()
    data=FixtureDataset(method,seed);pairs=fixture_pairs(data,45)
    run_segment(model=model,optimizer=opt,scaler=scaler,dataset=data,track=method,start=start,end=end,
      out=out,budget=budget,binding_sha='CPU_FIXTURE_ONLY',expected_pairs=pairs,device='cpu',authorize=lambda:None)
    return torch.load(out/'last.pt',map_location='cpu',weights_only=True)

def run_scientific_training(method,out):
    from retinal_m0 import maples_branch_science as rt
    binding=rt.verify_active_session()
    if Path(out).resolve()!=rt.OUT.resolve() or method not in TRACK_CAPS:raise GateError('wrong science output/method')
    cfg=read_json(rt.OUT/'configs'/(method+'.json'));ledger=BudgetLedger(rt.OUT/'science_budget.json',canonical_sha(binding))
    snap=ledger.snapshot();track=snap['tracks'].get(method,{})
    if snap['status']!='READY' or snap.get('compute_pending') or any(t.get('pending') for t in snap['tracks'].values()):raise GateError('unsafe prior state')
    start=track.get('verified_successes',0)
    if method=='B2' and snap['tracks'].get('B1',{}).get('verified_successes')!=2000:raise GateError('B1 must complete before B2')
    saved=load_committed_checkpoint(rt.OUT/'runs'/method,method,start,ledger,canonical_sha(binding),allow_later_arm=start==2000) if start else None
    for completed in range(200,start,200):
        if not rt.verify_completed_evaluation(method,completed):raise GateError('earlier milestone evaluation missing')
    if start==2000 and rt.verify_completed_evaluation(method,2000):
        return {'status':'ALREADY_COMPLETED_NO_NEW_UPDATES','method':method}
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed(2026)
    from retinal_m0.maples_core import CorePatches
    from retinal_m0.model import UNet
    base=CorePatches(cfg,'T0');arch=read_json(rt.CODE/'configs/development_v1.json')
    model=UNet(arch['widths'],arch['group_norm_groups']).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],betas=tuple(cfg['betas']),eps=cfg['eps'],foreach=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=cfg['amp_initial_scale'])
    if start==0:
        m0=torch.load(cfg['source_m0'],map_location='cpu',weights_only=True)
        if m0['step']!=10000:raise GateError('M0 step mismatch')
        model.load_state_dict(m0['model'],strict=True)
        from retinal_m0.state import rng_state,state_digest
        original=read_json(rt.A0/'summary.json')
        for k,v in [('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scaler',scaler.state_dict()),('rng',rng_state())]:
            if hashlib.sha256(state_digest(v).encode()).hexdigest()!=original['initial_'+k]:raise GateError('fresh original initialization mismatch:'+k)
        del m0
    if start:
        model.load_state_dict(saved['model'],strict=True)
        if start%200==0:rt.evaluate_checkpoint(model,method,start,binding)
        del saved
    if start==2000:return {'status':'ALREADY_COMPLETED_NO_NEW_UPDATES','method':method}
    data=BranchDataset(base,method,rt.SOURCE,read_json(rt.SOURCE/'artifact_sha256.json'))
    pairs=[json.loads(x) for x in (rt.A0/'pairing.jsonl').read_text().splitlines() if x.strip()]
    targets=read_json(rt.OUT/'pairing_8000_verification.json')['rows']
    # Segments end only at predeclared200 milestones; evaluations never select a model.
    result=[]
    for end in range((start//200+1)*200,2001,200):
        step_start=start
        result.append(run_segment(model=model,optimizer=optimizer,scaler=scaler,dataset=data,track=method,
          start=step_start,end=end,out=rt.OUT/'runs'/method,budget=ledger,binding_sha=canonical_sha(binding),
          expected_pairs=pairs,expected_targets=targets,device='cuda:0',authorize=rt.verify_active_session,clip_norm=cfg['clip_norm']))
        rt.evaluate_checkpoint(model,method,end,binding)
        start=end
    return {'status':'SCIENCE_ARM_COMPLETE','method':method,'step':start,'segments':result}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--fixture-output',type=Path,required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=45)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--method',choices=['B1','B2'],default='B1')
    a=p.parse_args();run_fixture_segment(a.fixture_output,start=a.start,end=a.end,seed=a.seed,method=a.method)

if __name__=='__main__':main()
