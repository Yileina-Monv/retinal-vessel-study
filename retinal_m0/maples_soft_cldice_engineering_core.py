"""Shared candidate-C engineering loop: production GPU and tiny CPU fixtures.

Hard limits are 50 CONT + 50 SPLIT charged updates. No scientific training CLI.
"""
from __future__ import annotations
import contextlib,os,time,math,random,json
from pathlib import Path
import numpy as np
import torch
from retinal_m0.maples_g1_safety import (GateError,atomic_json,exclusive_lock,read_json,file_sha,finite_tree)
from retinal_m0.maples_g2_engineering import optimizer_update,require_finite_tensors
from retinal_m0.state import state_digest
from retinal_m0.maples_core import weighted_masked_loss
from retinal_m0.maples_soft_cldice import soft_cldice_terms

TRACK_CAPS={'CONT':50,'SPLIT':50}

def empty_budget(binding):
    return {'schema':1,'binding_sha256':binding,'global_cap':100,'track_caps':TRACK_CAPS,'status':'READY',
            'charged_events':0,'verified_successes':0,'failed_events':0,'uncertain_events':0,
            'optimizer_call_starts':0,'forward_calls':0,'gradient_calls':0,'failed_compute':0,
            'tracks':{},'compute_pending':None,'gradient_calls_this_step':0}

class Budget:
    def __init__(self,path,binding):self.path=Path(path);self.binding=binding
    def _read(self):
        s=read_json(self.path)
        if s.get('schema')!=1 or s.get('binding_sha256')!=self.binding or s.get('global_cap')!=100 or s.get('track_caps')!=TRACK_CAPS:raise GateError('engineering budget identity/cap drift')
        totals={k:0 for k in ['charged_events','verified_successes','failed_events','uncertain_events','optimizer_call_starts']}
        for n,t in s['tracks'].items():
            if n not in TRACK_CAPS or t['charged_events']!=t['verified_successes']+t['failed_events']+t['uncertain_events'] or t['charged_events']>50:raise GateError('invalid track accounting')
            for k in totals:
                if type(t[k])is not int or t[k]<0:raise GateError('invalid track counter')
                totals[k]+=t[k]
        if any(s[k]!=v for k,v in totals.items()) or s['charged_events']>100 or s['forward_calls']>100 or s['gradient_calls']>300:raise GateError('aggregate budget mismatch/exhausted')
        return s
    def snapshot(self):
        with exclusive_lock(str(self.path)+'.lock'):return self._read()
    def _change(self,fn):
        with exclusive_lock(str(self.path)+'.lock'):
            s=self._read();fn(s);atomic_json(self.path,s);self._read()
    def initialize_fixture(self):
        if self.path.exists():raise GateError('budget exists; no reset')
        atomic_json(self.path,empty_budget(self.binding))
    def before_forward(self,track,step):
        def f(s):
            if s['status']!='READY' or s['compute_pending'] is not None or any(t.get('pending') for t in s['tracks'].values()):raise GateError('stopped/unresolved engineering work')
            if track not in TRACK_CAPS or type(step)is not int:raise GateError('unknown engineering trajectory')
            t=s['tracks'].setdefault(track,{'charged_events':0,'verified_successes':0,'failed_events':0,'uncertain_events':0,'optimizer_call_starts':0,'pending':None})
            if step!=t['verified_successes']+1 or t['charged_events']>=50 or s['charged_events']>=100 or s['forward_calls']>=100:raise GateError('nonsequential or exhausted update')
            s['forward_calls']+=1;s['compute_pending']={'track':track,'step':step};s['gradient_calls_this_step']=0
        self._change(f)
    def before_gradient(self,track,step):
        def f(s):
            if s['status']!='READY' or s['compute_pending']!={'track':track,'step':step} or s['gradient_calls_this_step']>=3 or s['gradient_calls']>=300:raise GateError('extra/unbound gradient call')
            s['gradient_calls']+=1;s['gradient_calls_this_step']+=1
        self._change(f)
    def reserve(self,track,step,attempt):
        def f(s):
            if attempt!=1 or s['status']!='READY' or s['compute_pending']!={'track':track,'step':step} or s['gradient_calls_this_step']!=3:raise GateError('optimizer without complete audited forward')
            t=s['tracks'][track]
            if t['pending'] is not None or t['charged_events']>=50 or s['charged_events']>=100:raise GateError('optimizer pending/cap')
            t['pending']={'step':step,'attempt':1,'started':False,'reserved_at':time.time()}
        self._change(f)
    def before_optimizer(self,track,step):
        def f(s):
            t=s['tracks'].get(track);p=None if t is None else t.get('pending')
            if s['status']!='READY' or p is None or p['step']!=step or p['started']:raise GateError('unauthorized/duplicate optimizer invocation')
            p['started']=True;t['optimizer_call_starts']+=1;s['optimizer_call_starts']+=1
        self._change(f)
    def resolve(self,track,outcome,*,amp_retry=False):
        def f(s):
            t=s['tracks'][track];p=t['pending']
            if p is None:raise GateError('no reserved optimizer outcome')
            if outcome=='success' and not p['started']:raise GateError('success without actual optimizer invocation')
            if outcome in ['success','failed','uncertain']:
                k={'success':'verified_successes','failed':'failed_events','uncertain':'uncertain_events'}[outcome]
                t[k]+=1;s[k]+=1;t['charged_events']+=1;s['charged_events']+=1
            elif outcome=='amp_overflow':
                s['failed_compute']+=1;s['amp_overflow']=s.get('amp_overflow',0)+1
            else:raise GateError('unknown optimizer outcome')
            t['last_outcome']={'outcome':outcome,'pending_record':p};t['pending']=None;s['compute_pending']=None
            if outcome!='success':s['status']='FAILED_STOP'
        self._change(f)
    def stop(self,reason):
        def f(s):
            if s['compute_pending'] is not None and s['status']!='FAILED_STOP':s['failed_compute']+=1
            s.setdefault('first_stop_reason',str(reason))
            s['status']='FAILED_STOP';s['stop_reason']=str(reason)
        self._change(f)

def rng_state(device):
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),*n[2:]],'torch':torch.get_rng_state(),
            'cuda0':torch.cuda.get_rng_state(0) if str(device).startswith('cuda') else None}
def restore_rng(state,device):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),*n[2:]));torch.set_rng_state(state['torch'].cpu())
    if str(device).startswith('cuda'):
        if state['cuda0'] is None:raise GateError('missing GPU0 RNG')
        torch.cuda.set_rng_state(state['cuda0'].cpu(),0)

def atomic_checkpoint(path,payload,*,replace):
    p=Path(path)
    if not replace and p.exists():raise GateError('immutable milestone exists')
    tmp=p.with_name(p.name+'.pending')
    with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def append(stream,row):
    stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
def pair_record(batch,step):
    return {'step':step,'draws':batch['draw'].tolist(),'filenames':list(batch['filename']),
            'geometry':batch['geometry'].tolist(),'input_sha256':list(batch['input_sha256'])}
def norm(gs):return math.sqrt(sum(float(g.double().square().sum().cpu()) for g in gs))

def audit(model,base,aux,scaler,budget,track,step,device):
    ps=tuple(model.parameters());before_rng=state_digest(rng_state(device));before_scaler=scaler.state_dict()
    scale=float(scaler.get_scale())
    def get(loss):
        budget.before_gradient(track,step)
        g=tuple(x.detach().float()/scale for x in torch.autograd.grad(scaler.scale(loss),ps,retain_graph=True,allow_unused=False))
        require_finite_tensors(g);return g
    gb,ga=get(base),get(aux);n0,n1=norm(gb),norm(ga)
    dot=sum(float((a.double()*b.double()).sum().cpu()) for a,b in zip(gb,ga))
    r={'A0_L2':n0,'weighted_aux_L2':n1,'aux_to_A0':n1/n0 if n0 else None,'cosine':dot/(n0*n1) if n0 and n1 else None,
       'pause_ratio_gt2':bool((n0==0 and n1>0) or (n0 and n1/n0>2)),'unscaled':True,'preclip':True}
    finite_tree(r)
    if any(p.grad is not None for p in ps) or state_digest(rng_state(device))!=before_rng or scaler.state_dict()!=before_scaler:raise GateError('gradient audit has side effect')
    return r

def run_segment(*,model,optimizer,scaler,dataset,track,start,end,out,budget,binding_sha,
                expected_pairs,device,eta,k,authorize=lambda:None):
    from torch.utils.data import DataLoader
    cuda=str(device).startswith('cuda')
    if track not in TRACK_CAPS or (track,start,end) not in [('CONT',0,50),('SPLIT',0,25),('SPLIT',25,50)]:raise GateError('only the approved 50 / 25+25 segments are allowed')
    if cuda:
        from retinal_m0.maples_soft_cldice_engineering import worker_authority
        authorize=lambda:worker_authority(track,start,end)
        cfg=authorize()
        if eta!=cfg['eta'] or k!=cfg['k']:raise GateError('accepted coefficient or k drift')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    loader_rng=torch.Generator().manual_seed(2026);resume_gen=None
    if start==0:out.mkdir(exist_ok=False)
    else:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if saved['track']!=track or saved['step']!=start or saved['next_draw']!=start*4 or saved['binding_sha256']!=binding_sha or saved['eta']!=eta or saved['k']!=k:raise GateError('resume checkpoint identity drift')
        if cuda and saved['pid']==os.getpid():raise GateError('production resume requires a new worker process')
        if saved['budget']!=budget.snapshot() or saved['train_sha256']!=file_sha(out/'train.jsonl') or saved['attempts_sha256']!=file_sha(out/'attempts.jsonl'):raise GateError('resume ledger/log prefix mismatch')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);restore_rng(saved['rng'],device)
        for key,value in [('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scaler',scaler.state_dict()),('rng',rng_state(device))]:
            if state_digest(value)!=saved['digests'][key]:raise GateError('restored state mismatch:'+key)
        resume_gen=saved['loader_generator'];loader_rng.set_state(resume_gen)
        atomic_json(out/f'restore_{start:03d}.json',{'passed':True,'new_pid':os.getpid(),'checkpoint_pid':saved['pid'],'checkpoint_sha256':file_sha(out/'last.pt'),'ledger_exact':True,'full_log_prefix_exact':True,'source':'planned new-process restore; not a replay'})
        del saved
    model.train();loader=DataLoader(dataset,batch_size=4,sampler=range(start*4,end*4),num_workers=0,pin_memory=cuda,drop_last=True,generator=loader_rng)
    iterator=iter(loader)
    if resume_gen is not None:loader_rng.set_state(resume_gen)
    active={'step':None}
    def pre_forward(*_):authorize();budget.before_forward(track,active['step'])
    def pre_optimizer(*_):authorize();budget.before_optimizer(track,active['step'])
    fh=model.register_forward_pre_hook(pre_forward);oh=optimizer.register_step_pre_hook(pre_optimizer)
    if cuda:torch.cuda.reset_peak_memory_stats(0)
    try:
        with (out/'train.jsonl').open('a' if start else 'x') as log,(out/'attempts.jsonl').open('a' if start else 'x') as attempts:
            step=start
            for batch in iterator:
                active['step']=step+1;pair=pair_record(batch,step+1)
                if pair!=expected_pairs[step]:raise GateError('actual input sampling/augmentation differs from original T0')
                optimizer.zero_grad(set_to_none=True)
                x,y,f=[batch[n].to(device) for n in ['image','mask','fov']]
                if cuda and (x.shape!=(4,3,512,512) or y.shape!=(4,1,512,512) or f.shape!=y.shape):raise GateError('production native crop/batch shape drift')
                with torch.autocast('cuda' if cuda else 'cpu',dtype=torch.float16 if cuda else torch.bfloat16,enabled=cuda):z=model(x)
                base=weighted_masked_loss(z,y,f,f);terms=soft_cldice_terms(z,y,f,k=k);aux=eta*terms['loss'];loss=base+aux
                require_finite_tensors(loss)
                check=audit(model,base,aux,scaler,budget,track,step+1,device)
                record={'step':step+1,'track':track,'pid':os.getpid(),'pair':pair,'A0_loss':float(base.detach()),'raw_aux_loss':float(terms['loss'].detach()),'weighted_aux_loss':float(aux.detach()),'loss':float(loss.detach()),'audit':check,'loss_scale_before':float(scaler.get_scale()),'GT_count':terms['GT_count'].detach().cpu().tolist(),'GT_skeleton_mass':terms['GT_skeleton_mass'].detach().cpu().tolist(),'FOV_count':f.sum((1,2,3)).detach().cpu().tolist()}
                append(attempts,record)
                if check['pause_ratio_gt2']:raise GateError('aux/A0 ratio>2; stop before parameter update')
                budget.before_gradient(track,step+1);scaler.scale(loss).backward();scaler.unscale_(optimizer)
                require_finite_tensors([p.grad for p in model.parameters() if p.grad is not None])
                clip=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                record['preclip_total_L2']=float(clip)
                outcome=optimizer_update(budget,track,step+1,1,optimizer,scaler,amp_enabled=cuda)
                if outcome['outcome']!='success':raise GateError('AMP overflow/step skipped: preserve counts and stop, no automatic retry')
                step+=1;optimizer.zero_grad(set_to_none=True)
                r=rng_state(device)
                digests={name:state_digest(value) for name,value in [('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scaler',scaler.state_dict()),('rng',r),('loader_generator',loader_rng.get_state()),('sampling',{'next_draw':step*4,'seed':2026})]}
                record.update(state_digests=digests,loss_scale_after=float(scaler.get_scale()),GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated(0)/1048576 if cuda else 0.,GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved(0)/1048576 if cuda else 0.)
                finite_tree(record);append(log,record)
                payload={'schema':1,'NOT_FOR_SCIENTIFIC_INITIALIZATION':True,'track':track,'step':step,'next_draw':step*4,'pid':os.getpid(),'binding_sha256':binding_sha,'eta':eta,'k':k,
                         'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'scheduler':None,'rng':r,'loader_generator':loader_rng.get_state(),'digests':digests,'budget':budget.snapshot(),
                         'train_sha256':file_sha(out/'train.jsonl'),'attempts_sha256':file_sha(out/'attempts.jsonl')}
                atomic_checkpoint(out/'last.pt',payload,replace=True)
                if step in [25,50]:atomic_checkpoint(out/f'step_{step:03d}.pt',payload,replace=False)
                if cuda and torch.cuda.max_memory_reserved(0)>8*1024**3:raise GateError('8GiB allocator cap exceeded')
                print('UPDATE',track,step,'/50','ratio',check['aux_to_A0'],'cos',check['cosine'],flush=True)
                del z,base,terms,aux,loss,x,y,f,payload
        return {'track':track,'start':start,'end':end,'pid':os.getpid(),'last_sha256':file_sha(out/'last.pt'),'budget':budget.snapshot()}
    except BaseException as e:
        budget.stop(repr(e));raise
    finally:fh.remove();oh.remove()
