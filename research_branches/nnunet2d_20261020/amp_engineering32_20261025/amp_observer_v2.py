"""Persistent observer for unchanged nnU-Net steps; import never initializes CUDA."""
from __future__ import annotations
import copy,hashlib,json,math,os,random,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from revised_policy import StepEvidence,classify_step,AmpLimits,RevisionStop
from revised_runtime import state_is_finite

def digest(value):
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        elif isinstance(v,np.ndarray):h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());add(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for x in v:add(x)
        else:h.update(repr(v).encode())
    add(value);return h.hexdigest()

def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.pending')
    with tmp.open('x',encoding='utf-8') as f:json.dump(value,f,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def rng_pack():
    v={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
    if torch.cuda.is_initialized():v['cuda0']=torch.cuda.get_rng_state(0)
    return v

def restore_rng(v):
    random.setstate(v['python']);np.random.set_state(v['numpy']);torch.set_rng_state(v['torch'])
    if 'cuda0' in v:
        if not torch.cuda.is_initialized():raise RevisionStop('no implicit CUDA restore')
        torch.cuda.set_rng_state(v['cuda0'],0)

class Journal:
    """A reserved attempt is already charged; process failure never refunds it."""
    def __init__(self,path,binding,epochs,train_per_epoch,monitor_per_epoch,limits):
        self.path=Path(path);self.binding=binding;self.schedule=(epochs,train_per_epoch,monitor_per_epoch);self.limits=limits
        if any(type(x)is not int or x<1 for x in self.schedule):raise ValueError('positive schedule')
    def initialize(self):
        if self.path.exists():raise RevisionStop('ledger exists; never reset')
        s={'schema':2,'binding':self.binding,'schedule':list(self.schedule),'limits':{'consecutive':self.limits.max_consecutive_skips,'total':self.limits.max_total_skips},'status':'READY','attempts':0,'success':0,'skip':0,'fatal':0,'unknown':0,'consecutive':0,'monitor_attempts':0,'monitor_complete':0,'train_forwards':0,'monitor_forwards':0,'backwards':0,'optimizer_calls':0,'pending':None,'epoch':0,'iteration':0,'monitor_iteration':0,'phase':'train'}
        atomic(self.path,s)
    def read(self):
        s=json.loads(self.path.read_text())
        if s['binding']!=self.binding or s['schedule']!=list(self.schedule) or s['limits']!={'consecutive':self.limits.max_consecutive_skips,'total':self.limits.max_total_skips}:raise RevisionStop('ledger identity drift')
        pending=1 if s['pending'] and s['pending']['kind']=='train' else 0
        if s['attempts']!=s['success']+s['skip']+s['fatal']+s['unknown']+pending:raise RevisionStop('attempt accounting mismatch')
        if not 0<=s['attempts']<=self.schedule[0]*self.schedule[1] or not 0<=s['monitor_attempts']<=self.schedule[0]*self.schedule[2]:raise RevisionStop('budget cap violated')
        if s['train_forwards']>s['attempts'] or s['backwards']>s['attempts'] or s['optimizer_calls']>s['attempts'] or s['monitor_forwards']>s['monitor_attempts']:raise RevisionStop('extra computation detected')
        return s
    def save(self,s):atomic(self.path,s);self.read()
    def start(self,kind,epoch):
        s=self.read()
        if s['status']!='READY' or s['pending'] is not None or s['epoch']!=epoch or s['phase']!=kind:raise RevisionStop('stopped, unresolved or wrong phase')
        key='attempts' if kind=='train' else 'monitor_attempts';cap=self.schedule[0]*self.schedule[1 if kind=='train' else 2]
        index=s['iteration' if kind=='train' else 'monitor_iteration'];per_epoch=self.schedule[1 if kind=='train' else 2]
        if s[key]>=cap or index>=per_epoch:raise RevisionStop('fixed attempt/monitor budget exhausted')
        s[key]+=1;s['pending']={'kind':kind,'number':s[key],'forward':0,'backward':0,'optimizer':0};self.save(s);return s[key]
    def compute(self,kind):
        s=self.read();p=s['pending']
        if s['status']!='READY' or p is None:raise RevisionStop('unauthorized computation')
        if kind=='forward':
            if p['forward']!=0:raise RevisionStop('extra forward')
            p['forward']=1;s['train_forwards' if p['kind']=='train' else 'monitor_forwards']+=1
        elif kind=='backward':
            if p['kind']!='train' or p['forward']!=1 or p['backward']!=0:raise RevisionStop('extra/unbound backward')
            p['backward']=1;s['backwards']+=1
        elif kind=='optimizer':
            if p['kind']!='train' or p['backward']!=1 or p['optimizer']!=0:raise RevisionStop('extra/unbound optimizer')
            p['optimizer']=1;s['optimizer_calls']+=1
        else:raise RevisionStop('unknown compute')
        self.save(s)
    def resolve_train(self,evidence,changed):
        s=self.read();p=s['pending']
        if p is None or p['kind']!='train' or p['forward']!=1 or p['backward']!=1 or evidence.optimizer_step_calls!=p['optimizer']:raise RevisionStop('step evidence mismatch')
        kind,reason=classify_step(evidence)
        if kind=='SUCCESS' and not changed:kind,reason='FATAL','optimizer_called_without_parameter_change'
        if kind=='AMP_SKIP' and changed:kind,reason='FATAL','parameters_or_optimizer_changed_during_skip'
        s[{'SUCCESS':'success','AMP_SKIP':'skip','FATAL':'fatal'}[kind]]+=1
        s['consecutive']=s['consecutive']+1 if kind=='AMP_SKIP' else 0;s['pending']=None;s['iteration']+=1
        if kind=='FATAL':s['status']='STOPPED';s['reason']=reason
        elif s['consecutive']>=self.limits.max_consecutive_skips or s['skip']>=self.limits.max_total_skips:s['status']='STOPPED';s['reason']='AMP_skip_limit'
        self.save(s);return kind,reason
    def resolve_monitor(self):
        s=self.read();p=s['pending']
        if p is None or p['kind']!='monitor' or p['forward']!=1 or p['optimizer'] or p['backward']:raise RevisionStop('monitor evidence mismatch')
        s['monitor_complete']+=1;s['monitor_iteration']+=1;s['pending']=None;self.save(s)
    def finish_train(self):
        s=self.read()
        if s['status']!='READY' or s['phase']!='train' or s['pending'] or s['iteration']!=self.schedule[1]:raise RevisionStop('unfinished train epoch')
        s['phase']='monitor';self.save(s)
    def finish_monitor(self):
        s=self.read()
        if s['status']!='READY' or s['phase']!='monitor' or s['pending'] or s['monitor_iteration']!=self.schedule[2]:raise RevisionStop('unfinished monitor epoch')
        s['phase']='finalize';self.save(s)
    def finish_epoch(self,new_epoch):
        s=self.read()
        if s['status']!='READY' or s['phase']!='finalize' or new_epoch!=s['epoch']+1:raise RevisionStop('incorrect epoch advancement')
        s.update(epoch=new_epoch,iteration=0,monitor_iteration=0,phase='complete' if new_epoch==self.schedule[0] else 'train');self.save(s)
    def stop(self,reason):
        s=self.read()
        if s['pending'] and s['pending']['kind']=='train':s['unknown']+=1
        s['pending']=None;s['status']='STOPPED';s.setdefault('reason',str(reason));self.save(s)

class Observer:
    """Wrap official steps and epoch hooks, with no extra model calls."""
    def __init__(self,trainer,journal,out,allowed_ids,authorize=lambda:None,force_overflow=None):
        self.t=trainer;self.j=journal;self.out=Path(out);self.names=set(allowed_ids);self.authorize=authorize;self.force=force_overflow;self.active=None;self.handles=[]
        self.original={n:getattr(trainer,n) for n in ['train_step','validation_step','on_train_epoch_start','on_train_epoch_end','on_validation_epoch_end','on_epoch_end']}
        trainer.train_step=self.train_step;trainer.validation_step=self.monitor
        trainer.on_train_epoch_start=self.begin_epoch;trainer.on_train_epoch_end=self.train_end;trainer.on_validation_epoch_end=self.monitor_end;trainer.on_epoch_end=self.epoch_end
        self.handles.append(trainer.network.register_forward_pre_hook(self.forward))
        self.handles.append(next(trainer.network.parameters()).register_hook(self.backward))
        self.handles.append(trainer.optimizer.register_step_pre_hook(self.optimizer))
    def forward(self,*_):self.authorize();self.j.compute('forward')
    def backward(self,g):
        self.authorize();self.j.compute('backward')
        return torch.full_like(g,float('inf')) if self.active==self.force and self.force is not None else g
    def optimizer(self,*_):self.authorize();self.j.compute('optimizer')
    def begin_epoch(self):
        s=self.j.read()
        if s['phase']!='train' or s['epoch']!=self.t.current_epoch or s['iteration']!=0:raise RevisionStop('epoch start cursor mismatch')
        result=self.original['on_train_epoch_start']()
        want=.01*(1-self.t.current_epoch/100)**.9
        if abs(float(self.t.optimizer.param_groups[0]['lr'])-want)>1e-12:raise RevisionStop('official Poly100 LR drift')
        return result
    def train_end(self,outputs):
        s=self.j.read()
        if len(outputs)!=self.j.schedule[1] or s['iteration']!=len(outputs):raise RevisionStop('partial train output mismatch')
        result=self.original['on_train_epoch_end'](outputs);self.j.finish_train();return result
    def monitor_end(self,outputs):
        if len(outputs)!=self.j.schedule[2]:raise RevisionStop('partial monitor output mismatch')
        result=self.original['on_validation_epoch_end'](outputs);self.j.finish_monitor();return result
    def epoch_end(self):
        if self.j.read()['phase']!='finalize':raise RevisionStop('premature epoch finalize')
        result=self.original['on_epoch_end']();self.j.finish_epoch(self.t.current_epoch);return result
    def ids(self,batch):
        ids=list(map(str,batch.get('keys',[])))
        if not ids or not set(ids)<=self.names:raise RevisionStop('non-Train input rejected')
        return ids
    def heartbeat(self):atomic(self.out/'heartbeat.json',{'time':time.time(),'budget':self.j.read(),'GPU_calls_by_heartbeat':0})
    def train_step(self,batch):
        self.authorize();ids=self.ids(batch);number=self.j.start('train',self.t.current_epoch);self.active=number
        before_model=digest(self.t.network.state_dict());before_opt=digest(self.t.optimizer.state_dict());before=float(self.t.grad_scaler.get_scale())
        old_clip=torch.nn.utils.clip_grad_norm_;norm={'value':float('nan')}
        def clip(params,maxnorm,*args,**kwargs):
            if maxnorm!=12:raise RevisionStop('original clip12 changed')
            v=old_clip(params,maxnorm,*args,**kwargs);norm['value']=float(v.detach().cpu()) if torch.is_tensor(v) else float(v);return v
        try:
            with patch('torch.nn.utils.clip_grad_norm_',clip):result=self.original['train_step'](batch)
            after_model=digest(self.t.network.state_dict());after_opt=digest(self.t.optimizer.state_dict());s=self.j.read()
            e=StepEvidence(float(np.asarray(result['loss']).reshape(-1)[0]),norm['value'],before,float(self.t.grad_scaler.get_scale()),s['pending']['optimizer'],state_is_finite(self.t.network.state_dict()),state_is_finite(self.t.optimizer.state_dict()))
            changed_model=before_model!=after_model;changed_opt=before_opt!=after_opt
            kind,reason=self.j.resolve_train(e,changed_model if e.optimizer_step_calls else changed_model or changed_opt)
            row={'attempt':number,'epoch':self.t.current_epoch,'iteration':self.j.read()['iteration']-1,'ids':ids,'input_sha256':digest(batch['data']),'target_sha256':digest(batch['target']),'classification':kind,'reason':reason,'loss':e.loss if math.isfinite(e.loss) else str(e.loss),'unscaled_preclip_L2':e.gradient_norm if math.isfinite(e.gradient_norm) else str(e.gradient_norm),'scale_before':before,'scale_after':e.scale_after,'optimizer_calls':e.optimizer_step_calls,'model_changed':changed_model,'optimizer_changed':changed_opt,'model_sha256':after_model,'optimizer_sha256':after_opt,'scaler_sha256':digest(self.t.grad_scaler.state_dict()),'RNG_sha256':digest(rng_pack()),'lr':float(self.t.optimizer.param_groups[0]['lr']),'injected_engineering_overflow':number==self.force if self.force else False}
            if torch.cuda.is_initialized():row['GPU_peak_reserved_MiB']=torch.cuda.max_memory_reserved(0)/1048576
            append(self.out/'attempts.jsonl',row);self.heartbeat()
            if self.j.read()['status']!='READY':raise RevisionStop(self.j.read()['reason'])
            return result
        except BaseException as exc:self.j.stop(repr(exc));self.heartbeat();raise
        finally:self.active=None
    def monitor(self,batch):
        self.authorize();ids=self.ids(batch);number=self.j.start('monitor',self.t.current_epoch)
        try:
            with torch.no_grad():result=self.original['validation_step'](batch)
            loss=float(np.asarray(result['loss']).reshape(-1)[0])
            if not math.isfinite(loss):raise RevisionStop('nonfinite Train monitor')
            self.j.resolve_monitor();append(self.out/'monitor.jsonl',{'number':number,'ids':ids,'loss':loss,'optimizer_calls':0});self.heartbeat();return result
        except BaseException as exc:self.j.stop(repr(exc));self.heartbeat();raise
    def close(self):
        for h in self.handles:h.remove()
        for name,fn in self.original.items():setattr(self.t,name,fn)

def save_full_state(path,trainer,journal):
    """Explicit metadata; never pretend native queued augmentation is restored."""
    s=journal.read();state={'schema':'nnunet_amp_observer_v2','model':trainer.network.state_dict(),'optimizer':trainer.optimizer.state_dict(),'scaler':trainer.grad_scaler.state_dict(),'scheduler':trainer.lr_scheduler.state_dict(),'rng':rng_pack(),'cursor':{k:s[k] for k in ['epoch','iteration','monitor_iteration','phase']},'budget':s,'native_queue_resume_supported':False,'native_current_epoch':trainer.current_epoch}
    state['digests']={k:digest(state[k]) for k in ['model','optimizer','scaler','scheduler','rng','cursor','budget']}
    tmp=Path(str(path)+'.pending')
    with tmp.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);return state['digests']
