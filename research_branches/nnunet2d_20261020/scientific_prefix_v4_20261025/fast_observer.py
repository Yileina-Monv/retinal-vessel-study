"""Equivalent training observer: full hashes at checkpoints/overflows, not every step."""
from __future__ import annotations
import sys,os,math,time,json,copy
from pathlib import Path
from unittest.mock import patch
import torch,numpy as np
BRANCH=Path('${RETINA_PROJECT_ROOT}/code/retinal-vessel-study/research_branches/nnunet2d_20261020')
sys.path.extend([str(BRANCH/'amp_engineering32_20261025'),str(BRANCH/'amp_recovery_revision_20261021')])
from amp_observer_v2 import Observer as AuditObserver,Journal,AmpLimits,digest,atomic,append,rng_pack,restore_rng,StepEvidence,RevisionStop

def states_finite(*values):
    groups={};scalars=[]
    def visit(v):
        if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex()):groups.setdefault(v.device,[]).append(torch.isfinite(v).all())
        elif isinstance(v,dict):
            for x in v.values():visit(x)
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
        elif isinstance(v,float):scalars.append(math.isfinite(v))
    for v in values:visit(v)
    return all(scalars) and all(bool(torch.stack(flags).all().item()) for flags in groups.values())

class FastObserver(AuditObserver):
    def __init__(self,*args,on_event=None,**kwargs):
        super().__init__(*args,**kwargs)
        if any(True for _ in self.t.network.buffers()):raise RevisionStop('fast observer requires frozen stateless normalization, no mutable model buffers')
        self.on_event=on_event;self.partial_train=[];self.partial_monitor=[];self.parameter_before=None
        self.clock=time.monotonic();self.overflow_before=None
        self.loss_guard=None
        if isinstance(self.t.loss,torch.nn.Module):self.loss_guard=self.t.loss.register_forward_hook(self.guard_loss)
    def guard_loss(self,module,args,result):
        if not torch.is_tensor(result) or result.numel()!=1 or not bool(torch.isfinite(result).all()):raise RevisionStop('nonfinite or invalid forward loss before backward')
    def optimizer(self,*args):
        super().optimizer(*args)
        # GPU-local copies plus one aggregate comparison prove a real update without CPU hashing.
        self.parameter_before=[p.detach().clone() for p in self.t.network.parameters()]
    def begin_epoch(self):
        result=super().begin_epoch();self.partial_train=[];self.partial_monitor=[];return result
    def train_end(self,outputs):
        if digest(outputs)!=digest(self.partial_train):raise RevisionStop('partial Train outputs do not match official epoch outputs')
        return super().train_end(outputs)
    def monitor_end(self,outputs):
        if digest(outputs)!=digest(self.partial_monitor):raise RevisionStop('partial monitor outputs do not match official epoch outputs')
        return super().monitor_end(outputs)
    def epoch_end(self):
        result=super().epoch_end();self.partial_train=[];self.partial_monitor=[]
        if self.on_event:self.on_event('epoch',self.j.read(),self)
        return result
    def train_step(self,batch):
        self.authorize();ids=self.ids(batch);number=self.j.start('train',self.t.current_epoch);self.active=number;self.parameter_before=None;self.overflow_before=None
        before=float(self.t.grad_scaler.get_scale());old_clip=torch.nn.utils.clip_grad_norm_;norm={'value':float('nan')}
        def clip(params,maxnorm,*args,**kwargs):
            if maxnorm!=12:raise RevisionStop('original clip12 changed')
            result=old_clip(params,maxnorm,*args,**kwargs);norm['value']=float(result.detach().cpu()) if torch.is_tensor(result) else float(result)
            if not math.isfinite(norm['value']):self.overflow_before={'model':digest(self.t.network.state_dict()),'optimizer':digest(self.t.optimizer.state_dict())}
            return result
        try:
            with patch('torch.nn.utils.clip_grad_norm_',clip):result=self.original['train_step'](batch)
            scale_after=float(self.t.grad_scaler.get_scale());s=self.j.read();calls=s['pending']['optimizer']
            if self.parameter_before is not None:
                changed_model=bool(torch.stack([torch.any(a!=p.detach()) for a,p in zip(self.parameter_before,self.t.network.parameters())]).any().item());changed_opt=None
            elif self.overflow_before is not None:
                changed_model=digest(self.t.network.state_dict())!=self.overflow_before['model'];changed_opt=digest(self.t.optimizer.state_dict())!=self.overflow_before['optimizer']
            else:raise RevisionStop('no actual optimizer update and no captured overflow evidence')
            e=StepEvidence(float(np.asarray(result['loss']).reshape(-1)[0]),norm['value'],before,scale_after,calls,states_finite(self.t.network.state_dict()),states_finite(self.t.optimizer.state_dict()))
            kind,reason=self.j.resolve_train(e,changed_model if calls else changed_model or changed_opt)
            row={'attempt':number,'epoch':self.t.current_epoch,'iteration':self.j.read()['iteration']-1,'ids':ids,'input_sha256':digest(batch['data']),'target_sha256':digest(batch['target']),'classification':kind,'reason':reason,'loss':e.loss if math.isfinite(e.loss) else str(e.loss),'unscaled_preclip_L2':e.gradient_norm if math.isfinite(e.gradient_norm) else str(e.gradient_norm),'scale_before':before,'scale_after':scale_after,'optimizer_calls':calls,'model_changed':changed_model,'optimizer_changed':changed_opt,'lr':float(self.t.optimizer.param_groups[0]['lr']),'elapsed_seconds':time.monotonic()-self.clock,'full_state_hash_scope':'checkpoints and overflow events','overflow_before':self.overflow_before}
            if torch.cuda.is_initialized():row['GPU_peak_reserved_MiB']=torch.cuda.max_memory_reserved(0)/1048576
            append(self.out/'attempts.jsonl',row);self.partial_train.append(copy.deepcopy(result));self.heartbeat()
            if self.j.read()['status']!='READY':raise RevisionStop(self.j.read()['reason'])
            if self.on_event:self.on_event('train',self.j.read(),self)
            return result
        except BaseException as exc:
            self.j.stop(repr(exc));self.heartbeat();raise
        finally:self.active=None;self.parameter_before=None
    def monitor(self,batch):
        result=super().monitor(batch);self.partial_monitor.append(copy.deepcopy(result))
        if self.on_event:self.on_event('monitor',self.j.read(),self)
        return result
    def save_bundle(self,path,source_binding):
        s=self.j.read();payload={'schema':'nnunet_science100_fast_observer_v3','binding_sha256':source_binding,'model':self.t.network.state_dict(),'optimizer':self.t.optimizer.state_dict(),'scaler':self.t.grad_scaler.state_dict(),'scheduler':self.t.lr_scheduler.state_dict(),'rng':rng_pack(),'cursor':{k:s[k] for k in ['epoch','iteration','monitor_iteration','phase']},'budget':s,'native_current_epoch':self.t.current_epoch,'partial_train':self.partial_train,'partial_monitor':self.partial_monitor,'native_queue_resume_supported':False}
        payload['log_prefixes']={name:__import__('hashlib').sha256((self.out/name).read_bytes()).hexdigest() if (self.out/name).exists() else None for name in ['attempts.jsonl','monitor.jsonl']}
        payload['digests']={k:digest(payload[k]) for k in ['model','optimizer','scaler','scheduler','rng','cursor','budget','partial_train','partial_monitor','log_prefixes']}
        path=Path(path);tmp=path.with_name(path.name+'.pending')
        with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);return payload['digests']
    def close(self):
        if self.loss_guard:self.loss_guard.remove()
        super().close()
