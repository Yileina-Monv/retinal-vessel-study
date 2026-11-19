"""NL adapter: native full loop, exact input audit, same-graph component diagnostics."""
import math,time,copy
import torch
from common import *
from fast_observer import FastObserver,digest,states_finite,rng_pack
from selection import choose_groups
from aux_loss import sparse_aux
from diagnostic import bounded_gradients
from pathlib import Path
class NLObserver(FastObserver):
 def __init__(self,trainer,*args,**kwargs):
  repair_dir=Path(__file__).resolve().parent
  release_path=GROUP/repair_dir.name/'TRAINING_RELEASE.json'
  if not release_path.exists():raise RuntimeError('independent repair training release required; old release is insufficient')
  release=read(release_path)
  if release.get('scope')!='NL_REPAIR_RESUME_ONLY' or release.get('adapter_sha256')!=sha(__file__) or release.get('diagnostic_sha256')!=sha(repair_dir/'diagnostic.py'):raise RuntimeError('repair release binding mismatch')
  raise RuntimeError('resume loop and revised physical budget require independent implementation/admission; adapter is CPU-reviewed only')
  self.diag_mode=False;self.component=None;self.groups=None;self.coefficient=card()['method']['lambda']
  self.history=(BASELINE/'runs'/f'seed_{SEED}'/'attempts.jsonl').open()
  self.monitor_history=(BASELINE/'runs'/f'seed_{SEED}'/'monitor.jsonl').open()
  self.paired_train=0;self.paired_monitor=0
  trainer.train_step=self.native_step
  super().__init__(trainer,*args,**kwargs)
 def backward(self,g):
  if self.diag_mode:
   self.authorize();self.diag_hook_calls+=1;return g
  return super().backward(g)
 def optimizer(self,*args):
  if self.diag_mode:raise RuntimeError('optimizer prohibited during diagnostics')
  return super().optimizer(*args)
 def train_step(self,batch):
  number=self.j.read()['attempts']+1;h=json.loads(next(self.history))
  compare_train(batch,h,number,self.t.current_epoch,float(self.t.optimizer.param_groups[0]['lr']),digest)
  records=batch.get('nl_candidates')
  if records is None or len(records)!=2 or [r['image'] for r in records]!=list(batch['keys']):raise RuntimeError('missing/misaligned auxiliary sidecar')
  before=digest(rng_pack());self.groups=[choose_groups(r['candidates'],SEED,number,i) for i,r in enumerate(records)]
  if before!=digest(rng_pack()):raise RuntimeError('private selection changed training RNG')
  self.paired_train+=1
  log(STAGE/'auxiliary_targets.jsonl',{'attempt':number,'ids':list(batch['keys']),'counts':[r['counts'] for r in records],'group_counts':[len(g) for g in self.groups],'selected_groups_sha256':digest(self.groups),'pairing':'PASS'})
  return super().train_step(batch)
 def monitor(self,batch):
  h=json.loads(next(self.monitor_history))
  if list(batch['keys'])!=h['ids'] or 'nl_candidates' in batch:raise RuntimeError('Train monitor ID/role mismatch')
  self.paired_monitor+=1;return super().monitor(batch)
 def diagnostic_grad(self,loss,component):
  initial=float(self.t.grad_scaler.get_scale()) if component=='base' else 128.
  before=digest(rng_pack());scaler_before=digest(self.t.grad_scaler.state_dict())
  self.diag_mode=True;self.component=component;self.diag_hook_calls=0
  try:
   def record(row):
    self.authorize()
    row.update(attempt=self.active,optimizer_calls_before_stop=self.j.read()['pending']['optimizer'])
    log(STAGE/'diagnostic_scaling.jsonl',row)
   result,meta=bounded_gradients(loss,tuple(self.t.network.named_parameters()),initial,component,record)
   if not hasattr(self,'diagnostic_meta'):self.diagnostic_meta={}
   self.diagnostic_meta[component]=meta
   return result
  finally:
   self.diag_mode=False;self.component=None
   if digest(rng_pack())!=before or digest(self.t.grad_scaler.state_dict())!=scaler_before:raise RuntimeError('diagnostic altered training RNG/scaler')
 def native_step(self,batch):
  t=self.t;data=batch['data'].to(t.device,non_blocking=True);target=[v.to(t.device,non_blocking=True) for v in batch['target']]
  t.optimizer.zero_grad(set_to_none=True)
  with torch.autocast(t.device.type,enabled=True):output=t.network(data);base=t.loss(output,target)
  if output[0].shape!=(2,2,1024,1536):raise RuntimeError('highest output shape')
  aux=sparse_aux(output[0],self.groups)
  if not states_finite(base,aux):raise RuntimeError('nonfinite base/aux loss')
  before=digest(rng_pack());g0=self.diagnostic_grad(base,'base');ga=self.diagnostic_grad(aux,'aux')
  aa=sum(float(x.double().square().sum()) for x in g0 if x is not None);bb=sum(float(x.double().square().sum()) for x in ga if x is not None)
  ab=sum(float(x.double().mul(y.double()).sum()) for x,y in zip(g0,ga) if x is not None and y is not None);del g0,ga
  if digest(rng_pack())!=before:raise RuntimeError('diagnostics changed training RNG')
  ratio=self.coefficient*math.sqrt(bb/aa) if aa else (0. if bb==0 else math.inf)
  row={'attempt':self.active,'epoch':t.current_epoch,'base_loss':float(base.detach()),'aux_loss':float(aux.detach()),'lambda':self.coefficient,'base_L2':math.sqrt(aa),'raw_aux_L2':math.sqrt(bb),'weighted_aux_L2':self.coefficient*math.sqrt(bb),'weighted_ratio':ratio if math.isfinite(ratio) else 'inf','cosine':ab/math.sqrt(aa*bb) if aa*bb else None,'base_diagnostic_scale':self.diagnostic_meta['base']['accepted_scale'],'aux_diagnostic_scale':self.diagnostic_meta['aux']['accepted_scale'],'diagnostic_details':self.diagnostic_meta.copy(),'groups':[len(g) for g in self.groups],'optimizer_calls_before_diagnostics':self.j.read()['pending']['optimizer']}
  log(STAGE/'auxiliary_gradients.jsonl',row)
  if not math.isfinite(ratio) or ratio>2:raise RuntimeError('weighted gradient ratio>2 before optimizer')
  loss=base+self.coefficient*aux
  if not states_finite(loss):raise RuntimeError('nonfinite total loss')
  t.grad_scaler.scale(loss).backward();t.grad_scaler.unscale_(t.optimizer)
  torch.nn.utils.clip_grad_norm_(t.network.parameters(),12)
  t.grad_scaler.step(t.optimizer);t.grad_scaler.update()
  return {'loss':loss.detach().cpu().numpy()}
 def close(self):
  super().close();self.history.close();self.monitor_history.close()
