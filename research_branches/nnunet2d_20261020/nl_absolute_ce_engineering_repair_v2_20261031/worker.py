import os,sys,copy,math,time,signal,ctypes,contextlib
from common import *
def main(role):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.getppid()!=int(os.environ['NL_CONTROLLER_PID']):raise RuntimeError('Admission')
 if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0:raise RuntimeError('parent guard')
 key=validate();ledger=Ledger(O/'budget.json',key);assert ledger.get()['role']==role
 import torch,numpy as np,random
 B=C/'research_branches/nnunet2d_20261020'
 sys.path.extend([str(C),str(B/'scientific_resume_v5_20261025'),str(B/'deep_supervision_ablation_20261028')])
 from fast_observer import digest,rng_pack,restore_rng,states_finite
 from candidate import legacy_digest
 # Immutable CPU implementation loaded under isolated module name; GPU adapter changes only device admission.
 import importlib.util
 sp=importlib.util.spec_from_file_location('gpu_aux',Path(__file__).with_name('gpu_aux.py'));aux=importlib.util.module_from_spec(sp);sp.loader.exec_module(aux)
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
 if torch.cuda.device_count()!=1:raise RuntimeError('GPU count')
 torch.cuda.set_per_process_memory_fraction(10240*1048576/torch.cuda.get_device_properties(0).total_memory)
 seed=2028 if '2028' in role else 2027
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 native=O/('native_'+role)
 os.environ.update(nnUNet_raw=str(BASE/'raw'),nnUNet_preprocessed=str(BASE/'preprocessed'),nnUNet_results=str(native),nnUNet_compile='false',nnUNet_n_proc_DA='1')
 from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs
 plans=read(BASE/'preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json');plans['continue_training']=False
 t=nnUNetTrainer_100epochs(plans,'2d','all',read(BASE/'preprocessed/Dataset501_MAPLES/dataset.json'),device=torch.device('cuda:0'));t.initialize()
 assert list(t.loss.weight_factors)==[64/127,32/127,16/127,8/127,4/127,2/127,1/127,0]
 assert t.batch_size==2 and list(t.configuration_manager.patch_size)==[1024,1536]
 assert sum(p.numel() for p in t.network.parameters())==59177872
 expected=read(BASE/f'replication_seed2027_2028_100epochs_v1_20261026/runs/seed_{seed}/initialization.json')
 assert legacy_digest(t.network.state_dict())==expected['model']
 assert digest(t.optimizer.state_dict())==expected['optimizer'] and digest(t.grad_scaler.state_dict())==expected['scaler']
 assert not any(True for _ in t.network.buffers())
 t.network.train();params=tuple(t.network.parameters())
 fixture=read(O/'CPU_FIXTURES.json');items={(x['seed'],x['attempt']):x for x in fixture['rows']}
 def get_batch(attempt):
  item=items[(seed,attempt)]
  if sha(item['path'])!=item['sha256']:raise RuntimeError('fixture SHA')
  b=torch.load(item['path'],map_location='cpu',weights_only=False)
  # same legacy byte digest, no model call here
  import hashlib
  def bd(v):
   h=hashlib.sha256()
   def add(v):
    if torch.is_tensor(v):
     a=v.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    else:
     h.update(type(v).__name__.encode())
     for a in v:add(a)
   add(v);return h.hexdigest()
  assert bd(b['data'])==item['input_sha256'] and bd(b['target'])==item['target_sha256'] and list(b['keys'])==item['ids']
  return b
 def grads(loss,retain,component):
  scale=float(t.grad_scaler.get_scale()) if component=='base' else 128.
  gs=torch.autograd.grad(loss*scale,params,allow_unused=True,retain_graph=retain)
  result=[None if g is None else g.detach().float().div(scale).cpu() for g in gs]
  if not states_finite(result):
   bad=[]
   for (name,_),g in zip(t.network.named_parameters(),result):
    if g is not None and not bool(torch.isfinite(g).all()):
     bad.append({'name':name,'nan':int(torch.isnan(g).sum()),'inf':int(torch.isinf(g).sum())})
   log('component_failures.jsonl',{'component':component,'scale':scale,'bad':bad})
   raise RuntimeError('Nonfinite diagnostic gradient '+component+' scale'+str(scale))
  return result
 def stats(a,b):
  aa=sum(float(x.double().square().sum()) for x in a if x is not None);bb=sum(float(x.double().square().sum()) for x in b if x is not None)
  ab=sum(float(x.double().mul(y.double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)
  return dict(base_L2=math.sqrt(aa),aux_L2=math.sqrt(bb),cosine=ab/math.sqrt(aa*bb) if aa*bb else None)
 def forward(batch,groups,amp=True):
  ledger.forward()
  x=batch['data'].to('cuda');ys=[v.to('cuda') for v in batch['target']]
  with (contextlib.nullcontext() if amp else torch.autograd.graph.save_on_cpu(pin_memory=False)):
   with torch.autocast('cuda',enabled=amp):
    outputs=t.network(x);base=t.loss(outputs,ys)
   if outputs[0].shape!=(2,2,1024,1536):raise RuntimeError('highest resolution shape')
   lossaux=aux.sparse_aux(outputs[0],groups)
  if not states_finite(base,lossaux):raise RuntimeError('nonfinite forward')
  return base,lossaux,outputs
 def snapshot(name):
  data={'model':t.network.state_dict(),'optimizer':t.optimizer.state_dict(),'scaler':t.grad_scaler.state_dict(),'scheduler':t.lr_scheduler.state_dict(),'rng':rng_pack(),'epoch':t.current_epoch}
  dig={k:digest(v) for k,v in data.items()}
  p=O/(name+'.pt');assert not p.exists();torch.save(data,p);atomic(O/(name+'.json'),{'digests':dig,'sha256':sha(p)},True)
  return dig
 step_guard=t.optimizer.register_step_pre_hook(lambda *_:ledger.optimizer())
 try:
  if role=='diag':
   initial=legacy_digest(t.network.state_dict());first=read(O/'REPAIR_CARRY.json')['first_calibration'];norms=[(first['base_L2'],first['aux_L2'])];records=[first]
   log('diagnostics.jsonl',dict(first,reused_probe=True))
   for j,attempt in enumerate(fixture['calibration_attempts'][1:],start=1):
    b=get_batch(attempt);ledger.reserve();l0,la,outs=forward(b,b['groups'])
    g0=grads(l0,True,"base");ga=grads(la,False,"aux");row=stats(g0,ga)
    if row['base_L2']<=0 or row['aux_L2']<=0:raise RuntimeError('zero calibration norm')
    norms.append((row['base_L2'],row['aux_L2']));row.update(kind='calibration',index=j+1,attempt=attempt,base_loss=float(l0.detach()),aux_loss=float(la.detach()),scale=float(t.grad_scaler.get_scale()),base_diagnostic_scale=float(t.grad_scaler.get_scale()),aux_diagnostic_scale=128.,peak_MiB=torch.cuda.max_memory_reserved()/1048576)
    log('diagnostics.jsonl',row);ledger.resolve();records.append(row)
    del b,l0,la,outs,g0,ga
   coefficient=.1*math.sqrt(sum(a*a for a,b in norms)/sum(b*b for a,b in norms))
   assert math.isfinite(coefficient) and coefficient>0
   if max(coefficient*b/a for a,b in norms)>2:raise RuntimeError('Calibration weighted ratio>2')
   atomic(O/'LAMBDA_CALIBRATION.json',{'lambda':coefficient,'reference_seed':2027,'Train_batches':32,'scope':'Initial all-parameter RMS; not Dev optimum','diagnostics':records,'auxiliary_diagnostic_scale':128,'training_GradScaler_unchanged':True,'frozen_for_both_seeds':True},True)
   for j in range(1,8):
    b=get_batch(fixture['calibration_attempts'][j//2]);groups=copy.deepcopy(b['groups'])
    case=j//2
    if case==0:groups=[[],[]]
    elif case==1:groups=[g[:1] for g in groups]
    elif case==2:
     for gs in groups:
      for g in gs:g['negatives']=g['negatives'][:1]
    ledger.reserve();l0,la,outs=forward(b,groups,amp=j%2==0)
    g0=grads(l0,True,"base");ga=grads(la,False,"aux");row=stats(g0,ga);ratio=coefficient*row['aux_L2']/row['base_L2']
    assert torch.equal(aux.total_loss(l0,outs[0],None,0),l0)
    if ratio>2:raise RuntimeError('Edge diagnostic weighted ratio>2')
    if case==0 and (float(la.detach())!=0 or row['aux_L2']!=0):raise RuntimeError('Empty auxiliary not zero')
    row.update(kind='edge',case=case,AMP=j%2==0,weighted_ratio=ratio,lambda0_loss_exact=True)
    log('diagnostics.jsonl',row);ledger.resolve()
    del b,l0,la,outs,g0,ga
   assert legacy_digest(t.network.state_dict())==initial and not t.optimizer.state
   atomic(O/'diag_complete.json',{'status':'PASS','zero_updates':True,'empty_AMP_check_reused_from_probe':True,'lambda':coefficient,'budget':ledger.get()},True);ledger.finish();return
  coefficient=read(O/'LAMBDA_CALIBRATION.json')['lambda'] if role in ['cont','split_a','split_b'] else 0
  start=16 if role=='split_b' else 0
  if start:
   meta=read(O/'split16.json');assert sha(O/'split16.pt')==meta['sha256']
   state=torch.load(O/'split16.pt',map_location='cpu',weights_only=False)
   for k in state:assert digest(state[k])==meta['digests'][k]
   t.network.load_state_dict(state['model']);t.optimizer.load_state_dict(state['optimizer']);t.grad_scaler.load_state_dict(state['scaler']);t.lr_scheduler.load_state_dict(state['scheduler']);t.current_epoch=state['epoch'];restore_rng(state['rng'])
   for k,v in [('model',t.network.state_dict()),('optimizer',t.optimizer.state_dict()),('scaler',t.grad_scaler.state_dict()),('scheduler',t.lr_scheduler.state_dict()),('rng',rng_pack())]:assert digest(v)==meta['digests'][k],k
   atomic(O/'RESTORE_CHECK.json',{'status':'PASS','fixed_inputs_only':True,'native_queue_resume':False},True)
  else:t.on_train_epoch_start()
  for index in range(start,start+LIMITS[role]):
   b=get_batch(index+1);ledger.reserve();t.optimizer.zero_grad(set_to_none=True);ledger.forward()
   x=b['data'].to('cuda');ys=[v.to('cuda') for v in b['target']]
   with torch.autocast('cuda',enabled=True):outputs=t.network(x);base=t.loss(outputs,ys)
   if not states_finite(base):raise RuntimeError('nonfinite base')
   row={'role':role,'attempt':index+1,'lambda':coefficient}
   if coefficient:
    la=aux.sparse_aux(outputs[0],b['groups'])
    if not states_finite(la):raise RuntimeError('nonfinite aux')
    g0=grads(base,True,"base");ga=grads(la,True,"aux");row.update(stats(g0,ga));del g0,ga
    ratio=coefficient*row['aux_L2']/row['base_L2'] if row['base_L2'] else (0 if row['aux_L2']==0 else math.inf)
    row['weighted_ratio']=ratio
    if not math.isfinite(ratio) or ratio>2:
     log('updates.jsonl',{**row,'status':'PAUSE_BEFORE_OPTIMIZER'});raise RuntimeError('Weighted aux/base ratio>2 or invalid')
    total=base+coefficient*la
   elif role.startswith('zero'):total=aux.total_loss(base,outputs[0],None,0)
   else:total=base
   scale=float(t.grad_scaler.get_scale());t.grad_scaler.scale(total).backward();t.grad_scaler.unscale_(t.optimizer)
   finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
   if not finite:
    before=digest(t.network.state_dict());optbefore=digest(t.optimizer.state_dict())
    t.grad_scaler.step(t.optimizer);t.grad_scaler.update()
    assert float(t.grad_scaler.get_scale())<scale and digest(t.network.state_dict())==before and digest(t.optimizer.state_dict())==optbefore
    ledger.resolve(skip=True);row['status']='SAFE_AMP_SKIP'
   else:
    norm=float(torch.nn.utils.clip_grad_norm_(params,12));assert math.isfinite(norm)
    t.grad_scaler.step(t.optimizer);t.grad_scaler.update()
    if not states_finite(t.network.state_dict(),t.optimizer.state_dict()):raise RuntimeError('Nonfinite postupdate')
    ledger.resolve();row.update(status='SUCCESS',total_unscaled_L2=norm)
   row.update(peak_MiB=torch.cuda.max_memory_reserved()/1048576,scale_before=scale,scale_after=float(t.grad_scaler.get_scale()))
   log('updates.jsonl',row)
   del b,x,ys,outputs,base,total
   if coefficient:del la
  name='split16' if role=='split_a' else 'split32' if role=='split_b' else 'cont32' if role=='cont' else role
  result=snapshot(name)
  compare='base'+str(seed) if role.startswith('zero') else 'cont32' if role=='split_b' else None
  if compare:
   wanted=read(O/(compare+'.json'))['digests']
   assert result==wanted,('Exact state mismatch',role,[k for k in result if result[k]!=wanted[k]])
  atomic(O/(role+'_complete.json'),{'status':'PASS','exact_compare':compare,'budget':ledger.get()},True);ledger.finish()
 finally:step_guard.remove()
if __name__=='__main__':
 role=sys.argv[1]
 try:main(role)
 except BaseException as e:
  import traceback
  key=sha(O/'GPU_RELEASE.json');ledger=Ledger(O/'budget.json',key)
  ledger.stop(repr(e));atomic(O/(role+'_failure.json'),{'error':repr(e),'traceback':traceback.format_exc(),'budget':ledger.get()});raise
