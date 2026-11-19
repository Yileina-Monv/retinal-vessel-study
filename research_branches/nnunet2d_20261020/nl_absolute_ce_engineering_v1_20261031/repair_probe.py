import os,sys,copy,math,time,signal,ctypes
from common import *
def main(role):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise RuntimeError('Admission')

 rel=read(O/'REPAIR_PROBE_RELEASE.json')
 for f,h in rel['bound_files'].items():
  if sha(f)!=h:raise RuntimeError('Probe binding changed '+f)
 assert read(O/'RESULT.json')['budget']['diag_reserved']==1 and read(O/'RESULT.json')['budget']['update_reserved']==0
 atomic(O/'PROBE_BUDGET.json',{'status':'RESERVED','diagnostic_forwards_reserved':1,'prior_diagnostics':1,'cumulative_reserved':2,'optimizer_calls':0,'forwards':0},True)
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

 names=[n for n,p in t.network.named_parameters()]
 initial=legacy_digest(t.network.state_dict());initial_scaler=digest(t.grad_scaler.state_dict())
 t.optimizer.register_step_pre_hook(lambda *_: (_ for _ in ()).throw(RuntimeError('Probe forbids parameter updates')))
 b=get_batch(fixture['calibration_attempts'][0]);x=b['data'].to('cuda');ys=[v.to('cuda') for v in b['target']]
 ledger=read(O/'PROBE_BUDGET.json');ledger['forwards']=1;atomic(O/'PROBE_BUDGET.json',ledger)
 began=time.monotonic()
 with torch.autocast('cuda',enabled=True):outputs=t.network(x);base=t.loss(outputs,ys)
 lossaux=aux.sparse_aux(outputs[0],b['groups']);assert states_finite(base,lossaux)
 vectors={};rows=[]
 def norm(gs):return math.sqrt(sum(float(g.double().square().sum()) for g in gs if g is not None))
 for label,loss in [('base',base),('aux',lossaux)]:
  for scale in [1.,128.,float(t.grad_scaler.get_scale())]:
   gs=torch.autograd.grad(loss*scale,params,retain_graph=True,allow_unused=True)
   cpu=[None if g is None else g.detach().float().div(scale).cpu() for g in gs];del gs
   bad=[];nan=inf=0
   for name,g in zip(names,cpu):
    if g is None:continue
    ni=int(torch.isnan(g).sum());ii=int(torch.isinf(g).sum());nan+=ni;inf+=ii
    if ni or ii:bad.append({'parameter':name,'nan':ni,'inf':ii})
   row={'component':label,'scale':scale,'finite':nan+inf==0,'nan_count':nan,'inf_count':inf,'first_bad_layers':bad[:5],'unscaled_L2':norm(cpu) if nan+inf==0 else None}
   rows.append(row);log('probe_gradients.jsonl',row)
   if scale in [1.,128.] and row['finite']:vectors[(label,scale)]=cpu
   else:del cpu
 diffs={}
 for label in ['base','aux']:
  if (label,1.) in vectors and (label,128.) in vectors:
   a=vectors[(label,1.)];b2=vectors[(label,128.)]
   diff=math.sqrt(sum(float((x-y).double().square().sum()) for x,y in zip(a,b2) if x is not None and y is not None))
   diffs[label]=diff/max(norm(b2),1e-30)
 empty=aux.sparse_aux(outputs[0],[[],[]]);eg=torch.autograd.grad(empty,params,retain_graph=False,allow_unused=True)
 assert float(empty.detach())==0 and all(g is None or bool((g==0).all()) for g in eg)
 assert torch.equal(aux.total_loss(base,outputs[0],None,0),base)
 assert legacy_digest(t.network.state_dict())==initial and digest(t.grad_scaler.state_dict())==initial_scaler and not t.optimizer.state
 coeff=None;first=None
 if ('base',128.) in vectors and ('aux',128.) in vectors:
  a=vectors[('base',128.)];b2=vectors[('aux',128.)];na=norm(a);nb=norm(b2)
  cos=sum(float(x.double().mul(y.double()).sum()) for x,y in zip(a,b2) if x is not None and y is not None)/(na*nb)
  first={'attempt':fixture['calibration_attempts'][0],'base_L2':na,'aux_L2':nb,'cosine':cos,'base_loss':float(base.detach()),'aux_loss':float(lossaux.detach()),'diagnostic_scale':128.}
 report={'status':'COMPLETE_ZERO_UPDATE_PROBE','rows':rows,'relative_L2_difference_scale1_vs128':diffs,'first_calibration_candidate':first,'empty_AMP_zero_gradient_pass':True,'lambda0_loss_exact':True,'model_unchanged':True,'optimizer_unchanged':True,'training_GradScaler_unchanged':True,'actual_GPU_forwards':1,'updates':0,'GPU_seconds':time.monotonic()-began,'peak_MiB':torch.cuda.max_memory_reserved()/1048576}
 atomic(O/'PROBE_RESULT.json',report,True)
 ledger.update(status='COMPLETE',optimizer_calls=0);atomic(O/'PROBE_BUDGET.json',ledger);print(report,flush=True)
if __name__=='__main__':
 import subprocess
 sys.path.insert(0,str(C))
 from retinal_m0.maples_g1_safety import exclusive_lock
 with exclusive_lock(R/'experiments/a14_gpu0.lock'),exclusive_lock(O/'PROBE_RUN.lock'):
  procs=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  if procs:raise RuntimeError('GPU0 busy')
  try:main('probe')
  except BaseException as e:
   import traceback
   atomic(O/'PROBE_FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'no_automatic_retry':True});raise
