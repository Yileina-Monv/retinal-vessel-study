"""Candidate C CPU preparation only. Cannot start a GPU or parameter updates."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse, contextlib, csv, hashlib, io, json, math, random, resource
import signal, sys, time, shutil
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_cdt, binary_erosion
from torch.utils.data._utils.collate import default_collate
from retinal_m0.maples_core import CorePatches, weighted_masked_loss
from retinal_m0.maples_soft_cldice import soft_cldice_terms, candidate_loss, calibrate_eta, soft_erode_zero
from retinal_m0.model import UNet
from retinal_m0.state import state_digest

ROOT=Path('${RETINA_PROJECT_ROOT}')
CODE=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1'
OUT=REVIEW/'soft_cldice_cpu_v1_20261023'
AUTH=OUT/'CPU_AUTHORIZATION.json'
PROPOSAL=REVIEW/'scientific_b1_b2_2000_v1_20261022/analysis/literature_mechanism_review_20261023/NEXT_METHOD_PROPOSAL_v1.md'
BATCHES=REVIEW/'scientific_review_20261021/attempt_01/G1_fixed_batches.json'
BASE=ROOT/'experiments/t0_tw_core_20261016/runs/seed_2026/T0'
TRAIN=ROOT/'docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json'
MARKER='SOFT_CLDICE_CPU_APPROVED_20261023'
MAX_RSS_KIB=12*1024*1024

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8388608),b''):h.update(c)
 return h.hexdigest()
def J(p):return json.loads(Path(p).read_text())
def atomic_json(p,obj):
 temp=p.with_name(p.name+'.pending')
 with temp.open('w',encoding='utf8') as f:
  json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
def write_new(p,obj):
 with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def rng():
 n=np.random.get_state()
 return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),*n[2:]],'torch':torch.get_rng_state()}
def restore_rng(s):
 random.setstate(s['python']);n=s['numpy']
 np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),*n[2:]));torch.set_rng_state(s['torch'])
def norm(gs):return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs))
def dot(a,b):return sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(a,b))
def peak_rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

@contextlib.contextmanager
def forbid_updates():
 patched=[]
 def no_step(*a,**kw):raise RuntimeError('CPU_ONLY_PARAMETER_UPDATES_FORBIDDEN')
 for value in vars(torch.optim).values():
  if isinstance(value,type) and issubclass(value,torch.optim.Optimizer) and 'step' in value.__dict__:
   patched.append((value,value.__dict__['step']));value.step=no_step
 try:yield
 finally:
  for cls,fn in patched:cls.step=fn

def authorize():
 a=J(AUTH)
 if a.get('phase')!='C_SOFT_CLDICE_CPU_ONLY' or a.get('user_approved') is not True or a.get('GPU_allowed') is not False or a.get('parameter_updates')!=0 or a.get('max_M0_forwards')!=10 or a.get('planned_verified_batches')!=8:
  raise RuntimeError('CPU authority scope mismatch')
 if sha(PROPOSAL)!=a['proposal_sha256']:raise RuntimeError('approved proposal drift')
 if MARKER not in (CODE/'STATUS.md').read_text():raise RuntimeError('STATUS CPU authority missing')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('GPU must remain invisible')
 for p,h in a['sources_sha256'].items():
  if sha(Path(p))!=h:raise RuntimeError('approved CPU source drift:'+p)
 return a

def train_k(dataset, rows, attempt, sources):
 result=[];start=time.monotonic()
 for i,row in enumerate(rows):
  assert row['index']==i and row['trial_split']=='train' and row['filename']==dataset.names[i]
  for key in ['RGB','V']:
   p=Path(row[key]['path']);h=sha(p);assert h==row[key]['sha256'];sources[str(p)]=h
  raw,g,f,_=dataset._full_arrays(i)
  assert list(g.shape)==row['native_shape']
  m=g.astype(bool)&f
  # Five-site cross erosion extinction depth equals padded L1 distance.
  dist=distance_transform_cdt(np.pad(m,1),metric='taxicab')[1:-1,1:-1]
  k=int(dist.max());assert k>0
  result.append({'filename':row['filename'],'shape':list(g.shape),'GT_FOV_pixels':int(m.sum()),'FOV_pixels':int(f.sum()),'cross_erosion_extinction_steps':k,'GT_FOV_sha256':hashlib.sha256(m.tobytes()).hexdigest()})
  dataset.cache.clear()
  if (i+1)%20==0:print('TRAIN_K',i+1,'/',len(rows),flush=True)
 k=max(r['cross_erosion_extinction_steps'] for r in result)
 # Independent exact-erode confirmation on the maximal GT, no soft approximation.
 j=max(range(len(result)),key=lambda j:result[j]['cross_erosion_extinction_steps'])
 _,g,f,_=dataset._full_arrays(j);remaining=g.astype(bool)&f
 cross=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)
 for _ in range(k-1):remaining=binary_erosion(remaining,structure=cross,border_value=0)
 assert remaining.any() and not binary_erosion(remaining,structure=cross,border_value=0).any()
 dataset.cache.clear()
 report={'status':'PASSED','Train_count':len(rows),'native_shapes':sorted({tuple(r['shape']) for r in result}),'k':k,'maximal_image':result[j]['filename'],'independent_binary_erosion_at_k_minus_1_and_k':True,'GT_only_no_Dev':True,'rows':result,'elapsed_seconds':time.monotonic()-start}
 write_new(attempt/'train_k.json',report)
 return k

def restoration_test(attempt):
 """No updates: nonempty synthetic optimizer state and actual RNG round-trip.

 This does NOT assert CUDA AMP/update trajectory equivalence.
 """
 saved_global=rng();random.seed(123);np.random.seed(123);torch.manual_seed(123)
 m=UNet((8,16),groups=8);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,foreach=False)
 # Explicit fixture only; no parameter step to manufacture optimizer history.
 for p in m.parameters():
  opt.state[p]={'step':torch.tensor(7.),'exp_avg':torch.full_like(p,.0125),'exp_avg_sq':torch.full_like(p,.001)}
 gen=torch.Generator().manual_seed(77);torch.rand(3,generator=gen)
 payload={'model':m.state_dict(),'optimizer':opt.state_dict(),'scaler':torch.amp.GradScaler('cpu',enabled=False).state_dict(),'scheduler':None,'rng':rng(),'loader_generator':gen.get_state(),'next_global_draw':456,'ledger':{'failed':2,'uncertain':1,'charged':3,'parameter_updates':0},'AMP_verified':False}
 with (attempt/'cpu_state_roundtrip.pt').open('xb') as f:torch.save(payload,f)
 draws=(random.random(),float(np.random.rand()),torch.rand(5),torch.rand(5,generator=gen))
 m2=UNet((8,16),groups=8);o2=torch.optim.AdamW(m2.parameters(),lr=1e-4,foreach=False)
 s=torch.load(attempt/'cpu_state_roundtrip.pt',map_location='cpu',weights_only=True)
 m2.load_state_dict(s['model']);o2.load_state_dict(s['optimizer']);restore_rng(s['rng'])
 gen2=torch.Generator();gen2.set_state(s['loader_generator'])
 after=(random.random(),float(np.random.rand()),torch.rand(5),torch.rand(5,generator=gen2))
 assert draws[:2]==after[:2] and torch.equal(draws[2],after[2]) and torch.equal(draws[3],after[3])
 assert state_digest(m.state_dict())==state_digest(m2.state_dict()) and state_digest(opt.state_dict())==state_digest(o2.state_dict())
 assert s['next_global_draw']==456 and s['ledger']==payload['ledger']
 with forbid_updates():
  try:o2.step()
  except RuntimeError as e:assert str(e)=='CPU_ONLY_PARAMETER_UPDATES_FORBIDDEN'
  else:raise AssertionError('optimizer.step guard ineffective')
 restore_rng(saved_global)
 write_new(attempt/'cpu_restore_check.json',{'status':'PASSED_CPU_STATE_ROUNDTRIP_ONLY','nonempty_optimizer_state':'synthetic fixture, not trained updates','model_optimizer_RNG_private_loader_and_cursor_exact':True,'failure_counters_preserved':True,'optimizer_step_denied':True,'parameter_updates':0,'CUDA_AMP_and_updated_trajectory_verified':False,'limitation':'actual candidate GPU update/resume requires separate approval and engineering verification'})

def run():
 auth=authorize();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 # CPU reference backend only; GPU numerical behavior remains unvalidated.
 # Preserve the 1e-4 global-gradient check instead of loosening it.
 torch.backends.mkldnn.enabled=False
 torch.use_deterministic_algorithms(True)
 random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
 prior=J(OUT/'attempt_02/cpu_ledger.json')
 if prior['status']!='FAILED_STOP' or prior['M0_forwards']!=2 or prior['autograd_calls']!=14 or prior['parameter_updates']!=0:raise RuntimeError('unexpected retained CPU failure state')
 if sha(OUT/'attempt_02/cpu_ledger.json')!=auth['retained_failed_ledger_sha256']:raise RuntimeError('failed CPU ledger drift')
 attempt=OUT/'attempt_03';attempt.mkdir(parents=True,exist_ok=False)
 ledger={'phase':'CPU_ONLY','M0_forward_cap':10,'M0_forwards':prior['M0_forwards'],'autograd_calls':prior['autograd_calls'],'completed_batches':0,'failed_batches':prior['failed_batches'],'pending':None,'parameter_updates':0,'GPU_calls':0,'status':'RUNNING','inherited_failed_ledger':prior,'note':'two failed CPU diagnostics retained; eight verified batches plus two failed forwards. CPU ATen reference backend, no training-budget or loss change'}
 atomic_json(attempt/'cpu_ledger.json',ledger)
 started=time.monotonic();sources=dict(auth['sources_sha256'])
 def tick():
  authorize()
  if peak_rss()>MAX_RSS_KIB:raise RuntimeError('CPU_peak_RSS_over_12_GiB')
 def timed_out(*_):raise TimeoutError('CPU calibration 30-minute cap reached; no retry')
 try:
  config=J(BASE/'config.json');architecture=J(CODE/'configs/development_v1.json')
  if sha(config['source_m0'])!=config['source_m0_sha256']:raise RuntimeError('M0 source mismatch')
  sources[config['source_m0']]=config['source_m0_sha256']
  train=J(TRAIN);assert train['count']==110
  ds=CorePatches(config,'T0')
  assert [r['filename'] for r in train['rows']]==ds.names
  # Block content reads from any non-Train data file, including Dev and Test.
  allow={str((Path(config['data_root'])/n).resolve()) for n in ['data_manifest.csv','split_record.json']}
  for row in ds.native.rows:
   for key in ['image_path','label_vessels','label_hemorrhages','label_brightuncertains','label_reduncertains']:
    p=(Path(config['data_root'])/row[key]).resolve();allow.add(str(p));sources[str(p)]=sha(p)
  data_prefix=str(Path(config['data_root']).resolve())+'/'
  def audit(event,args):
   if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
    p=str(Path(os.fsdecode(args[0])).resolve())
    if p.startswith(data_prefix) and p not in allow:raise RuntimeError('non-Train data content access forbidden:'+p)
  sys.addaudithook(audit)
  old_k=OUT/'attempt_01/train_k.json'
  if sha(old_k)!=auth['retained_train_k_sha256']:raise RuntimeError('retained Train-only k evidence drift')
  kr=J(old_k);assert kr['Train_count']==110 and kr['independent_binary_erosion_at_k_minus_1_and_k']
  # All underlying Train files are rehashed above; do not redo the completed GT analysis.
  k=kr['k'];shutil.copy2(old_k,attempt/'train_k.json');tick()
  restoration_test(attempt)
  fixed=J(BATCHES)['batches'];assert len(fixed)==8
  # All diagnostic draws must still equal original T0 evidence.
  batches=[]
  for spec in fixed:
   items=[ds[int(r['draw'])] for r in spec['draws']]
   for item,ref in zip(items,spec['draws']):
    assert item['filename']==ref['filename'] and item['geometry'].tolist()==ref['geometry'] and item['input_sha256']==ref['input_sha256_original_T0']
   batches.append((spec,default_collate(items)))
  ds.cache.clear()
  model=UNet(architecture['widths'],architecture['group_norm_groups']).cpu().train()
  c=torch.load(config['source_m0'],map_location='cpu',weights_only=True);assert c['step']==10000
  model.load_state_dict(c['model'],strict=True);assert state_digest(c['model'])==state_digest(model.state_dict());del c
  ps=tuple(model.parameters());names=[n for n,_ in model.named_parameters()]
  model_hash=state_digest(model.state_dict());rng_hash=state_digest(rng());records=[]
  signal.signal(signal.SIGALRM,timed_out);signal.alarm(1800)
  with forbid_updates():
   for j,(spec,batch) in enumerate(batches):
    tick();assert ledger['M0_forwards']<10 and ledger['pending'] is None
    ledger['pending']={'index':j,'original_T0_step':spec['step_original_T0_one_based']};ledger['M0_forwards']+=1;atomic_json(attempt/'cpu_ledger.json',ledger)
    t=time.monotonic();z=model(batch['image']);forward_time=time.monotonic()-t
    base=weighted_masked_loss(z,batch['mask'],batch['fov'],batch['fov'])
    t=time.monotonic();terms=soft_cldice_terms(z,batch['mask'],batch['fov'],k=k);loss_time=time.monotonic()-t
    def grad(loss):
     ledger['autograd_calls']+=1;atomic_json(attempt/'cpu_ledger.json',ledger)
     out=tuple(g.detach() for g in torch.autograd.grad(loss,ps+(z,),retain_graph=True,allow_unused=False))
     if not all(torch.isfinite(g).all() for g in out):raise RuntimeError('nonfinite real parameter gradient')
     return out[:-1],out[-1]
    t=time.monotonic();gb,zb=grad(base);nb=norm(gb)
    if not math.isfinite(nb) or nb<=1e-12:raise RuntimeError('A0 gradient too small/nonfinite for calibration')
    zero=candidate_loss(z,batch['mask'],batch['fov'],k=k,eta=0)
    gz,zz=grad(zero)
    if not torch.equal(base,zero) or not torch.equal(zb,zz) or not all(torch.equal(a,b) for a,b in zip(gb,gz)):raise RuntimeError('real eta0 mismatch')
    del gz,zz,zb
    # Measure true per-crop parameter contributions and sum at fixed batch size.
    ga=tuple(torch.zeros_like(g) for g in gb);za=torch.zeros_like(z);per_crop=[]
    for i in range(4):
     gi,zi=grad(terms['per_crop'][i]);ni=norm(gi)
     per_crop.append({'slot':i,'GT_count':int(terms['GT_count'][i]),'GT_skeleton_mass':float(terms['GT_skeleton_mass'][i]),'loss':float(terms['per_crop'][i].detach()),'raw_parameter_L2':ni,'cosine_with_batch_A0':dot(gb,gi)/(nb*ni) if ni else None})
     for acc,g in zip(ga,gi):acc.add_(g,alpha=.25)
     za.add_(zi,alpha=.25);del gi,zi
    direct,zd=grad(terms['loss'])
    difference=norm(tuple(a-b for a,b in zip(ga,direct)))
    direct_norm=norm(direct)
    reduction_check={'parameter_difference_L2':difference,'parameter_relative_L2_error':difference/max(direct_norm,1e-30),'parameter_max_absolute_error':max(float((a-b).abs().max()) for a,b in zip(ga,direct)),'original_componentwise_test_passed':all(torch.allclose(a,b,atol=1e-7,rtol=2e-4) for a,b in zip(ga,direct)),'logit_gradient_exact':torch.equal(za,zd),'logit_gradient_max_absolute_error':float((za-zd).abs().max()),'parameter_relative_L2_limit':1e-4,'note':'separate FP32 convolution/reduction accumulation is not required to match coordinatewise near cancellation; independently verify pre-network logit derivative and full gradient relative norm'}
    write_new(attempt/f'aggregation_{j:02d}.json',reduction_check)
    if not torch.allclose(za,zd,atol=1e-9,rtol=1e-5) or reduction_check['parameter_relative_L2_error']>1e-4:raise RuntimeError('logit or global parameter linearity check failed; no tolerance auto-retry')
    # Use directly differentiated auxiliary for the authoritative raw ratio.
    na=norm(direct);dp=dot(gb,direct)
    if not math.isfinite(na) or na<=0:raise RuntimeError('auxiliary gradient missing/zero')
    cosine=dp/(nb*na);ratio=na/nb
    layers={n:{'A0_L2':norm((a,)),'raw_aux_L2':norm((b,)),'dot':dot((a,),(b,))} for n,a,b in zip(names,gb,direct)}
    if any(p.grad is not None for p in ps) or state_digest(model.state_dict())!=model_hash or state_digest(rng())!=rng_hash:raise RuntimeError('CPU diagnostic state side effect')
    row={'batch_index':j,'original_T0_step':spec['step_original_T0_one_based'],'draws':batch['draw'].tolist(),'filenames':list(batch['filename']),'geometry':batch['geometry'].tolist(),'input_sha256':list(batch['input_sha256']),'loss_A0':float(base.detach()),'raw_aux_loss':float(terms['loss'].detach()),'base_parameter_L2':nb,'raw_aux_parameter_L2':na,'raw_aux_to_base_L2':ratio,'cosine':cosine,'unscaled':True,'preclip':True,'zero_eta_value_and_parameter_grad_exact':True,'per_crop_gradient_aggregation_verified':True,'aggregation_check':reduction_check,'per_crop':per_crop,'per_layer':layers,'forward_seconds':forward_time,'aux_loss_seconds':loss_time,'all_gradient_seconds':time.monotonic()-t,'peak_RSS_KiB':peak_rss()}
    records.append(row);write_new(attempt/f'batch_{j:02d}.json',row)
    ledger['pending']=None;ledger['completed_batches']+=1;atomic_json(attempt/'cpu_ledger.json',ledger)
    print('BATCH_COMPLETE',j+1,'/8','base',nb,'raw_aux',na,'ratio',ratio,'cos',cosine,'seconds',forward_time+time.monotonic()-t+loss_time,flush=True)
    del z,base,zero,terms,gb,ga,direct,za,zd
  signal.alarm(0)
  eta=calibrate_eta([r['raw_aux_to_base_L2'] for r in records])
  for row in records:
   row['eta']=eta;row['weighted_aux_to_base_L2']=eta*row['raw_aux_to_base_L2'];row['weighted_aux_parameter_L2']=eta*row['raw_aux_parameter_L2']
   n0,n1,c=row['base_parameter_L2'],row['weighted_aux_parameter_L2'],row['cosine']
   row['combined_parameter_L2_from_verified_linear_gradients']=math.sqrt(max(0,n0*n0+n1*n1+2*n0*n1*c))
  assert max(r['weighted_aux_to_base_L2'] for r in records)<=.10000000001
  for p,h in sources.items():assert sha(Path(p))==h,p
  report={'status':'PASSED_CPU_ONLY_GPU_NOT_RELEASED','CPU_backend':'ATen FP32, MKLDNN disabled for reference additivity check; not a GPU backend change','k':k,'eta':eta,'rho':.10,'eta_policy':'min(1,.10/max raw aux-to-A0 real parameter gradient ratios on original eight M0 batches)','batches':records,'M0_parameter_count':sum(p.numel() for p in ps),'M0_model_state_unchanged':True,'RNG_unchanged':True,'input_files_unchanged':len(sources),'peak_RSS_KiB':peak_rss(),'elapsed_seconds':time.monotonic()-started,'GPU_calls':0,'optimizer_steps':0,'parameter_updates':0,'official_Test_content_accessed':False,'non_Train_data_open_guard_active':True,'CUDA_AMP_verified':False,'updated_trajectory_resume_verified':False,'next':'Sol review -> user separately approves GPU-only diagnostics; never auto-launch'}
  write_new(attempt/'calibration.json',report);write_new(attempt/'source_manifest.json',sources)
  ledger['status']='COMPLETE_CPU_ONLY';atomic_json(attempt/'cpu_ledger.json',ledger)
  write_new(attempt/'completion.json',{k:v for k,v in report.items() if k!='batches'})
  print(json.dumps({k:v for k,v in report.items() if k not in ['batches']},ensure_ascii=False),flush=True)
 except BaseException as e:
  signal.alarm(0);ledger['status']='FAILED_STOP';ledger['failed_batches']+=bool(ledger['pending']);ledger['exception']=repr(e)
  atomic_json(attempt/'cpu_ledger.json',ledger)
  write_new(attempt/'failure.json',{'exception':repr(e),'ledger':ledger,'peak_RSS_KiB':peak_rss(),'elapsed_seconds':time.monotonic()-started,'GPU_calls':0,'parameter_updates':0})
  raise

def main():
 p=argparse.ArgumentParser(description='Candidate C CPU-only zero-update preparation; GPU unsupported')
 p.add_argument('--run-cpu',action='store_true');args=p.parse_args()
 if not args.run_cpu:p.error('explicit --run-cpu and bound CPU authorization required')
 run()

if __name__=='__main__':main()
