"""Isolated trainer and frozen evaluator for the MAPLES 2/1 exploration."""
from __future__ import annotations
import argparse,csv,fcntl,hashlib,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch.utils.data import DataLoader
import retinal_m0.maples_core_experiment as parent
from retinal_m0.maples_class_split import ClassSplitPatches,class_split_loss_with_contributions,contribution_records,align_contribution_log
from retinal_m0.maples_class_split_gate import current_binding,read_pairing
from retinal_m0.model import UNet
from retinal_m0.state import atomic_save,restore_rng,rng_state,state_digest
from retinal_data.maples_native import NativeMaplesDataset
from retinal_data.maples_fov import candidate_fov
from retinal_m0.inference import tiled_probability

ROOT=Path('${RETINA_PROJECT_ROOT}');CODE=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/t0_tw_class_split_20261018';PAIR=ROOT/'experiments/t0_tw_core_20261016'
PROTO=ROOT/'docs/t0_tw_class_split_20261018/protocol.json';PROTO_SHA='b486512cd3c3d9605fdb8f498b4e62c46c0666adcdf7544d0c7e901c8344775a'
SEEDS=(2026,2027,2028);METHOD='v2_b1';DEVICE_UUID='${GPU_UUID}'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def shastate(x):return hashlib.sha256(state_digest(x).encode()).hexdigest()
def writej(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def binding_sha():
 p=OUT/'execution_binding.json';return sha(p)
def require_binding():
 if sha(PROTO)!=PROTO_SHA:raise RuntimeError('Frozen 2/1 protocol hash mismatch')
 p=OUT/'execution_binding.json';v=OUT/'implementation_gate/verification.json'
 if not p.is_file() or not v.is_file():raise RuntimeError('Missing 2/1 execution binding or gate')
 if json.loads(p.read_text())!=current_binding():raise RuntimeError('Bound sources or data changed after gate')
 g=json.loads(v.read_text())
 if not g.get('all_passed') or g.get('execution_binding_sha256')!=sha(p):raise RuntimeError('Current 2/1 gate is not passed')
 return sha(p)
def gpu():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Expose only A7 GPU0 via CUDA_VISIBLE_DEVICES=0')
 raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
 if raw[1].strip()!=DEVICE_UUID:raise RuntimeError('A7 GPU0 UUID mismatch')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
 if any(x.split(',')[0].strip()==DEVICE_UUID for x in apps.splitlines() if x.strip()) or int(raw[2])>=512 or int(raw[3])>=10:
  raise RuntimeError('GPU0 is busy; no process was stopped')
 return {'index':0,'uuid':raw[1].strip(),'memory_mib':int(raw[2]),'util_percent':int(raw[3])}
def make_config(seed):
 if seed not in SEEDS:raise ValueError('Unapproved seed')
 c=parent.config(seed,'T0',False)
 c.update(kind='MAPLES_T0_TW_class_split_2_1_v1',method=METHOD,
  phase='approved_exploratory_class_split_2_1',loss='explicit_A_class_weighted_BCE_per_image_D_plus_original_unweighted_Dice',
  weight_definition='A=Rin|Rnear; A&V=2; A&~V=1; other FOV including U=1; outside FOV=0',
  max_steps=2000,save_every=200,protocol_sha256=PROTO_SHA,
  execution_binding_id='t0_tw_class_split_20261018',vessel_weight=2.0,background_weight=1.0,
  official_test=False)
 return c
def expected_pair(seed):
 t0=PAIR/f'runs/seed_{seed}/T0/pairing.jsonl';tw=PAIR/f'runs/seed_{seed}/TW/pairing.jsonl'
 if t0.read_bytes()!=tw.read_bytes():raise RuntimeError(f'Old T0/TW pairing differs seed={seed}')
 return read_pairing(t0)[1]
def assert_batch(batch,seed,records):
 for slot,draw in enumerate(batch['draw'].tolist()):
  expected=records[int(draw)]
  got=(int(draw),batch['filename'][slot],batch['geometry'][slot].tolist(),batch['input_sha256'][slot])
  if got!=expected:raise RuntimeError(f'Paired input changed seed={seed} draw={draw}')
def _old_initial(seed):
 t0=json.loads((PAIR/f'runs/seed_{seed}/T0/summary.json').read_text())
 tw=json.loads((PAIR/f'runs/seed_{seed}/TW/summary.json').read_text())
 for k in ('initial_model','initial_optimizer','initial_scaler','initial_rng'):
  if t0[k]!=tw[k]:raise RuntimeError(f'Old paired initial state differs seed={seed} field={k}')
 return t0
def _init(seed,c):
 torch.cuda.set_device(0);parent.setup(seed)
 mc=parent.modelcfg();model=UNet(mc['widths'],mc['group_norm_groups']).cuda().train()
 opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'],betas=tuple(c['betas']),eps=c['eps'],foreach=False)
 scaler=torch.amp.GradScaler('cuda',init_scale=c['amp_initial_scale'])
 return model,opt,scaler
def _load_m0(model,c):
 raw=torch.load(c['source_m0'],map_location='cpu',weights_only=True)
 if raw.get('step')!=10000:raise RuntimeError('Expected original M0 step 10000')
 model.load_state_dict(raw['model'],strict=True);del raw
def initial_state_check(seed,c,model,opt,scaler):
 _load_m0(model,c)
 got={'initial_model':shastate(model.state_dict()),'initial_optimizer':shastate(opt.state_dict()),
      'initial_scaler':shastate(scaler.state_dict()),'initial_rng':shastate(rng_state())}
 old=_old_initial(seed)
 if any(got[k]!=old[k] for k in got):raise RuntimeError(f'Fresh M0 initial state differs seed={seed}: {got}')
 return got
def _align_log(path,step):
 from retinal_m0.maples_t0 import align_log
 align_log(path,step)
def worker(config_path,out,resume=False):
 c=json.loads(Path(config_path).read_text());seed=int(c['seed']);out=Path(out)
 if seed not in SEEDS or c.get('method')!=METHOD or c.get('vessel_weight')!=2.0 or c.get('background_weight')!=1.0 or c.get('max_steps')!=2000:
  raise ValueError('Run settings outside the sole approved 2/1 condition')
 bsha=require_binding();runbind={'execution_binding_sha256':bsha,'config_sha256':sha(config_path),
  'source_m0_sha256':c['source_m0_sha256'],'protocol_sha256':PROTO_SHA,'method':METHOD,'seed':seed}
 if not out.resolve().is_relative_to((OUT/'runs').resolve()):raise ValueError('Run output is outside approved root')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Must expose only GPU0')
 with (ROOT/'experiments/a14_gpu0.lock').open('a+') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);snapshot=gpu();started=time.perf_counter()
  if resume and not (out/'last.pt').is_file():raise RuntimeError('Resume checkpoint missing')
  if not resume:
   out.mkdir(parents=True,exist_ok=False);writej(out/'config.json',c);writej(out/'run_binding.json',runbind)
  model,opt,scaler=_init(seed,c);dataset=ClassSplitPatches(c);records=expected_pair(seed)
  step=cursor=0;elapsed=0.;initial={}
  if resume:
   saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
   if saved['config']!=c or saved['binding']!=runbind:raise RuntimeError('Checkpoint binding mismatch')
   model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
   step=int(saved['step']);cursor=int(saved['next_global_draw']);elapsed=float(saved['elapsed_seconds'])
   initial={k:saved[k] for k in ('initial_model','initial_optimizer','initial_scaler','initial_rng')}
   _align_log(out/'train.jsonl',step);_align_log(out/'pairing.jsonl',step)
   tail=align_contribution_log(out/'bce_contributions.jsonl',step,cursor)
   restore_rng(saved['rng'])
   checks={'model':shastate(model.state_dict()),'optimizer':shastate(opt.state_dict()),'scaler':shastate(scaler.state_dict()),'rng':shastate(rng_state())}
   want={k:shastate(saved[k]) for k in ('model','optimizer','scaler','rng')}
   if checks!=want:raise RuntimeError('Restored state digest differs')
   writej(out/f'restore_at_{step:06d}.json',{'step':step,'next_global_draw':cursor,'contribution_alignment':tail,'passed':True,'digests':checks})
   del saved
  else:
   initial=initial_state_check(seed,c,model,opt,scaler)
  loader=DataLoader(dataset,batch_size=4,sampler=range(cursor,8000),num_workers=0,pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(seed))
  summary={'status':'running','method':METHOD,'seed':seed,'target':2000,'start_step':step,'next_global_draw':cursor,'gpu':snapshot,'run_binding':runbind,**initial}
  try:
   torch.cuda.reset_peak_memory_stats()
   with (out/'train.jsonl').open('a' if resume else 'x',encoding='utf-8') as log,(out/'pairing.jsonl').open('a' if resume else 'x',encoding='utf-8') as pair,(out/'bce_contributions.jsonl').open('a' if resume else 'x',encoding='utf-8') as blog:
    for batch in loader:
     if time.perf_counter()-started>=c['safe_stop_seconds']:break
     assert_batch(batch,seed,records)
     x,y,f,a=[batch[k].cuda(non_blocking=True) for k in ('image','mask','fov','area')]
     if x.shape!=(4,3,512,512) or y.shape!=(4,1,512,512) or f.shape!=y.shape or a.shape!=y.shape:raise RuntimeError('Training batch shape mismatch')
     before=model.head.weight.detach().clone();retry=0
     while True:
      opt.zero_grad(set_to_none=True)
      with torch.autocast('cuda',dtype=torch.float16):z=model(x)
      loss,bce_image,dice_image,parts=class_split_loss_with_contributions(z,y,f,a)
      if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
      scaler.scale(loss).backward();scaler.unscale_(opt)
      grad=torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm'],error_if_nonfinite=False)
      if torch.isfinite(grad):break
      scaler.update();retry+=1
      if retry>8:raise RuntimeError('AMP retry limit reached')
     scaler.step(opt);scaler.update();torch.cuda.synchronize()
     if torch.equal(before,model.head.weight):raise RuntimeError('Parameter did not update')
     step+=1;draws=batch['draw'].tolist();cursor+=4
     rows=contribution_records(parts,list(batch['filename']),seed,step,draws,batch['geometry'].tolist(),list(batch['input_sha256']),retry,bce_image,dice_image,protocol_sha256=PROTO_SHA,code_binding_sha256=bsha,run_binding_sha256=sha(out/'run_binding.json'))
     bce_batch=float(bce_image.mean().detach());dice_batch=float(dice_image.mean().detach());total=float(loss.detach())
     group_sum=sum(float(np.mean([r['groups'][g]['C'] for r in rows])) for g in ('A_V1','A_V0','O_V1','O_V0'))
     if not np.isclose(group_sum,bce_batch,rtol=1e-5,atol=1e-7) or not np.isclose(bce_batch+dice_batch,total,rtol=1e-5,atol=1e-7):raise RuntimeError('Contribution/loss reconciliation failed')
     row={'step':step,'global_draw_start':draws[0],'BCE_batch':bce_batch,'Dice_batch':dice_batch,'actual_total_loss':total,
       'gradient_norm':float(grad),'amp_scale':scaler.get_scale(),'amp_retries':retry,'draws':draws,
       'filenames':list(batch['filename']),'geometry':batch['geometry'].tolist(),'input_sha256':list(batch['input_sha256']),'parameters_updated':True}
     log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
     pair.write(json.dumps({k:row[k] for k in ('step','draws','filenames','geometry','input_sha256')})+'\n');pair.flush()
     for r in rows:blog.write(json.dumps(r,allow_nan=False)+'\n')
     blog.flush()
     if step%200==0 or step==2000:
      payload={'schema_version':1,'step':step,'next_global_draw':cursor,'epoch':cursor//440,'position_in_epoch':cursor%440,
       'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),
       'config':c,'binding':runbind,**initial,'elapsed_seconds':elapsed+time.perf_counter()-started,'pid':os.getpid()}
      atomic_save(payload,out/'last.pt');atomic_save(payload,out/f'step_{step:06d}.pt')
     if step%25==0:print(f'2/1 seed={seed} step={step}/2000 loss={total:.6f}',flush=True)
   final=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
   if final['step']!=step or final['next_global_draw']!=cursor:raise RuntimeError('Saved checkpoint cursor mismatch')
   summary.update(status='passed' if step==2000 else 'checkpointed',completed_steps=step,next_global_draw=cursor,
     model_digest=shastate(final['model']),optimizer_digest=shastate(final['optimizer']),scaler_digest=shastate(final['scaler']),rng_digest=shastate(final['rng']),
     checkpoint_sha256=sha(out/'last.pt'),elapsed_seconds=elapsed+time.perf_counter()-started,
     max_allocated_mib=torch.cuda.max_memory_allocated()/1048576,max_reserved_mib=torch.cuda.max_memory_reserved()/1048576)
   writej(out/'summary.json',summary);return summary
  except Exception:
   summary.update(status='failed',error=traceback.format_exc(),completed_steps=step,next_global_draw=cursor);writej(out/'summary.json',summary);raise

def evaluate(config_path,out,step):
 c=json.loads(Path(config_path).read_text());out=Path(out);seed=int(c['seed'])
 if seed not in SEEDS or c.get('method')!=METHOD or step not in (0,*range(200,2001,200)):raise ValueError('Evaluation point outside frozen schedule')
 bsha=require_binding();runbind={'execution_binding_sha256':bsha,'config_sha256':sha(config_path),'source_m0_sha256':c['source_m0_sha256'],'protocol_sha256':PROTO_SHA,'method':METHOD,'seed':seed}
 ckpt=out/f'step_{step:06d}.pt';csha=c['source_m0_sha256'] if step==0 else sha(ckpt)
 edir=out/'evaluation'/f'step_{step:06d}';mp=edir/'metrics.json'
 if mp.is_file():
  old=json.loads(mp.read_text())
  if old.get('checkpoint_sha256')==csha and old.get('run_binding')==runbind:return old
  raise RuntimeError('Stale evaluation binding')
 with (ROOT/'experiments/a14_gpu0.lock').open('a+') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);gs=gpu()
  dev_manifest=json.loads((ROOT/'experiments/a15_2_z0_t0_calibration_20261015/dev_manifest.json').read_text())
  ds=NativeMaplesDataset('${RETINA_PROJECT_ROOT}/data/maples_handoff_20261008',split='dev')
  refs={r['filename']:r for r in dev_manifest['images']}
  if len(refs)!=28 or {r['filename'] for r in ds.rows}!=set(refs):raise RuntimeError('Frozen dev28 membership mismatch')
  rec=[]
  if step==0:
   zdir=ROOT/'experiments/a15_2_z0_t0_calibration_20261015/z0/probabilities'
   for n in sorted(refs):rec.append(dict(filename=n,**parent.eval_prob(np.load(zdir/f'{n}.npy',allow_pickle=False),refs[n],.5)))
   source='shared frozen Z0 probabilities; no test-set access'
  else:
   saved=torch.load(ckpt,map_location='cpu',weights_only=True)
   if saved.get('step')!=step or saved.get('binding')!=runbind:raise RuntimeError('Checkpoint binding mismatch')
   mc=parent.modelcfg();model=UNet(mc['widths'],mc['group_norm_groups']).cuda();model.load_state_dict(saved['model'],strict=True);model.eval()
   norm=json.loads((CODE/'releases/development_v1/normalization.json').read_text());mu=torch.tensor(norm['mean'])[:,None,None];sd=torch.tensor(norm['std'])[:,None,None]
   pdir=edir/'probabilities'
   for i,row in enumerate(ds.rows,1):
    item=ds[i-1];raw=item['image'].permute(1,2,0).numpy();fov=candidate_fov(np.rint(raw*255).astype(np.uint8))
    image=((item['image']-mu)/sd)*torch.from_numpy(fov[None]).float()
    prob,tiles=tiled_probability(model,image,patch=512,stride=256,batch_size=4,device='cuda',amp=False)
    arr=prob.numpy().astype(np.float32);m=parent.eval_prob(arr,refs[row['filename']],.5);m.update(filename=row['filename'],tiled_count=tiles);rec.append(m)
    if step==2000:
     pdir.mkdir(parents=True,exist_ok=True);np.save(pdir/f'{row["filename"]}.npy',arr,allow_pickle=False)
    if i%7==0:print(f'2/1 seed={seed} eval={step} images={i}/28',flush=True)
   del model,saved;source=str(ckpt)
  names=('dice','precision','recall','fp_density','near_pixel_recall','near_thin_recall_d1','far_recall')
  aggregate={}
  for name in names:
   values=[r[name]['value'] for r in rec if r[name]['value'] is not None]
   aggregate[name]={'mean':float(np.mean(values)) if values else None,'n':len(values),'NA':len(rec)-len(values)}
  result={'status':'passed','method':METHOD,'seed':seed,'step':step,'threshold':.5,'checkpoint_sha256':csha,
   'run_binding':runbind,'source':source,'gpu':gs,'dev_images':len(rec),'aggregate':aggregate,
   'images':rec,'endpoint_probabilities_saved':step==2000}
  writej(mp,result);edir.mkdir(parents=True,exist_ok=True)
  fields=['filename']+[name+suf for name in aggregate for suf in ('_value','_numerator','_denominator','_NA_reason')]
  with (edir/'per_image.csv').open('w',newline='',encoding='utf-8-sig') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
   for r in rec:
    row={'filename':r['filename']}
    for name in aggregate:
     for suf,key in (('_value','value'),('_numerator','numerator'),('_denominator','denominator'),('_NA_reason','NA_reason')):row[name+suf]=r[name][key]
    writer.writerow(row)
  return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');p.add_argument('--evaluate-step',type=int)
 p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
 p.add_argument('--resume',action='store_true');a=p.parse_args()
 if not a.train and a.evaluate_step is None:raise SystemExit('Must specify --train or --evaluate-step; no implicit training')
 if a.train and a.evaluate_step is not None:raise SystemExit('Choose one operation')
 result=worker(a.config,a.output,a.resume) if a.train else evaluate(a.config,a.output,a.evaluate_step)
 print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
if __name__=='__main__':main()
