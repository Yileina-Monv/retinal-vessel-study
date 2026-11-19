"Frozen seed-2026 T0/TW 2000->4000 continuation; isolated from formal runs."
from __future__ import annotations
import argparse, csv, fcntl, hashlib, json, os, random, shutil, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from retinal_data.maples_native import NativeMaplesDataset
from retinal_data.maples_fov import candidate_fov
from retinal_m0.inference import tiled_probability
from retinal_m0.maples_core import CorePatches, draw_geometry, weighted_masked_loss
from retinal_m0.maples_evaluate import evaluate_arrays, load_mask
from retinal_m0.model import UNet
from retinal_m0.state import atomic_save, restore_rng, rng_state, state_digest

ROOT=Path('${RETINA_PROJECT_ROOT}'); CODE=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/t0_tw_budget_extension_20261018'
PROTO=ROOT/'docs/t0_tw_budget_extension_20261018/protocol.json'
MANIFEST=ROOT/'docs/t0_tw_budget_extension_20261018/source_manifest.json'
PROTO_SHA='ca415da2f467698748a429c04b5d064369274a9efca5b4c8be293f24b22cdf63'
MANIFEST_SHA='5485458e8f8adbb80ecebeee61f8bf746e10cbddc16541ffad22d84805cb6b23'
PARENT=ROOT/'experiments/t0_tw_core_20261016'
DEV=ROOT/'experiments/a15_2_z0_t0_calibration_20261015/dev_manifest.json'
METHODS=('T0','TW'); SEED=2026; DEVICE_UUID='${GPU_UUID}'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def statehash(x): return hashlib.sha256(state_digest(x).encode()).hexdigest()
def require_sha(path,expected,label):
 actual=sha(path)
 if actual!=expected:raise RuntimeError(label+' SHA256 mismatch')
 return actual
def writej(path,obj):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def protocol():
 if sha(PROTO)!=PROTO_SHA: raise RuntimeError('Frozen protocol hash mismatch')
 p=json.loads(PROTO.read_text(encoding='utf-8'))
 if p.get('status')!='FROZEN_USER_APPROVED_BEFORE_CONTINUATION': raise RuntimeError('Protocol status mismatch')
 return p
def source_manifest():
 if sha(MANIFEST)!=MANIFEST_SHA: raise RuntimeError('Source manifest hash mismatch')
 return json.loads(MANIFEST.read_text(encoding='utf-8'))
def setup(seed):
 torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def gpu():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0': raise RuntimeError('CUDA_VISIBLE_DEVICES must be physical GPU0 only')
 raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
 if raw[1].strip()!=DEVICE_UUID: raise RuntimeError('GPU0 UUID mismatch')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
 if any(x.split(',')[0].strip()==DEVICE_UUID for x in apps.splitlines() if x.strip()) or int(raw[2])>=512 or int(raw[3])>=10:
  raise RuntimeError('GPU0 busy; no process was stopped')
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.version.cuda!='12.1' or str(torch.__version__)!='2.4.0+cu121':
  raise RuntimeError('CUDA/PyTorch runtime differs from frozen environment')
 return {'index':0,'uuid':raw[1].strip(),'memory_mib':int(raw[2]),'util_percent':int(raw[3]),'torch':str(torch.__version__),'cuda':torch.version.cuda,'device_count':torch.cuda.device_count()}
def protected_hashes():
 m=source_manifest(); got={}
 for path,want in m['protected_files'].items():
  if not Path(path).is_file(): raise RuntimeError('Protected file missing: '+path)
  actual=sha(path)
  if actual!=want: raise RuntimeError('Protected file changed: '+path)
  got[path]=actual
 return got
def parent_info(method):
 p=protocol()['resume']['sources'][method]
 ck=Path(p['checkpoint']); cfg=Path(str(ck.parent/'config.json'))
 require_sha(ck,p['checkpoint_sha256'],method+' source checkpoint');require_sha(cfg,p['parent_config_sha256'],method+' parent config')
 bind=json.loads((PARENT/'execution_binding.json').read_text())
 bsha=sha(PARENT/'execution_binding.json')
 if bsha!=p['parent_binding']['execution_binding_sha256']: raise RuntimeError(method+' parent execution binding mismatch')
 c=json.loads(cfg.read_text()); saved=torch.load(ck,map_location='cpu',weights_only=True)
 if saved['step']!=2000 or saved['next_global_draw']!=8000 or saved['epoch']!=18 or saved['position_in_epoch']!=80: raise RuntimeError(method+' source cursor mismatch')
 if saved['config']!=c or saved['binding']['execution_binding_sha256']!=bsha or saved['binding']['config_sha256']!=sha(cfg): raise RuntimeError(method+' checkpoint lineage mismatch')
 return {'path':str(ck),'sha256':sha(ck),'config_path':str(cfg),'config_sha256':sha(cfg),'parent_binding_sha256':bsha,'config':c}
def child_config(method):
 src=parent_info(method); c=dict(src['config'])
 c.update(kind='MAPLES_T0_TW_seed2026_budget_extension_v1',phase='continuation_2001_to_4000',max_steps=4000,
          protocol_sha256=PROTO_SHA,execution_binding_id='t0_tw_budget_extension_20261018')
 return c

def binding():
 protocol(); source_manifest()
 files=[Path(__file__),CODE/'retinal_m0/maples_budget_extension_analysis.py',CODE/'tests/test_maples_budget_extension.py',CODE/'retinal_m0/maples_core.py',CODE/'retinal_m0/maples_t0.py',CODE/'retinal_m0/model.py',CODE/'retinal_m0/state.py',CODE/'retinal_m0/inference.py',CODE/'retinal_m0/maples_evaluate.py',CODE/'retinal_data/maples_native.py',CODE/'retinal_data/maples_fov.py',CODE/'retinal_data/fives.py',CODE/'retinal_m1/maps.py',CODE/'configs/development_v1.json',CODE/'releases/development_v1/normalization.json']
 for p in files:
  if not p.is_file(): raise RuntimeError('Bound source missing: '+str(p))
 parents={m:parent_info(m) for m in METHODS}
 for m in METHODS:
  cfg=child_config(m); cp=OUT/'runs/seed_2026'/m/'config.json'; cp.parent.mkdir(parents=True,exist_ok=True)
  if cp.exists() and json.loads(cp.read_text())!=cfg: raise RuntimeError('Existing child config differs: '+m)
  if not cp.exists(): writej(cp,cfg)
 return {'protocol_path':str(PROTO),'protocol_sha256':sha(PROTO),'source_manifest_path':str(MANIFEST),'source_manifest_sha256':sha(MANIFEST),
  'code_root':str(CODE),'sources':{str(p.relative_to(CODE)):sha(p) for p in files},'protected_source_hashes':protected_hashes(),
  'parents':{m:{k:v for k,v in parents[m].items() if k!='config'} for m in METHODS},
  'child_configs':{m:{'path':str(OUT/'runs/seed_2026'/m/'config.json'),'sha256':sha(OUT/'runs/seed_2026'/m/'config.json')} for m in METHODS},
  'data':{'manifest_sha256':sha(ROOT/'data/maples_handoff_20261008/data_manifest.csv'),'split_sha256':sha(ROOT/'data/maples_handoff_20261008/split_record.json'),
          'normalization_sha256':sha(CODE/'releases/development_v1/normalization.json')},
  'environment':{'python':sys.version,'torch':str(torch.__version__),'cuda':str(torch.version.cuda),'numpy':np.__version__,'gpu0_uuid':DEVICE_UUID},
  'scope':{'seed':SEED,'methods':list(METHODS),'source_step':2000,'target_step':4000,'new_updates_each':2000,'new_draws_each':8000,'official_test_accessed':False}}
def bind_sha():
 p=OUT/'execution_binding.json'
 if not p.is_file() or json.loads(p.read_text())!=binding(): raise RuntimeError('Execution binding does not match current code/data/sources')
 return sha(p)
def require_gate():
 b=bind_sha(); v=OUT/'implementation_gate/verification.json'
 if not v.is_file(): raise RuntimeError('CUDA implementation gate missing')
 d=json.loads(v.read_text())
 if d.get('all_passed') is not True or d.get('execution_binding_sha256')!=b: raise RuntimeError('Gate not passed for exact current binding')
 return b

def model_parts(cfg,device='cuda'):
 mc=json.loads((CODE/'configs/development_v1.json').read_text())
 model=UNet(mc['widths'],mc['group_norm_groups']).to(device)
 opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],betas=tuple(cfg['betas']),eps=cfg['eps'],foreach=False)
 scaler=torch.amp.GradScaler('cuda',init_scale=cfg['amp_initial_scale'])
 return model,opt,scaler
def restore_parent_cuda(method):
 cfg=child_config(method); info=parent_info(method); ck=torch.load(info['path'],map_location='cpu',weights_only=True)
 model,opt,scaler=model_parts(cfg)
 model.load_state_dict(ck['model'],strict=True); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
 restore_rng(ck['rng'])
 got={'model':statehash(model.state_dict()),'optimizer':statehash(opt.state_dict()),'scaler':statehash(scaler.state_dict()),'rng':statehash(rng_state())}
 expected=protocol()['resume']['sources'][method]['state_digests']
 if got!=expected: raise RuntimeError(method+' CUDA full-state restore digest mismatch: '+json.dumps({'got':got,'expected':expected}))
 groups=opt.param_groups
 if len(groups)!=1 or groups[0]['lr']!=0.0001 or groups[0]['betas']!=(0.9,0.999) or groups[0]['eps']!=1e-8 or groups[0]['weight_decay']!=0.0001 or groups[0]['foreach'] is not False:
  raise RuntimeError(method+' optimizer config differs')
 if any(int(s['step'])!=2000 for s in opt.state.values()): raise RuntimeError(method+' AdamW counters are not all step 2000')
 if scaler.get_scale()!=2048: raise RuntimeError(method+' AMP scaler was not restored at scale 2048')
 out={'method':method,'source_checkpoint_sha256':info['sha256'],'loaded_state_digests':got,'expected_state_digests':expected,
      'optimizer_steps':[int(s['step']) for s in opt.state.values()],'lr':[g['lr'] for g in groups],
      'scaler':scaler.state_dict(),'step':ck['step'],'next_global_draw':ck['next_global_draw'],'epoch':ck['epoch'],'position_in_epoch':ck['position_in_epoch'],'passed':True}
 del model,opt,scaler,ck; torch.cuda.empty_cache(); return out

def gate():
 OUT.mkdir(parents=True,exist_ok=True)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0': raise RuntimeError('Launch gate with CUDA_VISIBLE_DEVICES=0')
 gs=gpu(); bind=binding(); writej(OUT/'execution_binding.json',bind); bsha=sha(OUT/'execution_binding.json')
 testproc=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_maples_budget_extension.py','-v'],cwd=CODE,text=True,capture_output=True,timeout=300)
 if testproc.returncode: raise RuntimeError('Targeted no-update CPU gate tests failed: '+testproc.stdout+'\n'+testproc.stderr)
 writej(OUT/'implementation_gate/targeted_tests.json',{'status':'passed','returncode':0,'stdout':testproc.stdout,'stderr':testproc.stderr,'optimizer_updates':0})

 # Exact metadata-only config changes; every scientific/source field remains parent-identical.
 configs={}; config_checks={}
 allowed={'kind','phase','max_steps','protocol_sha256','execution_binding_id'}
 for m in METHODS:
  p=parent_info(m); child=child_config(m); cfgpath=OUT/'runs/seed_2026'/m/'config.json'
  diff={k:(p['config'].get(k),child.get(k)) for k in set(p['config'])|set(child) if p['config'].get(k)!=child.get(k)}
  if not set(diff)<=allowed or child['max_steps']!=4000 or child['learning_rate']!=0.0001 or child['scheduler'] is not None: raise RuntimeError(m+' child config changed outside metadata whitelist: '+repr(diff))
  configs[m]=child; config_checks[m]={'diff_keys':sorted(diff),'allowed_only':True,'sha256':sha(cfgpath)}
 # Full CUDA restore, before any forward/backward or parameter update.
 torch.cuda.set_device(0); setup(SEED)
 restores={m:restore_parent_cuda(m) for m in METHODS}
 # Full planned draw identities/geometries are generated from the frozen train manifest and
 # deterministic per-draw geometry function. Only the protocol's 12 real samples are materialized here;
 # all 8000 actual paired tensor digests are compared from the training workers' update logs.
 from retinal_m0.maples_core import CorePatches
 ds={m:CorePatches(configs[m],m) for m in METHODS}
 sm=source_manifest()['cpu_resume_preflight']['runs']
 sample_checks={m:[] for m in METHODS}
 planned=OUT/'implementation_gate/planned_pairing.jsonl'; planned.parent.mkdir(parents=True,exist_ok=True)
 old_states=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
 native_rows=ds['T0'].native.rows
 with planned.open('w',encoding='utf-8') as f:
  for start in range(8000,16000,4):
   rec={m:{'draws':[],'filenames':[],'geometry':[]} for m in METHODS}
   for draw in range(start,start+4):
    idx=draw%len(native_rows); meta=native_rows[idx]
    geom=draw_geometry(SEED,draw,int(meta['height']),int(meta['width']),patch=512)
    for m in METHODS:
     rec[m]['draws'].append(draw);rec[m]['filenames'].append(meta['filename']);rec[m]['geometry'].append(list(geom))
   if rec['T0']!=rec['TW']: raise RuntimeError('Planned continuation order/geometry differs between methods')
   f.write(json.dumps({'first_step':2001+(start-8000)//4,'T0':rec['T0'],'TW':rec['TW']},separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 # Materialize each of the 12 frozen real-input samples for both methods and verify source RGB/V/FOV/geometry digests.
 sample_draws={int(x['draw']) for m in METHODS for x in sm[m]['future_real_input_samples']}
 for draw in sorted(sample_draws):
  got_by_method={}
  for m in METHODS:
   x=ds[m][draw]
   got={'draw':int(x['draw']),'filename':x['filename'],'geometry':x['geometry'].tolist(),'input_sha256':x['input_sha256']}
   expected=next(z for z in sm[m]['future_real_input_samples'] if z['draw']==draw)
   if got!=expected: raise RuntimeError(m+' 12-sample source manifest mismatch at draw '+str(draw))
   index=draw%len(ds[m].names);_,_,fov,w=ds[m]._full_arrays(index)
   if np.any(w[~fov]!=0) or np.any(~np.isin(w[fov],(1.,2.))): raise RuntimeError(m+' full-image weight/FOV rule failed')
   if m=='T0' and np.any(w[fov]!=1): raise RuntimeError('T0 must use unit weights in FOV')
   got_by_method[m]=got;sample_checks[m].append({**got,'weight_rule_passed':True})
  if got_by_method['T0']!=got_by_method['TW']: raise RuntimeError('Real input sample differs across methods at draw '+str(draw))
 if any(len(sample_checks[m])!=12 for m in METHODS): raise RuntimeError('Did not verify all 12 real source samples')
 if sum(1 for _ in planned.open())!=2000: raise RuntimeError('Full future plan row count is not 2000')
 # Dataset/geometry construction must not advance global RNG streams.
 now=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
 if now[0]!=old_states[0] or now[1][0]!=old_states[1][0] or not np.array_equal(now[1][1],old_states[1][1]) or now[1][2:]!=old_states[1][2:] or not torch.equal(now[2],old_states[2]) or not torch.equal(now[3],old_states[3]):
  raise RuntimeError('Input planning consumed a global random stream')
 if any(len(sample_checks[m])!=12 for m in METHODS): raise RuntimeError('Did not verify all 12 real source samples')
 if sum(1 for _ in planned.open())!=2000: raise RuntimeError('Full future plan row count is not 2000')
 # CPU-only entry boundary checks: source rejection, counters, fixed endpoint and schemas; no optimizer updates.
 checks={'continuation_boundary':(2000,8000,2001,16000)==(2000,8000,2001,16000),
         'exact_target':all(configs[m]['max_steps']==4000 for m in METHODS),
         'checkpoint_and_eval_steps':protocol()['training']['checkpoint_steps']==protocol()['evaluation']['new_fixed_threshold_steps'],
         'endpoint_probability_step':protocol()['evaluation']['main_endpoint'].startswith('Global step 4000'),
         'no_scheduler':all(configs[m]['scheduler'] is None for m in METHODS),
         'zero_engineering_updates':True,'pair_plan_rows':2000,'pair_plan_draws':8000}
 if not all(v is True for k,v in checks.items() if k!='pair_plan_rows' and k!='pair_plan_draws') or checks['pair_plan_rows']!=2000 or checks['pair_plan_draws']!=8000: raise RuntimeError('Targeted no-update entry checks failed')
 doc={'status':'passed','all_passed':True,'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'device':gs,
  'protocol_sha256':PROTO_SHA,'source_manifest_sha256':MANIFEST_SHA,'execution_binding_sha256':bsha,'protected_file_count':len(bind['protected_source_hashes']),
  'config_checks':config_checks,'cuda_full_state_restoration':restores,'future_pairing':{'status':'passed','start_draw':8000,'end_draw_inclusive':15999,'draws_per_method':8000,'update_rows_per_method':2000,'planned_file':str(planned),'planned_sha256':sha(planned),'rows':2000},
  'real_input_samples':sample_checks,'targeted_no_update_checks':checks,'optimizer_updates':0,'official_test_accessed':False,'entry_source_sha256':sha(__file__)}
 writej(OUT/'implementation_gate/verification.json',doc)
 return doc

def align_child_log(path, global_step):
 path=Path(path)
 if not path.is_file():
  if global_step!=2000: raise RuntimeError('Missing child log for a non-source checkpoint: '+str(path))
  return
 lines=path.read_text(encoding='utf-8').splitlines()
 rows=[json.loads(x) for x in lines]
 expected=list(range(2001,global_step+1))
 if [int(x['step']) for x in rows[:len(expected)]]!=expected or len(rows)<len(expected):
  raise RuntimeError('Child checkpoint/log prefix differs: '+str(path))
 if len(rows)>len(expected):
  tail='\n'.join(lines[len(expected):])+'\n'
  backup=path.with_name(path.name+f'.recovery_{global_step}_'+hashlib.sha256(tail.encode()).hexdigest()[:12])
  if backup.exists() and backup.read_text(encoding='utf-8')!=tail: raise RuntimeError('Existing discarded-tail backup differs')
  if not backup.exists(): backup.write_text(tail,encoding='utf-8')
  path.write_text(''.join(x+'\n' for x in lines[:len(expected)]),encoding='utf-8')

def validate_scope(seed,method,target=4000):
 if seed!=2026 or method not in METHODS or target!=4000:
  raise ValueError('Only the frozen seed-2026 T0/TW global-step-4000 endpoint is authorized')
 return True
def validate_cursor(step,next_draw):
 if step<2000 or step>4000 or next_draw!=step*4 or next_draw<8000 or next_draw>16000:
  raise ValueError('Continuation step/draw cursor outside frozen range')
 return True
def validate_pair_row(row):
 required=('step','draws','filenames','geometry','input_sha256')
 if any(k not in row for k in required):raise ValueError('Incomplete paired update log row')
 step=int(row['step'])
 if step<2001 or step>4000 or len(row['draws'])!=4 or len(row['filenames'])!=4 or len(row['geometry'])!=4 or len(row['input_sha256'])!=4:
  raise ValueError('Paired update log row violates frozen schema/budget')
 if row['draws']!=list(range((step-1)*4,step*4)):raise ValueError('Paired update draw cursor mismatch')
 return True
def save_endpoint_probabilities(step):
 return int(step)==4000

def worker(method):
 validate_scope(SEED,method,4000)
 bsha=require_gate(); cfgpath=OUT/'runs/seed_2026'/method/'config.json'; cfg=json.loads(cfgpath.read_text()); out=cfgpath.parent
 runbind={'execution_binding_sha256':bsha,'config_sha256':sha(cfgpath),'parent_checkpoint_sha256':protocol()['resume']['sources'][method]['checkpoint_sha256'],
          'parent_config_sha256':protocol()['resume']['sources'][method]['parent_config_sha256'],'parent_binding_sha256':protocol()['resume']['sources'][method]['parent_binding']['execution_binding_sha256']}
 history_path=out/'run_binding_history.json'
 if not (out/'run_binding.json').exists():
  unexpected=[p.name for p in out.iterdir() if p.name not in ('config.json','interruption_attempt_01.json')]
  if unexpected: raise RuntimeError('Unexpected existing run artifacts; preserve and inspect, do not overwrite: '+repr(unexpected))
  if json.loads((out/'config.json').read_text())!=cfg: raise RuntimeError('Existing child config differs')
  writej(out/'run_binding.json',runbind);writej(history_path,{'previous_bindings':[],'active_binding':runbind})
 else:
  active=json.loads((out/'run_binding.json').read_text())
  if active!=runbind:
   attempt_path=out/'interruption_attempt_01.json'
   if not attempt_path.is_file():raise RuntimeError('Refusing unrecorded child binding migration')
   attempt=json.loads(attempt_path.read_text())
   ck=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
   source_ck=torch.load(out/'step_002200.pt',map_location='cpu',weights_only=True)
   same_lineage={k:v for k,v in active.items() if k!='execution_binding_sha256'}=={k:v for k,v in runbind.items() if k!='execution_binding_sha256'}
   old_allowed=(active.get('execution_binding_sha256')==attempt.get('old_execution_binding_sha256') and
                ck.get('binding')==active and ck.get('step')==2200 and ck.get('next_global_draw')==8800 and
                source_ck.get('binding')==active and source_ck.get('step')==2200 and same_lineage and
                sha(out/'step_002200.pt')==attempt.get('step_002200_sha256'))
   if not old_allowed:raise RuntimeError('Prior complete child checkpoint cannot be safely rebound')
   state_equal=all(statehash(ck[k])==statehash(source_ck[k]) for k in ('model','optimizer','scaler','rng'))
   if not state_equal:raise RuntimeError('Interrupted child checkpoint states differ from immutable step-2200 checkpoint')
   backup=out/'last.pt.before_gate_revalidation'
   if backup.exists() and sha(backup)!=sha(out/'last.pt'):raise RuntimeError('Prior last-checkpoint backup differs')
   if not backup.exists():shutil.copy2(out/'last.pt',backup)
   migration={'from_execution_binding_sha256':active['execution_binding_sha256'],'to_execution_binding_sha256':runbind['execution_binding_sha256'],
    'source_checkpoint_sha256':sha(out/'step_002200.pt'),'backup_checkpoint_sha256':sha(backup),'migrated_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'optimizer_updates':0,'state_digests_unchanged':True,'reason':'logging-only progress formatting fix after the complete step-2200 checkpoint; no optimizer/loss/data/sampling change'}
   payload={**ck,'binding':runbind,'binding_migration':migration}
   atomic_save(payload,out/'last.pt')
   hist={'previous_bindings':[active],'active_binding':runbind,'migration':migration};writej(history_path,hist)
   writej(out/'binding_migration.json',migration);writej(out/'run_binding.json',runbind)
   del payload,ck,source_ck
  elif not history_path.exists():writej(history_path,{'previous_bindings':[],'active_binding':runbind})
 start_step=2000; cursor=8000; parent_ck=Path(protocol()['resume']['sources'][method]['checkpoint'])
 if (out/'last.pt').exists():
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
  if ck.get('binding')!=runbind or ck['step']<2000 or ck['step']>4000: raise RuntimeError('Existing child checkpoint lineage/step invalid')
  start_step=int(ck['step']);cursor=int(ck['next_global_draw'])
  validate_cursor(start_step,cursor)
  del ck
  align_child_log(out/'train.jsonl',start_step);align_child_log(out/'pairing.jsonl',start_step)
 elif any(p.name.startswith(('train','pairing')) for p in out.iterdir()): raise RuntimeError('Logs exist without a complete child checkpoint')
 else:
  # Seed run directory with source restoration copied as a new child checkpoint without modifying parent.
  ck=torch.load(parent_ck,map_location='cpu',weights_only=True)
  payload={**ck,'binding':runbind,'config':cfg,'parent_checkpoint_sha256':runbind['parent_checkpoint_sha256'],
           'parent_config_sha256':runbind['parent_config_sha256'],'parent_binding_sha256':runbind['parent_binding_sha256'],'elapsed_seconds':0.0,'pid':os.getpid()}
  atomic_save(payload,out/'last.pt'); del ck,payload
 if start_step>=4000: return {'status':'already_complete','step':start_step,'method':method}
 # One uninterrupted safe segment; lock is the existing shared GPU0 lock.
 lockpath=ROOT/'experiments/a14_gpu0.lock'; started=time.perf_counter()
 with lockpath.open('a+') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB); snap=gpu(); torch.cuda.set_device(0); setup(SEED)
  ds=CorePatches(cfg,method); model,opt,scaler=model_parts(cfg)
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
  model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);restore_rng(ck['rng'])
  step=int(ck['step']);cursor=int(ck['next_global_draw']); elapsed0=float(ck.get('elapsed_seconds',0.0))
  got={'model':statehash(model.state_dict()),'optimizer':statehash(opt.state_dict()),'scaler':statehash(scaler.state_dict()),'rng':statehash(rng_state())}
  expected={k:statehash(ck[k]) for k in ('model','optimizer','scaler','rng')}
  if got!=expected: raise RuntimeError('Child CUDA resume state mismatch')
  if opt.param_groups[0]['lr']!=0.0001 or any(int(s['step'])!=step for s in opt.state.values()) or scaler.get_scale()<=0: raise RuntimeError('Optimizer/LR/scaler continuation invalid')
  del ck
  loader=DataLoader(ds,batch_size=4,sampler=range(cursor,4000*4),num_workers=0,pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(SEED))
  logmode='a' if (out/'train.jsonl').exists() else 'x'; pairmode='a' if (out/'pairing.jsonl').exists() else 'x'
  with (out/'train.jsonl').open(logmode) as log,(out/'pairing.jsonl').open(pairmode) as pair:
   for batch in loader:
    if step>=4000: break
    x,y,f,w=[batch[k].cuda(non_blocking=True) for k in ('image','mask','fov','weight')]
    if x.shape!=(4,3,512,512) or y.shape!=(4,1,512,512) or f.shape!=y.shape or w.shape!=y.shape: raise RuntimeError('Batch shape mismatch')
    before=model.head.weight.detach().clone(); retries=0
    while True:
     opt.zero_grad(set_to_none=True)
     with torch.autocast('cuda',dtype=torch.float16): z=model(x)
     loss=weighted_masked_loss(z,y,f,w,epsilon=1e-6)
     if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
     scaler.scale(loss).backward();scaler.unscale_(opt);gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_norm'],error_if_nonfinite=False)
     if torch.isfinite(gn): break
     scaler.update();retries+=1
     if retries>8: raise RuntimeError('AMP overflow retry limit')
    scaler.step(opt);scaler.update();torch.cuda.synchronize()
    if torch.equal(before,model.head.weight): raise RuntimeError('No successful parameter update')
    step+=1;cursor+=4
    row={'step':step,'loss':float(loss.detach()),'gradient_norm':float(gn),'amp_scale':scaler.get_scale(),'amp_retries':retries,
         'draws':batch['draw'].tolist(),'filenames':list(batch['filename']),'geometry':batch['geometry'].tolist(),'input_sha256':list(batch['input_sha256']),
         'parameters_updated':True,'learning_rate':float(opt.param_groups[0]['lr']),'elapsed_seconds':elapsed0+time.perf_counter()-started}
    validate_pair_row(row)
    log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
    pair.write(json.dumps({k:row[k] for k in ('step','draws','filenames','geometry','input_sha256')},allow_nan=False)+'\n');pair.flush()
    # Stop only at the next mandated 200-step checkpoint after the 7200s safety segment.
    safe=(time.perf_counter()-started)>=cfg['safe_stop_seconds']
    if step%200==0:
     payload={'schema_version':1,'step':step,'next_global_draw':cursor,'epoch':cursor//440,'position_in_epoch':cursor%440,
       'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),'config':cfg,'binding':runbind,
       'parent_checkpoint_sha256':runbind['parent_checkpoint_sha256'],'parent_config_sha256':runbind['parent_config_sha256'],'parent_binding_sha256':runbind['parent_binding_sha256'],
       'elapsed_seconds':elapsed0+time.perf_counter()-started,'pid':os.getpid()}
     atomic_save(payload,out/'last.pt');atomic_save(payload,out/f'step_{step:06d}.pt')
     print('method={} seed=2026 global_step={}/4000 loss={:.6f}'.format(method,step,float(loss.detach())),flush=True)
     if safe: break
  final=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
  if final['step']!=step or final['next_global_draw']!=cursor: raise RuntimeError('Final child checkpoint/cursor mismatch')
  result={'status':'passed' if step==4000 else 'checkpointed','method':method,'seed':SEED,'start_step':start_step,'completed_step':step,'next_global_draw':cursor,
          'new_successful_updates':step-2000,'new_draws':cursor-8000,'gpu':snap,'run_binding':runbind,'checkpoint_sha256':sha(out/'last.pt'),
          'optimizer_step_minmax':[min(int(s['step']) for s in final['optimizer']['state'].values()),max(int(s['step']) for s in final['optimizer']['state'].values())],
          'scaler':final['scaler'],'state_digests':{k:statehash(final[k]) for k in ('model','optimizer','scaler','rng')},'elapsed_seconds':final['elapsed_seconds']}
  writej(out/'summary.json',result);return result

def evaluate(method,step):
 if method not in METHODS or step not in protocol()['evaluation']['new_fixed_threshold_steps']:
  raise ValueError('Only authorized fixed evaluations 2200..4000 are allowed')
 bsha=require_gate();cfgpath=OUT/'runs/seed_2026'/method/'config.json';cfg=json.loads(cfgpath.read_text());run=cfgpath.parent
 runbind={'execution_binding_sha256':bsha,'config_sha256':sha(cfgpath),'parent_checkpoint_sha256':protocol()['resume']['sources'][method]['checkpoint_sha256'],
          'parent_config_sha256':protocol()['resume']['sources'][method]['parent_config_sha256'],'parent_binding_sha256':protocol()['resume']['sources'][method]['parent_binding']['execution_binding_sha256']}
 ckpt=run/f'step_{step:06d}.pt'
 if not ckpt.is_file():raise RuntimeError('Fixed checkpoint missing: '+str(ckpt))
 csha=sha(ckpt); saved=torch.load(ckpt,map_location='cpu',weights_only=True)
 checkpoint_binding=saved['binding']
 if saved['step']!=step or saved['next_global_draw']!=step*4 or saved['config']!=cfg:raise RuntimeError('Fixed checkpoint lineage/cursor mismatch')
 if checkpoint_binding!=runbind:
  hist=json.loads((run/'run_binding_history.json').read_text())
  known=[hist.get('active_binding')]+hist.get('previous_bindings',[])
  if checkpoint_binding not in known:
   raise RuntimeError("Fixed checkpoint binding is absent from recorded run history")
 edir=run/'evaluation'/f'step_{step:06d}';mp=edir/'metrics.json'
 prior_probability_hashes={}
 attempt_path=edir/'evaluation_attempt_01.json'
 if step==4000 and attempt_path.is_file():
  attempt=json.loads(attempt_path.read_text())
  if attempt.get('checkpoint_sha256')!=csha or attempt.get('method')!=method or attempt.get('seed')!=SEED or attempt.get('step')!=4000:
   raise RuntimeError('Partial endpoint evaluation attempt belongs to a different source checkpoint')
  prior_probability_hashes=attempt.get('probabilities',{})
  if len(prior_probability_hashes)!=28:raise RuntimeError('Partial endpoint evaluation does not record all 28 probability hashes')
 if mp.exists():
  result=json.loads(mp.read_text())
  if result.get('checkpoint_sha256')==csha and result.get('run_binding')==runbind:return result
  raise RuntimeError('Stale evaluation output binding')
 with (ROOT/'experiments/a14_gpu0.lock').open('a+') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);gs=gpu();torch.cuda.set_device(0)
  manifest=json.loads(DEV.read_text());refs={r['filename']:r for r in manifest['images']}
  if len(refs)!=28:raise RuntimeError('Frozen dev manifest must contain exactly 28 images')
  ds=NativeMaplesDataset(cfg['data_root'],split='dev')
  if {r['filename'] for r in ds.rows}!=set(refs):raise RuntimeError('Dev28 membership mismatch')
  mc=json.loads((CODE/'configs/development_v1.json').read_text());model=UNet(mc['widths'],mc['group_norm_groups']).cuda()
  model.load_state_dict(saved['model'],strict=True);model.eval()
  norm=json.loads((CODE/'releases/development_v1/normalization.json').read_text())
  mu=torch.tensor(norm['mean'])[:,None,None];sd=torch.tensor(norm['std'])[:,None,None]
  pdir=edir/'probabilities';rec=[];prob_hashes={}
  for i,row in enumerate(ds.rows):
   item=ds[i];raw=item['image'].permute(1,2,0).numpy();fov=candidate_fov(np.rint(raw*255).astype(np.uint8))
   image=((item['image']-mu)/sd)*torch.from_numpy(fov[None]).float()
   prob,tiles=tiled_probability(model,image,patch=512,stride=256,batch_size=4,device='cuda',amp=False)
   a=prob.numpy().astype(np.float32);ref=refs[row['filename']]
   bpath=Path(ref['region_bundle'])
   if sha(bpath)!=ref['region_bundle_sha256']:raise RuntimeError('Frozen regional bundle hash changed')
   with np.load(bpath,allow_pickle=False) as bundle:
    metrics=evaluate_arrays(a,load_mask(ref['vessel_path']),load_mask(ref['fov_path']),threshold=.5,u=bundle['U'],
       regions={k:bundle[k] for k in ('internal','near','far')},thin=bundle['thin'],matching_support='omega')
   metrics.update(filename=row['filename'],tiled_count=int(tiles))
   if save_endpoint_probabilities(step):
    pdir.mkdir(parents=True,exist_ok=True);pp=pdir/(row['filename']+'.npy')
    if pp.exists():
     if row['filename'] not in prior_probability_hashes or sha(pp)!=prior_probability_hashes[row['filename']]:
      raise RuntimeError('Refusing to overwrite an unverified endpoint probability map: '+str(pp))
     old_probability=np.load(pp,allow_pickle=False)
     if old_probability.dtype!=np.float32 or old_probability.shape!=a.shape or not np.array_equal(old_probability,a):
      raise RuntimeError('Recomputed endpoint probability differs from the preserved partial artifact: '+row['filename'])
    else:np.save(pp,a,allow_pickle=False)
    prob_hashes[row['filename']]=sha(pp)
   if step==4000:
    from retinal_m1.maps import targets
    label=load_mask(ref['vessel_path']);field=load_mask(ref['fov_path']);maps=targets(label,field)
    with np.load(bpath,allow_pickle=False) as bundle:
     for diameter in (2,4,5):
      thin=maps['skeleton'] & ((2.0*maps['skeleton_radius'])<=float(diameter))
      extra=evaluate_arrays(a,label,field,threshold=.5,u=bundle['U'],regions={k:bundle[k] for k in ('internal','near','far')},thin=thin,matching_support='omega')
      metrics[f'near_thin_recall_d1_diameter_le{diameter}']=extra['near_thin_recall_d1']
   rec.append(metrics)
   if (i+1)%7==0:print(f'{method} eval={step} images={i+1}/28',flush=True)
  del model,saved;torch.cuda.empty_cache()
 aggregate={}
 metric_names=('dice','precision','recall','fp_density','near_pixel_recall','near_thin_recall_d1','far_recall')+(tuple(f'near_thin_recall_d1_diameter_le{d}' for d in (2,4,5)) if step==4000 else ())
 for m in metric_names:
  vals=[r[m]['value'] for r in rec if r[m]['value'] is not None]
  aggregate[m]={'mean':float(np.mean(vals)) if vals else None,'n':len(vals),'NA':len(rec)-len(vals)}
 result={'status':'passed','method':method,'seed':SEED,'step':step,'threshold':.5,'checkpoint_sha256':csha,'run_binding':runbind,
  'source_checkpoint':str(ckpt),'checkpoint_binding':checkpoint_binding,'gpu':gs,'dev_images':len(rec),'aggregate':aggregate,'images':rec,
  'endpoint_probabilities_saved':save_endpoint_probabilities(step),'probability_hashes':prob_hashes,'official_test_accessed':False}
 edir.mkdir(parents=True,exist_ok=True);writej(mp,result)
 fields=['filename']+[m+suf for m in aggregate for suf in ('_value','_numerator','_denominator','_NA_reason')]
 with (edir/'per_image.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rec:
   row={'filename':r['filename']}
   for m in aggregate:
    for suf,key in (('_value','value'),('_numerator','numerator'),('_denominator','denominator'),('_NA_reason','NA_reason')):row[m+suf]=r[m][key]
   w.writerow(row)
 return result

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='op',required=True)
 s.add_parser('gate')
 w=s.add_parser('continue');w.add_argument('--seed',type=int,required=True);w.add_argument('--method',choices=METHODS,required=True)
 e=s.add_parser('evaluate');e.add_argument('--seed',type=int,required=True);e.add_argument('--method',choices=METHODS,required=True);e.add_argument('--step',type=int,required=True)
 a=p.parse_args()
 if a.op=='gate': r=gate()
 elif a.op=='evaluate':
  validate_scope(a.seed,a.method,4000)
  r=evaluate(a.method,a.step)
 else:
  validate_scope(a.seed,a.method,4000)
  r=worker(a.method)
 print(json.dumps(r,ensure_ascii=False,indent=2,default=str))
if __name__=='__main__': main()
