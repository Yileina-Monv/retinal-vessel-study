"""Terminal-only evaluation adapter; parent science files remain immutable."""
import importlib.util,sys,os,time
from pathlib import Path
PARENT=Path(__file__).resolve().parent.parent
_spec=importlib.util.spec_from_file_location('_nl_parent_common',PARENT/'common.py')
_parent=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_parent)
for _name in ['ROOT','CODE','DATA_BASE','P','BASELINE','SEED','read','sha','atomic','log','protocol','card','authority_entry']:
 globals()[_name]=getattr(_parent,_name)
SCIENCE=_parent.GROUP
BRANCH=Path(__file__).resolve().parent
GROUP=SCIENCE/'terminal_evaluation_20261103'
GPU='${GPU_UUID}'
MARKER='NL_FIXED_ENDPOINT_EVALUATION_RELEASED_20261103'
STAGE=GROUP/'runs'/f'seed_{SEED}'
def terminal_ok(t,p):
 b=t['counts']
 if t['status']!='TRAINING_COMPLETE_FIXED_ENDPOINT_EVALUATION_PENDING' or t['epochs']!=100 or b['epoch']!=100 or b['phase']!='complete' or b['pending'] is not None:raise RuntimeError('incomplete training')
 if b['attempts']!=25001 or b['success']+b['skip']!=25000 or b['unknown']!=1 or b['inherited_unknown']!=1 or b['fatal'] or b['skip']>=25 or b['optimizer_calls']!=b['success'] or b['monitor_complete']!=5000:raise RuntimeError('invalid inherited cumulative budget')
 if p['status']!='PASS' or p['logical_train_pairs']!=25000 or p['physical_attempts']!=25001 or p['monitor_ID_pairs']!=5000:raise RuntimeError('incomplete pairing')
 return True
_cache={'time':0}
def validate(full=False):
 r=read(GROUP/'EVALUATION_RELEASE.json')
 if r['scope']!='TWO_FIXED_ENDPOINT_DEV28_ONLY' or r['GPU']!=GPU or r['seeds']!=[2027,2028] or r['new_training'] or r['Test_allowed'] or r['automatic_retry']:raise RuntimeError('wrong evaluation scope')
 if r['forward_cap_per_seed']!=67 or r['hours_cap_per_seed']!=.5 or r['total_forward_cap']!=134:raise RuntimeError('budget drift')
 if MARKER not in (CODE/'STATUS.md').read_text()[:16000]:raise RuntimeError('release revoked by STATUS')
 if sha(GROUP/'CPU_REVIEW.json')!=r['cpu_review_sha256'] or read(GROUP/'CPU_REVIEW.json')['status']!='PASS':raise RuntimeError('CPU review binding')
 if full or time.monotonic()-_cache['time']>30:
  for path,h in r['runtime_sources'].items():
   if sha(path)!=h:raise RuntimeError('runtime drift '+path)
  _cache['time']=time.monotonic()
 if full:
  for path,h in r['immutable_sources'].items():
   if sha(path)!=h:raise RuntimeError('immutable drift '+path)
  for seed in [2027,2028]:
   s=GROUP/'runs'/f'seed_{seed}'
   t=read(s/'training_completion.json');terminal_ok(t,read(s/'PAIRING_COMPLETE.json'))
   if t['counts']!=read(s/'budget.json'):raise RuntimeError('completion vs cumulative budget mismatch')
   if sha(t['checkpoint_final'])!=t['checkpoint_final_sha256']:raise RuntimeError('wrong final checkpoint')
 return sha(GROUP/'EVALUATION_RELEASE.json')
def runtime_authorizer(key,role):
 from retinal_m0.maples_g1_safety import process_identity
 if role!='infer':raise RuntimeError('only terminal inference admitted')
 p=STAGE/'INFERENCE_SESSION.json'
 for _ in range(100):
  if p.exists():break
  time.sleep(.05)
 s=read(p);h=sha(p)
 def check():
  if sha(p)!=h or process_identity()!=s['worker'] or process_identity(os.getppid())!=s['controller'] or os.environ.get('NNUNET100_NONCE')!=s['nonce'] or s['seed']!=SEED or s['role']!='infer':raise RuntimeError('session binding failed')
  job=read(GROUP/'job_state.json')
  if job.get('active_seed')!=SEED or job.get('role')!='infer' or job.get('status')!='RUNNING' or validate()!=key:raise RuntimeError('revoked evaluation')
  if time.time()-s['started_epoch']>1800:raise RuntimeError('inference30min exhausted')
 check();return check
