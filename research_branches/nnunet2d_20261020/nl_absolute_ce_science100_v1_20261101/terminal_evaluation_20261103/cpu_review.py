import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
import copy,itertools,ast
import infer,controller
checks=[]
def passed(name):checks.append(name)
for seed in [2027,2028]:
 s=GROUP/'runs'/f'seed_{seed}';t=read(s/'training_completion.json');p=read(s/'PAIRING_COMPLETE.json')
 terminal_ok(t,p);assert t['counts']==read(s/'budget.json')
 assert sha(t['checkpoint_final'])==t['checkpoint_final_sha256']
 old=SCIENCE/'runs'/f'seed_{seed}'
 assert sha(old/'budget.json')==t['counts']['parent_budget_sha256']
 def rows(path):
  with path.open() as f:
   for line in f:
    if line.strip():yield readrow(line)
 import json
 def readrow(line):return json.loads(line)
 valid=lambda r:r['classification'] in ['SUCCESS','AMP_SKIP']
 hist=[r for r in rows(old/'attempts.jsonl') if valid(r)]
 tail=[r for r in rows(s/'attempts.jsonl') if valid(r)]
 n0=list(rows(BASELINE/'runs'/f'seed_{seed}'/'attempts.jsonl'))
 actual=hist+tail
 assert len(hist)==p['parent_training_pairs'] and len(tail)==p['tail_training_pairs']
 assert len(actual)==len(n0)==25000
 for i,(a,b) in enumerate(zip(actual,n0),1):
  assert a['attempt']==b['attempt']==i
  for k in ['epoch','iteration','ids','input_sha256','target_sha256','lr']:assert a[k]==b[k],(seed,i,k)
 assert sum(r['classification']=='SUCCESS' for r in actual)==t['counts']['success']
 assert sum(r['classification']=='AMP_SKIP' for r in actual)==t['counts']['skip']
 indexes=list(rows(s/'logical_physical_index.jsonl'))
 assert len(indexes)==len(tail)
 for r,idx in zip(tail,indexes):
  assert r['attempt']==idx['logical_input'] and idx['physical_attempt']==idx['logical_input']+1 and idx['inherited_unknown']==1
 for field,bad in [('attempts',25000),('unknown',0),('success',0),('fatal',1),('skip',25),('monitor_complete',4999),('pending',{'x':1})]:
  fake=copy.deepcopy(t);fake['counts'][field]=bad
  try:terminal_ok(fake,p)
  except RuntimeError:pass
  else:raise AssertionError('bad terminal admitted '+field)
 q=read(BASELINE/'runs'/f'seed_{seed}'/'protocol.json')
 assert len(q['Dev_ids'])==28 and sum(q['inference']['expected_tiles_by_image'].values())==67
 assert not any((s/name).exists() for name in ['dev_predictions','INFERENCE_SESSION.json','inference_budget.json','infer.log'])
 passed(f'seed{seed}:full25000input_GT_ID_LR_pairing_and_budget_lineage_finalSHA_7rejections')
for x in [0,5,-1,True]:
 try:infer.verify_tile_count(x,4)
 except RuntimeError:pass
 else:raise AssertionError('bad tile count')
infer.verify_tile_count(1,4);infer.verify_tile_count(4,4);passed('native_tile_limit')
class Dummy:
 def _internal_predict_sliding_window_return_logits(self):raise ValueError('original_error')
d=Dummy();state={'running':True,'calls':0}
def fail(e):state.update(running=False,calls=state['calls']+1)
infer.prevent_native_retry(d,fail,lambda:state['running'])
try:d._internal_predict_sliding_window_return_logits()
except ValueError:pass
else:raise AssertionError()
try:d._internal_predict_sliding_window_return_logits()
except RuntimeError:pass
else:raise AssertionError()
assert state['calls']==1;passed('failure_no_native_retry')
try:controller.run_child(2027,'train')
except RuntimeError as e:assert 'forbidden' in str(e)
else:raise AssertionError('training accepted')
passed('train_role_denied')
try:validate()
except FileNotFoundError:pass
else:raise AssertionError('missing release accepted')
passed('missing_release_denied')
assert (BRANCH/'analyze.py').read_bytes()==(PARENT/'analyze.py').read_bytes()
old=(PARENT/'infer.py').read_text();new=(BRANCH/'infer.py').read_text()
assert old[old.index('    doc=json.loads'):]==new[new.index('    doc=json.loads'):]
passed('original_prediction_body_and_analysis_unchanged')
assert 'a14_gpu1.lock' in (BRANCH/'controller.py').read_text()
assert "for seed in [2027,2028]:completed.append(run_child(seed,'train'))" not in (BRANCH/'controller.py').read_text()
assert "'started_epoch':time.time()" in (BRANCH/'controller.py').read_text()
passed('resource_lock_session_deadline_and_no_train_loop')
atomic(GROUP/'CPU_REVIEW.json',{'status':'PASS','reviewer':'Sol','checks':checks,'terminal_log_pairs_per_seed':25000,'GPU_calls':0,'parameter_updates':0,'historical_unknown_retained':2,'monitor_tensor_historical_limit':True,'time':time.time()},True)
print('CPU_PASS',checks,flush=True)
