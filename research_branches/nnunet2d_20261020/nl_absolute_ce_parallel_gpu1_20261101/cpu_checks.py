import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
import common,controller as new,ast,tempfile,importlib.util,types,copy,fcntl
from unittest.mock import patch
OLD=BRANCH.parent/'nl_absolute_ce_science100_v1_20261101'
checks=[]
def ok(v,n):
 if not v:raise AssertionError(n)
 checks.append(n)
def refuses(fn,n):
 try:fn()
 except (RuntimeError,ValueError,FileExistsError,BlockingIOError):checks.append(n);return
 raise AssertionError('not rejected '+n)
q=card();base=read(ORIGINAL_GROUP/'protocol.json');validate_document(q)
for k in ['method','training','evaluation','gradients','checkpoints','inference','pairing']:ok(q[k]==base[k],'unchanged scientific '+k)
ok(GPU=='${GPU_UUID}','explicit physicalGPU1 UUID')
ok((GROUP/'runs').resolve()==(ORIGINAL_GROUP/'runs').resolve(),'one canonical output and budgets')
for n in ['geometry.py','aux_loss.py','selection.py','sidecar.py','nl_observer.py','infer.py','analyze.py']:ok(sha(BRANCH/n)==sha(OLD/n),'unchanged '+n)
line="    if SEED!=2028:raise RuntimeError('GPU1 delegation permits training seed2028 only')\n"
ok((BRANCH/'train.py').read_text().replace(line,'')==(OLD/'train.py').read_text(),'train only adds seed2028 admission guard')
for section,key,val in [('method','lambda',.03),('runtime','device','${GPU_UUID}'),('training','train_attempts_per_seed',25001),('authorization','science_training_approved',False)]:
 b=copy.deepcopy(q);b[section][key]=val;refuses(lambda:validate_document(b),'scope refusal '+section+'.'+key)
# Exercise actual old run_child exclusive-open-before-Popen behavior with CPU mocks.
spec=importlib.util.spec_from_file_location('old_queue',OLD/'controller.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
with tempfile.TemporaryDirectory(prefix='nl-gpu1-delegation-') as d:
 d=Path(d);stage=d/'runs/seed_2028';stage.mkdir(parents=True);(stage/'train.log').write_text('actual delegated stdout reservation')
 with patch.object(old,'GROUP',d),patch.object(old,'validate',lambda **k:'mockrelease'),patch.object(old,'resource_check',lambda:None),patch.object(old.subprocess,'Popen') as spawn:
  refuses(lambda:old.run_child(2028,'train'),'native old queue refuses canonical claimed2028')
  ok(spawn.call_count==0,'duplicate prevented BEFORE child/GPU launch')
 (stage/'train.log').unlink()
 class NoRealSpawn(RuntimeError):pass
 def marker(*a,**k):raise NoRealSpawn('mock child reached')
 with patch.object(old,'GROUP',d),patch.object(old,'validate',lambda **k:'mockrelease'),patch.object(old,'resource_check',lambda:None),patch.object(old.subprocess,'Popen',marker):
  try:old.run_child(2028,'train')
  except NoRealSpawn:ok(True,'unclaimed fake stage reaches single spawn gate')
 claim={'completed_phases':[{'seed':2027,'role':'train','exit_code':0}],'error':"FileExistsError: "+str(ORIGINAL_GROUP/'runs/seed_2028/train.log')}
 with patch.object(new,'complete2027',lambda:True):
  ok(new.expected_retirement(claim),'precise delegated queue retirement accepted')
  for name,doc in [('training_fault',{**claim,'error':'training failure'}),('wrong_exit',{**claim,'completed_phases':[{'seed':2027,'role':'train','exit_code':1}]}),('wrong_seed',{**claim,'completed_phases':[{'seed':2028,'role':'train','exit_code':0}]}),('partial_phase',{**claim,'completed_phases':[]})]:ok(not new.expected_retirement(doc),'not suppressed '+name)
 with patch.object(new,'complete2027',lambda:False):ok(not new.expected_retirement(claim),'incomplete2027 never retired as success')
 lock1=(d/'lock').open('a+');lock2=(d/'lock').open('a+');fcntl.flock(lock1,fcntl.LOCK_EX|fcntl.LOCK_NB);refuses(lambda:fcntl.flock(lock2,fcntl.LOCK_EX|fcntl.LOCK_NB),'device lock duplicate refused');lock1.close();lock2.close()
for f in BRANCH.glob('*.py'):ast.parse(f.read_text());ok(True,'syntax '+f.name)
# Reuse verified stream and full-network engineering, never launch another GPU diagnostic.
r=read(ORIGINAL_GROUP/'cpu_integration/RESULT.json');ok(r['status']=='PASS' and next(x for x in r['seeds'] if x['seed']==2028)['matched_training_batches']==32,'reuse exact native2028 stream32batch check')
ok(read(ORIGINAL_GROUP/'CPU_ENTRY_TESTS.json')['status']=='PASS','reuse persistent counters and realCPU gradient tests')
for p,h in read(ORIGINAL_GROUP/'SOURCE_BINDING.json')['runtime_sources'].items():ok(sha(p)==h,'running source unchanged '+Path(p).name)
import torch
ok(not torch.cuda.is_initialized(),'no CUDA initialization')
result={'status':'PASS','checks':len(checks),'passed':checks,'GPU_calls':0,'updates':0,'original_running_sources_untouched':True,'no_Dev_Test_read':True,'reused_native_stream_batches2028':32,'limits':['Parallel CPU/IO contention measured by actual progress only','Cross-device initialized model/optimizer/scaler/RNG must match beforefirst forward','Expected old queue retirement only accepted after complete2027 and exact duplicate-log exception']}
atomic(GROUP/'CPU_TESTS.json',result,True);print(json.dumps({k:v for k,v in result.items() if k!='passed'}),flush=True)
