from pathlib import Path
import sys,importlib.util,tempfile,subprocess,time,ast
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('handoff_guard',D/'queue_handoff_guard.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
from common import *
checks=[]
def ok(v,n):
 if not v:raise AssertionError(n)
 checks.append(n)
spec=importlib.util.spec_from_file_location('oldqueue_test',g.OLD/'controller.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as d:
 d=Path(d);stage=d/'runs/seed_2028';stage.mkdir(parents=True);(stage/'train.log').write_text('delegated canonical output')
 m.GROUP=d;m.validate=lambda **k:'CPU';m.resource_check=lambda:None
 e=g.prove_duplicate_rejection(stage,m)
 ok(e['duplicate_child_spawn_calls']==0,'real old duplicate gate prevents Popen')
 ok(e['filename']==str(stage/'train.log'),'real exception filename retained')
 ok(str(stage/'train.log') not in e['repr_original'],'actual repr omits filename, risk reproduced')
 ok(str(stage/'train.log') in e['str_with_actual_filename'],'actual str retains filename')
 src=(D/'controller.py').read_text();node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='expected_retirement');ctx={'complete2027':lambda:True,'ORIG':d};exec(compile(ast.Module(body=[node],type_ignores=[]),'parser','exec'),ctx)
 phases=[{'seed':2027,'role':'train','exit_code':0}];doc={'completed_phases':phases,'error':e['type']+': '+e['str_with_actual_filename']}
 ok(ctx['expected_retirement'](doc),'unchanged live controller accepts evidence enriched true exception')
 ctx['complete2027']=lambda:False;ok(not ctx['expected_retirement'](doc),'incomplete scientific run never accepted')
for exitcode in [0,3]:
 p=subprocess.Popen([sys.executable,'-B','-c','import sys;sys.exit('+str(exitcode)+')'])
 for _ in range(200):
  state,code=g.terminal_exit(p.pid)
  if state=='Z':break
  time.sleep(.02)
 ok(state=='Z' and code==exitcode*256,'kernel exit evidence '+str(exitcode));p.wait()
for p,h in read(GROUP/'SOURCE_BINDING.json')['runtime_sources'].items():ok(sha(p)==h,'original running code unchanged '+Path(p).name)
for p,h in read(g.DELEGATED/'SOURCE_BINDING.json')['runtime_sources'].items():ok(sha(p)==h,'GPU1 running code unchanged '+Path(p).name)
ast.parse((D/'queue_handoff_guard.py').read_text());ok(True,'guard syntax')
result={'status':'PASS','checks':len(checks),'GPU_calls':0,'parameter_updates':0,'actual_OS_exception_and_Z_exit_tested':True,'no_change_to_live_release_or_worker_sources':True,'passed':checks}
atomic(g.DELEGATED/'QUEUE_HANDOFF_CPU.json',result,True);print('HANDOFF_CPU_PASS',len(checks),flush=True)
