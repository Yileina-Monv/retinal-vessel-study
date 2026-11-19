"""CPU integration, immutable data/source admission and genuine CLI refusals."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent;REV=HERE.parent/'amp_recovery_revision_20261021';OLD=HERE.parent/'no_wall_limit_restart_02';sys.path[:0]=[str(HERE),str(REV),str(OLD)]
import json,subprocess,hashlib,importlib.util,importlib.metadata as md
from amp32_gate import STAGE,CODE,BRANCH,PHASE,sha,canonical,protocol
from amp_observer_v2 import atomic
import torch
torch.set_num_threads(1)
q=protocol();ev=STAGE/'cpu_preparation';ev.mkdir(exist_ok=False)
legacy=STAGE.parent/'scientific_seed2026_100epochs_v2_no_wall_limit/repair_amp_cpu_20261021/attempt_01/verification.json';old=json.loads(legacy.read_text())
for p,h in old['protected_sha256_before'].items():assert sha(p)==h,p
for n,h in old['candidate_code_sha256'].items():assert sha(REV/n)==h,n
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(HERE),str(REV),str(OLD),str(CODE)]),CUDA_VISIBLE_DEVICES='')
t=subprocess.run([sys.executable,'-B',str(HERE/'test_amp_observer_v2.py'),'-v'],env=env,cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
(ev/'integration_tests.log').write_text(t.stdout)
if t.returncode:raise RuntimeError('CPU integration tests failed; retained log')
assert 'Ran 8 tests' in t.stdout and '\nOK' in t.stdout
print('CPU_INTEGRATION8_PASSED',flush=True)
base=STAGE.parent;pre=base/'preprocessed/Dataset501_MAPLES';plans=pre/'nnUNetPlans_8GBtarget.json';assert sha(plans)==q['plans_sha256']
versions={n:md.version(n) for n in ['nnunetv2','torch','batchgenerators','batchgeneratorsv2']};assert versions=={'nnunetv2':'2.8.1','torch':'2.4.0','batchgenerators':'0.25.3','batchgeneratorsv2':'0.3.5'}
os.environ.update(nnUNet_raw=str(base/'raw'),nnUNet_preprocessed=str(base/'preprocessed'),nnUNet_results=str(ev/'constructor_only'),nnUNet_compile='false',nnUNet_n_proc_DA='1')
from nnunet2d_scientific_constructor import build_fresh_trainer
trainer=build_fresh_trainer(json.loads(plans.read_text()),json.loads((pre/'dataset.json').read_text()),torch.device('cpu'))
assert trainer.network is None and trainer.optimizer is None and not torch.cuda.is_initialized()
from amp32_runner import bind_train_datasets
actual_train,actual_monitor=bind_train_datasets(trainer,q['train_ids'])
assert set(actual_train.identifiers)==set(actual_monitor.identifiers)==set(q['train_ids'])
names=set(q['train_ids']);sources={str(plans):sha(plans),str(pre/'dataset.json'):sha(pre/'dataset.json')};folder=pre/'nnUNetPlans_8GBtarget_2d';counts={'.b2nd':0,'.pkl':0}
for p in folder.iterdir():
 if not p.is_file():raise RuntimeError('unexpected preprocessed directory')
 if p.name.endswith('_seg.b2nd'):name=p.name[:-len('_seg.b2nd')]
 elif p.suffix in ['.b2nd','.pkl']:name=p.stem
 else:raise RuntimeError('unexpected preprocessed file: '+p.name)
 if name not in names:raise RuntimeError('non-Train file: '+p.name)
 sources[str(p)]=sha(p);counts[p.suffix]+=1
assert counts=={'.b2nd':220,'.pkl':110},counts
official=Path(list(importlib.util.find_spec('nnunetv2').submodule_search_locations)[0]);code={str(p):sha(p) for p in HERE.glob('*.py')}
for n in ['revised_policy.py','revised_runtime.py','revised_recovery.py']:code[str(REV/n)]=sha(REV/n)
for n in ['nnunet2d_scientific_constructor.py','nnunet2d_scientific_runtime.py']:code[str(OLD/n)]=sha(OLD/n)
for p in official.rglob('*.py'):code[str(p)]=sha(p)
code[str(STAGE/'protocol.json')]=sha(STAGE/'protocol.json')
runtime={p:h for p,h in code.items() if str(official) not in p or p.endswith(('/nnUNetTrainer.py','/polylr.py'))}
binding={'phase':PHASE,'protocol_sha256':sha(STAGE/'protocol.json'),'code_sources':code,'runtime_code_sources':runtime,'data_sources':sources,'versions':versions,'source_root_actual':str(official),'CPU_tests':8,'legacy37_evidence_sha256':sha(legacy),'Train_only_counts':counts};key=canonical(binding);binding['binding_sha256']=key;atomic(STAGE/'cpu_binding.json',binding)
denials=[]
for argv in [['--run-authorized'],['--worker','--parent-pid','1','--nonce','wrong'],['--run-authorized','--steps','33'],['--run-authorized','--epochs','100']]:
 r=subprocess.run([sys.executable,'-B',str(HERE/'amp32_runner.py'),*argv],env=env,cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
 assert r.returncode!=0,(argv,r.stdout);denials.append({'argv':argv,'exit_code':r.returncode,'output':r.stdout[-2500:]})
atomic(ev/'CLI_refusals.json',denials)
assert not torch.cuda.is_initialized() and not (STAGE/'RUN_SESSION.json').exists()
atomic(STAGE/'CPU_REVIEW.json',{'status':'CPU_ACCEPTED','reviewer_executor':'Sol','binding_sha256':key,'new_tests':8,'legacy_tests_reused':37,'legacy_sources_unchanged':True,'actual_official_run_training_train_step_fixture':True,'fixed_input_new_process_restore_exact':True,'native_queue_resume_supported':False,'actual_CPU_constructor_passed':True,'actual_retinal_model_initialized':False,'preprocessed_Train_files':330,'versions':versions,'CLI_denials':4,'GPU_calls':0,'Dev_Test_pixels_read':False,'scientific100epochs_authorized':False,'integration_log_sha256':sha(ev/'integration_tests.log'),'CLI_log_sha256':sha(ev/'CLI_refusals.json')})
print('CPU_REVIEW_COMPLETE',key,flush=True)
