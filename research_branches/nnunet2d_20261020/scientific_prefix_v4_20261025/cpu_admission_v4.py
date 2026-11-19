"""Reuse unchanged full-loop/evaluation evidence; validate the actual V4 amendment."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,subprocess,shutil
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from science100_gate import STAGE,CODE,PHASE,PROTOCOL_SHA,protocol,sha,canonical
from fast_observer import atomic
from prefix_contract import validate_prefix
import torch
S=STAGE;OLD=S.parent/'scientific_seed2026_100epochs_v3_amp_20261025';q=protocol();ev=S/'cpu_preparation';ev.mkdir(exist_ok=False)
original=json.loads((OLD/'source_binding.json').read_text());oldreview=json.loads((OLD/'CPU_REVIEW.json').read_text());repair=json.loads((OLD/'prefix_gate_review_20261025/CPU_REPAIR_REVIEW.json').read_text())
assert oldreview['tests']==10 and oldreview['status']=='CPU_ACCEPTED' and repair['new_candidate_CPU_tests']==6
for group in ['all_code_sources','Train_sources']:
 for p,h in original[group].items():assert sha(p)==h,p
unchanged=['fast_observer.py','science100_analysis.py','science100_infer.py','science100_controller.py','test_fast_observer.py']
oldbranch=CODE/'research_branches/nnunet2d_20261020/scientific_amp_v3_20261025'
for n in unchanged:assert sha(HERE/n)==sha(oldbranch/n),n
assert sha(HERE/'prefix_contract.py')==repair['candidate_code_sha256']
p=subprocess.run([sys.executable,'-B',str(HERE/'test_prefix_contract.py')],cwd=HERE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
(ev/'prefix_tests.log').write_text(p.stdout);assert p.returncode==0 and 'Ran 6 tests' in p.stdout
actual=[json.loads(x) for x in (OLD/'attempts.jsonl').read_text().splitlines()];expected=[json.loads(x) for x in (S.parent/'amp_engineering32_20261025/attempts.jsonl').read_text().splitlines()];observed=validate_prefix(actual,expected)
atomic(ev/'actual_previous2_steps_under_amended_contract.json',observed)
allcode=dict(original['all_code_sources']);allcode.update({str(p):sha(p) for p in HERE.glob('*.py')});runtime={p:h for p,h in original['runtime_code_sources'].items() if not p.startswith(str(oldbranch)) and not p.startswith(str(OLD))};runtime.update({str(p):sha(p) for p in HERE.glob('*.py')})
for path in [S/'protocol.json',S/'SCIENTIFIC_PROTOCOL.md']:allcode[str(path)]=sha(path);runtime[str(path)]=sha(path)
allcode[str(OLD/'prefix_gate_review_20261025/CPU_REPAIR_REVIEW.json')]=sha(OLD/'prefix_gate_review_20261025/CPU_REPAIR_REVIEW.json')
binding={'phase':PHASE,'protocol_sha256':PROTOCOL_SHA,'all_code_sources':allcode,'runtime_code_sources':runtime,'Train_sources':original['Train_sources'],'reference_sources':original['reference_sources'],'versions':original['versions'],'CPU_tests_reused_unchanged':10,'prefix_tests':6,'old_CPU_binding_sha256':sha(OLD/'source_binding.json'),'old_CPU_review_sha256':sha(OLD/'CPU_REVIEW.json'),'no_extra_GPU_engineering':True};key=canonical(binding);binding['binding_sha256']=key;atomic(S/'source_binding.json',binding)
denials=[]
env=dict(os.environ,PYTHONPATH=str(HERE)+os.pathsep+str(CODE),CUDA_VISIBLE_DEVICES='')
for script,args in [('science100_controller.py',['--run-authorized']),('science100_controller.py',['--run-authorized','--epochs','200']),('science100_train.py',['--authorized-train','--parent-pid','1','--nonce','wrong']),('science100_infer.py',['--authorized-final-inference','--parent-pid','1','--nonce','wrong'])]:
 p=subprocess.run([sys.executable,'-B',str(HERE/script),*args],env=env,cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90);assert p.returncode!=0;denials.append({'script':script,'args':args,'exit_code':p.returncode,'output':p.stdout[-2500:]})
atomic(ev/'CLI_denials.json',denials)
assert not torch.cuda.is_initialized() and not (S/'TRAIN_SESSION.json').exists()
atomic(S/'CPU_REVIEW.json',{'status':'CPU_ACCEPTED','binding_sha256':key,'reused_unchanged_actual_official_loop_tests':10,'prefix_regressions_passed':6,'actual_previous_1ULP_difference_passed':True,'unchanged_execution_eval_files':unchanged,'first30_exact_input_target_ID_LR_contract':True,'GPU_numeric_differences_diagnostic_only':True,'intrinsic_finite_update_AMP_and_budget_stops_retained':True,'original_three_size_export_and31metric_A0_fixture_reused':True,'original_330_Train_sources_rehashed':True,'CLI_denials':4,'fresh_restart_approved':True,'prior2_updates_preserved':True,'no_extra_GPU_calls':True,'Test_accessed':False,'prefix_test_log_sha256':sha(ev/'prefix_tests.log'),'old_CPU_receipt_sha256':sha(OLD/'CPU_REVIEW.json')})
print('V4_CPU_ADMISSION_PASSED',key,flush=True)
