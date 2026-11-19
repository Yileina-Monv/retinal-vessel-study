"""Actual saved-state, logger, ledger and continuation CPU admission; no GPU work."""
import os,sys
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import json,subprocess,ast,copy
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from science100_gate import STAGE,CODE,PHASE,PROTOCOL_SHA,protocol,sha,canonical
from fast_observer import atomic,digest,states_finite
from resume_support import migrate_budget,rebuild_logger
import torch,numpy as np
q=protocol();S=STAGE;OLD=Path(q['continuation']['parent_stage']);oldbranch=CODE/'research_branches/nnunet2d_20261020/scientific_prefix_v4_20261025'
if (S/'CPU_REVIEW.json').exists() or (S/'TRAIN_SESSION.json').exists():raise RuntimeError('CPU stage already frozen/executed')
os.environ['PYTHONPATH']=os.pathsep.join([str(HERE),str(HERE.parent/'amp_engineering32_20261025'),str(HERE.parent/'amp_recovery_revision_20261021'),str(CODE)])
ev=S/'cpu_preparation';ev.mkdir(exist_ok=True);original=json.loads((OLD/'source_binding.json').read_text());oldreview=json.loads((OLD/'CPU_REVIEW.json').read_text());assert oldreview['status']=='CPU_ACCEPTED'
for group in ['all_code_sources','Train_sources']:
    for p,h in original[group].items():assert sha(p)==h,p
oldbudget=json.loads((OLD/'budget.json').read_text());assert sha(OLD/'budget.json')==q['continuation']['old_budget_sha256']
source=Path(q['continuation']['checkpoint']);assert sha(source)==q['continuation']['checkpoint_sha256'];saved=torch.load(source,weights_only=False,map_location='cpu')
for k,h in saved['digests'].items():assert digest(saved[k])==h,k
assert states_finite(saved['model'],saved['optimizer']);assert saved['native_current_epoch']==3 and saved['cursor']=={'epoch':3,'iteration':250,'monitor_iteration':2,'phase':'monitor'}
for name,count in [('attempts.jsonl',1000),('monitor.jsonl',152)]:
    assert sha(S/name)==sha(OLD/name)==saved['log_prefixes'][name];assert len((S/name).read_text().splitlines())==count
rows=[json.loads(x) for x in (S/'attempts.jsonl').read_text().splitlines()];assert all(r['attempt']==i+1 and r['classification']=='SUCCESS' and r['optimizer_calls']==1 for i,r in enumerate(rows))
assert len(saved['partial_train'])==250 and len(saved['partial_monitor'])==2
for output,row in zip(saved['partial_train'],rows[750:]):assert float(np.asarray(output['loss']))==row['loss']
native_path=Path(q['continuation']['native_logger_checkpoint']);assert sha(native_path)==q['continuation']['native_logger_checkpoint_sha256'];native=torch.load(native_path,weights_only=False,map_location='cpu')
logging,meta=rebuild_logger(native,saved,(OLD/'train.log').read_text())
logger={'logging':logging,'logging_digest':digest(logging),'best_ema':native['_best_ema'],'inference_allowed_mirroring_axes':native['inference_allowed_mirroring_axes'],'provenance':meta}
torch.save(logger,S/'logger_reconstruction.pt');atomic(ev/'logger_reconstruction.json',meta)
del native,saved
def methods(path):
    tree=ast.parse(path.read_text());c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FastObserver');return {n.name:ast.dump(n,include_attributes=False) for n in c.body if isinstance(n,ast.FunctionDef)}
a=methods(oldbranch/'fast_observer.py');b=methods(HERE/'fast_observer.py');assert {k:v for k,v in a.items() if k!='save_bundle'}=={k:v for k,v in b.items() if k!='save_bundle'}
assert sha(HERE/'science100_infer.py')==sha(oldbranch/'science100_infer.py')
oldanalysis=(oldbranch/'science100_analysis.py').read_text();newanalysis=(HERE/'science100_analysis.py').read_text()
import difflib
diff=list(difflib.unified_diff(oldanalysis.splitlines(),newanalysis.splitlines()));(ev/'evaluation_metadata_only.diff').write_text('\n'.join(diff)+'\n')
assert sum(x.startswith('+') and not x.startswith('+++') for x in diff)==2 and sum(x.startswith('-') and not x.startswith('---') for x in diff)==1
for name,count in [('test_fast_observer.py',10),('test_resume_support.py',4)]:
    p=subprocess.run([sys.executable,'-B',str(HERE/name)],cwd=HERE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    log=ev/(name+'.log')
    if log.exists():log.rename(ev/(name+'.previous_failed.log'))
    log.write_text(p.stdout);assert p.returncode==0 and ('Ran '+str(count)+' tests') in p.stdout,p.stdout
allcode=dict(original['all_code_sources']);allcode.update({str(p):sha(p) for p in HERE.glob('*.py')})
runtime={p:h for p,h in original['runtime_code_sources'].items() if not p.startswith(str(oldbranch)) and not p.startswith(str(OLD))};runtime.update({str(p):sha(p) for p in HERE.glob('*.py')})
for p in [S/'protocol.json',S/'SCIENTIFIC_PROTOCOL.md',S/'continuation_authorization.json',S/'logger_reconstruction.pt']:allcode[str(p)]=sha(p);runtime[str(p)]=sha(p)
for p in [source,native_path,OLD/'budget.json']:allcode[str(p)]=sha(p)
binding={'phase':PHASE,'protocol_sha256':PROTOCOL_SHA,'all_code_sources':allcode,'runtime_code_sources':runtime,'Train_sources':original['Train_sources'],'reference_sources':original['reference_sources'],'versions':original['versions'],'old_binding_sha256':original['binding_sha256'],'saved1000_state_verified':True,'logger_checkpoint_sha256':sha(S/'logger_reconstruction.pt'),'no_extra_GPU_engineering':True}
key=canonical(binding);binding['binding_sha256']=key;atomic(S/'source_binding.json',binding)
migrated=migrate_budget(oldbudget,key);assert (migrated['attempts'],migrated['monitor_complete'])==(1000,152)
atomic(S/'budget.json',migrated);atomic(S/'budget_migration.json',{'old_budget_path':str(OLD/'budget.json'),'old_budget_sha256':sha(OLD/'budget.json'),'old_consumed_budget':oldbudget,'migrated_budget':migrated,'changed_fields':['binding','status','reason removed'],'no_refund_no_reset_no_replay':True,'original_total_train_cap':25000,'remaining_train_attempts':24000,'remaining_monitor_batches':4848,'user_approval':'你别停啊'})
denials=[];env=dict(os.environ,PYTHONPATH=str(HERE)+os.pathsep+str(CODE),CUDA_VISIBLE_DEVICES='')
# Missing release/CPU receipt and invalid direct-child identities must reject before CUDA.
for name,args in [('science100_controller.py',['--run-authorized']),('science100_controller.py',['--run-authorized','--epochs','200']),('science100_train.py',['--authorized-train','--parent-pid','1','--nonce','wrong']),('science100_infer.py',['--authorized-final-inference','--parent-pid','1','--nonce','wrong'])]:
    p=subprocess.run([sys.executable,'-B',str(HERE/name),*args],env=env,cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90);assert p.returncode!=0;denials.append({'script':name,'returncode':p.returncode,'tail':p.stdout[-1800:]})
atomic(ev/'CLI_denials.json',denials)
assert not torch.cuda.is_initialized() and not (S/'TRAIN_SESSION.json').exists()
atomic(S/'CPU_REVIEW.json',{'status':'CPU_ACCEPTED','binding_sha256':key,'budget_migration_sha256':sha(S/'budget_migration.json'),'actual_saved1000_states_and_log_prefixes_verified':True,'all_training_math_AST_unchanged':True,'official_inference_unchanged':True,'evaluation_metadata_only_changed':True,'CPU_official_equivalence_and_recovery_tests':10,'new_resume_migration_logger_watchdog_tests':4,'actual_logger_rebuilt_from_native3epochs_plus250_outputs':True,'native_queue_resume_supported':False,'original_Train330_rehashed':True,'CLI_denials':4,'no_GPU_work':True,'Test_accessed':False})
print('V5_CPU_ADMISSION_ACCEPTED',key,flush=True)
