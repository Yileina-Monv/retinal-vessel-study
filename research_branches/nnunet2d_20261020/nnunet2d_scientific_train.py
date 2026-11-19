from __future__ import annotations
import argparse, hashlib, json, os, pathlib, signal, sys, time, traceback
import numpy as np
import torch
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunet2d_scientific_constructor import build_fresh_trainer, validate_initialized_batch_size
from nnunet2d_scientific_runtime import (HardBudget, StopScientificRun,
    install_seeded_worker_adapter, install_step_instrumentation, set_all_seeds)

BASE=pathlib.Path('${RETINA_PROJECT_ROOT}')
FREEZE=BASE/'docs/nnunet2d_seed2026_100epochs_20261020'
PROTO=json.loads((FREEZE/'protocol.json').read_text())
ROOT=pathlib.Path(PROTO['paths']['scientific_root'])
PREFLIGHT=pathlib.Path(os.environ.get('SCIENTIFIC_PREFLIGHT_PATH',str(ROOT/'preflight/verification.json')))
MODEL_DIR=pathlib.Path(PROTO['paths']['nnUNet_results'])/'Dataset501_MAPLES/nnUNetTrainer_100epochs__nnUNetPlans_8GBtarget__2d'
RUN=MODEL_DIR/'fold_all'

def digest_state(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode())
        a=value.detach().cpu().contiguous().numpy()
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--execute-scientific-run',action='store_true',help='required explicit guard for the authorized single run')
    args=ap.parse_args()
    if not args.execute_scientific_run: raise SystemExit('Refusing: pass --execute-scientific-run only after verified preflight and supervisor approval')
    pre=json.loads(PREFLIGHT.read_text())
    if pre.get('status')!='PASS': raise SystemExit('preflight is not PASS')
    binding_path=pathlib.Path(os.environ.get('SCIENTIFIC_BINDING_PATH',str(ROOT/'execution_binding.json')))
    if not binding_path.is_file(): raise SystemExit('source-bound scientific execution binding missing')
    if os.environ.get('PYTHONHASHSEED')!='2026': raise SystemExit('PYTHONHASHSEED must be set before interpreter startup to 2026')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0': raise SystemExit('GPU0-only mapping required')
    if os.environ.get('nnUNet_n_proc_DA')!='1': raise SystemExit('exactly one worker per Train/monitor loader required')
    if os.environ.get('nnUNet_compile','false').lower()!='false': raise SystemExit('compile must remain disabled')
    for name,path in [('nnUNet_raw',PROTO['paths']['raw']),('nnUNet_preprocessed',PROTO['paths']['preprocessed']),('nnUNet_results',PROTO['paths']['nnUNet_results'])]:
        os.environ[name]=path
    set_all_seeds(2026)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise SystemExit('visible CUDA device not available exactly once')
    torch.cuda.set_device(0); torch.cuda.init()
    dev=torch.device('cuda:0')
    plans_path=pathlib.Path(PROTO['paths']['preprocessed'])/'Dataset501_MAPLES/nnUNetPlans_8GBtarget.json'
    dataset_path=pathlib.Path(PROTO['paths']['preprocessed'])/'Dataset501_MAPLES/dataset.json'
    plans=json.loads(plans_path.read_text()); dataset=json.loads(dataset_path.read_text())
    trainer=build_fresh_trainer(plans,dataset,dev)
    used_seeds=install_seeded_worker_adapter()
    trainer.initialize()  # fresh random model/optimizer, no checkpoint load path is used
    validate_initialized_batch_size(trainer)
    init_digest=digest_state(trainer.network.state_dict())
    init_record={'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'seed':2026,'initialization':'fresh official nnUNetTrainer_100epochs after all seed setup; no checkpoint loaded','trainer_class':trainer.__class__.__name__,'model_state_sha256':init_digest,'optimizer_state_sha256':hashlib.sha256(json.dumps(trainer.optimizer.state_dict(),sort_keys=True,default=str).encode()).hexdigest(),'rng_state_torch_cpu_sha256':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cuda_uuid':torch.cuda.get_device_name(0),'worker_seed_adapter':used_seeds,'attempts_before_first_update':0,'checkpoint_load_calls':0}
    init_path=ROOT/'initial_state.json'; init_path.parent.mkdir(parents=True,exist_ok=True); init_path.write_text(json.dumps(init_record,indent=2)+'\n')
    started=float(os.environ['SCIENTIFIC_PROCESS_START_MONOTONIC'])
    install_step_instrumentation(trainer,ROOT,HardBudget(),started)
    # run_training is called only here, behind explicit mode and preflight gates.
    def watchdog_stop(signum,frame): raise StopScientificRun(f'external watchdog signal {signum}; incomplete, no automatic resume')
    signal.signal(signal.SIGTERM,watchdog_stop)
    try:
        trainer.run_training()
    except StopScientificRun as exc:
        interrupted=RUN/'checkpoint_interrupted_unresumable.pth'
        saved=False; save_error=None
        try:
            if trainer.was_initialized and trainer.optimizer is not None:
                trainer.save_checkpoint(str(interrupted)); saved=interrupted.exists()
        except Exception as save_exc: save_error=repr(save_exc)
        for obj in (getattr(trainer,'dataloader_train',None),getattr(trainer,'dataloader_val',None)):
            if hasattr(obj,'_finish'):
                try: obj._finish()
                except Exception: pass
        stop={'status':'INCOMPLETE_STOPPED','reason':str(exc),'attempts':trainer.scientific_budget.attempts,'successful_updates':sum(1 for _ in open(trainer.attempt_log_path,encoding='utf-8')) if trainer.attempt_log_path.exists() else 0,'unresumable_snapshot_written':saved,'snapshot_save_error':save_error,'automatic_resume':False}
        (ROOT/'stop_record.json').write_text(json.dumps(stop,indent=2)+'\n')
        return 3
    except BaseException as exc:
        stop={'status':'INCOMPLETE_EXCEPTION','exception':repr(exc),'traceback':traceback.format_exc(),'attempts':getattr(getattr(trainer,'scientific_budget',None),'attempts',None),'automatic_resume':False}
        (ROOT/'stop_record.json').write_text(json.dumps(stop,indent=2)+'\n')
        raise
    attempts=sum(1 for _ in open(trainer.attempt_log_path,encoding='utf-8'))
    monitors=sum(1 for _ in open(trainer.monitor_log_path,encoding='utf-8'))
    final=RUN/'checkpoint_final.pth'
    if attempts!=25000 or monitors!=5000 or not final.is_file(): raise RuntimeError(f'completion mismatch: attempts={attempts}, monitor={monitors}, final={final.is_file()}')
    import hashlib as _h
    report={'status':'TRAINING_COMPLETE','epochs':trainer.num_epochs,'iterations_per_epoch':trainer.num_iterations_per_epoch,'attempts':attempts,'successful_updates':trainer.scientific_budget.attempts,'internal_train_monitor_batches':monitors,'checkpoint_final':str(final),'checkpoint_final_sha256':_h.sha256(final.read_bytes()).hexdigest(),'fresh_initial_model_state_sha256':init_digest,'worker_seed_pools':used_seeds,'cuda_elapsed_process_seconds':time.monotonic()-started,'official_test_accessed':False,'dev_inference_performed_before_terminal_checkpoint':False}
    (ROOT/'training_completion.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
