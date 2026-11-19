"""Official loop continued from committed1000; new augmentation queues disclosed."""
import os,sys,json,time,ctypes,signal,hashlib,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'no_wall_limit_restart_02';ENG=HERE.parent/'amp_engineering32_20261025';sys.path.extend([str(OLD),str(ENG)])
from science100_gate import STAGE,CODE,GPU,protocol,validate,runtime_authorizer,sha,update_status
from fast_observer import FastObserver,Journal,AmpLimits,atomic,digest,rng_pack,restore_rng,states_finite
from resume_support import continue_mid_monitor_then_official

def execute(parent,nonce):
    if os.getppid()!=parent or os.environ.get('NNUNET100_NONCE')!=nonce or len(nonce)<32:raise RuntimeError('not authenticated scientific child')
    key=validate(full=True);q=protocol();authorize=runtime_authorizer(key,'train')
    if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0 or os.getppid()!=parent:raise RuntimeError('parent death guard failed')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('PYTHONHASHSEED')!='2026':raise RuntimeError('wrong GPU/seed environment')
    import torch,numpy as np
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before scientific admission')
    from nnunet2d_scientific_constructor import build_fresh_trainer,validate_initialized_batch_size
    from nnunet2d_scientific_runtime import set_all_seeds,install_seeded_worker_adapter
    from amp32_runner import bind_train_datasets
    base=STAGE.parent;os.environ.update(nnUNet_raw=str(base/'raw'),nnUNet_preprocessed=str(base/'preprocessed'),nnUNet_results=str(STAGE/'nnunet_results'),nnUNet_compile='false',nnUNet_n_proc_DA='1')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);set_all_seeds(2026)
    if torch.cuda.device_count()!=1:raise RuntimeError('exactly released GPU0 must be visible')
    torch.cuda.set_device(0)
    plans=json.loads((base/'preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json').read_text());data=json.loads((base/'preprocessed/Dataset501_MAPLES/dataset.json').read_text())
    t=build_fresh_trainer(plans,data,torch.device('cuda:0'));seeds=install_seeded_worker_adapter();t.initialize();validate_initialized_batch_size(t)
    names=set(q['train_ids']);tr,va=bind_train_datasets(t,names)
    atomic(STAGE/'actual_loader_sets.json',{'Train':sorted(tr.identifiers),'Train_internal_monitor':sorted(va.identifiers),'source':'fold=all; no Dev','seed_metadata_phase':'before official on_train_start builds augmentation pools; see actual_seed_pools.json after first update'})
    j=Journal(STAGE/'budget.json',key,100,250,50,AmpLimits(5,25));initial=j.read()
    expected=json.loads((STAGE/'budget_migration.json').read_text())['migrated_budget']
    if initial!=expected:raise RuntimeError('resume budget differs from CPU-approved1000 boundary')
    t.plot_network_architecture=lambda:None
    t.on_train_start()  # official setup ONCE; reconstructed native worker queues
    expected_seeds=[{'pool':'train','worker_count':1,'seeds':[2026]},{'pool':'internal_train_monitor','worker_count':1,'seeds':[2027]}]
    if seeds!=expected_seeds or t.dataloader_train.num_processes!=1 or t.dataloader_val.num_processes!=1:raise RuntimeError('actual reconstructed augmentation pools drift')
    atomic(STAGE/'actual_seed_pools.json',{'adapter_calls':seeds,'train_loader_processes':1,'monitor_loader_processes':1,'queue_state':'rebuilt, not exact V4 continuation','native_queue_resume_supported':False})
    saved=torch.load(q['continuation']['checkpoint'],weights_only=False,map_location='cpu')
    logs=torch.load(STAGE/'logger_reconstruction.pt',weights_only=False,map_location='cpu')
    t.network.load_state_dict(saved['model']);t.optimizer.load_state_dict(saved['optimizer']);t.grad_scaler.load_state_dict(saved['scaler']);t.lr_scheduler.load_state_dict(saved['scheduler'])
    t.current_epoch=saved['native_current_epoch'];t.logger.load_checkpoint(logs['logging']);t._best_ema=logs['best_ema'];t.inference_allowed_mirroring_axes=logs['inference_allowed_mirroring_axes']
    restore_rng(saved['rng'])  # restore main RNG AFTER worker setup, not unrecorded prefetched queues
    actual={k:digest(v) for k,v in {'model':t.network.state_dict(),'optimizer':t.optimizer.state_dict(),'scaler':t.grad_scaler.state_dict(),'scheduler':t.lr_scheduler.state_dict(),'rng':rng_pack()}.items()}
    if any(actual[k]!=saved['digests'][k] for k in actual) or not states_finite(t.network.state_dict(),t.optimizer.state_dict()):raise RuntimeError('actual restored model/optimizer/AMP/LR/RNG mismatch')
    if digest(t.logger.get_checkpoint())!=logs['logging_digest']:raise RuntimeError('actual logger load mismatch')
    atomic(STAGE/'restoration_receipt.json',{'status':'RESTORED_BEFORE_ANY_NEW_FORWARD','available_state_digests':actual,'old_checkpoint_sha256':sha(q['continuation']['checkpoint']),'epoch':t.current_epoch,'counts':initial,'logging_digest':logs['logging_digest'],'native_queue_resume_supported':False,'queues_rebuilt_seeds':expected_seeds,'old1000_updates_retained':True,'new_GPU_forward_calls':0})
    def event(kind,counts,o):
        save=(kind=='train' and (counts['attempts'] in [1010,1100] or counts['attempts']%250==0)) or kind=='epoch'
        if save:
            o.save_bundle(STAGE/'state_last.pt',key)
            if kind=='train' and counts['attempts'] in [1010,1100]:shutil.copy2(STAGE/'state_last.pt',STAGE/f'state_resume_{counts["attempts"]:05d}.pt')
            if kind=='epoch' and counts['epoch'] in [20,40,60,80,100]:shutil.copy2(STAGE/'state_last.pt',STAGE/f'state_epoch_{counts["epoch"]:03d}.pt')
        if kind=='train' and (counts['attempts']==1 or counts['attempts']%100==0):print('SCIENCE_UPDATE',counts['attempts'],counts['success'],counts['skip'],'seconds',round(time.monotonic()-o.clock,2),flush=True)
        if kind=='epoch':update_status(execution_started=True,completed_epochs=counts['epoch'],charged_attempts=counts['attempts'],successful_updates=counts['success'],AMP_skips=counts['skip'])
        if shutil.disk_usage(STAGE).free<50*2**30:raise RuntimeError('free disk below50GiB; no deletion or budget change')
    observer=FastObserver(t,j,STAGE,names,authorize,on_event=event)
    observer.partial_train=saved['partial_train'];observer.partial_monitor=saved['partial_monitor']
    elapsed_prior=json.loads((STAGE/'attempts.jsonl').read_text().splitlines()[-1])['elapsed_seconds']
    observer.clock-=elapsed_prior
    del saved
    # The optional architecture graph is a debug forward, not training. Disable it explicitly.
    t.plot_network_architecture=lambda:None
    def stop_signal(signum,frame):raise RuntimeError('external stop '+str(signum)+'; no automatic resume')
    signal.signal(signal.SIGTERM,stop_signal)
    try:
        continue_mid_monitor_then_official(t,observer,j)
        counts=j.read();final=Path(t.output_folder)/'checkpoint_final.pth'
        if counts['attempts']!=25000 or counts['monitor_complete']!=5000 or counts['success']+counts['skip']!=25000 or counts['fatal'] or counts['unknown'] or counts['skip']>=25 or counts['status']!='READY' or counts['phase']!='complete' or t.current_epoch!=100 or not final.exists():raise RuntimeError('incomplete terminal scientific criteria')
        observer.save_bundle(STAGE/'state_last.pt',key)
        atomic(STAGE/'training_completion.json',{'status':'TRAINING_COMPLETE','epochs':100,'attempts':25000,'successful_updates':counts['success'],'AMP_skips':counts['skip'],'internal_train_monitor_batches':5000,'counts':counts,'binding_sha256':key,'checkpoint_final':str(final),'checkpoint_final_sha256':sha(final),'elapsed_seconds_excluding_interruption':time.monotonic()-observer.clock,'native_queue_resume_supported':False,'Dev_Test_calls_before_final':0,'fresh_initialization_this_process':False,'lineage':'V4 fresh seed2026 first1000 + V5 remaining24000 with disclosed rebuilt augmentation queues','replayed_training_updates':0})
    except BaseException as exc:
        j.stop(repr(exc))
        try:observer.save_bundle(STAGE/'state_interrupted.pt',key)
        except BaseException as se:atomic(STAGE/'snapshot_failure.json',{'error':repr(se)})
        atomic(STAGE/'training_failure.json',{'error':repr(exc),'counts':j.read(),'automatic_restart':False});raise
    finally:
        observer.close()
        for loader in [getattr(t,'dataloader_train',None),getattr(t,'dataloader_val',None)]:
            if hasattr(loader,'_finish'):
                try:loader._finish()
                except BaseException:pass

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--authorized-train',action='store_true',required=True);p.add_argument('--parent-pid',type=int,required=True);p.add_argument('--nonce',required=True);a=p.parse_args();execute(a.parent_pid,a.nonce)
