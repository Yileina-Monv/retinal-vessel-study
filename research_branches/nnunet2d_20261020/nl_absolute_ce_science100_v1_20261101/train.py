"""Official full100-epoch loop, guarded observer, fresh initialization only."""
import os,sys,json,time,ctypes,signal,hashlib,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'no_wall_limit_restart_02';ENG=HERE.parent/'amp_engineering32_20261025';sys.path.extend([str(OLD),str(ENG)])
from common import STAGE,CODE,GPU,protocol,validate,runtime_authorizer,sha,update_status,SEED,DATA_BASE
from fast_observer import FastObserver,Journal,AmpLimits,atomic,digest,rng_pack

def execute(parent,nonce):
    if os.getppid()!=parent or os.environ.get('NNUNET100_NONCE')!=nonce or len(nonce)<32:raise RuntimeError('not authenticated scientific child')
    key=validate(full=True);q=protocol();authorize=runtime_authorizer(key,'train')
    if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0 or os.getppid()!=parent:raise RuntimeError('parent death guard failed')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('PYTHONHASHSEED')!=str(SEED):raise RuntimeError('wrong GPU/seed environment')
    import torch,numpy as np
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before scientific admission')
    from nnunet2d_scientific_constructor import build_fresh_trainer,validate_initialized_batch_size
    from seeded_runtime import set_all_seeds,install_seeded_worker_adapter
    from amp32_runner import bind_train_datasets
    base=DATA_BASE;os.environ.update(nnUNet_raw=str(base/'raw'),nnUNet_preprocessed=str(base/'preprocessed'),nnUNet_results=str(STAGE/'nnunet_results'),nnUNet_compile='false',nnUNet_n_proc_DA='1')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);set_all_seeds(SEED)
    if torch.cuda.device_count()!=1:raise RuntimeError('exactly released GPU0 must be visible')
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(10240*1048576/torch.cuda.get_device_properties(0).total_memory)
    plans=json.loads((base/'preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json').read_text());data=json.loads((base/'preprocessed/Dataset501_MAPLES/dataset.json').read_text())
    t=build_fresh_trainer(plans,data,torch.device('cuda:0'));seeds=install_seeded_worker_adapter(SEED);t.initialize();validate_initialized_batch_size(t)
    legacy=hashlib.sha256()
    for name,value in sorted(t.network.state_dict().items()):
        a=value.detach().cpu().contiguous().numpy();legacy.update(name.encode());legacy.update(str(a.dtype).encode());legacy.update(str(a.shape).encode());legacy.update(memoryview(a).cast('B'))
    atomic(STAGE/'initialization.json',{'seed':SEED,'fresh_random_official':True,'weights_loaded':False,'legacy_model_sha256':legacy.hexdigest(),'model':digest(t.network.state_dict()),'optimizer':digest(t.optimizer.state_dict()),'scaler':digest(t.grad_scaler.state_dict()),'RNG':digest(rng_pack()),'batch':t.batch_size,'patch':list(t.configuration_manager.patch_size),'epochs':t.num_epochs,'binding_sha256':key,'native_queue_resume_supported':False})
    expected=__import__('common').read(__import__('common').BASELINE/'runs'/f'seed_{SEED}'/'initialization.json')
    actual=__import__('common').read(STAGE/'initialization.json')
    for k in ['model','optimizer','scaler','RNG']:
        if actual[k]!=expected[k]:raise RuntimeError('fresh initialization mismatch '+k)
    if list(t.loss.weight_factors)!=[64/127,32/127,16/127,8/127,4/127,2/127,1/127,0]:raise RuntimeError('native DS weights changed')
    from sidecar import install
    install()
    names=set(q['train_ids']);tr,va=bind_train_datasets(t,names)
    atomic(STAGE/'actual_loader_sets.json',{'Train':sorted(tr.identifiers),'Train_internal_monitor':sorted(va.identifiers),'source':'fold=all; no Dev','seed_metadata_phase':'before official on_train_start builds augmentation pools; see actual_seed_pools.json after first update'})
    j=Journal(STAGE/'budget.json',key,100,250,50,AmpLimits(5,25));j.initialize();initial=j.read()
    if initial['attempts'] or initial['status']!='READY':raise RuntimeError('fresh prereleased25000 budget required; no resume/reset')
    def event(kind,counts,o):
        if kind=='train' and counts['attempts']==1:
            expected_seeds=[{'pool':'train','worker_count':1,'seeds':[SEED]},{'pool':'internal_train_monitor','worker_count':1,'seeds':[SEED+1]}]
            if seeds!=expected_seeds:raise RuntimeError('actual constructed augmentation seed pools drift')
            atomic(STAGE/'actual_seed_pools.json',{'phase':'after official loader construction','adapter_calls':seeds,'train_loader_processes':t.dataloader_train.num_processes,'monitor_loader_processes':t.dataloader_val.num_processes})
        save=(kind=='train' and (counts['attempts'] in [30,200] or counts['attempts']%250==0)) or kind=='epoch'
        if save:
            o.save_bundle(STAGE/'state_last.pt',key)
            if kind=='train' and counts['attempts'] in [30,200]:shutil.copy2(STAGE/'state_last.pt',STAGE/f'state_start_{counts["attempts"]:05d}.pt')
            if kind=='epoch' and counts['epoch'] in [20,40,60,80,100]:shutil.copy2(STAGE/'state_last.pt',STAGE/f'state_epoch_{counts["epoch"]:03d}.pt')
        if kind=='train' and (counts['attempts']==1 or counts['attempts']%100==0):print('SCIENCE_UPDATE',counts['attempts'],counts['success'],counts['skip'],'seconds',round(time.monotonic()-o.clock,2),flush=True)
        if kind=='epoch':update_status(execution_started=True,completed_epochs=counts['epoch'],charged_attempts=counts['attempts'],successful_updates=counts['success'],AMP_skips=counts['skip'])
        if shutil.disk_usage(STAGE).free<50*2**30:raise RuntimeError('free disk below50GiB; no deletion or budget change')
    from nl_observer import NLObserver
    observer=NLObserver(t,j,STAGE,names,authorize,on_event=event)
    # The optional architecture graph is a debug forward, not training. Disable it explicitly.
    t.plot_network_architecture=lambda:None
    def stop_signal(signum,frame):raise RuntimeError('external stop '+str(signum)+'; no automatic resume')
    signal.signal(signal.SIGTERM,stop_signal)
    try:
        t.run_training()  # the actual official100-epoch loop; only instance observers are wrapped
        counts=j.read();final=Path(t.output_folder)/'checkpoint_final.pth'
        if counts['attempts']!=25000 or counts['monitor_complete']!=5000 or counts['success']+counts['skip']!=25000 or counts['fatal'] or counts['unknown'] or counts['skip']>=25 or counts['status']!='READY' or counts['phase']!='complete' or t.current_epoch!=100 or not final.exists():raise RuntimeError('incomplete terminal scientific criteria')
        if observer.paired_train!=25000 or observer.paired_monitor!=5000:raise RuntimeError('pairing count incomplete')
        atomic(STAGE/'PAIRING_COMPLETE.json',{'status':'PASS','train_input_GT_ID_LR_pairs':25000,'monitor_ID_pairs':5000,'monitor_tensor_hash_equality_proved':False})
        observer.save_bundle(STAGE/'state_last.pt',key)
        atomic(STAGE/'training_completion.json',{'seed':SEED,'status':'TRAINING_COMPLETE','epochs':100,'attempts':25000,'successful_updates':counts['success'],'AMP_skips':counts['skip'],'internal_train_monitor_batches':5000,'counts':counts,'binding_sha256':key,'checkpoint_final':str(final),'checkpoint_final_sha256':sha(final),'elapsed_seconds':time.monotonic()-observer.clock,'native_queue_resume_supported':False,'Dev_Test_calls_before_final':0,'fresh_initialization':True})
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
