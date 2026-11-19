"""Only the authorized32-attempt engineering mode; no scientific entry exists."""
import os,sys,json,time,subprocess,signal,hashlib,ctypes
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'no_wall_limit_restart_02';REV=HERE.parent/'amp_recovery_revision_20261021'
sys.path[:0]=[str(HERE),str(REV),str(OLD)]
from amp32_gate import STAGE,CODE,PHASE,GPU,protocol,validate,resources,runtime_authorizer,update_status,sha
sys.path.insert(0,str(CODE))
from retinal_m0.maples_g1_safety import exclusive_lock,process_identity,owner_alive
from amp_observer_v2 import Journal,Observer,AmpLimits,atomic,digest,rng_pack,save_full_state

def bind_train_datasets(trainer,names):
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    if trainer.dataset_class is None:trainer.dataset_class=infer_dataset_class(trainer.preprocessed_dataset_folder)
    train,monitor=trainer.get_tr_and_val_datasets()
    if set(train.identifiers)!=set(names) or set(monitor.identifiers)!=set(names):raise RuntimeError('both actual loader sets must be exactly Train110')
    return train,monitor

def worker(parent,nonce):
    if os.getppid()!=parent or nonce!=os.environ.get('NNUNET_AMP32_NONCE') or len(nonce)<32:raise RuntimeError('worker must be authenticated direct child')
    key=validate(full=True);q=protocol()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('PYTHONHASHSEED')!='2026':raise RuntimeError('GPU/seed process environment drift')
    if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0 or os.getppid()!=parent:raise RuntimeError('parent death guard failed')
    authorize=runtime_authorizer(key)
    import torch,numpy as np
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before worker admission')
    from nnunet2d_scientific_constructor import build_fresh_trainer,validate_initialized_batch_size
    from nnunet2d_scientific_runtime import set_all_seeds,install_seeded_worker_adapter
    base=STAGE.parent
    os.environ.update(nnUNet_raw=str(base/'raw'),nnUNet_preprocessed=str(base/'preprocessed'),nnUNet_results=str(STAGE/'nnunet_results'),nnUNet_compile='false',nnUNet_n_proc_DA='1')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);set_all_seeds(2026)
    if torch.cuda.device_count()!=1:raise RuntimeError('only released GPU0 may be visible')
    torch.cuda.set_device(0)
    plans=json.loads((base/'preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json').read_text());data=json.loads((base/'preprocessed/Dataset501_MAPLES/dataset.json').read_text())
    t=build_fresh_trainer(plans,data,torch.device('cuda:0'));seeds=install_seeded_worker_adapter();t.initialize();validate_initialized_batch_size(t)
    original=json.loads((base/'scientific_seed2026_100epochs_v2_no_wall_limit/initial_state.json').read_text())
    legacy=hashlib.sha256()
    for name,value in sorted(t.network.state_dict().items()):
        a=value.detach().cpu().contiguous().numpy();legacy.update(name.encode());legacy.update(str(a.dtype).encode());legacy.update(str(a.shape).encode());legacy.update(memoryview(a).cast('B'))
    if legacy.hexdigest()!=original['model_state_sha256']:raise RuntimeError('fresh initialization differs from frozen seed2026')
    atomic(STAGE/'initialization.json',{'fresh':True,'old_weights_loaded':False,'model_sha256':digest(t.network.state_dict()),'legacy_model_sha256':legacy.hexdigest(),'optimizer':digest(t.optimizer.state_dict()),'scaler':digest(t.grad_scaler.state_dict()),'RNG':digest(rng_pack()),'real_architecture':type(t.network).__name__,'batch':t.batch_size,'patch':list(t.configuration_manager.patch_size),'lr_horizon':t.num_epochs})
    names=set(q['train_ids']);tr,va=bind_train_datasets(t,names)
    train_loader=monitor_loader=None;o=None
    j=Journal(STAGE/'budget.json',key,1,32,5,AmpLimits(5,25))
    if j.read()['attempts'] or j.read()['status']!='READY':raise RuntimeError('fresh prereleased ledger required')
    t.disable_checkpointing=True  # engineering produces explicit full states, never best/final scientific weights
    # All actual model forwards are now guarded; loader prefetch does not call the model.
    o=Observer(t,j,STAGE,names,authorize,force_overflow=31)
    try:
        train_loader,monitor_loader=t.get_dataloaders()
        atomic(STAGE/'loader_binding.json',{'Train110_train':sorted(tr.identifiers),'Train110_monitor':sorted(va.identifiers),'seed_pools':seeds,'prefetch_batches_discarded_by_official_loader':2,'prefetch_model_forwards':0})
        t.on_epoch_start();t.on_train_epoch_start();outputs=[]
        for i in range(32):
            outputs.append(t.train_step(next(train_loader)))
            if i+1 in [30,31,32]:save_full_state(STAGE/f'state_attempt_{i+1:02d}.pt',t,j)
            print('ENGINEERING_ATTEMPT',i+1,j.read()['success'],j.read()['skip'],flush=True)
        t.on_train_epoch_end(outputs);t.on_validation_epoch_start();mon=[]
        with torch.no_grad():
            for i in range(5):mon.append(t.validation_step(next(monitor_loader)))
        t.on_validation_epoch_end(mon);t.on_epoch_end()
        dg=save_full_state(STAGE/'state_complete.pt',t,j);counts=j.read()
        if any(counts[k]!=v for k,v in {'attempts':32,'success':31,'skip':1,'fatal':0,'unknown':0,'monitor_attempts':5,'monitor_complete':5,'train_forwards':32,'monitor_forwards':5,'backwards':32,'optimizer_calls':31}.items()) or counts['phase']!='complete':raise RuntimeError('engineering counts/phase incomplete')
        rows=[json.loads(x) for x in (STAGE/'attempts.jsonl').read_text().splitlines()]
        if [x['classification'] for x in rows]!=['SUCCESS']*30+['AMP_SKIP','SUCCESS']:raise RuntimeError('unexpected actual skip sequence')
        if rows[30]['model_changed'] or rows[30]['optimizer_changed'] or not rows[31]['model_changed']:raise RuntimeError('skip/recovery state check failed')
        atomic(STAGE/'completion.json',{'status':'ENGINEERING32_COMPLETE_PENDING_SOL_REVIEW','binding_sha256':key,'counts':counts,'full_state_digests':dg,'checkpoint_sha256':sha(STAGE/'state_complete.pt'),'peak_reserved_MiB':torch.cuda.max_memory_reserved(0)/1048576,'peak_allocated_MiB':torch.cuda.max_memory_allocated(0)/1048576,'Dev_Test_calls':0,'scientific_training_started':False,'all_model_forwards_guarded':True})
    except BaseException as exc:
        j.stop(repr(exc))
        try:save_full_state(STAGE/'state_interrupted.pt',t,j)
        except BaseException as se:atomic(STAGE/'snapshot_failure.json',{'error':repr(se)})
        raise
    finally:
        if o:o.close()
        for loader in [train_loader,monitor_loader]:
            if hasattr(loader,'_finish'):
                try:loader._finish()
                except BaseException:pass

def controller():
    key=validate(full=True);q=protocol()
    with exclusive_lock(STAGE/'ENGINEERING.lock'),exclusive_lock(CODE.parents[1]/'experiments/a14_gpu0.lock'):
        if (STAGE/'RUN_SESSION.json').exists():raise RuntimeError('one session already attempted; no automatic restart')
        gpu=resources();nonce=os.urandom(24).hex();env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONHASHSEED='2026',NNUNET_AMP32_NONCE=nonce,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',nnUNet_compile='false',nnUNet_n_proc_DA='1',MPLCONFIGDIR=str(STAGE/'matplotlib'))
        env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        started=time.monotonic()
        with (STAGE/'worker.log').open('x',encoding='utf-8') as log:
            p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)), '--worker','--parent-pid',str(os.getpid()),'--nonce',nonce],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(STAGE/'RUN_SESSION.json',{'controller':process_identity(),'worker':process_identity(p.pid),'nonce':nonce,'binding_sha256':key,'GPU0':gpu})
            update_status(execution_started=True,controller_pid=os.getpid(),worker_pid=p.pid)
            timed_out=False
            while p.poll() is None:
                if time.monotonic()-started>=900:
                    timed_out=True
                    if owner_alive({'pid':p.pid,'start_time':json.loads((STAGE/'RUN_SESSION.json').read_text())['worker']['start_time']}):os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    break
                time.sleep(1)
        elapsed=time.monotonic()-started;code=p.returncode
        atomic(STAGE/'controller_result.json',{'worker_exit':code,'elapsed_seconds':elapsed,'wall_cap_seconds':900,'timed_out':timed_out,'extra_attempts_or_restarts':0})
        if code!=0 or timed_out or not (STAGE/'completion.json').exists():
            j=Journal(STAGE/'budget.json',key,1,32,5,AmpLimits(5,25));j.stop('worker_exit_'+str(code));atomic(STAGE/'failure.json',{'worker_exit':code,'counts':j.read(),'no_restart':True});update_status(state='FAILED_STOP',GPU_released=False,next='Stop; Sol review exact failed/unknown costs, no implicit retry')
            raise RuntimeError('engineering stopped; preserve logs and costs')
        update_status(state='CONSUMED',GPU_released=False,run_state='ENGINEERING_COMPLETE_PENDING_SOL_REVIEW',next='Sol engineering review; scientific100epochs still unapproved')
        print('ENGINEERING32_WORKER_COMPLETED',round(elapsed,2),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--run-authorized',action='store_true');g.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--parent-pid',type=int);p.add_argument('--nonce',default='');a=p.parse_args()
    try:
        if a.worker:worker(a.parent_pid,a.nonce)
        else:controller()
    except BaseException as e:
        print('ENGINEERING_STOP',repr(e),flush=True);raise
