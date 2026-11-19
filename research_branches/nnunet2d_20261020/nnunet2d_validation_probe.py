"""Measure a few fold=all Train-only validation batches; no optimizer calls."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch

ROOT=Path('${RETINA_PROJECT_ROOT}'); EXP=ROOT/'experiments/nnunet2d_branch_20261020'
PLAN=EXP/'preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json'
DATA=EXP/'raw/Dataset501_MAPLES/dataset.json'
CKPT=EXP/'engineering/attempts/train50/checkpoint_stage50.pth'
OUT=EXP/'engineering/train_internal_val_probe.json'

def digest(obj):
    h=hashlib.sha256()
    def walk(x):
        if torch.is_tensor(x):
            t=x.detach().cpu().contiguous(); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str): h.update(str(k).encode()); walk(x[k])
        elif isinstance(x,(list,tuple)):
            for y in x: walk(y)
        else: h.update(repr(x).encode())
    walk(obj); return h.hexdigest()

def main():
    os.environ.update({'CUDA_VISIBLE_DEVICES':'0','nnUNet_raw':str(EXP/'raw'),
      'nnUNet_preprocessed':str(EXP/'preprocessed'),'nnUNet_results':str(EXP/'results'),
      'nnUNet_compile':'false','nnUNet_n_proc_DA':'1','OMP_NUM_THREADS':'1',
      'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs
    plans=load_json(str(PLAN)); plans['continue_training']=False
    trainer=nnUNetTrainer_100epochs(plans,'2d','all',load_json(str(DATA)),torch.device('cuda:0'))
    trainer.initialize(); trainer.load_checkpoint(str(CKPT))
    model_before=digest(trainer.network.state_dict()); opt_before=digest(trainer.optimizer.state_dict())
    train_loader,val_loader=trainer.get_dataloaders()
    trainer.network.eval(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(0)
    records=[]; start=time.monotonic()
    with torch.no_grad():
        for i in range(5):
            batch=next(val_loader); tick=time.monotonic()
            result=trainer.validation_step(batch)
            torch.cuda.synchronize(0); elapsed=time.monotonic()-tick
            loss=float(np.asarray(result['loss']).reshape(()))
            if not np.isfinite(loss): raise FloatingPointError('nonfinite Train-only validation loss')
            records.append({'batch':i+1,'case_ids':list(map(str,batch.get('keys',[]))),
              'input_shape':list(batch['data'].shape),'loss':loss,'elapsed_seconds':elapsed})
    elapsed=time.monotonic()-start
    model_after=digest(trainer.network.state_dict()); opt_after=digest(trainer.optimizer.state_dict())
    report={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
      'split':'fold=all internal Train set only','batches':records,'batch_count':len(records),
      'elapsed_seconds':elapsed,'median_validation_batch_seconds':float(np.median([r['elapsed_seconds'] for r in records])),
      'validation_batches_per_100_epochs_from_trainer':int(trainer.num_val_iterations_per_epoch)*100,
      'measured_validation_gpu_hours_estimate':float(np.median([r['elapsed_seconds'] for r in records]))*int(trainer.num_val_iterations_per_epoch)*100/3600,
      'peak_memory_allocated_bytes':int(torch.cuda.max_memory_allocated(0)),
      'peak_memory_reserved_bytes':int(torch.cuda.max_memory_reserved(0)),
      'model_unchanged':model_before==model_after,'optimizer_unchanged':opt_before==opt_after,
      'optimizer_calls':0,'dev_or_test_read':False,'dev_model_calls':0,'training_run_called':False,
      'checkpoint_sha256':hashlib.sha256(CKPT.read_bytes()).hexdigest()}
    assert report['model_unchanged'] and report['optimizer_unchanged']
    OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for loader in (train_loader,val_loader):
        if hasattr(loader,'_finish'): loader._finish()
    torch.cuda.synchronize(0); torch.cuda.empty_cache()
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
