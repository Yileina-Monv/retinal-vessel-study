"""Hard-capped nnU-Net engineering updates. Never calls run_training()."""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path('${RETINA_PROJECT_ROOT}')
EXP = ROOT / 'experiments/nnunet2d_branch_20261020'
PLANS = EXP/'preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json'
DATASET_JSON = EXP/'raw/Dataset501_MAPLES/dataset.json'
TRAIN_MANIFEST = ROOT/'docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json'

def jsonl(path, obj):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(obj,sort_keys=True,default=str)+'\n'); f.flush(); os.fsync(f.fileno())

def state_digest(obj):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            t=x.detach().cpu().contiguous()
            h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str): h.update(str(k).encode()); visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for y in x: visit(y)
        elif isinstance(x,np.ndarray): h.update(str(x.dtype).encode()); h.update(x.tobytes())
        else: h.update(repr(x).encode())
    visit(obj); return h.hexdigest()

def rng_pack(trainer):
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda_0':torch.cuda.get_rng_state(0),
            'scheduler':trainer.lr_scheduler.state_dict()}

def restore_rng(trainer, packed):
    random.setstate(packed['python']); np.random.set_state(packed['numpy'])
    torch.set_rng_state(packed['torch_cpu']); torch.cuda.set_rng_state(packed['torch_cuda_0'],device=0)
    trainer.lr_scheduler.load_state_dict(packed['scheduler'])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['train50','fixture25','split12','resume13'])
    ap.add_argument('--output',required=True)
    ap.add_argument('--checkpoint-in')
    ap.add_argument('--state-in')
    ap.add_argument('--fixture')
    ap.add_argument('--checkpoint-out',required=True)
    ap.add_argument('--state-out',required=True)
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    os.environ.update({'CUDA_VISIBLE_DEVICES':'0','nnUNet_compile':'false','nnUNet_n_proc_DA':'1',
                       'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.mode!='train50':
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in bounded worker')
    torch.cuda.set_device(0)
    cuda_start=time.monotonic()
    device=torch.device('cuda:0')
    # Seed applies only to this disposable engineering model; no scientific run is created.
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed(2026)
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs
    plans=load_json(str(PLANS)); plans['continue_training']=False; dataset_json=load_json(str(DATASET_JSON))
    trainer=nnUNetTrainer_100epochs(plans,'2d','all',dataset_json,device)
    trainer.initialize()
    if trainer.num_epochs!=100 or trainer.configuration_manager.batch_size!=2 or tuple(map(int,trainer.configuration_manager.patch_size))!=(1024,1536):
        raise RuntimeError('Unexpected plans/trainer config; refusing to update')
    if torch.cuda.current_device()!=0: raise RuntimeError('not on CUDA device 0')
    shape_info={'gpu_name':torch.cuda.get_device_name(0),'cuda_version':torch.version.cuda,
                'patch_size':[1024,1536],'batch_size':2,'trainer':'nnUNetTrainer_100epochs',
                'run_training_called':False,'compile_enabled':False,'mode':args.mode}
    summary={'started_utc':time.time(),'cuda_context_start_elapsed':cuda_start,'configuration':shape_info,
             'optimizer_attempts':0,'successful_updates':0,'amp_skips':0,'steps':[]}
    events=out/'attempts.jsonl'

    if args.mode=='train50':
        trainer.lr_scheduler.step(0)
        train_loader,_val_loader=trainer.get_dataloaders()
        all_cases=json.loads(TRAIN_MANIFEST.read_text(encoding='utf-8'))['rows']
        sizes={r['filename']:tuple(r['native_shape']) for r in all_cases}
        fixed_batch=None
        iterator=iter(train_loader)
        count=50
        start_index=0
    else:
        if not args.checkpoint_in or not args.state_in or not args.fixture: raise ValueError('resume mode needs checkpoint/state/fixture')
        trainer.load_checkpoint(args.checkpoint_in)
        packed=torch.load(args.state_in,map_location='cpu',weights_only=False)
        restore_rng(trainer,packed['rng'])
        fixture=torch.load(args.fixture,map_location='cpu',weights_only=False)
        fixed_batch=fixture['batch']
        sizes={}
        iterator=None
        count={'fixture25':25,'split12':12,'resume13':13}[args.mode]
        start_index=int(packed.get('fixture_cursor',0))

    hook_shapes=[]
    def hook(_module,_inputs,output):
        def getshape(x): return list(x.shape) if torch.is_tensor(x) else [getshape(y) for y in x]
        hook_shapes[:] = getshape(output)
    hook_handle=trainer.network.register_forward_hook(hook)
    clip_orig=torch.nn.utils.clip_grad_norm_
    norm_holder=[]
    def clip_capture(parameters,*a,**kw):
        norm=clip_orig(parameters,*a,**kw); norm_holder.append(float(norm.detach().cpu()) if torch.is_tensor(norm) else float(norm)); return norm
    torch.nn.utils.clip_grad_norm_=clip_capture
    start=time.monotonic()
    for local_idx in range(count):
        gidx=start_index+local_idx
        batch=next(iterator) if args.mode=='train50' else fixed_batch
        if args.mode=='train50' and fixed_batch is None:
            fixed_batch={'batch':{k:(v.detach().cpu().clone() if torch.is_tensor(v) else [x.detach().cpu().clone() for x in v] if isinstance(v,list) else v) for k,v in batch.items()},
                         'case_ids':list(map(str,batch.get('keys',[])))}
            torch.save(fixed_batch,out/'train_fixture_first_batch.pt')
        keys=list(map(str,batch.get('keys',[]))) if args.mode=='train50' else list(map(str,fixed_batch.get('case_ids',[])))
        scale_before=float(trainer.grad_scaler.get_scale()) if trainer.grad_scaler is not None else None
        record={'attempt':summary['optimizer_attempts']+1,'mode':args.mode,'case_ids':keys,
                'patch_shape':list(batch['data'].shape),'target_shapes':[list(t.shape) for t in batch['target']] if isinstance(batch['target'],list) else list(batch['target'].shape),
                'lr':float(trainer.optimizer.param_groups[0]['lr']),'scale_before':scale_before}
        jsonl(events,{'event':'optimizer_attempt_start',**record})
        norm_holder.clear(); hook_shapes.clear()
        try:
            result=trainer.train_step(batch)
        except BaseException as e:
            jsonl(events,{'event':'optimizer_attempt_exception','attempt':record['attempt'],'error':repr(e)})
            raise
        loss=float(np.asarray(result['loss']).reshape(()))
        norm=norm_holder[-1] if norm_holder else float('nan')
        scale_after=float(trainer.grad_scaler.get_scale()) if trainer.grad_scaler is not None else None
        finite=bool(np.isfinite(loss) and np.isfinite(norm))
        skipped=bool(scale_before is not None and scale_after is not None and scale_after<scale_before)
        summary['optimizer_attempts']+=1
        step={'attempt':summary['optimizer_attempts'],'loss':loss,'gradient_norm_before_clip':norm,
              'gradient_finite':bool(np.isfinite(norm)),'loss_finite':bool(np.isfinite(loss)),
              'amp_scale_before':scale_before,'amp_scale_after':scale_after,'amp_skipped':skipped,
              'network_output_shapes':hook_shapes.copy(),'case_ids':keys,
              'input_shape':list(batch['data'].shape),'target_shapes':record['target_shapes'],
              'elapsed_seconds':time.monotonic()-start}
        summary['steps'].append(step)
        if not finite or skipped:
            summary['amp_skips']+=int(skipped)
            jsonl(events,{'event':'optimizer_attempt_failed',**step})
            raise FloatingPointError('nonfinite loss/gradient or AMP skip; stop without retry')
        summary['successful_updates']+=1
        jsonl(events,{'event':'optimizer_attempt_success',**step})
        if args.mode=='train50':
            for case in keys: sizes.setdefault(case,None)
    torch.nn.utils.clip_grad_norm_=clip_orig
    hook_handle.remove()
    if args.mode=='train50':
        # Every case observed by the actual 50-step Train loader is mapped back to native dimensions.
        seen={case:sizes.get(case) for s in summary['steps'] for case in s['case_ids']}
        summary['native_size_coverage']={f'{w}x{h}':sum(1 for x in seen.values() if x==(h,w)) for h,w in set(sizes.values()) if (h,w) is not None}
        # The first genuine transformed Train batch becomes the immutable, same-input restore fixture.
        fixture_path=EXP/'engineering/fixtures/train_batch_fixture.pt'; fixture_path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(fixed_batch,fixture_path)
        summary['fixture_path']=str(fixture_path)
        summary['fixture_case_ids']=fixed_batch['case_ids']
    elif args.mode in ('fixture25','split12','resume13'):
        summary['fixture_cursor_start']=start_index
        summary['fixture_cursor_end']=start_index+count

    # Save nnU-Net checkpoint plus the scheduler/RNG data nnU-Net itself does not serialize.
    original_epoch=trainer.current_epoch
    trainer.current_epoch=0
    trainer.save_checkpoint(args.checkpoint_out)
    trainer.current_epoch=original_epoch
    packed={'rng':rng_pack(trainer),'fixture_cursor':start_index+count if args.mode!='train50' else 0,
            'scheduler_digest':state_digest(trainer.lr_scheduler.state_dict()),'mode':args.mode}
    torch.save(packed,args.state_out)
    summary.update({'finished_utc':time.time(),'elapsed_seconds':time.monotonic()-start,
                    'checkpoint_out':args.checkpoint_out,'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint_out).read_bytes()).hexdigest(),
                    'state_out':args.state_out,'scheduler_digest':packed['scheduler_digest'],
                    'model_digest':state_digest(trainer.network.state_dict()),
                    'optimizer_digest':state_digest(trainer.optimizer.state_dict()),
                    'scaler_digest':state_digest(trainer.grad_scaler.state_dict() if trainer.grad_scaler else None),
                    'rng_digest':state_digest(packed['rng'])})
    (out/'worker_summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n',encoding='utf-8')
    if args.mode=='train50':
        if set(summary['native_size_coverage'].values()) and min(summary['native_size_coverage'].values())==0:
            raise RuntimeError('actual Train updates did not cover all three native sizes')
    print(json.dumps({k:v for k,v in summary.items() if k!='steps'},indent=2,default=str),flush=True)
    if args.mode=='train50':
        for loader in (train_loader,_val_loader):
            if hasattr(loader,'_finish'): loader._finish()
    torch.cuda.synchronize(); torch.cuda.empty_cache()

if __name__=='__main__': main()
