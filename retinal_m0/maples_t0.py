"""A14 MAPLES T0: unchanged native reader/U-Net/FOV/loss, bounded GPU0 training."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from retinal_data.maples_native import NativeMaplesDataset
from retinal_data.maples_fov import candidate_fov
from retinal_data.fives import paired_crop_transform
from retinal_m0.model import UNet
from retinal_m0.state import atomic_save, rng_state, restore_rng, state_digest
from retinal_release.data import masked_loss

ROOT=Path('${RETINA_PROJECT_ROOT}')
CODE=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def validate_model_settings(settings):
    if settings['widths']!=[32,64,128,256,512] or settings['group_norm_groups']!=8:
        raise ValueError('Frozen M0 architecture differs')

def align_log(path,step):
    path=Path(path);lines=path.read_text().splitlines()
    rows=[json.loads(line) for line in lines]
    if len(rows)<step or [row['step'] for row in rows[:step]]!=list(range(1,step+1)):
        raise ValueError('Checkpoint/log prefix differs')
    if len(rows)>step:
        tail='\n'.join(lines[step:])+'\n'
        backup=path.with_name(path.name+f'.recovery_{step}_'+hashlib.sha256(tail.encode()).hexdigest()[:12])
        if backup.exists():
            if backup.read_text()!=tail:raise ValueError('Recovery backup differs')
        else:backup.write_text(tail)
        path.write_text(''.join(line+'\n' for line in lines[:step]))

def draw_geometry(seed,draw,height,width):
    if min(height,width)<512 or draw<0:raise ValueError('Invalid source or draw')
    rng=np.random.default_rng(np.random.SeedSequence([seed,draw]))
    return (int(rng.integers(height-511)),int(rng.integers(width-511)),
            bool(rng.random()<0.5),bool(rng.random()<0.5),int(rng.integers(4)))

class T0Patches(Dataset):
    """Four round-robin visits/image/epoch; every global draw has its own geometry."""
    def __init__(self,config):
        self.native=NativeMaplesDataset(config['data_root'],split='train')
        if self.native.split_counts!={'train':110,'dev':28,'official_test':60}:raise ValueError('Split differs')
        self.names=[r['filename'] for r in self.native.rows]
        self.config=config
        norm=Path(config['normalization_source'])
        if digest(norm)!=config['normalization_sha256']:raise ValueError('Normalization hash differs')
        values=json.loads(norm.read_text())
        if values['mean']!=config['mean'] or values['std']!=config['std']:raise ValueError('FIVES statistics differ')
        self.mean=np.array(values['mean'],dtype=np.float32)[None,None,:]
        self.std=np.array(values['std'],dtype=np.float32)[None,None,:]
        self.cache=OrderedDict()
    def __len__(self):return len(self.names)*4
    def __getitem__(self,draw):
        draw=int(draw);idx=draw%len(self.names)
        if idx not in self.cache:
            item=self.native[idx]
            image=item['image'].permute(1,2,0).numpy()
            mask=item['vessel_mask'][0].numpy()
            fov=candidate_fov(np.rint(image*255).astype(np.uint8)).astype(np.uint8)
            self.cache[idx]=(image,mask,fov)
            if len(self.cache)>2:self.cache.popitem(last=False)
        image,mask,fov=self.cache[idx]
        top,left,hflip,vflip,turns=draw_geometry(self.config['seed'],draw,*mask.shape)
        kw=dict(top=top,left=left,size=512,hflip=hflip,vflip=vflip,quarter_turns=turns)
        x,y=paired_crop_transform(image,mask,**kw)
        _,field=paired_crop_transform(image,fov,**kw)
        if field.sum()==0:raise ValueError('Empty FOV crop: no silent resampling')
        x=((x-self.mean)/self.std)*field[:,:,None]
        return dict(image=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))),
                    mask=torch.from_numpy(np.ascontiguousarray(y[None])).float(),
                    fov=torch.from_numpy(np.ascontiguousarray(field[None])).float(),
                    filename=self.names[idx],draw=draw,
                    geometry=torch.tensor([top,left,int(hflip),int(vflip),turns]))

def gpu_snapshot():
    raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    v=[x.strip() for x in raw.split(',')]
    return dict(index=int(v[0]),uuid=v[1],name=v[2],used_mib=int(v[3]),total_mib=int(v[4]),util_percent=int(v[5]),temperature_c=int(v[6]))

def require_gpu_free():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Only assigned physical GPU0 allowed')
    snap=gpu_snapshot()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    if any(row.split(',')[0].strip()==snap['uuid'] for row in apps.splitlines()):raise RuntimeError('GPU0 has another compute process')
    if snap['used_mib']>=512 or snap['util_percent']>=10:raise RuntimeError('GPU0 is busy')
    return snap

def validate_config(c):
    if c['kind']!='A14_engineering_T0' or c['loss']!='existing_masked_BCE_plus_Dice_unit_weights':raise ValueError('T0 loss/config kind differs')
    if (c['batch_size'],c['patch'],c['crops_per_image'],c['amp'],c['accumulation'])!=(4,512,4,True,1):raise ValueError('Frozen batch/crop/AMP profile differs')
    if (c['learning_rate'],c['weight_decay'],c['clip_norm'])!=(1e-4,1e-4,1.0):raise ValueError('Frozen optimizer values differ')
    if not 1<=c['max_steps']<=2000:raise ValueError('Update limit must be <=2000')
    if c['seed'] not in (2026,2027,2028):raise ValueError('Unapproved seed')
    if c['scheduler'] is not None or c['num_workers']!=0 or c['hflip_probability']!=0.5 or c['vflip_probability']!=0.5 or c['quarter_turn_probability_each']!=0.25 or c['photometric'] or c['resize'] or c['official_test']:raise ValueError('This implementation uses constant LR and zero workers')
    if digest(c['source_m0'])!=c['source_m0_sha256']:raise ValueError('M0 hash differs')

def train(config_path,output,resume=False,stop_at=None):
    config=json.loads(Path(config_path).read_text());validate_config(config)
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'experiments'):raise ValueError('Outputs must be in project experiments')
    target=config['max_steps'] if stop_at is None else stop_at
    if not 1<=target<=config['max_steps']:raise ValueError('Invalid stop step')
    lock_path=ROOT/'experiments/a14_gpu0.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        initial_gpu=require_gpu_free()
        if resume:
            if not (output/'last.pt').is_file():raise ValueError('No checkpoint to resume')
        else:
            output.mkdir(parents=True,exist_ok=False)
            write_json(output/'config.json',config)
        binding={'config_sha256':digest(config_path),'source_m0_sha256':digest(config['source_m0']),
                 'manifest_sha256':digest(Path(config['data_root'])/'data_manifest.csv'),
                 'split_sha256':digest(Path(config['data_root'])/'split_record.json'),
                 'code':{str(p.relative_to(CODE)):digest(p) for p in [Path(__file__),CODE/'retinal_m0/model.py',CODE/'configs/development_v1.json',CODE/'retinal_m0/state.py',CODE/'retinal_release/data.py',CODE/'retinal_data/fives.py',CODE/'retinal_data/maples_fov.py',CODE/'retinal_data/maples_native.py']},
                 'gpu_uuid':initial_gpu['uuid'],'torch':str(torch.__version__),'cuda':str(torch.version.cuda)}
        write_json(output/('binding_resume.json' if resume else 'binding.json'),binding)
        started=time.perf_counter();status={'status':'running','resumed':resume,'pid':os.getpid(),'target_steps':target,'completed_steps':0,'initial_gpu':initial_gpu}
        try:
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
            dataset=T0Patches(config)
            model_config=json.loads((CODE/'configs/development_v1.json').read_text())
            validate_model_settings(model_config)
            model=UNet(model_config['widths'],model_config['group_norm_groups']).cuda().train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'],betas=tuple(config['betas']),eps=config['eps'],foreach=False)
            scaler=torch.amp.GradScaler('cuda',init_scale=config['amp_initial_scale'])
            step=0;cursor=0;elapsed=0.;initial_digest=None
            if resume:
                saved=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
                if saved['config']!=config or saved['binding']!=binding:raise ValueError('Resume config, data, code or hardware binding differs')
                model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
                step=saved['step'];cursor=saved['next_global_draw'];elapsed=saved['elapsed_seconds'];initial_digest=saved['initial_model_digest']
                if cursor!=step*4 or step>=target:raise ValueError('Resume cursor/target differs')
                align_log(output/'train.jsonl',step)
                align_log(output/'gpu.jsonl',step)
                restore_rng(saved['rng'])
                restored={k:state_digest(v) for k,v in dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng_state()).items()}
                expected={k:state_digest(saved[k]) for k in restored}
                if restored!=expected:raise RuntimeError('State restore verification failed')
                write_json(output/f'restore_at_{step}.json',dict(restored_digests=restored,expected_digests=expected,step=step,next_global_draw=cursor,pid=os.getpid(),previous_pid=saved['pid'],passed=True))
                del saved
            else:
                saved=torch.load(config['source_m0'],map_location='cpu',weights_only=True)
                if saved['step']!=10000:raise ValueError('Wrong M0 source step')
                model.load_state_dict(saved['model'],strict=True);del saved
                initial_digest=state_digest(model.state_dict())
            # Private DataLoader generator avoids consuming the model's RNG during iterator creation.
            loader=DataLoader(dataset,batch_size=4,sampler=range(cursor,target*4),num_workers=0,pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(config['seed']))
            torch.cuda.reset_peak_memory_stats()
            status['completed_steps']=step
            with (output/'train.jsonl').open('a' if resume else 'x') as log, (output/'gpu.jsonl').open('a' if resume else 'x') as resources:
                for batch in loader:
                    if time.perf_counter()-started>config['safe_stop_seconds']:raise TimeoutError('Bounded runtime limit exceeded')
                    tick=time.perf_counter();x,y,fov=[batch[k].cuda(non_blocking=True) for k in ('image','mask','fov')]
                    if x.shape!=(4,3,512,512) or y.shape!=(4,1,512,512) or fov.shape!=y.shape:raise ValueError('Incorrect batch shape')
                    if not torch.isfinite(x).all() or not ((y==0)|(y==1)).all() or not ((fov==0)|(fov==1)).all():raise ValueError('Invalid input values')
                    before=model.head.weight.detach().clone();retries=0
                    while True:
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.float16):logits=model(x)
                        if logits.shape!=y.shape:raise ValueError('Incorrect forward shape')
                        # Existing loss casts logits to float32 and implements A13 per-image FOV BCE + Dice.
                        loss=masked_loss(logits,y,fov,epsilon=1e-6)
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                        scaler.scale(loss).backward();scaler.unscale_(optimizer)
                        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'],error_if_nonfinite=False)
                        if torch.isfinite(grad_norm):break
                        scaler.update();retries+=1
                        if retries>8:raise RuntimeError('Repeated AMP gradient overflow')
                    scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
                    if torch.equal(before,model.head.weight):raise RuntimeError('Parameters did not update')
                    step+=1;cursor+=4
                    row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad_norm),learning_rate=optimizer.param_groups[0]['lr'],amp_scale=scaler.get_scale(),amp_retries=retries,
                             draws=batch['draw'].tolist(),filenames=batch['filename'],geometry=batch['geometry'].tolist(),image_shape=list(x.shape),output_shape=list(logits.shape),parameters_updated=True,
                             step_seconds=time.perf_counter()-tick,elapsed_seconds=elapsed+time.perf_counter()-started,
                             max_allocated_mib=torch.cuda.max_memory_allocated()/1048576,max_reserved_mib=torch.cuda.max_memory_reserved()/1048576)
                    log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                    snap=gpu_snapshot();snap.update(step=step,pid=os.getpid());resources.write(json.dumps(snap)+'\n');resources.flush()
                    status['completed_steps']=step
                    if step%config['save_every']==0 or step==target:
                        payload=dict(schema_version=1,step=step,next_global_draw=cursor,epoch=cursor//len(dataset),position_in_epoch=cursor%len(dataset),model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=None,scaler=scaler.state_dict(),rng=rng_state(),config=config,binding=binding,elapsed_seconds=row['elapsed_seconds'],initial_model_digest=initial_digest,pid=os.getpid())
                        atomic_save(payload,output/'last.pt')
                        atomic_save(payload,output/f'step_{step:06d}.pt')
                    print(f"T0 {step}/{config['max_steps']} loss={row['loss']:.6f} GPU={snap['used_mib']}MiB",flush=True)
            final=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            if final['step']!=step or state_digest(final['model'])!=state_digest(model.state_dict()):raise RuntimeError('Checkpoint verification failed')
            if initial_digest==state_digest(final['model']):raise RuntimeError('No change from M0')
            status.update(status='passed',elapsed_seconds=elapsed+time.perf_counter()-started,next_global_draw=cursor,checkpoint_sha256=digest(output/'last.pt'),model_digest=state_digest(final['model']),max_allocated_mib=torch.cuda.max_memory_allocated()/1048576,max_reserved_mib=torch.cuda.max_memory_reserved()/1048576,formal_experiment=False)
        except Exception:
            status.update(status='failed',error=traceback.format_exc());raise
        finally:
            write_json(output/'summary.json',status)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--stop-at',type=int)
    a=p.parse_args();train(a.config,a.output,a.resume,a.stop_at)
if __name__=='__main__':main()
