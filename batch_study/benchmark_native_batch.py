"""Supplementary GPU-resident batch-capacity benchmark; not an accuracy experiment."""
from pathlib import Path
import argparse
import gc
import time
import torch
from retinal_batchstudy.run import CONFIG, read, setup, sequence, get_batch, dump
from retinal_data.fives import trial_dataset
from retinal_m0.model import UNet
from retinal_m0.objectives import pixel_loss

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device-label',required=True)
    args=parser.parse_args()
    if not args.device_label.replace('_','').replace('-','').isalnum(): raise ValueError('Invalid label')
    config=read(CONFIG)
    setup(config,2026)
    dataset=trial_dataset('smoke_train')
    dataset.seed=2026
    dataset.cache_size=32
    inputs,targets,signatures=get_batch(dataset,sequence(dataset,16),0,16)
    inputs,targets=inputs.cuda(),targets.cuda()
    results=[]
    for micro in (1,2,4,8,16):
        row={'microbatch':micro,'accumulation':16//micro,'effective_batch':16}
        model=optimizer=scaler=None
        try:
            setup(config,2026)
            model=UNet(config['widths'],config['group_norm_groups']).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'],foreach=False)
            scaler=torch.amp.GradScaler('cuda',init_scale=config['amp_initial_scale'])
            model.train()
            times=[]
            torch.cuda.reset_peak_memory_stats()
            for step in range(12):
                torch.cuda.synchronize()
                started=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0,16,micro):
                    with torch.autocast('cuda',dtype=torch.float16):
                        logits=model(inputs[offset:offset+micro])
                        loss,_=pixel_loss(logits,targets[offset:offset+micro],config['dice_epsilon'])
                    scaler.scale(loss*micro/16).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=True)
                old=scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale()<old: raise RuntimeError('AMP skipped update')
                torch.cuda.synchronize()
                if step>=2: times.append(time.perf_counter()-started)
            row.update(status='completed',timed_updates=len(times),seconds_per_effective_batch=sum(times)/len(times),
                       crops_per_second=16*len(times)/sum(times),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        except torch.cuda.OutOfMemoryError:
            row.update(status='out_of_memory',note='No automatic batch reduction; this configuration is not available in the observed process')
        except Exception as exc:
            row.update(status='failed',error=str(exc))
        finally:
            del model,optimizer,scaler
            # References to last logits/loss can retain memory after a completed backward.
            if 'logits' in locals(): del logits
            if 'loss' in locals(): del loss
            gc.collect()
            torch.cuda.empty_cache()
        results.append(row)
        print(row,flush=True)
    output={'gpu':torch.cuda.get_device_name(),'scope':'Fixed 16 actual training crops already on GPU; 2 warmup and 10 timed updates per layout',
            'not_accuracy_evidence':True,'not_end_to_end_loader_throughput':True,'input_signatures':signatures,'rows':results}
    dump(ROOT/'batch_study'/f'native_batch_capacity_{args.device_label}.json',output)


if __name__=='__main__': main()
