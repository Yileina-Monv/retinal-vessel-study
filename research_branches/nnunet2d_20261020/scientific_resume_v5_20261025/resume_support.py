"""Controlled continuation; native asynchronous augmentation queues are NOT exact-restorable."""
import copy,re,datetime
import torch

def migrate_budget(old,new_binding):
    if old['pending'] is not None or any(old[k] for k in ['fatal','unknown','skip']):raise RuntimeError('unsafe pending/failed budget migration')
    if old['attempts']!=old['success'] or old['phase']!='monitor' or old['iteration']!=old['schedule'][1] or not 0<old['monitor_iteration']<old['schedule'][2]:raise RuntimeError('not a committed mid-monitor boundary')
    new=copy.deepcopy(old);new['binding']=new_binding;new['status']='READY';new.pop('reason',None)
    return new

def rebuild_logger(native,bundle,console):
    from nnunetv2.utilities.collate_outputs import collate_outputs
    epoch=bundle['cursor']['epoch'];logging=copy.deepcopy(native['logging'])
    if native['current_epoch']!=epoch or any(len(x)!=epoch for x in logging.values()):raise RuntimeError('native logger boundary mismatch')
    if len(bundle['partial_train'])!=bundle['budget']['schedule'][1] or len(bundle['partial_monitor'])!=bundle['cursor']['monitor_iteration']:raise RuntimeError('partial output counts mismatch')
    matches=re.findall(r'^(\d{4}-\d\d-\d\d \d\${LOCAL_PATH} '+str(epoch)+r'\s*$',console,re.M)
    if len(matches)!=1:raise RuntimeError('unique recorded epoch-start line required')
    stamp=datetime.datetime.fromisoformat(matches[0]).timestamp()
    loss=collate_outputs(bundle['partial_train'])['loss'].mean()
    logging['train_losses'].append(loss)
    logging['lrs'].append(bundle['optimizer']['param_groups'][0]['lr'])
    logging['epoch_start_timestamps'].append(stamp)
    return logging,{'past_complete_epochs':epoch,'partial_train_batches':len(bundle['partial_train']),'partial_monitor_batches':len(bundle['partial_monitor']),'current_train_loss':float(loss),'epoch_start_source':'console Epoch line; approximate timestamp, wall duration includes interruption; not used for speed claims','native_queue_resume_supported':False}

def continue_mid_monitor_then_official(t,o,j):
    s=j.read()
    if s['phase']!='monitor' or t.current_epoch!=s['epoch'] or len(o.partial_train)!=s['iteration'] or len(o.partial_monitor)!=s['monitor_iteration']:raise RuntimeError('resume cursor mismatch')
    t.on_validation_epoch_start()
    with torch.no_grad():
        while j.read()['monitor_iteration']<s['schedule'][2]:t.validation_step(next(t.dataloader_val))
    t.on_validation_epoch_end(o.partial_monitor.copy());t.on_epoch_end()
    start=t.on_train_start;t.on_train_start=lambda:None
    try:t.run_training()
    finally:t.on_train_start=start
