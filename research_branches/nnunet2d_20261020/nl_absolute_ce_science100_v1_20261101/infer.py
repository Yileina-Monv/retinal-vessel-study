"""Exactly one terminal-only Dev28 inference, no retries/ensemble/TTA."""
import os,sys,json,time,ctypes,signal,copy
from pathlib import Path
from common import STAGE,CODE,GPU,protocol,validate,runtime_authorizer,sha
from fast_observer import atomic

def prevent_native_retry(predictor,on_error,is_running):
    original=predictor._internal_predict_sliding_window_return_logits
    def once(*args,**kwargs):
        if not is_running():raise RuntimeError('native inference fallback retry refused before any model call')
        try:return original(*args,**kwargs)
        except BaseException as exc:on_error(exc);raise
    predictor._internal_predict_sliding_window_return_logits=once

def verify_tile_count(actual,cap):
    # Official nonzero cropping may reduce the grid; native geometry is its fixed upper bound.
    if type(actual)is not int or type(cap)is not int or not 1<=actual<=cap:raise RuntimeError('invalid/exceeded native geometry tile cap')

def execute(parent,nonce):
    if os.getppid()!=parent or nonce!=os.environ.get('NNUNET100_NONCE') or len(nonce)<32:raise RuntimeError('not direct authenticated inference child')
    key=validate(full=True);q=protocol();authorize=runtime_authorizer(key,'infer')
    if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0 or os.getppid()!=parent:raise RuntimeError('inference parent-death guard failed')
    from common import GROUP,read
    for seed in [2027,2028]:
        if read(GROUP/'runs'/f'seed_{seed}'/'PAIRING_COMPLETE.json')['status']!='PASS':raise RuntimeError('both paired terminal runs required')
        if read(GROUP/'runs'/f'seed_{seed}'/'training_completion.json')['status']!='TRAINING_COMPLETE':raise RuntimeError('both terminal runs required')
    training=json.loads((STAGE/'training_completion.json').read_text());b=training['counts']
    if training['status']!='TRAINING_COMPLETE' or training['attempts']!=25000 or training['internal_train_monitor_batches']!=5000 or b['success']+b['skip']!=25000 or b['fatal'] or b['unknown'] or b['skip']>=25:raise RuntimeError('terminal training incomplete; Dev remains closed')
    final=Path(training['checkpoint_final'])
    if sha(final)!=training['checkpoint_final_sha256']:raise RuntimeError('final checkpoint digest mismatch')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise RuntimeError('wrong inference GPU')
    import torch,numpy as np
    from PIL import Image
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(10240*1048576/torch.cuda.get_device_properties(0).total_memory)
    if torch.cuda.device_count()!=1:raise RuntimeError('only GPU0 may be visible')
    doc=json.loads(Path(q['references']['Dev_RGB_manifest']).read_text());rows=doc['images']
    if len(rows)!=28 or [r['filename'] for r in rows]!=q['Dev_ids']:raise RuntimeError('Dev whitelist/order drift')
    pred=STAGE/'dev_predictions';inp=STAGE/'dev_input';export=pred/'official_export';pred.mkdir(exist_ok=False);inp.mkdir(exist_ok=False);export.mkdir()
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,perform_everything_on_device=True,device=torch.device('cuda:0'),verbose=False,verbose_preprocessing=False,allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(final.parent.parent),use_folds=('all',),checkpoint_name='checkpoint_final.pth')
    budget={'binding_sha256':key,'forward_cap':sum(q['inference']['expected_tiles_by_image'].values()),'forwards':0,'images':{},'status':'RUNNING'};active={'name':None}
    def hook(*_):
        authorize();name=active['name'];record=budget['images'].get(name)
        if budget['status']!='RUNNING' or record is None or record['state']!='PENDING' or record['forwards']>=q['inference']['expected_tiles_by_image'][name] or budget['forwards']>=budget['forward_cap']:
            budget['status']='FAILED_STOP';atomic(STAGE/'inference_budget.json',budget);raise RuntimeError('extra/unapproved terminal inference forward; no fallback retry')
        record['forwards']+=1;budget['forwards']+=1;atomic(STAGE/'inference_budget.json',budget)
    def inference_error(exc):
        budget['status']='FAILED_STOP';budget['reason']=repr(exc);atomic(STAGE/'inference_budget.json',budget)
    prevent_native_retry(predictor,inference_error,lambda:budget['status']=='RUNNING')
    handle=predictor.network.register_forward_pre_hook(hook);records=[]
    try:
        for row in rows:
            authorize();name=row['filename'];active['name']=name;source=row['inputs']['RGB']
            if sha(source['path'])!=source['sha256']:raise RuntimeError('Dev RGB digest drift')
            rgb=np.asarray(Image.open(source['path']).convert('RGB'))
            if list(rgb.shape[:2])!=row['native_shape']:raise RuntimeError('native RGB shape mismatch')
            files=[]
            for ch in range(3):
                p=inp/(name+f'_{ch:04d}.png');Image.fromarray(rgb[:,:,ch],mode='L').save(p);files.append(str(p))
            if name in budget['images']:raise RuntimeError('image already attempted')
            budget['images'][name]={'state':'PENDING','attempts':1,'forwards':0};atomic(STAGE/'inference_budget.json',budget)
            # A single case per call makes one-attempt accounting explicit. Official preprocessing/inverse export retained.
            predictor.predict_from_files([files],[str(export/name)],save_probabilities=True,overwrite=False,num_processes_preprocessing=1,num_processes_segmentation_export=1)
            with np.load(export/(name+'.npz'),allow_pickle=False) as z:arr=z['probabilities']
            if arr.ndim==4 and arr.shape[1]==1:arr=arr[:,0]
            if arr.shape!=(2,*row['native_shape']):raise RuntimeError('class/orientation/native inverse export mismatch')
            prob=np.asarray(arr[1],dtype=np.float32)
            if not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):raise RuntimeError('invalid probability range')
            verify_tile_count(budget['images'][name]['forwards'],q['inference']['expected_tiles_by_image'][name])
            out=pred/(name+'.npy');np.save(out,prob,allow_pickle=False);budget['images'][name]['state']='COMPLETE';atomic(STAGE/'inference_budget.json',budget);records.append({'filename':name,'shape':list(prob.shape),'dtype':str(prob.dtype),'sha256':sha(out),'checkpoint_sha256':training['checkpoint_final_sha256'],'actual_tile_forwards':budget['images'][name]['forwards'],'native_geometry_tile_cap':q['inference']['expected_tiles_by_image'][name]})
            atomic(STAGE/'heartbeat_inference.json',{'time':time.time(),'images_complete':len(records),'forward_calls':budget['forwards']})
        if not 0<budget['forwards']<=budget['forward_cap'] or len(records)!=28:raise RuntimeError('terminal evaluation quota incomplete/exceeded')
        budget['status']='COMPLETE';atomic(STAGE/'inference_budget.json',budget);atomic(STAGE/'prediction_verification.json',{'status':'PASS','images':records,'threshold':.5,'checkpoint':'checkpoint_final.pth','TTA':False,'ensemble':False,'postprocessing':False,'Test_accessed':False})
    except BaseException as exc:
        budget['status']='FAILED_STOP';budget['reason']=repr(exc);atomic(STAGE/'inference_budget.json',budget);raise
    finally:handle.remove()
    predictor.network.to('cpu');del predictor;torch.cuda.empty_cache()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--authorized-final-inference',action='store_true',required=True);p.add_argument('--parent-pid',type=int,required=True);p.add_argument('--nonce',required=True);a=p.parse_args();execute(a.parent_pid,a.nonce)
