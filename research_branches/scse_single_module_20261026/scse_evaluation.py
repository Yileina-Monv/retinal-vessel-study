from pathlib import Path
import os,json
from scse_gate import STAGE,CODE,ROOT,OUT,protocol as check_protocol,sha,validate_release,runtime_authorizer
from retinal_m0.maples_g1_safety import read_json,atomic_json,GateError

def _load_arrays(npz):
    import numpy as np
    with np.load(npz,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}

def hard_scalar_details(hard):
    result={k:hard[k] for k in ('pred_skeleton_count','gt_skeleton_count','pred_hit_count','gt_hit_count')}
    result.update({k+'_NA_reason':v for k,v in hard['NA_reason'].items()})
    return result

def validate_probability_records(rows,refs,probs):
    import numpy as np
    expected=[r['filename'] for r in refs]
    if len(expected)!=28 or len(set(expected))!=28 or len(rows)!=28 or [r.get('filename') for r in rows]!=expected: raise GateError('evaluation must contain exactly frozen Dev28 in order')
    for row,ref in zip(rows,refs):
        path=Path(probs)/(row['filename']+'.npy')
        if not path.is_file() or sha(path)!=row.get('probability_sha256'): raise GateError('probability digest mismatch')
        a=np.load(path,allow_pickle=False)
        if a.dtype!=np.float32 or list(a.shape)!=ref['native_shape'] or not np.isfinite(a).all() or np.any((a<0)|(a>1)): raise GateError('probability dtype/shape/range mismatch')
    return True

def reserve_evaluation_forward(budget,slot_key,name,batch_tiles,protocol):
    """Charge before a real inference forward; no retries/refunds or extra slots."""
    if slot_key not in {'SCSE:'+str(x) for x in range(200,2001,200)}:
        raise GateError('unapproved evaluation slot')
    profile=protocol['runtime']['evaluation_image_profile'][name]
    state=budget['slots'][slot_key]['images'][name]
    if state.get('status')!='INCOMPLETE' or state.get('attempts')!=1 or type(batch_tiles) is not int or batch_tiles<1:
        raise GateError('unapproved evaluation image/attempt')
    calls=sum(v.get('forward_calls',0) for s in budget['slots'].values() for v in s['images'].values())
    tiles=sum(v.get('tiles',0) for s in budget['slots'].values() for v in s['images'].values())
    if (calls+1>protocol['runtime']['evaluation_forward_cap'] or tiles+batch_tiles>protocol['runtime']['evaluation_tile_cap']
        or state['forward_calls']+1>profile['forward_calls'] or state['tiles']+batch_tiles>profile['tiles']):
        raise GateError('fixed evaluation forward/tile budget exceeded')
    state['forward_calls']+=1;state['tiles']+=batch_tiles

def evaluate_checkpoint(model,step,binding_sha):
    import csv, numpy as np, torch
    from PIL import Image
    from retinal_m0.inference import tiled_probability
    from retinal_m0.maples_branch_science_analysis import evaluate_image,metric_values
    from retinal_m0.maples_focal_dice_science_analysis import hard_cldice
    from retinal_m0.maples_focal_dice_science_core import rng_state,restore_rng
    if step not in range(200,2001,200): raise GateError('unplanned C evaluation step')
    ref_manifest=read_json(Path(check_protocol()['reference']['science_reference_manifest']))
    source_manifest=read_json(ROOT/'docs/full_fov_thin_skeleton_bce_20261019/dev_reference_manifest.json')
    sources={r['filename']:r for r in source_manifest['images']}
    dest=OUT/'evaluation'/f'step_{step:06d}'; probs=dest/'probabilities'; dest.mkdir(parents=True,exist_ok=True); probs.mkdir(exist_ok=True)
    metrics_path=dest/'metrics.json'; ckpt=OUT/f'step_{step:04d}.pt'; checkpoint_sha=sha(ckpt); reference_sha=sha(Path(check_protocol()['reference']['science_reference_manifest']))
    if len(ref_manifest['images'])!=28 or len(sources)!=28 or set(sources)!={r['filename'] for r in ref_manifest['images']}: raise GateError('Dev input/reference membership differs')
    if metrics_path.exists():
        saved=read_json(metrics_path)
        if saved.get('method')!='SCSE' or saved.get('step')!=step or saved.get('threshold')!=.5 or saved.get('binding_sha256')!=binding_sha or saved.get('checkpoint_sha256')!=checkpoint_sha or saved.get('reference_manifest_sha256')!=reference_sha: raise GateError('existing C evaluation binding drift')
        validate_probability_records(saved.get('images',[]),ref_manifest['images'],probs)
        return saved
    evaluation_authorize=runtime_authorizer(binding_sha)
    budget_path=STAGE/'evaluation_budget.json'
    if not budget_path.exists(): atomic_json(budget_path,{'schema':1,'slots':{}})
    budget=read_json(budget_path); slot_key='SCSE:'+str(step); slot=budget['slots'].setdefault(slot_key,{'checkpoint_sha256':checkpoint_sha,'images':{}})
    if slot['checkpoint_sha256']!=checkpoint_sha: raise GateError('evaluation checkpoint changed')
    model_snapshot={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    before=rng_state('cuda'); before_digest=__import__('retinal_m0.state',fromlist=['state_digest']).state_digest(before); was=model.training; model_before=__import__('retinal_m0.state',fromlist=['state_digest']).state_digest(model.state_dict()); rows=[]
    try:
        from scse_core import verify_checkpoint
        validate_release(full_sources=True)
        saved_ckpt=verify_checkpoint(ckpt,expected_step=step,binding_sha=binding_sha,run_dir=OUT)
        model.load_state_dict(saved_ckpt['model'],strict=True); del saved_ckpt; model.eval()
        norm=read_json(CODE/'releases/development_v1/normalization.json'); mu=torch.tensor(norm['mean'])[:,None,None]; sd=torch.tensor(norm['std'])[:,None,None]
        for row in ref_manifest['images']:
            name=row['filename']; destp=probs/(name+'.npy'); state=slot['images'].get(name)
            if state and state.get('status')=='COMPLETE':
                if sha(destp)!=state['probability_sha256']: raise GateError('completed probability file digest changed')
                prob=np.load(destp,allow_pickle=False)
            else:
                if state is not None: raise GateError('incomplete evaluation image cannot be retried automatically')
                inp=sources[name]['inputs']['RGB']
                if sha(inp['path'])!=inp['sha256']: raise GateError('frozen Dev RGB source changed')
                with Image.open(inp['path']) as im: raw=np.asarray(im.convert('RGB')).copy()
                arrays=_load_arrays(row['reference_npz']); fov=arrays['FOV'].astype(bool)
                image=(torch.from_numpy(raw.transpose(2,0,1).copy()).float()/255-mu)/sd
                image*=torch.from_numpy(fov[None].astype(np.float32))
                state={'status':'INCOMPLETE','attempts':1,'forward_calls':0,'tiles':0}; slot['images'][name]=state; atomic_json(budget_path,budget)
                def count_forward(module,inputs):
                    evaluation_authorize()
                    reserve_evaluation_forward(budget,slot_key,name,int(inputs[0].shape[0]),check_protocol());atomic_json(budget_path,budget)
                hook=model.register_forward_pre_hook(count_forward)
                try: prediction,tiles=tiled_probability(model,image,patch=512,stride=256,batch_size=4,device='cuda',amp=False)
                finally: hook.remove()
                prob=prediction.detach().cpu().numpy().astype(np.float32)
                with destp.with_name(destp.name+'.pending').open('wb') as f: np.save(f,prob,allow_pickle=False); f.flush(); os.fsync(f.fileno())
                os.replace(destp.with_name(destp.name+'.pending'),destp)
                profile=check_protocol()['runtime']['evaluation_image_profile'][name]
                if state['tiles']!=profile['tiles'] or state['forward_calls']!=profile['forward_calls'] or int(tiles)!=profile['tiles']:raise GateError('inference geometry differs from frozen evaluation profile')
                state.update(status='COMPLETE',probability_sha256=sha(destp),tiled_count=int(tiles)); atomic_json(budget_path,budget)
            if sha(row['reference_npz'])!=row['reference_npz_sha256']: raise GateError('sealed GT reference drift')
            arrays=_load_arrays(row['reference_npz'])
            if prob.dtype!=np.float32 or list(prob.shape)!=row['native_shape'] or not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)): raise GateError('new/saved prediction dtype/shape/range mismatch')
            record=evaluate_image(prob,arrays,'B2',threshold=.5); record['method']='SCSE'
            hard=hard_cldice((prob>=.5)&arrays['FOV'].astype(bool),arrays['GT'],arrays['FOV'])
            record.update(hard_scalar_details(hard))
            record['hard_cldice']=hard['cldice']; record['topology_precision']=hard['topology_precision']; record['topology_sensitivity']=hard['topology_sensitivity']
            rows.append({'filename':name,'probability_sha256':sha(destp),'RGB_sha256':sources[name]['inputs']['RGB']['sha256'],'reference_npz_sha256':row['reference_npz_sha256'],**record})
            atomic_json(STAGE/'heartbeat.json',{'status':'EVALUATING','step':step,'image':name,'completed_images':len(rows),'pid':os.getpid(),'unix_time':__import__('time').time()})
            print('EVALUATION',step,len(rows),name,flush=True)
        result={'method':'SCSE','step':step,'threshold':.5,'checkpoint_sha256':checkpoint_sha,'reference_manifest_sha256':reference_sha,'binding_sha256':binding_sha,'images':rows}
        validate_probability_records(rows,ref_manifest['images'],probs)
        atomic_json(metrics_path,result)
        return result
    finally:
        model.load_state_dict(model_snapshot,strict=True); del model_snapshot
        model.train(was); restore_rng(before,'cuda')
        after=__import__('retinal_m0.state',fromlist=['state_digest']).state_digest(rng_state('cuda'))
        if after!=before_digest: raise GateError('evaluation changed RNG after restoration')
        if __import__('retinal_m0.state',fromlist=['state_digest']).state_digest(model.state_dict())!=model_before: raise GateError('evaluation changed model parameters')
