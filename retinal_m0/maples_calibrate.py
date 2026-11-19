"""A15.2 bound dev28 Z0 inference and isolated200-step A14 resume segments."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time,traceback
import numpy as np
import torch
from retinal_m0.maples_t0 import ROOT,CODE,digest,write_json,require_gpu_free,gpu_snapshot,validate_model_settings
from retinal_m0.model import UNet
from retinal_m0.inference import tiled_probability
from retinal_m0.maples_protocol import proposed_references,select_checkpoint
from retinal_m0.maples_evaluate import load_mask,evaluate_manifest,validate_dev_membership,evaluate_arrays,aggregate
from retinal_m1.maps import targets
from retinal_data.maples_native import NativeMaplesDataset

BASE=ROOT/'experiments/a15_2_z0_t0_calibration_20261015'
PRIOR=ROOT/'experiments/a15_1_maples_dev_evaluation_20261015'

def flatten(report,step,reference_id):
    a=report['aggregate'];ids=[r['filename'] for r in report['images'] if r['near_thin_recall_d1']['value'] is not None]
    return dict(step=step,**{k:v['mean'] for k,v in a.items()},valid_n={k:v['n'] for k,v in a.items()},NA={k:v['NA'] for k,v in a.items()},primary_n=len(ids),primary_ids=ids,reference_id=reference_id)

def prepare():
    protocol=json.loads((BASE/'evaluation_protocol_v1.json').read_text())
    if protocol['status']!='APPROVED_BY_USER_A15_2' or protocol['primary_diameter_max']!=3:raise ValueError('Protocol differs')
    doc=json.loads((PRIOR/'dev_evaluation_manifest.json').read_text());validate_dev_membership(doc['images'])
    refs=BASE/'references';refs.mkdir(exist_ok=False)
    for row in doc['images']:
        v=load_mask(row['vessel_path']);f=load_mask(row['fov_path'])
        if digest(row['vessel_path'])!=row['vessel_path_sha256'] or digest(row['fov_path'])!=row['fov_path_sha256']:raise ValueError('Reference changed')
        fingerprints=json.loads((PRIOR/'source_input_fingerprints.json').read_text())
        if any(digest(row[k])!=fingerprints[row[k]]['sha256'] for k in ('H_path','U_bright_path','U_red_path')):raise ValueError('H/U differ from approved prior inputs')
        masks,meta=proposed_references(v,f,load_mask(row['H_path']),load_mask(row['U_bright_path']),load_mask(row['U_red_path']),diameter_max=3)
        maps=targets(v,f)
        for cap in (2,4,5):masks['thin_'+str(cap)]=maps['skeleton']&(2*maps['skeleton_radius']<=cap)
        path=refs/(row['filename']+'.npz');np.savez_compressed(path,**masks)
        row.update(region_bundle=str(path),region_bundle_sha256=digest(path),regional_source_hashes={k:digest(row[k]) for k in ('H_path','U_bright_path','U_red_path')})
        write_json(refs/(row['filename']+'.json'),dict(meta, status='APPROVED_BY_USER_A15_2',primary_target_pixels=int((masks['thin']&masks['near']).sum())))
    doc.update(definition_status='approved_maples_a15_2_eval_v1',regional_definition=dict(matching_support='omega',diameter_max=3,width_formula='2*EDT(V&FOV) on Zhang skeleton before U clipping',U='BrightUncertains union RedUncertains',radius='max(1,ceil(nativeRGBwidth/100))',kernel='euclidean_disk',d1_tolerance=1,protocol_sha256=digest(BASE/'evaluation_protocol_v1.json')))
    write_json(BASE/'dev_manifest.json',doc)
    write_json(BASE/'binding.json',dict(reference_id=digest(BASE/'dev_manifest.json'),protocol_sha256=digest(BASE/'evaluation_protocol_v1.json'),training_config_sha256=digest(BASE/'train_config.json'),source_files={str(p):digest(p) for p in [Path(__file__),CODE/'retinal_m0/maples_t0.py',CODE/'retinal_m0/maples_evaluate.py',CODE/'retinal_m0/maples_protocol.py',CODE/'retinal_m0/inference.py',CODE/'retinal_m0/model.py',CODE/'configs/development_v1.json',CODE/'retinal_data/maples_fov.py',CODE/'retinal_release/data.py']}))

def verify_binding():
    b=json.loads((BASE/'binding.json').read_text())
    if digest(BASE/'evaluation_protocol_v1.json')!=b['protocol_sha256'] or digest(BASE/'train_config.json')!=b['training_config_sha256'] or digest(BASE/'dev_manifest.json')!=b['reference_id']:raise ValueError('Frozen configuration changed')
    if any(digest(f)!=h for f,h in b['source_files'].items()):raise ValueError('Frozen source changed')
    return b

def macro_density_grid(densities):
    # Match aggregate() one-dimensional reduction exactly, including equality.
    matrix=np.asarray(densities)
    return np.array([float(np.mean(matrix[:,i].tolist())) for i in range(matrix.shape[1])])

def supplementary(out,doc,primary,step):
    grid=np.linspace(0,1,1001,dtype=np.float32);densities=[]
    for row in doc['images']:
        p=np.load(out/'probabilities'/(row['filename']+'.npy'),allow_pickle=False);v=load_mask(row['vessel_path']);f=load_mask(row['fov_path'])
        ranks=np.searchsorted(grid,p[f&~v],side='right');hist=np.bincount(ranks,minlength=1002);tail=np.cumsum(hist[::-1])[::-1];densities.append(tail[1:1002]/f.sum())
    mean_fp=macro_density_grid(densities)
    target=primary['aggregate']['fp_density']['mean'] if step==0 else json.loads((BASE/'z0/metrics/results.json').read_text())['aggregate']['fp_density']['mean']
    feasible=np.flatnonzero(mean_fp<=target);chosen=float(grid[feasible[0]]) if feasible.size else None
    all_fixed={str(cap):[] for cap in (2,4,5)};matched=[]
    for row in doc['images']:
        p=np.load(out/'probabilities'/(row['filename']+'.npy'),allow_pickle=False);v=load_mask(row['vessel_path']);f=load_mask(row['fov_path'])
        with np.load(row['region_bundle'],allow_pickle=False) as bundle:
            kwargs=dict(u=bundle['U'],regions={k:bundle[k] for k in ('internal','near','far')},matching_support='omega')
            for cap in (2,4,5):
                z=evaluate_arrays(p,v,f,threshold=.5,thin=bundle['thin_'+str(cap)],**kwargs);z['filename']=row['filename'];all_fixed[str(cap)].append(z)
            if chosen is not None:
                z=evaluate_arrays(p,v,f,threshold=chosen,thin=bundle['thin'],**kwargs);z['filename']=row['filename'];matched.append(z)
    write_json(out/'supplementary_bins.json',dict(threshold=.5,selection_use=False,cumulative_bins={k:dict(aggregate=aggregate(v),images=v) for k,v in all_fixed.items()}))
    write_json(out/'matched_FP.json',dict(selection_use=False,target_FP_density=target,threshold=chosen,threshold_grid='0:0.001:1',actual_FP_density=aggregate(matched)['fp_density']['mean'] if matched else None,aggregate=aggregate(matched) if matched else None,images=matched,development_descriptive_only=True))

def infer(step):
    binding=verify_binding();cfg=json.loads((BASE/'train_config.json').read_text());doc=json.loads((BASE/'dev_manifest.json').read_text())
    with (ROOT/'experiments/a14_gpu0.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);initial_gpu=require_gpu_free()
        started=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        source=Path(cfg['source_m0']) if step==0 else BASE/'training'/f'step_{step:06d}.pt'
        if step==0 and digest(source)!=cfg['source_m0_sha256']:raise ValueError('Wrong source M0')
        saved=torch.load(source,map_location='cpu',weights_only=True)
        if saved['step']!=(10000 if step==0 else step):raise ValueError('Wrong checkpoint step')
        mc=json.loads((CODE/'configs/development_v1.json').read_text());validate_model_settings(mc);model=UNet(mc['widths'],mc['group_norm_groups']);model.load_state_dict(saved['model'],strict=True);del saved
        model=model.cuda().eval();out=BASE/('z0' if step==0 else f'validation_{step:06d}');out.mkdir(exist_ok=False);probs=out/'probabilities';probs.mkdir()
        native=NativeMaplesDataset(cfg['data_root'],split='dev');assert [r['filename'] for r in native.rows]==[r['filename'] for r in doc['images']]
        mean=torch.tensor(cfg['mean'])[:,None,None];std=torch.tensor(cfg['std'])[:,None,None]
        if digest(cfg['normalization_source'])!=cfg['normalization_sha256']:raise ValueError('Normalization changed')
        timing=[];torch.cuda.reset_peak_memory_stats()
        for index,row in enumerate(doc['images']):
            tick=time.perf_counter();item=native[index];f=load_mask(row['fov_path']);x=((item['image']-mean)/std)*torch.from_numpy(f[None]).float()
            probability,tiles=tiled_probability(model,x,patch=512,stride=256,batch_size=4,device='cuda',amp=False)
            probability=probability.numpy().astype(np.float32,copy=False)
            if probability.shape!=tuple(row['native_shape']) or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:raise ValueError('Invalid native probability')
            np.save(probs/(row['filename']+'.npy'),probability,allow_pickle=False)
            timing.append(dict(filename=row['filename'],tiles=tiles,seconds=time.perf_counter()-tick))
            print(f'VAL step={step} image={index+1}/28 tiles={tiles}',flush=True)
        torch.cuda.synchronize();report=evaluate_manifest(BASE/'dev_manifest.json',probs,out/'metrics',threshold=.5)
        flat=flatten(report,step,binding['reference_id'])
        supplementary(out,doc,report,step)
        if step==0 and any(flat[k] is None for k in ('precision','fp_density','far_recall','near_thin_recall_d1')):raise ValueError('Z0 primary/protection undefined; stop before training')
        write_json(out/'summary.json',dict(metrics=flat,source_checkpoint=str(source),source_sha256=digest(source),inference=dict(patch=512,stride=256,batch=4,amp=False,TTA=False,resize=False),probabilities_dtype='float32',probabilities=28,elapsed_seconds=time.perf_counter()-started,max_allocated_mib=torch.cuda.max_memory_allocated()/1048576,max_reserved_mib=torch.cuda.max_memory_reserved()/1048576,gpu=initial_gpu,timing=timing))
        print('VALIDATION_COMPLETE',step,json.dumps(flat),flush=True)

def orchestrate():
    started=time.time();status=dict(status='running',stage='prepare',completed_updates=0,validated_steps=[],started_unix=started)
    def update(**kw):status.update(kw);write_json(BASE/'status.json',status)
    def run(name,args):
        verify_binding()
        with (BASE/(name+'.log')).open('x') as f:
            r=subprocess.run([sys.executable,'-B']+args,cwd=CODE,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,timeout=10800)
        if r.returncode:raise RuntimeError(name+' failed; see log, protocol not altered')
    try:
        prepare();update(stage='Z0_inference')
        run('Z0_inference',['-m','retinal_m0.maples_calibrate','--infer-step','0']);update(validated_steps=[0])
        baseline=json.loads((BASE/'z0/summary.json').read_text())['metrics'];write_json(BASE/'protection_rules.json',dict(frozen_before_T0=True,threshold=.5,baseline=baseline,precision_min=baseline['precision'],FP_density_max=baseline['fp_density'],far_recall_min=baseline['far_recall'],primary='near_thin_recall_d1_diameter_le3',tie='earliest_step',no_eligible='None_no_relaxation'))
        for stop in range(200,2001,200):
            update(stage='training',target_step=stop)
            args=['-m','retinal_m0.maples_t0','--config',str(BASE/'train_config.json'),'--output',str(BASE/'training'),'--stop-at',str(stop)]
            if stop>200:args+=['--resume']
            run(f'train_to_{stop:06d}',args);update(completed_updates=stop,stage='validation',target_step=stop)
            run(f'eval_{stop:06d}',['-m','retinal_m0.maples_calibrate','--infer-step',str(stop)])
            update(validated_steps=status['validated_steps']+[stop])
            candidates=[json.loads((BASE/f'validation_{s:06d}/summary.json').read_text())['metrics'] for s in status['validated_steps'] if s]
            best=select_checkpoint(candidates,baseline)
            write_json(BASE/'selection.json',dict(best_step=best['step'] if best else None,eligible_steps=[r['step'] for r in candidates if select_checkpoint([r],baseline) is not None],selection=best,rule='frozen_all3_Z0_gates_then_max_primary_earliest_tie'))
        update(status='completed',stage='completed',elapsed_seconds=time.time()-started)
    except Exception:
        update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.time()-started);raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--infer-step',type=int);a=p.parse_args()
    if a.infer_step is not None:infer(a.infer_step)
    else:orchestrate()
if __name__=='__main__':main()
