from __future__ import annotations
import argparse, csv, hashlib, json, os, pathlib, subprocess, sys, time
from datetime import datetime, timezone
import numpy as np
import torch
from PIL import Image
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

BASE=pathlib.Path('${RETINA_PROJECT_ROOT}')
FREEZE=BASE/'docs/nnunet2d_seed2026_100epochs_20261020/no_wall_limit_restart_02'
PROTO=json.loads((FREEZE/'protocol.json').read_text())
ROOT=pathlib.Path(PROTO['paths']['scientific_root'])
MODEL_DIR=pathlib.Path(PROTO['paths']['nnUNet_results'])/'Dataset501_MAPLES/nnUNetTrainer_100epochs__nnUNetPlans_8GBtarget__2d'
FOLD_DIR=MODEL_DIR/'fold_all'
DEV=BASE/'docs/full_fov_thin_skeleton_bce_20261019/dev_reference_manifest.json'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def convert_dev_rgb(rows, input_dir):
    input_dir.mkdir(parents=True,exist_ok=False)
    file_lists=[]; records=[]
    for row in rows:
        item=row['inputs']['RGB']; p=pathlib.Path(item['path'])
        if sha(p)!=item['sha256']: raise RuntimeError('Dev RGB source hash changed: '+row['filename'])
        rgb=np.asarray(Image.open(p).convert('RGB'))
        if list(rgb.shape[:2])!=row['native_shape'] or rgb.shape[2]!=3: raise RuntimeError('Dev RGB shape/channel mismatch: '+row['filename'])
        channels=[]
        for c in range(3):
            out=input_dir/f"{row['filename']}_{c:04d}.png"
            Image.fromarray(rgb[:,:,c],mode='L').save(out)
            channels.append(str(out))
        file_lists.append(channels)
        records.append({'filename':row['filename'],'native_shape':row['native_shape'],'source_rgb_sha256':item['sha256'],'converted_channels':[{'path':x,'sha256':sha(x)} for x in channels]})
    return file_lists,records

def write_eval_manifest(rows, path):
    images=[]
    for row in rows:
        inp=row['inputs']
        images.append({'filename':row['filename'],'split':'dev','source_split':'train','native_shape':row['native_shape'],
          'vessel_path':inp['GT']['path'],'vessel_path_sha256':inp['GT']['sha256'],
          'fov_path':inp['FOV']['path'],'fov_path_sha256':inp['FOV']['sha256'],
          'region_bundle':inp['regions']['path'],'region_bundle_sha256':inp['regions']['sha256']})
    doc={'scope':'MAPLES_dev28','definition_status':'frozen_nnUNet_seed2026_100epochs_protocol','regional_definition':{'matching_support':'omega'},'prediction_provenance':'nnUNet seed2026 fresh 100epochs checkpoint_final; exact terminal-only evaluation','images':images}
    path.write_text(json.dumps(doc,indent=2,ensure_ascii=False)+'\n')

def bootstrap_paired(final_rows, base_rows):
    metrics=['dice','precision','fp_density','recall','far_recall','near_pixel_recall','near_thin_recall_d1','fov_thin_recall_d0_w3','fov_thin_recall_d1_w3']
    by_n={}; out={}
    for metric in metrics:
        diffs=[]; used=[]; na=[]
        for f,b in zip(final_rows,base_rows):
            fv=f.get(metric); bv=b.get(metric)
            if fv is None or bv is None: na.append(f['filename'])
            else: diffs.append(float(fv)-float(bv)); used.append(f['filename'])
        n=len(diffs)
        if n:
            if n not in by_n:
                rng=np.random.Generator(np.random.PCG64(20261016))
                # Required distinct-n streams restart from the frozen seed; same-n metrics share this index matrix.
                by_n[n]=rng.integers(0,n,size=(1000,n),endpoint=False)
            idx=by_n[n]
            means=np.mean(np.asarray(diffs)[idx],axis=1)
            ci=np.quantile(means,[.025,.975],method='linear')
            out[metric]={'n':n,'mean_paired_difference':float(np.mean(diffs)),'ci95':[float(ci[0]),float(ci[1])],'improved':sum(x>0 for x in diffs),'worsened':sum(x<0 for x in diffs),'ties':sum(x==0 for x in diffs),'NA_images':na,'included_images':used,'bootstrap_index_key':str(n)}
        else: out[metric]={'n':0,'mean_paired_difference':None,'ci95':None,'improved':0,'worsened':0,'ties':0,'NA_images':na,'included_images':[],'bootstrap_index_key':None}
    return out,{str(n):v.tolist() for n,v in by_n.items()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-final-dev-inference',action='store_true',help='required explicit terminal-only inference mode')
    a=ap.parse_args()
    if not a.run_final_dev_inference: raise SystemExit('Refusing: explicit final-only Dev inference flag required')
    preflight_path=pathlib.Path(os.environ.get('SCIENTIFIC_PREFLIGHT_PATH',str(ROOT/'preflight/verification.json')))
    if json.loads(preflight_path.read_text()).get('status')!='PASS': raise SystemExit('preflight not PASS')
    training=json.loads((ROOT/'training_completion.json').read_text())
    if training.get('status')!='TRAINING_COMPLETE' or training.get('attempts')!=25000 or training.get('successful_updates')!=25000 or training.get('internal_train_monitor_batches')!=5000: raise SystemExit('terminal training criteria incomplete; refusing Dev access')
    checkpoint=FOLD_DIR/'checkpoint_final.pth'
    if sha(checkpoint)!=training['checkpoint_final_sha256']: raise SystemExit('final checkpoint hash mismatch')
    if os.environ.get('PYTHONHASHSEED')!='2026' or os.environ.get('CUDA_VISIBLE_DEVICES')!=PROTO['operational_amendment']['gpu0_uuid']: raise SystemExit('seed/device environment mismatch')
    torch.set_num_threads(1); torch.cuda.set_device(0)
    doc=json.loads(DEV.read_text()); rows=doc['images']
    if len(rows)!=28 or doc['count']!=28: raise RuntimeError('Frozen Dev28 manifest count changed')
    inp_dir=ROOT/'dev_input/imagesTs'; export_dir=ROOT/'dev_predictions/nnunet_export'; pred_dir=ROOT/'dev_predictions'
    if pred_dir.exists(): raise RuntimeError('Prediction output directory already exists; preserving prior outputs')
    pred_dir.mkdir(parents=True)
    files,source_records=convert_dev_rgb(rows,inp_dir)
    export_dir.mkdir(parents=True)
    outputs=[str(export_dir/r['filename']) for r in rows]
    predictor=nnUNetPredictor(tile_step_size=0.5,use_gaussian=True,use_mirroring=False,perform_everything_on_device=True,device=torch.device('cuda:0'),verbose=False,verbose_preprocessing=False,allow_tqdm=False)
    if predictor.use_mirroring is not False or predictor.tile_step_size!=0.5 or predictor.use_gaussian is not True: raise RuntimeError('Frozen predictor inference config mismatch')
    predictor.initialize_from_trained_model_folder(str(MODEL_DIR),use_folds=('all',),checkpoint_name='checkpoint_final.pth')
    start=time.monotonic()
    predictor.predict_from_files(files,outputs,save_probabilities=True,overwrite=False,num_processes_preprocessing=1,num_processes_segmentation_export=1)
    inference_seconds=time.monotonic()-start
    torch.cuda.synchronize(); peak_alloc=int(torch.cuda.max_memory_allocated()); peak_reserved=int(torch.cuda.max_memory_reserved())
    predictor.network.to('cpu'); del predictor; torch.cuda.empty_cache()
    pred_records=[]
    for row in rows:
        name=row['filename']; npz=export_dir/(name+'.npz')
        if not npz.is_file(): raise RuntimeError('Missing official exported probability archive: '+name)
        with np.load(npz,allow_pickle=False) as z: arr=z['probabilities']
        if arr.ndim==4 and arr.shape[1]==1: arr=arr[:,0]
        if arr.shape!=(2,*row['native_shape']): raise RuntimeError(f'probability channel/orientation/native shape mismatch {name}: {arr.shape}')
        p=np.asarray(arr[1],dtype=np.float32)
        if not np.isfinite(p).all() or float(p.min())<0 or float(p.max())>1: raise RuntimeError('invalid class1 vessel probability: '+name)
        out=pred_dir/(name+'.npy'); np.save(out,p,allow_pickle=False)
        pred_records.append({'filename':name,'native_shape':list(p.shape),'dtype':str(p.dtype),'min':float(p.min()),'max':float(p.max()),'finite':True,'class1_channel':'vessel','probability_sha256':sha(out),'official_export_archive_sha256':sha(npz)})
    (ROOT/'dev_input/manifest.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'source_scope':'frozen Dev28 RGB only, accessed after terminal checkpoint','images':source_records},indent=2)+'\n')
    (ROOT/'dev_predictions/prediction_export_verification.json').write_text(json.dumps({'status':'PASS','prediction_count':len(pred_records),'checkpoint_sha256':training['checkpoint_final_sha256'],'checkpoint_name':'checkpoint_final.pth','fold':'all','tile_step_size':0.5,'gaussian':True,'mirroring':False,'tta':False,'postprocessing':False,'inference_seconds':inference_seconds,'gpu_peak_allocated_bytes':peak_alloc,'gpu_peak_reserved_bytes':peak_reserved,'images':pred_records},indent=2)+'\n')
    # CPU endpoint metrics and the frozen original T0 comparison are computed only after terminal prediction exists.
    sys.path.insert(0,str(BASE/'code/retinal-vessel-study'))
    from retinal_m0.maples_evaluate import evaluate_arrays, load_mask
    from retinal_m0.maples_thin_skeleton_bce_cpu_gate import s3_target
    from scipy.ndimage import binary_dilation
    CROSS=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=bool)
    eval_rows=[]; t0_rows=[]
    t0_metrics=json.loads(pathlib.Path('${RETINA_PROJECT_ROOT}/experiments/t0_tw_core_20261016/runs/seed_2026/T0/evaluation/step_002000/metrics.json').read_text())
    t0_probs=pathlib.Path('${RETINA_PROJECT_ROOT}/experiments/t0_tw_core_20261016/runs/seed_2026/T0/evaluation/step_002000/probabilities')
    for row in rows:
        name=row['filename']; inp=row['inputs']
        for role in ('GT','FOV','regions'):
            if sha(pathlib.Path(inp[role]['path']))!=inp[role]['sha256']: raise RuntimeError('Frozen Dev reference hash changed '+name+'/'+role)
        v=load_mask(inp['GT']['path']); fov=load_mask(inp['FOV']['path'])
        g,sk,radius,s3=s3_target(v,fov)
        with np.load(inp['regions']['path'],allow_pickle=False) as b:
            u=b['U'].astype(bool); regions={k:b[k].astype(bool) for k in ('internal','near','far')}; thin=b['thin'].astype(bool)
        if not np.array_equal(thin,s3): raise RuntimeError('Frozen full-native S3 mismatch '+name)
        omega=fov & ~u
        denom=int((s3 & omega).sum())
        if denom!=row['new_full_domain_S3_Omega_denominator']: raise RuntimeError('Full-domain denominator changed '+name)
        for label,prob,path,dest in [('nnunet',np.load(pred_dir/(name+'.npy'),allow_pickle=False),pred_dir/(name+'.npy'),eval_rows),('T0',np.load(t0_probs/(name+'.npy'),mmap_mode='r',allow_pickle=False),t0_probs/(name+'.npy'),t0_rows)]:
            base=evaluate_arrays(prob,v,fov,threshold=0.5,u=u,regions=regions,thin=s3,matching_support='omega')
            pred=(prob>=0.5)&omega; pred_d1=binary_dilation(pred,structure=CROSS)
            d0=int((s3&omega&pred).sum()); d1=int((s3&omega&pred_d1).sum())
            rec={k:val['value'] for k,val in base.items() if isinstance(val,dict) and 'value' in val}
            rec.update({k+'_numerator':val['numerator'] for k,val in base.items() if isinstance(val,dict) and 'numerator' in val})
            rec.update({k+'_denominator':val['denominator'] for k,val in base.items() if isinstance(val,dict) and 'denominator' in val})
            rec.update({'filename':name,'native_shape':row['native_shape'],'new_full_domain_d0_hits':d0,'new_full_domain_d1_hits':d1,'new_full_domain_denominator':denom,'fov_thin_recall_d0_w3':(d0/denom if denom else None),'fov_thin_recall_d1_w3':(d1/denom if denom else None),'probability_sha256':sha(path)})
            dest.append(rec)
        if label=='T0':
            saved=next(x for x in t0_metrics['images'] if x['filename']==name)
            for key in ('dice','precision','recall','fp_density','near_pixel_recall','near_thin_recall_d1','far_recall'):
                for field in ('numerator','denominator','NA_reason'):
                    if base[key][field]!=saved[key][field]: raise RuntimeError('T0 frozen CPU recalculation mismatch '+name+'/'+key+'/'+field)
        if denom==0 and row['new_full_domain_S3_Omega_denominator']!=0: raise RuntimeError('unexpected NA')
    if len(eval_rows)!=28 or sum(x['new_full_domain_denominator'] for x in eval_rows)!=7291: raise RuntimeError('final full-domain cohort mismatch')
    if sum(x['near_thin_recall_d1_denominator'] for x in eval_rows)!=77 or sum(x['near_thin_recall_d1_denominator']==0 for x in eval_rows)!=19: raise RuntimeError('original near-denominator NA status changed')
    # Evaluate baseline exact counts and preserve existing T0 metrics.
    for row in t0_rows:
        old=next(x for x in t0_metrics['images'] if x['filename']==row['filename'])
        for metric in ('dice','precision','recall','fp_density','near_pixel_recall','near_thin_recall_d1','far_recall'):
            if row.get(metric+'_numerator')!=old[metric]['numerator'] or row.get(metric+'_denominator')!=old[metric]['denominator']:
                raise RuntimeError('T0 CPU baseline recalculation differs: '+row['filename']+'/'+metric)
    comparison,indices=bootstrap_paired(eval_rows,t0_rows)
    analysis=ROOT/'analysis'; analysis.mkdir(parents=True,exist_ok=False)
    (analysis/'nnunet_per_image.json').write_text(json.dumps(eval_rows,indent=2,allow_nan=False)+'\n')
    (analysis/'t0_recalculated_per_image.json').write_text(json.dumps(t0_rows,indent=2,allow_nan=False)+'\n')
    (analysis/'paired_comparison.json').write_text(json.dumps({'contrast':'nnUNet seed2026 final minus T0 seed2026 step2000','threshold':0.5,'bootstrap_seed':20261016,'bootstrap_replicates':1000,'metrics':comparison},indent=2,allow_nan=False)+'\n')
    (analysis/'bootstrap_indices.json').write_text(json.dumps(indices,separators=(',',':'))+'\n')
    report={'status':'EVALUATION_COMPLETE','protocol_id':PROTO['protocol_id'],'model':'nnUNet seed2026 fresh standard 2D fold=all 100epochs checkpoint_final','scope':'Dev28 reused development set; not independent validation','threshold':0.5,'inference':{'checkpoint_sha256':training['checkpoint_final_sha256'],'tile_step_size':0.5,'gaussian':True,'mirroring':False,'tta':False,'postprocessing':False,'seconds':inference_seconds},'counts':{'Dev_images':28,'full_domain_S3_Omega_positions':7291,'original_near_valid_images':9,'original_near_positions':77,'original_near_NA_images':19},'paired_comparison':comparison,'limits':['single-seed process comparison only','different FOV support/loss/sampling/preprocessing than T0','Dev reused for endpoint evaluation','no official Test access','one nnU-Net result does not establish independent validation'],'official_test_accessed':False}
    (analysis/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (analysis/'paired_per_image.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['filename','native_shape','probability_sha256']
        for metric in ('dice','precision','fp_density','recall','far_recall','near_pixel_recall','near_thin_recall_d1','fov_thin_recall_d0_w3','fov_thin_recall_d1_w3'):
            fields += [metric,metric+'_numerator',metric+'_denominator']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for row in eval_rows: w.writerow({k:(json.dumps(row[k]) if k=='native_shape' else row.get(k)) for k in fields})
    verification={'status':'PASS','training_complete_sha256':sha(ROOT/'training_completion.json'),'checkpoint_final_sha256':training['checkpoint_final_sha256'],'dev_probability_count':len(pred_records),'all_probabilities_native_float32_finite':all(x['dtype']=='float32' and x['finite'] and x['native_shape'] for x in pred_records),'dev_count':28,'full_domain_points':sum(x['new_full_domain_denominator'] for x in eval_rows),'full_domain_d0_denominator':sum(x['new_full_domain_denominator'] for x in eval_rows),'full_domain_d1_denominator':sum(x['new_full_domain_denominator'] for x in eval_rows),'old_near_positions':sum(x['near_thin_recall_d1_denominator'] for x in eval_rows),'old_near_NA':sum(x['near_thin_recall_d1_denominator']==0 for x in eval_rows),'t0_exact_cpu_recalculation':True,'bootstrap_n_keys':list(indices.keys()),'official_test_accessed':False,'new_dev_model_calls_after_final_checkpoint':28}
    verification['checks']={'prediction_count':verification['dev_probability_count']==28,'full_domain_7291':verification['full_domain_points']==7291,'old_near_9_77_19NA':verification['old_near_positions']==77 and verification['old_near_NA']==19,'all_probabilities':verification['all_probabilities_native_float32_finite'],'t0_recalculation':verification['t0_exact_cpu_recalculation']}
    verification['status']='PASS' if all(verification['checks'].values()) else 'FAIL'
    (ROOT/'completion_verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False)+'\n')
    if verification['status']!='PASS': raise RuntimeError('completion verification failed')
    print(json.dumps({'status':'EVALUATION_COMPLETE','prediction_count':len(pred_records),'inference_seconds':inference_seconds,'analysis':str(analysis),'comparison_metrics':list(comparison)},indent=2))


if __name__=='__main__': main()
