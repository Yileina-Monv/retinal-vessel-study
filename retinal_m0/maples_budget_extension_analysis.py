"Reproducible fixed-threshold and same-FP analysis for the frozen 2026 budget run."
from __future__ import annotations
import csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from retinal_data.maples_native import NativeMaplesDataset
from retinal_m0.maples_evaluate import evaluate_arrays, load_mask
from retinal_m0.state import state_digest
from retinal_m0 import maples_budget_extension as run

ROOT=run.ROOT; OUT=run.OUT; METHODS=('T0','TW')
CORE_METRICS=('near_thin_recall_d1','dice','precision','fp_density','far_recall','recall','near_pixel_recall')
DIAMETER_METRICS=tuple(f'near_thin_recall_d1_diameter_le{d}' for d in (2,4,5))
METRICS=CORE_METRICS+DIAMETER_METRICS
CONTRASTS=('TW4000_minus_T04000','T04000_minus_T02000','TW4000_minus_TW2000','interaction_4000_minus_2000')

def writej(path,obj):run.writej(path,obj)
def sha(path):return run.sha(path)
def readj(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows_csv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def ref_load(ref):
 b=Path(ref['region_bundle'])
 if sha(b)!=ref['region_bundle_sha256']:raise RuntimeError('Frozen region bundle hash mismatch: '+str(b))
 return load_mask(ref['vessel_path']),load_mask(ref['fov_path']),np.load(b,allow_pickle=False)
def eval_prob(prob,ref,threshold=.5,include_diameters=False):
 from retinal_m1.maps import targets
 y,f,b=ref_load(ref)
 with b as z:
  result=evaluate_arrays(prob,y,f,threshold=float(threshold),u=z['U'],regions={k:z[k] for k in ('internal','near','far')},thin=z['thin'],matching_support='omega')
  if float(threshold)==.5 or include_diameters:
   maps=targets(y,f)
   for diameter in (2,4,5):
    thin=maps['skeleton'] & ((2.0*maps['skeleton_radius'])<=float(diameter))
    extra=evaluate_arrays(prob,y,f,threshold=float(threshold),u=z['U'],regions={k:z[k] for k in ('internal','near','far')},thin=thin,matching_support='omega')
    result[f'near_thin_recall_d1_diameter_le{diameter}']=extra['near_thin_recall_d1']
  return result
def load_specs():
 doc=readj(run.DEV); refs=doc['images']; names=[x['filename'] for x in refs]
 if len(names)!=28 or len(set(names))!=28:raise RuntimeError('Frozen dev manifest membership invalid')
 return refs,names
def paths(method,step):
 if step==2000:
  base=run.PARENT/'runs/seed_2026'/method/'evaluation/step_002000'
 else:base=OUT/'runs/seed_2026'/method/'evaluation'/f'step_{step:06d}'
 return base/'metrics.json',base/'probabilities'
def load_metric_images(method,step,names):
 mp,pdir=paths(method,step); d=readj(mp)
 if d.get('status')!='passed' or d.get('method')!=method or d.get('seed')!=2026 or d.get('step')!=step or d.get('threshold')!=.5:raise RuntimeError('Fixed evaluation metadata mismatch: '+str(mp))
 im={x['filename']:x for x in d['images']}
 if list(x['filename'] for x in d['images'])!=names:raise RuntimeError('Evaluation image order differs from frozen dev manifest')
 probs={}
 # The frozen protocol stores predictions only at the reused 2000 baseline and new 4000 endpoint.
 # Intermediate checkpoints are fully evaluated from their persisted per-image count records.
 if step in (2000,4000):
  for name in names:
   p=pdir/f'{name}.npy'
   if not p.is_file():raise RuntimeError('Required probability map missing: '+str(p))
   probs[name]=np.load(p,allow_pickle=False)
 return d,im,probs
def image_metric(row,m):
 z=row[m]
 return None if z['value'] is None else float(z['value'])
def bootstrap(values,ids,indices):
 a=np.asarray(values,dtype=np.float64)
 if not len(a) or len(a)!=len(ids):return None
 est=a[indices].mean(axis=1)
 return {'mean':float(a.mean()),'n':len(a),'ci95':[float(x) for x in np.quantile(est,[.025,.975],method='linear')],
         'bootstrap_mean_sd':float(est.std(ddof=1))}
def fp_density(prob,y,fov,threshold):
 pred=prob>=np.float32(threshold)
 den=int(fov.sum()); num=int((pred&fov&~y).sum())
 return (num/den if den else None),num,den
def calc_samefp(method,step,names,refs,probs,target,grid):
 fp_values=[]
 for name in names:
  y=load_mask(refs[name]['vessel_path']);f=load_mask(refs[name]['fov_path']);p=probs[name]
  if p.shape!=y.shape or p.shape!=f.shape:raise RuntimeError('Same-FP native image shape mismatch')
  fp_values.append(np.asarray([fp_density(p,y,f,t)[0] for t in grid],dtype=np.float64))
 macro=np.stack(fp_values).mean(axis=0)
 feasible=macro<=target
 if not feasible.any():return {'method':method,'step':step,'threshold':None,'macro_fp_density':None,'target':target,'feasible':False},macro
 ix=int(np.flatnonzero(feasible)[0]);threshold=float(grid[ix]);detail=[]
 for name in names:
  r=refs[name];p=probs[name];metrics=eval_prob(p,r,threshold,include_diameters=True)
  fp,fpn,den=fp_density(p,load_mask(r['vessel_path']),load_mask(r['fov_path']),threshold)
  detail.append({'filename':name,'fp_density':fp,'fp_numerator':fpn,'fov_denominator':den,'metrics':metrics})
 return {'method':method,'step':step,'threshold':threshold,'macro_fp_density':float(macro[ix]),'target':target,'target_gap':float(macro[ix]-target),'feasible':True,'images':detail},macro
def analyzer():
 if not (OUT/'implementation_gate/verification.json').is_file():raise RuntimeError('Implementation gate evidence absent')
 gate=readj(OUT/'implementation_gate/verification.json')
 if gate.get('all_passed') is not True:raise RuntimeError('Implementation gate did not pass')
 refs_list,names=load_specs();refs={r['filename']:r for r in refs_list}
 metric={};probs={};evals={}
 for method in METHODS:
  for step in (2000,*range(2200,4001,200)):
   d,im,pm=load_metric_images(method,step,names)
   if step in (2000,4000):
    for name in names:
     fresh=eval_prob(pm[name],refs[name],.5)
     for extra in DIAMETER_METRICS:
      if step==4000:
       stored=im[name].get(extra)
       if stored is None or stored['numerator']!=fresh[extra]['numerator'] or stored['denominator']!=fresh[extra]['denominator']:
        raise RuntimeError(f'Endpoint diameter supplement mismatch {method} {name} {extra}')
      im[name][extra]=fresh[extra]
   metric[method,step]=im;probs[method,step]=pm;evals[method,step]=d
 # Verify all 28 images at every evaluation point, fixed 9/77 primary and 19 expected NA.
 pconf=run.protocol()['evaluation']['primary'];primary_ids=pconf['primary_ids']
 for m in METHODS:
  for step in (2000,*range(2200,4001,200)):
   im=metric[m,step]
   valid=[n for n in names if image_metric(im[n],'near_thin_recall_d1') is not None]
   if set(valid)!=set(primary_ids) or len(valid)!=len(primary_ids):raise RuntimeError(f'{m} step {step} primary effective-ID set changed')
   if sum(image_metric(im[n],'near_thin_recall_d1') is None for n in names)!=19:raise RuntimeError('Primary NA count changed')
 # Recompute 2000 and 4000 fixed metrics from saved probabilities to verify stored evaluation records.
 recalc=[]
 for m in METHODS:
  for step in (2000,4000):
   for n in names:
    fresh=eval_prob(probs[m,step][n],refs[n],.5);stored=metric[m,step][n]
    for met in METRICS:
     if met not in stored: continue
     if fresh[met]['numerator']!=stored[met]['numerator'] or fresh[met]['denominator']!=stored[met]['denominator']:
      raise RuntimeError(f'Fixed metric count mismatch {m} {step} {n} {met}')
     recalc.append({'method':m,'step':step,'filename':n,'metric':met,'numerator':fresh[met]['numerator'],'denominator':fresh[met]['denominator'],'value':fresh[met]['value']})
 # Paired temporal and method contrasts, including exact per-image numerators/denominators.
 per=[];values={}
 for met in METRICS:
  rows=[];ids=[]
  for n in names:
   v={k:image_metric(metric[m,st][n],met) for m,st,k in [('T0',2000,'T02000'),('T0',4000,'T04000'),('TW',2000,'TW2000'),('TW',4000,'TW4000')]}
   nums={}
   for m,st in [('T0',2000),('T0',4000),('TW',2000),('TW',4000)]:
    z=metric[m,st][n][met];nums[f'{m}{st}_numerator']=z['numerator'];nums[f'{m}{st}_denominator']=z['denominator']
   missing=[x is None for x in v.values()]
   out={'metric':met,'filename':n,**v,**nums}
   if any(missing):
    if not all(missing):raise RuntimeError(f'Asymmetric NA across paired method/time for {met} image {n}')
    out.update({c:None for c in CONTRASTS});per.append(out)
    if met=='near_thin_recall_d1' and n in primary_ids:raise RuntimeError('Expected primary image became NA: '+n)
    if met=='near_thin_recall_d1' and n not in primary_ids:continue
    if met=='near_pixel_recall' or met in DIAMETER_METRICS:
     for method,step in [('T0',2000),('T0',4000),('TW',2000),('TW',4000)]:
      z=metric[method,step][n][met]
      if z['denominator']!=0 or z['NA_reason']!='zero_denominator':raise RuntimeError(f'Invalid zero-denominator NA for {met} image {n}')
     continue
    raise RuntimeError(f'Unexpected NA for {met} image {n}')
   c={'TW4000_minus_T04000':v['TW4000']-v['T04000'],'T04000_minus_T02000':v['T04000']-v['T02000'],
      'TW4000_minus_TW2000':v['TW4000']-v['TW2000'],'interaction_4000_minus_2000':(v['TW4000']-v['T04000'])-(v['TW2000']-v['T02000'])}
   out.update(c);per.append(out);rows.append((n,v,c));ids.append(n)
  values[met]=(ids,rows)
 # Reuse the same frozen image bootstrap index table for every contrast with identical n.
 stats=[];boot_rows=[];boot_estimates=[];index_tables={}
 for met,(ids,rows) in values.items():
  n=len(ids)
  if n==0:
   for con in CONTRASTS:stats.append({'metric':met,'contrast':con,'mean':None,'n':0,'ci95':None,'bootstrap_mean_sd':None,'training_seed_sd':'NA','seed_scope':'no eligible images'})
   continue
  if n not in index_tables:
   rng=np.random.Generator(np.random.PCG64(20261016));index_tables[n]=rng.integers(0,n,size=(1000,n))
  ix=index_tables[n]
  for con in CONTRASTS:
   arr=np.asarray([r[2][con] for r in rows],dtype=np.float64);b=bootstrap(arr,ids,ix)
   estimates=arr[ix].mean(axis=1)
   for rep,value in enumerate(estimates):boot_estimates.append({'metric':met,'contrast':con,'n':n,'replicate':rep,'estimate':float(value)})
   stats.append({'metric':met,'contrast':con,**b,'training_seed_sd':'NA','seed_scope':'single seed 2026; descriptive paired-image interval'})
 # Fixed-reference Z0@0.5 recalculation for descriptive absolute context.
 zdir=run.ROOT/'experiments/a15_2_z0_t0_calibration_20261015/z0/probabilities';z0=[]
 for n in names:
  p=np.load(zdir/f'{n}.npy',allow_pickle=False);z0.append({'filename':n,**eval_prob(p,refs[n],.5)})
 z0_macro={}
 for met in METRICS:
  vals=[r[met]['value'] for r in z0 if r[met]['value'] is not None]
  z0_macro[met]={'mean':float(np.mean(vals)) if vals else None,'n':len(vals),'NA':28-len(vals)}
 # Same-FP supplementary grid, using the frozen common target and exact float32 1001-point grid.
 sp=run.protocol()['same_FP'];target=float(sp['target_macro_FP_per_FOV']);grid=np.linspace(0,1,1001,dtype=np.float32)
 grid_rows=[];same=[];same_per=[]
 for m in METHODS:
  for st in (2000,4000):
   result,macro=calc_samefp(m,st,names,refs,probs[m,st],target,grid);same.append({k:v for k,v in result.items() if k!='images'})
   for j,t in enumerate(grid):grid_rows.append({'method':m,'step':st,'threshold':float(t),'macro_fp_density':float(macro[j]),'target':target,'feasible':bool(macro[j]<=target)})
   if result['feasible']:
    for row in result['images']:same_per.append({'method':m,'step':st,'threshold':result['threshold'],**{k:v for k,v in row.items() if k!='metrics'},
      **{met:row['metrics'][met]['value'] for met in METRICS}})
 # Save each effective-n index table once and each metric/contrast bootstrap estimate.
 for n,table in index_tables.items():
  for rep in range(table.shape[0]):
   for pos in range(table.shape[1]):boot_rows.append({'n':n,'replicate':rep,'position':pos,'sampled_position':int(table[rep,pos])})
 # Write reproducible analysis artifacts.
 ad=OUT/'analysis';ad.mkdir(parents=True,exist_ok=True)
 def write_csv(name,rows_):
  if not rows_ and name=='same_fp_selected_per_image.csv': rows_=[{'method':None,'step':None,'threshold':None,'filename':None,'fp_density':None,'fp_numerator':None,'fov_denominator':None}]
  if not rows_:raise RuntimeError('Refusing to write empty table '+name)
  keys=list(dict.fromkeys(k for r in rows_ for k in r))
  with (ad/name).open('w',newline='',encoding='utf-8-sig') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows_)
 write_csv('fixed_0p5_per_image.csv',recalc)
 write_csv('paired_temporal_interaction_per_image.csv',per)
 write_csv('paired_contrast_summary.csv',stats)
 write_csv('bootstrap_indices.csv',boot_rows)
 write_csv('bootstrap_estimates.csv',boot_estimates)
 write_csv('same_fp_full_grid.csv',grid_rows)
 write_csv('same_fp_selected_per_image.csv',same_per)
 write_csv('same_fp_selected_thresholds.csv',same)
 # Copy complete 4000 evaluation curves/loss traces into a compact source manifest; logs remain authoritative.
 curve=[]
 for m in METHODS:
  for step in (2000,*range(2200,4001,200)):
   curve.append({'method':m,'step':step,**{k:evals[m,step]['aggregate'][k]['mean'] for k in CORE_METRICS}})
 # Verify the complete paired logs before acceptance.
 logs={}; parsed={}
 for m in METHODS:
  rd=OUT/'runs/seed_2026'/m
  train=[json.loads(x) for x in (rd/'train.jsonl').read_text().splitlines()]
  pair=[json.loads(x) for x in (rd/'pairing.jsonl').read_text().splitlines()]
  if len(train)!=2000 or len(pair)!=2000:raise RuntimeError(m+' must have exactly 2000 new update/pairing rows')
  if [int(x['step']) for x in train]!=list(range(2001,4001)) or [int(x['step']) for x in pair]!=list(range(2001,4001)):raise RuntimeError(m+' global-step log range mismatch')
  if any(not x.get('parameters_updated') for x in train):raise RuntimeError(m+' unsuccessful update was counted')
  if any(abs(float(x.get('learning_rate',.0001))-.0001)>1e-15 for x in train):raise RuntimeError(m+' learning rate changed during continuation')
  logs[m]=(train,pair);parsed[m]=pair
 loss_raw=[];loss_cycles=[]
 for m in METHODS:
  train=logs[m][0]
  for x in train:loss_raw.append({'method':m,'step':x['step'],'loss':x['loss'],'learning_rate':x.get('learning_rate',.0001)})
  cycle_ids=sorted({(int(x['step'])-1)//110 for x in train})
  for c in cycle_ids:
   part=[x for x in train if (int(x['step'])-1)//110==c]
   loss_cycles.append({'method':m,'global_110_update_cycle':c,'start_step':part[0]['step'],'end_step':part[-1]['step'],'updates':len(part),'mean_loss':float(np.mean([x['loss'] for x in part]))})
 write_csv('train_loss_raw.csv',loss_raw);write_csv('train_loss_110_update_cycles.csv',loss_cycles)
 for i,(a,b) in enumerate(zip(parsed['T0'],parsed['TW'])):
  if {k:a[k] for k in ('step','draws','filenames','geometry','input_sha256')}!={k:b[k] for k in ('step','draws','filenames','geometry','input_sha256')}:
   raise RuntimeError('Full runtime paired input mismatch at update '+str(i+2001))
 if parsed['T0'][0]['draws']!=[8000,8001,8002,8003] or parsed['T0'][-1]['draws']!=[15996,15997,15998,15999]:raise RuntimeError('Draw boundary mismatch')
 # Build immutable reuse manifest for all existing 2000-step inputs.
 reuse={'status':'read_only_reuse_verified','sources':[]}
 for m in METHODS:
  rd=run.PARENT/'runs/seed_2026'/m
  for rel in ('step_002000.pt','config.json','evaluation/step_002000/metrics.json','evaluation/step_002000/per_image.csv'):
   p=rd/rel;reuse['sources'].append({'method':m,'path':str(p),'sha256':sha(p)})
  for p in sorted((rd/'evaluation/step_002000/probabilities').glob('*.npy')):reuse['sources'].append({'method':m,'path':str(p),'sha256':sha(p)})
 writej(OUT/'reuse_manifest.json',reuse)
 # Main paired effect and guardrail summaries.
 keystats={(x['metric'],x['contrast']):x for x in stats}
 protection={}
 for met,direction in (('precision','higher'),('fp_density','lower'),('far_recall','higher')):
  z=keystats[(met,'TW4000_minus_T04000')]
  bound=z['ci95'][0] if direction=='higher' else z['ci95'][1]
  protection[met]={'direction':direction,'mean_change':z['mean'],'ci95':z['ci95'],'support_zero_degradation':bool(bound>=0 if direction=='higher' else bound<=0)}
 protection['all_three_supported']=all(protection[k]['support_zero_degradation'] for k in ('precision','fp_density','far_recall'))
 summary={'protocol_id':run.protocol()['protocol_id'],'seed':2026,'status':'passed','primary_metric':'near_thin_recall_d1','primary_ids':primary_ids,'primary_n':9,'primary_target_points':77,'primary_NA_other_images':19,'protection_support':protection,
  'fixed_threshold':.5,'single_seed_sd':'NA','z0_at_0p5':z0_macro,'contrast_summary':stats,'same_fp':same,'curve':curve,
  'interpretation_limits':run.protocol()['statistics']['limits'],'official_test_accessed':False}
 writej(ad/'analysis_summary.json',summary)
 # A self-contained readable HTML review report.
 html='''<!doctype html><meta charset=utf-8><title>T0/TW 4000-step budget extension</title><style>body{font:15px system-ui;max-width:1200px;margin:36px auto;padding:0 20px;color:#18212b}table{border-collapse:collapse;margin:18px 0;width:100%}th,td{border:1px solid #ccd3dc;padding:6px;text-align:left}th{background:#eef2f7}.note{background:#f6f8fa;padding:12px}</style><h1>T0/TW seed 2026: 2000→4000 budget extension</h1><p class=note>Post-hoc, single-seed development-set analysis. Fixed endpoint 4000; threshold 0.5. The original 2000-step formal result remains unchanged. No official Test data were accessed.</p><h2>Paired image contrasts</h2><table><thead><tr><th>Metric</th><th>Contrast</th><th>Mean</th><th>n</th><th>95% image-bootstrap interval</th></tr></thead><tbody>'''
 for x in stats:
  mean_text='NA' if x['mean'] is None else format(x['mean'],'.6f')
  ci_text='NA' if x['ci95'] is None else '['+format(x['ci95'][0],'.6f')+', '+format(x['ci95'][1],'.6f')+']'
  html+='<tr><td>'+str(x['metric'])+'</td><td>'+str(x['contrast'])+'</td><td>'+mean_text+'</td><td>'+str(x['n'])+'</td><td>'+ci_text+'</td></tr>'
 html+='</tbody></table><h2>Same-FP endpoint supplement</h2><table><tr><th>Method</th><th>Step</th><th>Threshold</th><th>Macro FP/FOV</th><th>Common target</th></tr>'
 for x in same:
  html+='<tr><td>'+str(x['method'])+'</td><td>'+str(x['step'])+'</td><td>'+str(x['threshold'])+'</td><td>'+str(x['macro_fp_density'])+'</td><td>'+str(x['target'])+'</td></tr>'
 html+='</table><p>Intervals resample images and are conditional on seed 2026; they do not quantify training-seed or patient-level uncertainty. Same-FP is descriptive and does not replace fixed-threshold results.</p><p>Full per-image data, bootstrap indices, and threshold grid are saved beside this report.</p>'
 (OUT/'report.html').write_text(html,encoding='utf-8')
 # Checkpoint, evaluation, probability and source closure checks.
 ckpts=[];eval_count=0;prob_files=[]
 for m in METHODS:
  rd=OUT/'runs/seed_2026'/m
  last=torch_checkpoint_info(rd/'last.pt',m)
  if last['step']!=4000 or last['next_global_draw']!=16000 or last['optimizer_step']!=4000:raise RuntimeError(m+' final continuation state invalid')
  for step in range(2200,4001,200):
   cp=rd/f'step_{step:06d}.pt'
   if not cp.is_file():raise RuntimeError('Missing mandated checkpoint '+str(cp))
   ckp=torch.load(cp,map_location='cpu',weights_only=True);cb=ckp.get('binding');hist=readj(rd/'run_binding_history.json')
   current=hist['active_binding']
   if cb!=current and not (m=='T0' and step==2200 and cb in hist.get('previous_bindings',[])):
    raise RuntimeError('Checkpoint implementation binding is not in the recorded exact-code history: '+str(cp))
   ckpts.append({'method':m,'step':step,'path':str(cp),'sha256':sha(cp),'checkpoint_execution_binding_sha256':cb['execution_binding_sha256']})
   mp=rd/'evaluation'/f'step_{step:06d}'/'metrics.json'
   if not mp.is_file():raise RuntimeError('Missing mandated fixed evaluation '+str(mp))
   ev=readj(mp)
   if ev.get('dev_images')!=28 or ev.get('threshold')!=.5:raise RuntimeError('Evaluation coverage/threshold mismatch')
   eval_count+=1
  pdir=rd/'evaluation/step_004000/probabilities';files=sorted(pdir.glob('*.npy'))
  if len(files)!=28:raise RuntimeError(m+' endpoint must have 28 probability maps')
  for pp in files:
   a=np.load(pp,allow_pickle=False);name=pp.stem;shape=next(x['native_shape'] for x in refs_list if x['filename']==name)
   if list(a.shape)!=shape or a.dtype!=np.float32 or not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise RuntimeError('Invalid endpoint probability '+str(pp))
   prob_files.append({'method':m,'path':str(pp),'sha256':sha(pp),'shape':list(a.shape),'dtype':str(a.dtype)})
 # Recheck all 237 protected parent inputs before claiming no overwrite.
 protected=run.protected_hashes()
 cv={'status':'passed','all_passed':True,'protocol_sha256':run.PROTO_SHA,'source_manifest_sha256':run.MANIFEST_SHA,
  'execution_binding_sha256':run.bind_sha(),'protected_parent_files_unchanged':len(protected),'source_protected_hashes':protected,
  'binding_histories':{m:readj(OUT/'runs/seed_2026'/m/'run_binding_history.json') for m in METHODS},
  'methods':{},'paired_runtime_rows':2000,'paired_runtime_draws':8000,'paired_runtime_input_digests_match':True,
  'checkpoints':ckpts,'new_checkpoint_count':len(ckpts),'new_fixed_evaluation_count':eval_count,'endpoint_probability_maps':prob_files,'endpoint_probability_count':len(prob_files),
  'primary_valid_images':9,'primary_target_points':77,'primary_NA_images':19,'fixed_threshold_metrics_recomputed':len(recalc),
  'analysis_files':[str(x) for x in sorted(ad.iterdir()) if x.is_file()],'report_html':str(OUT/'report.html'),
  'official_test_accessed':False,'best_checkpoint_selection':False,'additional_training':False}
 for m in METHODS:
  rd=OUT/'runs/seed_2026'/m;ck=torch_checkpoint_info(rd/'last.pt',m)
  cv['methods'][m]={'status':'passed','successful_updates':2000,'new_draws':8000,'start_step':2000,'end_step':4000,'next_global_draw':16000,
    'train_rows':len(logs[m][0]),'pairing_rows':len(logs[m][1]),'checkpoint':str(rd/'last.pt'),'checkpoint_sha256':sha(rd/'last.pt'),'optimizer_step':ck['optimizer_step'],
    'lr_start_end':[logs[m][0][0].get('learning_rate',.0001),logs[m][0][-1].get('learning_rate',.0001)],'amp_retry_count':sum(int(x.get('amp_retries',0)) for x in logs[m][0])}
 writej(OUT/'completion_verification.json',cv)
 return cv

def torch_checkpoint_info(path,method):
 import torch
 ck=torch.load(path,map_location='cpu',weights_only=True)
 proto=run.protocol();src=proto['resume']['sources'][method]
 hist=readj(OUT/'runs/seed_2026'/method/'run_binding_history.json')
 known=[hist.get('active_binding')]+hist.get('previous_bindings',[])
 rb=ck.get('binding')
 expected_lineage={'config_sha256':sha(OUT/'runs/seed_2026'/method/'config.json'),
  'parent_checkpoint_sha256':src['checkpoint_sha256'],'parent_config_sha256':src['parent_config_sha256'],'parent_binding_sha256':src['parent_binding']['execution_binding_sha256']}
 if rb not in known or any(rb.get(k)!=v for k,v in expected_lineage.items()):raise RuntimeError(method+' checkpoint binding is absent from its recorded history or has invalid parent lineage')
 steps={int(x['step']) for x in ck['optimizer']['state'].values()}
 if len(steps)!=1:raise RuntimeError(method+' optimizer step counters diverged')
 return {'step':int(ck['step']),'next_global_draw':int(ck['next_global_draw']),'optimizer_step':steps.pop(),'model_digest':run.statehash(ck['model']),
  'optimizer_digest':run.statehash(ck['optimizer']),'scaler_digest':run.statehash(ck['scaler']),'rng_digest':run.statehash(ck['rng'])}

if __name__=='__main__':
 result=analyzer();print(json.dumps({'status':result['status'],'all_passed':result['all_passed'],'new_checkpoint_count':result['new_checkpoint_count'],'new_fixed_evaluation_count':result['new_fixed_evaluation_count'],'endpoint_probability_count':result['endpoint_probability_count']},indent=2))
