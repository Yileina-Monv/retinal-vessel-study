"""CPU final comparison against bound historical N0, original metrics plus fixed local FP."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
from common import *
import numpy as np,csv
sys.path.insert(0,str(CODE))
from retinal_m0.maples_branch_science_analysis import evaluate_image,metric_values
from retinal_m0.maples_stratified_science_analysis import hard_cldice

def main():
 validate(full=True)
 if read(GROUP/'job_state.json')['role']!='analysis':raise RuntimeError('wrong phase')
 out=GROUP/'analysis';out.mkdir(exist_ok=False)
 ref=read(P/'SCIENCE_EVALUATION_REFERENCE.json');ids=dict(ref['fixed_ids_by_metric']);ids['local_fp']=ref['local_FP_ids'];idx=ref['bootstrap_indices_by_n']
 refs=read(protocol()['references']['science_reference_manifest'])['images'];seedrows={};means={};stats={}
 def ci(v):
  ar=np.asarray(v,float);ix=np.asarray(idx[str(len(v))]);lo,hi=np.quantile(np.mean(ar[ix],axis=1),[.025,.975],method='linear')
  return {'estimate':float(np.mean(ar)),'lower':float(lo),'upper':float(hi),'n_images':len(v)}
 for seed in [2027,2028]:
  stage=GROUP/'runs'/f'seed_{seed}'
  if read(stage/'prediction_verification.json')['status']!='PASS' or read(stage/'PAIRING_COMPLETE.json')['status']!='PASS':raise RuntimeError('incomplete pair/prediction')
  ep=read(BASELINE/'runs'/f'seed_{seed}'/'analysis/endpoint_100epochs.json');baseline={r['filename']:r for r in ep['per_image']};rows={}
  for r in refs:
   name=r['filename']
   if sha(r['reference_npz'])!=r['reference_npz_sha256']:raise RuntimeError('reference drift')
   with np.load(r['reference_npz'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
   f=stage/'dev_predictions'/(name+'.npy');prob=np.load(f,allow_pickle=False)
   if prob.dtype!=np.float32 or list(prob.shape)!=r['native_shape'] or not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):raise RuntimeError('probability invalid')
   full=evaluate_image(prob,arrays,'B2',threshold=.5);v=metric_values(full)
   h=hard_cldice((prob>=.5)&arrays['FOV'].astype(bool),arrays['GT'],arrays['FOV']);v['hard_cldice']=h['cldice'];full['hard_cldice_diagnostic']=h
   area=arrays['near'].astype(bool)&arrays['Omega'].astype(bool);gt=arrays['GT'].astype(bool);fp=int(((prob>=.5)&~gt&area).sum());den=int(area.sum());bg=int((area&~gt).sum());oldr=ref['baseline_region_records'][str(seed)][name]
   if den!=oldr['near_area'] or bg!=oldr['near_background_pixels']:raise RuntimeError('local FP support drift')
   v['local_fp']=fp/den if den else None;base=dict(baseline[name]['methods']['NNUNET']);base['local_fp']=oldr['N0_near_FP']/den if den else None
   rows[name]={'filename':name,'N0':base,'NL':v,'NL_full_record':full,'local_FP':{'NL':fp,'N0':oldr['N0_near_FP'],'area':den,'background':bg,'NL_FP_per_background':fp/bg if bg else None},'probability_sha256':sha(f)}
  seedrows[str(seed)]=rows;means[str(seed)]={m:{} for m in ['N0','NL']};stats[str(seed)]={}
  for metric,names in ids.items():
   for m in ['N0','NL']:
    vals=[rows[n][m][metric] for n in names]
    if any(v is None or not np.isfinite(v) for v in vals):raise RuntimeError('NA on fixed valid set '+metric)
    means[str(seed)][m][metric]=float(np.mean(vals))
   stats[str(seed)][metric]=ci([rows[n]['NL'][metric]-rows[n]['N0'][metric] for n in names])
 pooled={k:ci([sum(seedrows[str(s)][n]['NL'][k]-seedrows[str(s)][n]['N0'][k] for s in [2027,2028])/2 for n in ns]) for k,ns in ids.items()}
 loo=[]
 primary_ids=ids['near_thin_recall_d1']
 for omitted in primary_ids:
  kept=[n for n in primary_ids if n!=omitted]
  v=[sum(seedrows[str(s)][n]['NL']['near_thin_recall_d1']-seedrows[str(s)][n]['N0']['near_thin_recall_d1'] for s in [2027,2028])/2 for n in kept]
  loo.append({'omitted_image':omitted,'remaining_n':len(kept),'delta':float(np.mean(v)),'diagnostic_only':True})
 primary=all(stats[str(s)]['near_thin_recall_d1']['estimate']>0 for s in [2027,2028]) and pooled['near_thin_recall_d1']['lower']>0
 direction={'dice':1,'precision':1,'far_recall':1,'fp_density':-1,'local_fp':-1}
 point={k:all(direction[k]*stats[str(s)][k]['estimate']>=0 for s in [2027,2028]) for k in direction}
 interval={k:(pooled[k]['lower']>=0 if d==1 else pooled[k]['upper']<=0) for k,d in direction.items()}
 result={'status':'COMPLETE','comparison':'NL-N0; image paired then averaged within image across2seeds','primary_signal':primary,'point_guards':point,'interval_guards':interval,'classification':'LOCAL_SIGNAL_WITH_POINT_PROTECTION' if primary and all(point.values()) else 'LOCAL_GAIN_WITH_QUALITY_COST' if primary else 'NO_RELIABLE_LOCAL_GAIN_OR_INSUFFICIENT_EVIDENCE','formal_noninferiority':False,'independent_validation':False,'means':means,'per_seed':stats,'within_image_seed_average':pooled,'fixed_ids_by_metric':ids,'per_image':seedrows,'leave_one_image_out_primary':loo,'Test_access':False,'limitations':['9valid primary images77positions19NA','repeatedDev28','noNR','medical review0/105 and0/26','historical monitor lacks tensorSHA']}
 atomic(out/'NL_N0_COMPARISON.json',result,True)
 with (out/'paired_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['seed','metric','N0_macro','NL_macro','delta','CI_low','CI_high','n_images'])
  for seed in [2027,2028]:
   for metric,d in stats[str(seed)].items():w.writerow([seed,metric,means[str(seed)]['N0'][metric],means[str(seed)]['NL'][metric],d['estimate'],d['lower'],d['upper'],d['n_images']])
 lines=['# N0/NL固定100epochs终点：科研审查材料','','主指标增量信号：'+str(primary)+'；逐seed点保护全部通过：'+str(all(point.values()))+'；区间保护：'+str(interval),'','|seed|指标|N0 %|NL %|差值 pp|95%配对区间 pp|','|---|---|---:|---:|---:|---|']
 for seed in [2027,2028]:
  for k in ['near_thin_recall_d1','dice','precision','fp_density','local_fp','far_recall','hard_cldice']:
   d=stats[str(seed)][k];lines.append(f'|{seed}|{k}|{means[str(seed)]["N0"][k]*100:.4f}|{means[str(seed)]["NL"][k]*100:.4f}|{d["estimate"]*100:+.4f}|[{d["lower"]*100:+.4f},{d["upper"]*100:+.4f}]|')
 lines+=['','点保护通过不等于区间证明无恶化。主指标只有9图，两个seed不增加独立样本数；Dev反复使用，不是独立验证。原始逐图记录、NA、局部分母和不利图均保留。没有NR，不声明病灶位置特异性或医学连接恢复。','无论正负，先由Sol审查；不自动启动新实验。']
 (out/'科研结论_待Sol审查.md').write_text('\n'.join(lines))
 print('FINAL_ANALYSIS_COMPLETE',result['classification'],flush=True)
if __name__=='__main__':main()
