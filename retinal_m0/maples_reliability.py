"""Retrospective A15.2 reliability checks over frozen native predictions."""
from __future__ import annotations
import argparse, csv, hashlib, html, json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from retinal_m0.maples_evaluate import evaluate_arrays, load_mask
from retinal_m1.maps import targets
ROOT=Path('${RETINA_PROJECT_ROOT}'); B=ROOT/'experiments/a15_2_z0_t0_calibration_20261015'; DATA=ROOT/'data/maples_handoff_20261008'
GRID=np.linspace(0,1,1001,dtype=np.float32)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def write_json(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def metric_from_counts(numerator,denominator):
 n,d=int(numerator),int(denominator)
 return {'value':float(n/d) if d else None,'numerator':n,'denominator':d,'NA_reason':None if d else 'zero_denominator'}
def macro(values):
 v=list(values); return float(np.mean(v)) if v else None
def choose_threshold_from_counts(fp_counts,fov_pixels,z0_fp_density,fit_indices,grid):
 c=np.asarray(fp_counts,dtype=np.int64); f=np.asarray(fov_pixels,dtype=np.int64); z=np.asarray(z0_fp_density,dtype=np.float64); g=np.asarray(grid); ix=np.asarray(list(fit_indices),dtype=np.int64)
 if c.ndim!=2 or c.shape!=(len(f),len(g)) or len(z)!=len(f): raise ValueError('Misaligned FP counts, FOV, or baseline')
 if not len(ix) or np.any((ix<0)|(ix>=len(f))) or np.any(f<=0): raise ValueError('Valid fitting images and positive FOV required')
 target=float(np.mean(z[ix].tolist())); d=c[ix]/f[ix,None]; mg=np.asarray([float(np.mean(d[:,j].tolist())) for j in range(len(g))]); feasible=np.flatnonzero(mg<=target)
 if not len(feasible): return {'threshold':None,'target_fp_density':target,'actual_fp_density':None,'previous_threshold':float(g[-1]),'previous_fp_density':float(mg[-1]),'grid_index':None}
 j=int(feasible[0]); return {'threshold':float(g[j]),'target_fp_density':target,'actual_fp_density':float(mg[j]),'previous_threshold':float(g[j-1]) if j else None,'previous_fp_density':float(mg[j-1]) if j else None,'grid_index':j}
def paired_bootstrap(differences,*,reps=1000,seed=20261016,rng=None):
 d=np.asarray(list(differences),dtype=np.float64)
 if d.ndim!=1 or not len(d) or not np.isfinite(d).all(): raise ValueError('Finite paired image differences required')
 rng=rng if rng is not None else np.random.default_rng(seed); ix=rng.integers(0,len(d),size=(int(reps),len(d))); means=d[ix].mean(axis=1)
 return {'n_images':int(len(d)),'reps':int(reps),'seed':int(seed),'rng':'NumPy PCG64','interval':'empirical percentile, linear quantile, 2.5% and 97.5%','mean_difference':float(d.mean()),'ci95_low':float(np.quantile(means,.025,method='linear')),'ci95_high':float(np.quantile(means,.975,method='linear')),'draw_means':means}
def target_hit_distance(probability,target_rc,fov,u,threshold):
 p=np.asarray(probability); f=np.asarray(fov,bool); u=np.asarray(u,bool); r,c=map(int,target_rc)
 if p.ndim!=2 or p.shape!=f.shape or p.shape!=u.shape or not(0<=r<p.shape[0] and 0<=c<p.shape[1]): raise ValueError('Native coordinate/masks must align')
 support=(p>=threshold)&f&~u
 if not support.any(): return {'target_row':r,'target_col':c,'nearest_row':None,'nearest_col':None,'distance':None,'hit_d1':False}
 dist,ind=distance_transform_edt(~support,return_indices=True); return {'target_row':r,'target_col':c,'nearest_row':int(ind[0,r,c]),'nearest_col':int(ind[1,r,c]),'distance':float(dist[r,c]),'hit_d1':bool(dist[r,c]<=1)}
def lookup(prob,fov,u,threshold):
 s=(prob>=threshold)&fov&~u
 return (distance_transform_edt(~s,return_indices=True) if s.any() else (None,None))
def hit_at(dm,ind,rc):
 if dm is None:return None,None,None,False
 r,c=map(int,rc); return int(ind[0,r,c]),int(ind[1,r,c]),float(dm[r,c]),bool(dm[r,c]<=1)
def probability_fp_grid(p,v,f,grid=GRID):
 ranks=np.searchsorted(grid,p[f&~v],side='right'); hist=np.bincount(ranks,minlength=len(grid)+1); tail=np.cumsum(hist[::-1])[::-1]; return tail[1:len(grid)+1].astype(np.int64)
def flat(z):
 o={'threshold':z['threshold']}; o.update({k:int(z['counts'][k]) for k in ('tp','fp','fn','fov_pixels','vessel_pixels')})
 for k in ('precision','fp_density','far_recall','near_thin_recall_d1'):
  o[k]=z[k]['value']; o[k+'_num']=z[k]['numerator']; o[k+'_den']=z[k]['denominator']; o[k+'_NA_reason']=z[k]['NA_reason']
 return o
def direction(t,z,lower=False):
 if t is None or z is None:return 'NA'
 if t==z:return 'unchanged'
 return 'improved' if (t<z if lower else t>z) else 'declined'
def load_inputs():
 mp=B/'dev_manifest.json'; manifest=json.loads(mp.read_text(encoding='utf-8')); drows={r['filename']:r for r in csv.DictReader((DATA/'data_manifest.csv').open(encoding='utf-8-sig'))}
 if len(manifest['images'])!=28: raise ValueError('Expected frozen dev28 manifest')
 paths=[B/x for x in ('binding.json','dev_manifest.json','evaluation_protocol_v1.json','protection_rules.json','primary_denominators.json','selection.json','completion_verification_v2.json','train_config.json','z0/summary.json','z0/metrics/results.json','z0/matched_FP.json','validation_002000/summary.json','validation_002000/metrics/results.json','validation_002000/matched_FP.json','matched_FP_comparison.csv')]
 paths += [ROOT/'code/retinal-vessel-study/retinal_m0'/x for x in ('maples_evaluate.py','maples_protocol.py','maples_calibrate.py')]
 for r in manifest['images']:
  n=r['filename']; d=drows[n]
  if d['trial_split']!='dev' or d['source_split']!='train' or [int(d['height']),int(d['width'])]!=r['native_shape']: raise ValueError('Split/shape changed: '+n)
  img=DATA/d['image_path']
  if sha(img)!=d['image_sha256']: raise ValueError('RGB hash changed: '+n)
  for k in ('vessel_path','fov_path'):
   if sha(r[k])!=r[k+'_sha256']: raise ValueError(k+' hash changed: '+n)
  if sha(r['region_bundle'])!=r['region_bundle_sha256']: raise ValueError('Reference bundle hash changed: '+n)
  for k in ('H_path','U_bright_path','U_red_path'):
   if sha(r[k])!=r['regional_source_hashes'][k]: raise ValueError(k+' hash changed: '+n)
  for v in ('z0','validation_002000'):
   p=B/v/'probabilities'/(n+'.npy')
   if not p.is_file(): raise FileNotFoundError(p)
   paths.append(p)
  paths += [Path(r['region_bundle']),img]+[Path(r[k]) for k in ('vessel_path','fov_path','H_path','U_bright_path','U_red_path')]
 return manifest,drows,{str(p):sha(p) for p in dict.fromkeys(paths)}
def load_record(r,d):
 n=r['filename']
 with Image.open(DATA/d['image_path']) as im: rgb=np.asarray(im.convert('RGB'))
 v=load_mask(r['vessel_path']); f=load_mask(r['fov_path']); h=load_mask(r['H_path'])
 with np.load(r['region_bundle'],allow_pickle=False) as z: b={k:z[k] for k in z.files}
 z0=np.load(B/'z0/probabilities'/(n+'.npy'),allow_pickle=False); t0=np.load(B/'validation_002000/probabilities'/(n+'.npy'),allow_pickle=False)
 if rgb.shape[:2]!=tuple(r['native_shape']) or any(x.shape!=tuple(r['native_shape']) for x in (v,f,h,z0,t0)): raise ValueError('Native arrays misaligned: '+n)
 if not np.array_equal(b['internal']|b['near']|b['far'],f&~b['U']) or np.any(b['thin']&~v) or np.any(b['thin']&~f): raise ValueError('Frozen region/target mismatch: '+n)
 return {'name':n,'rgb':rgb,'v':v,'f':f,'h':h,'b':b,'z0':z0,'t0':t0}
def evaluate(rec,p,t):
 b=rec['b']; return evaluate_arrays(p,rec['v'],rec['f'],threshold=float(t),u=b['U'],regions={k:b[k] for k in ('internal','near','far')},thin=b['thin'],matching_support='omega')
def write_csv(path,rows):
 if not rows: raise ValueError('Empty CSV: '+str(path))
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def evaluate_all(records,same_t):
 rows=[]; ev={}
 for r in records:
  n=r['name']; ev[n]={}
  for wp,zt,tt in (('fixed_0p5',.5,.5),('original_same_fp',.5,same_t)):
   ev[n][wp]={'z0':evaluate(r,r['z0'],zt),'t0':evaluate(r,r['t0'],tt)}
  out={'filename':n}
  for wp in ('fixed_0p5','original_same_fp'):
   z,t=flat(ev[n][wp]['z0']),flat(ev[n][wp]['t0'])
   for model,s in (('z0',z),('t0',t)):
    for k,val in s.items():out[f'{wp}_{model}_{k}']=val
   out[wp+'_fp_direction']=direction(t['fp'],z['fp'],True)
   zd,td=z['near_thin_recall_d1_den'],t['near_thin_recall_d1_den']
   if zd!=td:raise ValueError('Primary denominator mismatch: '+n)
   out[wp+'_primary_direction']=direction(t['near_thin_recall_d1_num'],z['near_thin_recall_d1_num']) if zd else 'NA'
   out[wp+'_primary_delta']=(t['near_thin_recall_d1']-z['near_thin_recall_d1']) if zd else None
   out[wp+'_primary_macro_contribution']=out[wp+'_primary_delta']/9 if zd else None
  rows.append(out)
 return rows,ev
def point_audit(records,ev):
 rows=[]; coords_by={}
 for r in records:
  b=r['b']; coords=np.argwhere(b['thin']&b['near'])
  if not len(coords):continue
  maps=targets(r['v'],r['f']); coords_by[r['name']]=coords; maps4={}
  for wp in ('fixed_0p5','original_same_fp'):
   for model in ('z0','t0'):
    p=r[model]; t=ev[r['name']][wp][model]['threshold']; maps4[wp,model]=(*lookup(p,r['f'],b['U'],t),p,t)
  for ix,(rr,cc) in enumerate(coords,1):
   rr,cc=int(rr),int(cc); diameter=float(2*maps['skeleton_radius'][rr,cc])
   if not maps['skeleton'][rr,cc] or diameter>3+1e-8:raise ValueError('Target width/center mismatch '+r['name'])
   o={'filename':r['name'],'target_id':ix,'row_zero_based':rr,'col_zero_based':cc,'V':int(r['v'][rr,cc]),'FOV':int(r['f'][rr,cc]),'H':int(r['h'][rr,cc]),'U':int(b['U'][rr,cc]),'near':int(b['near'][rr,cc]),'thin_target':int(b['thin'][rr,cc]),'diameter_proxy_native_px':diameter,'diameter_cap_native_px':3,'matching_support':'omega=FOV_and_not_U','d1_tolerance_native_px':1,'z0_probability':float(r['z0'][rr,cc]),'t0_probability':float(r['t0'][rr,cc])}
   for wp in ('fixed_0p5','original_same_fp'):
    for model in ('z0','t0'):
     dm,ind,p,t=maps4[wp,model]; nr,nc,dist,hit=hit_at(dm,ind,(rr,cc)); pre=f'{wp}_{model}_'
     o[pre+'threshold']=float(t); o[pre+'probability_at_target']=float(p[rr,cc]); o[pre+'nearest_row']=nr; o[pre+'nearest_col']=nc; o[pre+'nearest_distance']=dist; o[pre+'hit_d1']=int(hit)
   rows.append(o)
 return rows,coords_by
def influence(per_image,wp):
 eligible=[r for r in per_image if r[wp+'_z0_near_thin_recall_d1_den']>0]
 if len(eligible)!=9:raise ValueError('Expected 9 primary images')
 order=sorted(eligible,key=lambda r:(-r[wp+'_primary_macro_contribution'],r['filename'])); named=['20051205_35276_0400_PP','20060411_61214_0200_PP']
 scenarios=[('all_9',[]),('drop_designated_max',[named[0]]),('drop_designated_pair',named)]; ans={}
 for key,dropped in scenarios:
  items=[r for r in eligible if r['filename'] not in dropped]; z=[r[wp+'_z0_near_thin_recall_d1'] for r in items]; t=[r[wp+'_t0_near_thin_recall_d1'] for r in items]; dif=[b-a for a,b in zip(z,t)]; dirs=[r[wp+'_primary_direction'] for r in items]
  ans[key]={'dropped_images':dropped,'n_images':len(items),'total_target_positions':sum(r[wp+'_z0_near_thin_recall_d1_den'] for r in items),'z0_macro_recall':macro(z),'t0_macro_recall':macro(t),'paired_macro_increment':macro(dif),'improved_images':[r['filename'] for r in items if r[wp+'_primary_direction']=='improved'],'declined_images':[r['filename'] for r in items if r[wp+'_primary_direction']=='declined'],'unchanged_images':[r['filename'] for r in items if r[wp+'_primary_direction']=='unchanged'],'directions_count':{x:dirs.count(x) for x in ('improved','declined','unchanged')},'_diffs':dif}
 return ans,[r['filename'] for r in order]
def blend_errors(rec,p,threshold):
 rgb=rec['rgb'].astype(np.float32)/255; v,f=rec['v'],rec['f']; pred=(p>=threshold)&f; canvas=rgb*.38
 for mask,color in ((pred&v,(0,.85,.32)),(pred&~v&f,(1,.28,.02)),(~pred&v&f,(.12,.42,1))):canvas[mask]=color
 return canvas
def create_figures(out,records,coords_by,per,loo):
 fd=out/'figures';fd.mkdir(); colors={'TP':'#00d954','FP':'#ff4705','FN':'#1f6bff'}
 for r in records:
  coords=coords_by.get(r['name'])
  if coords is None:continue
  fig,ax=plt.subplots(1,4,figsize=(19,6),constrained_layout=True); ax[0].imshow(r['rgb'],interpolation='nearest')
  for i,(y,x) in enumerate(coords,1):ax[0].text(x,y,str(i),color='white',fontsize=7,bbox={'boxstyle':'circle,pad=.12','facecolor':'#185adb','edgecolor':'white','linewidth':.5})
  ax[0].set_title('Native RGB + target IDs'); ax[0].axis('off'); b=r['b']; c=r['rgb'].astype(np.float32)/255*.42
  c[b['near']]=.35*c[b['near']]+.65*np.array((.1,.35,1)); c[r['h']&r['f']]=(0.95,.15,.12); c[b['U']&r['f']]=(1,.78,0); c[r['v']&r['f']]=(.05,.9,.35); ax[1].imshow(c,interpolation='nearest'); ax[1].set_title('V green / H red / U yellow / near blue'); ax[1].axis('off')
  for a,model,title in ((ax[2],'z0','Z0 @ 0.5'),(ax[3],'t0','T0 step 2000 @ 0.5')):
   a.imshow(blend_errors(r,r[model],.5),interpolation='nearest')
   for i,(y,x) in enumerate(coords,1):a.text(x,y,str(i),color='white',fontsize=7,bbox={'boxstyle':'circle,pad=.12','facecolor':'#185adb','edgecolor':'white','linewidth':.5})
   a.set_title(title+'; TP green, FP orange, FN blue'); a.axis('off')
  fig.suptitle(r['name']+' — native pixel coordinates'); fig.savefig(fd/(r['name']+'_overview.png'),dpi=130); plt.close(fig)
 focus={'20051205_35276_0400_PP','20060411_61214_0200_PP','20060530_54390_0100_PP'}
 for r in records:
  coords=coords_by.get(r['name'])
  if r['name'] not in focus or coords is None:continue
  fig,ax=plt.subplots(len(coords),2,figsize=(9,max(3,len(coords)*2.5)),squeeze=False,constrained_layout=True)
  for i,(y,x) in enumerate(coords,1):
   y,x=int(y),int(x); y0,y1=max(0,y-32),min(r['rgb'].shape[0],y+33); x0,x1=max(0,x-32),min(r['rgb'].shape[1],x+33); crop=r['rgb'][y0:y1,x0:x1]
   ax[i-1,0].imshow(crop,interpolation='nearest'); ax[i-1,0].scatter([x-x0],[y-y0],s=70,facecolors='none',edgecolors='#ffea00'); ax[i-1,0].set_title(f'Target {i}: row={y}, col={x} native px'); ax[i-1,0].axis('off')
   canvas=blend_errors(r,r['t0'],.5)[y0:y1,x0:x1]; ax[i-1,1].imshow(canvas,interpolation='nearest'); ax[i-1,1].scatter([x-x0],[y-y0],s=70,facecolors='none',edgecolors='#ffea00'); ax[i-1,1].set_title('T0 @ 0.5; TP green, FP orange, FN blue'); ax[i-1,1].axis('off')
  fig.suptitle(r['name']+' — clipped 65×65 native pixel crops'); fig.savefig(fd/(r['name']+'_local_targets.png'),dpi=145); plt.close(fig)
 elig=[r for r in per if r['fixed_0p5_z0_near_thin_recall_d1_den']>0]; names=[r['filename'].replace('_0400_PP','').replace('_0200_PP','').replace('_0100_PP','') for r in elig]; x=np.arange(len(elig))
 fig,axs=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
 for a,wp,title in zip(axs,('fixed_0p5','original_same_fp'),('Fixed 0.5','Original whole-dev same-FP')):
  ds=[r[wp+'_primary_delta'] for r in elig]; a.bar(x,ds,color=['#168c55' if d>=0 else '#c34343' for d in ds]); a.axhline(0,color='black',lw=.8); a.set_xticks(x,names,rotation=75,ha='right',fontsize=7); a.set_ylabel('T0 - Z0 d1 recall'); a.set_title(title)
 fig.savefig(fd/'primary_per_image_delta.png',dpi=145);plt.close(fig)
 th=[r['chosen_threshold'] for r in loo if r['chosen_threshold'] is not None];fig,a=plt.subplots(figsize=(8,4));a.hist(th,bins=np.arange(0,1.051,.05),color='#386cb0',edgecolor='white');a.set_xlim(0,1);a.set_xlabel('Chosen T0 threshold');a.set_ylabel('Held-out folds');a.set_title(f'LOO threshold distribution ({len(th)}/28 feasible)');fig.tight_layout();fig.savefig(fd/'loo_threshold_distribution.png',dpi=145);plt.close(fig)
def write_report(out,per,loo,inf,points,threshold):
 elig=[r for r in per if r['fixed_0p5_z0_near_thin_recall_d1_den']>0]; delta05=macro([r['fixed_0p5_primary_delta'] for r in elig]); deltam=macro([r['original_same_fp_primary_delta'] for r in elig]); chosen=[r['chosen_threshold'] for r in loo if r['chosen_threshold'] is not None]; over=[r['filename'] for r in loo if r['heldout_over_fit_target']==1]
 pc={w:{k:sum(r[w+'_primary_direction']==k for r in elig) for k in ('improved','declined','unchanged')} for w in ('fixed_0p5','original_same_fp')}; fc={w:{k:sum(r[w+'_fp_direction']==k for r in per) for k in ('improved','declined','unchanged')} for w in ('fixed_0p5','original_same_fp')}
 rows=''.join('<tr><td>{}</td><td>{}</td><td>{}/{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}/{}</td></tr>'.format(html.escape(r['filename']),r['fixed_0p5_z0_fp'],r['fixed_0p5_t0_fp'],r['fixed_0p5_z0_fov_pixels'],r['fixed_0p5_primary_direction'],r['fixed_0p5_primary_delta'] if r['fixed_0p5_primary_delta'] is not None else 'NA',r['original_same_fp_primary_direction'],r['fixed_0p5_z0_near_thin_recall_d1_num'],r['fixed_0p5_z0_near_thin_recall_d1_den']) for r in per)
 gallery=''.join(f'<figure><a href="figures/{r["filename"]}_overview.png"><img src="figures/{r["filename"]}_overview.png" loading="lazy"></a><figcaption>{html.escape(r["filename"])}</figcaption></figure>' for r in elig)
 gallery+=''.join(f'<figure><a href="figures/{n}_local_targets.png"><img src="figures/{n}_local_targets.png" loading="lazy"></a><figcaption>{n} — target crops at native scale</figcaption></figure>' for n in ('20051205_35276_0400_PP','20060411_61214_0200_PP','20060530_54390_0100_PP'))
 text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A15.2 可靠性检查</title><style>body{{font:16px/1.55 system-ui,"Microsoft YaHei",sans-serif;max-width:1280px;margin:32px auto;padding:0 22px;color:#17202b;background:#f5f7fa}}h1,h2{{line-height:1.2}}section{{background:white;padding:20px 24px;margin:16px 0;border-radius:12px;box-shadow:0 1px 5px #ccd3dc}}.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.kpi{{padding:14px;background:#f0f4f8;border-radius:8px}}.kpi b{{display:block;font-size:1.2em}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #dce2e8;padding:6px;text-align:left}}.scroll{{overflow:auto;max-height:620px}}figure{{margin:8px 0;background:white;padding:8px;border:1px solid #ddd;flex:1 1 390px}}figure img{{width:100%;height:auto}}.gallery{{display:flex;flex-wrap:wrap;gap:12px}}</style><body><h1>A15.2 可靠性检查</h1><p>仅使用已有 Z0 与 T0 第 2000 步概率开展 MAPLES 开发集回顾性检查；未重新推理、训练、选模或改变评价规则。</p><section><h2>结果概览</h2><div class="kpis"><div class="kpi">固定 0.5，9 图主指标差值<b>{delta05:+.4f}</b></div><div class="kpi">全开发集同误报（T0 阈值 {threshold:.3f}）<b>{deltam:+.4f}</b></div><div class="kpi">主指标位置<b>{len(points)} / 77</b></div><div class="kpi">留一图可行折<b>{len(chosen)} / 28</b></div></div><p>固定 0.5 主指标改善/下降/不变：{pc['fixed_0p5']}；误报改善/增加/不变：{fc['fixed_0p5']}。同误报工作点主指标：{pc['original_same_fp']}；误报：{fc['original_same_fp']}。</p><p>留一图阈值 {min(chosen):.3f}–{max(chosen):.3f}；被留图误报高于拟合目标 {len(over)}/28。折间拟合集重叠，结果仍来自开发集，不能视作独立测试。</p></section><section><h2>高贡献图影响和抽样区间</h2><p>按原同误报逐图贡献顺序暂扣第一图或前两图，原阈值保持固定；另显示下降图 20060530_54390_0100_PP。详情见 <a href="influence_summary.json">影响汇总</a> 和 <a href="bootstrap_draws.csv">1000 次抽样明细</a>。</p></section><section><h2>留一图阈值</h2><p>每折仅用另 27 图拟合 Z0@0.5 的宏平均误报密度目标，在原 0.001 网格选择不超过目标的最低 T0 阈值；被留图只在选阈后评价。</p><p><a href="loo_thresholds.csv">逐折结果</a> · <a href="loo_threshold_grid.csv">完整网格计算</a> · <a href="figures/loo_threshold_distribution.png">阈值分布图</a></p></section><section><h2>逐图核对</h2><p>CSV 覆盖全部 28 图的 FP 计数/FOV、精确率、远处召回和主指标分子/分母/NA；19 张主指标 NA 的图仍保留在全图误报统计。</p><div class="scroll"><table><thead><tr><th>图像</th><th>Z0 FP@0.5</th><th>T0 FP / FOV</th><th>@0.5 主方向</th><th>差值</th><th>同误报方向</th><th>Z0 主分子/分母</th></tr></thead><tbody>{rows}</tbody></table></div><p><a href="per_image.csv">逐图统计</a> · <a href="point_audit.csv">77 个位置</a></p></section><section><h2>原生像素位置图</h2><p>逐点记录含零基坐标、V/FOV/H/U/邻域、宽度代理、目标点概率、最近匹配位置与 d1 命中状态。概览和裁切图只用于查看，不用于测量。</p><div class="gallery">{gallery}</div></section><section><h2>验证和局限</h2><p><a href="verification.json">验收记录</a> · <a href="source_hashes.json">输入前后摘要</a>。单次训练轨迹、9 张有效主指标图、患者独立性未知；留一图结果相互相关。未新增必须改善或显著的门槛。</p></section></body></html>'''
 (out/'report.html').write_text(text,encoding='utf-8')
def run(output):
 out=Path(output)
 if out.exists():raise FileExistsError('Choose a fresh run directory: '+str(out))
 manifest,drows,hashes_before=load_inputs();out.mkdir(parents=True);write_json(out/'source_hashes.json',{'before':hashes_before})
 protocol=json.loads((B/'evaluation_protocol_v1.json').read_text(encoding='utf-8')); zs=json.loads((B/'z0/summary.json').read_text(encoding='utf-8')); ts=json.loads((B/'validation_002000/summary.json').read_text(encoding='utf-8')); zm=json.loads((B/'z0/matched_FP.json').read_text(encoding='utf-8')); tm=json.loads((B/'validation_002000/matched_FP.json').read_text(encoding='utf-8'))
 if protocol.get('status')!='APPROVED_BY_USER_A15_2' or protocol.get('primary_diameter_max')!=3:raise ValueError('Protocol identity/status differs')
 if ts['metrics']['step']!=2000 or zm.get('selection_use') is not False or tm.get('selection_use') is not False:raise ValueError('Input prediction/workpoint identity differs')
 same_t=float(tm['threshold']); config={'analysis_kind':'retrospective_reliability_not_formal_protocol','scope':'approved A15.2 dev28 reliability analysis','objects':{'baseline':'Z0 original M0','endpoint':'T0 saved step 2000; analysis endpoint only, not selected/formal model'},'workpoints':{'fixed':{'Z0':.5,'T0':.5},'original_whole_dev_same_fp':{'Z0':.5,'T0':same_t}},'protocol_sha256':sha(B/'evaluation_protocol_v1.json'),'protection_rules_sha256':sha(B/'protection_rules.json'),'regional_definition':json.loads((B/'dev_manifest.json').read_text(encoding='utf-8'))['regional_definition'],'loo':{'folds':28,'fit_images_per_fold':27,'grid':'float32 linspace 0:0.001:1 (1001 points)','choice':'minimum T0 threshold with fitting-set macro FP/FOV <= fitting-set Z0@0.5 macro FP/FOV; same 1D macro reduction'},'bootstrap':{'unit':'paired image resampling','reps':1000,'seed':20261016,'interval':'empirical percentile linear quantile, 2.5%-97.5%','new_success_gate':False},'forbidden_operations_performed':[]}
 write_json(out/'analysis_config.json',config); records=[load_record(r,drows[r['filename']]) for r in manifest['images']]; per,ev=evaluate_all(records,same_t); points,coords=point_audit(records,ev)
 denrows=json.loads((B/'primary_denominators.json').read_text(encoding='utf-8')); expected={x['filename']:int(x['denominator']) for x in denrows}; actual={r['name']:int((r['b']['thin']&r['b']['near']).sum()) for r in records if (r['b']['thin']&r['b']['near']).any()}
 if expected!=actual or sum(actual.values())!=77 or len(actual)!=9 or len(points)!=77:raise ValueError('Primary target set differs from frozen 9/77 record')
 # Reconcile recomputation to the source aggregates before interpreting effects.
 old={'fixed_0p5':(json.loads((B/'z0/metrics/results.json').read_text()),json.loads((B/'validation_002000/metrics/results.json').read_text())),'original_same_fp':(zm,tm)}
 for wp,(oz,ot) in old.items():
  for model,report in (('z0',oz),('t0',ot)):
   for metric in ('precision','fp_density','far_recall','near_thin_recall_d1'):
    vals=[r[f'{wp}_{model}_{metric}'] for r in per]; val=macro(v for v in vals if v is not None); prior=report['aggregate'][metric]['mean']
    if prior is None or val is None or not math.isclose(float(prior),float(val),rel_tol=0,abs_tol=1e-12):raise ValueError(f'Aggregate mismatch {wp}/{model}/{metric}: {val} != {prior}')
 write_csv(out/'per_image.csv',per);write_csv(out/'point_audit.csv',points)
 fpcounts=np.stack([probability_fp_grid(r['t0'],r['v'],r['f']) for r in records]); zfp=np.asarray([ev[r['name']]['fixed_0p5']['z0']['fp_density']['value'] for r in records]); fov=np.asarray([int(r['f'].sum()) for r in records],dtype=np.int64); loo=[]; grid_rows=[]
 for held in range(28):
  fit=[i for i in range(28) if i!=held]; choice=choose_threshold_from_counts(fpcounts,fov,zfp,fit,GRID); dens=fpcounts[fit]/fov[fit,None]; mg=np.asarray([float(np.mean(dens[:,j].tolist())) for j in range(1001)])
  for j,g in enumerate(GRID):grid_rows.append({'heldout_filename':records[held]['name'],'grid_index':j,'threshold':float(g),'fit_target_fp_density':choice['target_fp_density'],'fit_t0_fp_density':float(mg[j]),'feasible':int(mg[j]<=choice['target_fp_density'])})
  z=flat(ev[records[held]['name']]['fixed_0p5']['z0']); t=flat(evaluate(records[held],records[held]['t0'],choice['threshold'])) if choice['threshold'] is not None else {}; fp=t.get('fp'); fd=t.get('fov_pixels')
  loo.append({'filename':records[held]['name'],'fit_n':27,'fit_target_fp_density':choice['target_fp_density'],'chosen_threshold':choice['threshold'],'fit_actual_fp_density':choice['actual_fp_density'],'fit_gap':choice['actual_fp_density']-choice['target_fp_density'] if choice['actual_fp_density'] is not None else None,'previous_grid_threshold':choice['previous_threshold'],'previous_grid_fp_density':choice['previous_fp_density'],'heldout_fp_count':fp,'heldout_fov_denominator':fd,'heldout_fp_density':fp/fd if fd else None,'heldout_over_fit_target':int(fp/fd>choice['target_fp_density']) if fd else None,'z0_fixed_threshold':.5,'z0_fp_count':z['fp'],'z0_fov_denominator':z['fov_pixels'],'z0_precision':z['precision'],'z0_far_recall':z['far_recall'],'z0_primary_numerator':z['near_thin_recall_d1_num'],'z0_primary_denominator':z['near_thin_recall_d1_den'],'t0_precision':t.get('precision'),'t0_far_recall':t.get('far_recall'),'t0_primary_numerator':t.get('near_thin_recall_d1_num'),'t0_primary_denominator':t.get('near_thin_recall_d1_den'),'t0_primary_recall':t.get('near_thin_recall_d1'),'t0_primary_NA_reason':t.get('near_thin_recall_d1_NA_reason')})
 write_csv(out/'loo_thresholds.csv',loo);write_csv(out/'loo_threshold_grid.csv',grid_rows)
 inf={'workpoint_definition':config['workpoints'],'top_contribution_order':{},'fixed_0p5':{},'original_same_fp':{},'loo':{}}; draws=[]; rng=np.random.default_rng(20261016)
 for wp in ('fixed_0p5','original_same_fp'):
  cases,order=influence(per,wp)
  if wp=='original_same_fp' and order[:2]!=['20051205_35276_0400_PP','20060411_61214_0200_PP']:raise ValueError('Original contribution ranking differs from named cases')
  inf['top_contribution_order'][wp]=order;inf[wp]=cases
  for case,detail in cases.items():
   boot=paired_bootstrap(detail['_diffs'],rng=rng); means=boot.pop('draw_means');detail['bootstrap']=boot
   draws += [{'workpoint':wp,'scenario':case,'draw':i+1,'paired_macro_increment':float(v)} for i,v in enumerate(means)]
   del detail['_diffs']
 valid=[r for r in loo if r['t0_primary_recall'] is not None]; chosen=[r['chosen_threshold'] for r in loo if r['chosen_threshold'] is not None]
 inf['loo']={'feasible_folds':len(chosen),'threshold_min':min(chosen) if chosen else None,'threshold_max':max(chosen) if chosen else None,'threshold_values':[r['chosen_threshold'] for r in loo],'heldout_fp_over_target_images':[r['filename'] for r in loo if r['heldout_over_fit_target']==1],'primary_valid_n':len(valid),'primary_NA_n':28-len(valid),'primary_paired_macro_increment':macro(r['t0_primary_recall']-r['z0_primary_numerator']/r['z0_primary_denominator'] for r in valid),'limitation':'Overlapping fitting folds and development-set reuse'}
 write_json(out/'influence_summary.json',inf);write_csv(out/'bootstrap_draws.csv',draws);create_figures(out,records,coords,per,loo)
 for wp in ('fixed_0p5','original_same_fp'):
  eligible=[r for r in per if r[wp+'_z0_near_thin_recall_d1_den']>0]
  if len(eligible)!=9 or sum(r[wp+'_z0_near_thin_recall_d1_den'] for r in eligible)!=77:raise ValueError('Primary denominator changed: '+wp)
 after={p:sha(p) for p in hashes_before}; changed=[p for p in hashes_before if hashes_before[p]!=after[p]]
 if changed:raise ValueError('Protected source changed: '+repr(changed))
 write_json(out/'source_hashes.json',{'before':hashes_before,'after':after,'changed':changed})
 checks={'status':'passed','protocol_status':protocol.get('status'),'primary_diameter_max':protocol.get('primary_diameter_max'),'dev_images':28,'primary_images':len(actual),'primary_positions':len(points),'primary_denominators_match_frozen_record':True,'recomputed_aggregates_match_abs_tolerance_1e-12':True,'loo_folds':len(loo),'loo_grid_rows':len(grid_rows),'heldout_excluded_from_loo_fit':True,'zero_denominator_preserved_as_NA':all(r['fixed_0p5_z0_near_thin_recall_d1_den']==0 and r['fixed_0p5_z0_near_thin_recall_d1'] is None for r in per if r['filename'] not in actual),'bootstrap_reps_per_case':1000,'bootstrap_seed':20261016,'bootstrap_cases':len(draws)//1000,'protected_input_hashes_unchanged':not changed,'official_test_read':False,'new_prediction_or_training':False,'limitations':['single T0 trajectory','primary n=9 and 77 target positions','19 images primary NA','patient independence unknown','LOO folds overlap']}
 if not checks['zero_denominator_preserved_as_NA'] or checks['bootstrap_cases']!=6:raise ValueError('Verification failed')
 write_json(out/'verification.json',checks);write_report(out,per,loo,inf,points,same_t);print(json.dumps(checks,ensure_ascii=False,indent=2));return checks
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
if __name__=='__main__':main()
