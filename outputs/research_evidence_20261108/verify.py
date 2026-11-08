"""Read-only verification of this public evidence package; no model or network."""
from pathlib import Path
import hashlib, json, re
from urllib.parse import unquote
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
checks=0
def check(ok,label):
 global checks
 if not bool(ok):raise AssertionError(label)
 checks+=1
def close(a,b,label,tol=1e-11):
 global checks
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 if not np.allclose(a,b,atol=tol,rtol=0,equal_nan=True):raise AssertionError(label)
 checks+=max(a.size,b.size)
def frame(p):return pd.read_csv(ROOT/p)
def data(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))

src=data('source_manifest.json')
for x in src['files']:
 p=ROOT/x['path'];check(hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],'import SHA '+x['path'])
check(len(src['exact_existing_files'])==10,'10 exact duplicates linked instead of copied')
for p in ROOT.rglob('*.json'):
 json.loads(p.read_text(encoding='utf-8'));checks+=1
for p in ROOT.rglob('*.png'):
 with Image.open(p) as im: im.verify()
 checks+=1

f=frame('t0_tw_formal/fixed_0p5_per_image.csv')
check(len(f)==196,'formal196 rows')
for metric in ['dice','precision','recall','fp_density','near_pixel_recall','near_thin_recall_d1','far_recall']:
 expected=f[metric+'_numerator']/f[metric+'_denominator'].replace(0,np.nan)
 close(f[metric],expected,'formal numerator denominator '+metric)
paired=frame('t0_tw_formal/paired_seed_macro.csv')
close(paired['TW_minus_T0'],paired['TW_macro']-paired['T0_macro'],'formal paired differences')
close(paired.query("metric=='near_thin_recall_d1'")['TW_minus_T0'].mean(),-.006001371742112484,'formal primary')

obs=frame('plan_b/analysis/per_point_observations.csv')
check(len(obs)==847 and obs.point_id.nunique()==77 and obs.condition_id.nunique()==11,'Plan B77x11')
check(obs.groupby('point_id').d1_hit.sum().eq(0).sum()==27,'27 persistent misses')
check(obs.groupby('point_id').d1_local_max_probability.max().lt(.1).sum()==24,'24 always below .1')
pairs=frame('plan_b/analysis/formal_paired_point_changes.csv')
check(len(pairs)==231,'231 paired observations')
check(((~pairs.T0_hit)&pairs.TW_hit).sum()==0,'0 recovered')
check((pairs.T0_hit&(~pairs.TW_hit)).sum()==2,'2 lost')
check((pairs.T0_hit&pairs.TW_hit).sum()==104,'104 both hit')
check(((~pairs.T0_hit)&(~pairs.TW_hit)).sum()==125,'125 both miss')
matrix=pd.read_csv(ROOT/'plan_b/all77_conditions.csv',index_col=0)
source_matrix=obs.pivot(index='point_id',columns='condition_id',values='d1_hit').astype(int).sort_index()
close(matrix,source_matrix,'public heatmap counts')

m=frame('b1_b2_fp/per_image_metrics.csv');check(len(m)==84,'84 endpoint rows')
for metric,exp in [('dice',2*m.TP/(2*m.TP+m.FP+m.FN)),('precision',m.TP/(m.TP+m.FP)),('recall',m.TP/(m.TP+m.FN)),('fp_density',m.FP/m.FOV_pixels)]:close(m[metric],exp,'endpoint '+metric)
s=frame('b1_b2_fp/per_image_sets.csv');w=s.pivot(index='filename',columns='set',values='area')
check(len(w)==28,'all28 FP cases')
for meth,total,removed in [('B1',57856,7257),('B2',52495,7450)]:
 close(w['FP_'+meth]-w.FP_A0,w['new_'+meth]-w['removed_'+meth],'FP balance '+meth)
 close(w['FP_'+meth],w['retained_'+meth]+w['new_'+meth],'FP decomposition '+meth)
 check(w['new_'+meth].sum()==total and w['removed_'+meth].sum()==removed,'pooled new and removed '+meth)
close(s.near1_pixels+s.far1_pixels,s.area,'distance split')

c=frame('cldice_geometry/cldice_per_image.csv');cs=data('cldice_geometry/cldice_summary.json')
close(c.topology_precision,c.pred_skeleton_in_GT/c.pred_skeleton_denominator,'topology precision')
close(c.topology_sensitivity,c.GT_skeleton_in_pred/c.GT_skeleton_denominator,'topology sensitivity')
close(c.clDice,2*c.topology_precision*c.topology_sensitivity/(c.topology_precision+c.topology_sensitivity),'hard clDice')
for meth,g in c.groupby('method'):
 for metric in ['clDice','topology_precision','topology_sensitivity','Dice']:
  close(g[metric].mean(),cs['absolute_image_macro'][meth][metric],'clDice macro '+meth+metric)

t=frame('threshold_diagnosis/threshold_curves_macro.csv');ts=data('threshold_diagnosis/diagnostic_summary.json')
for meth,g in t.groupby('method'):
 row=g.loc[np.abs(g.threshold-.5).idxmin()];check(abs(row.threshold-.5)<1e-7,'fixed threshold')
 for metric in ['dice','precision','recall','fp_density']:close(row[metric],ts['methods'][meth]['fixed_05'][metric],meth+' fixed .5 '+metric)
 close(g.dice.max(),ts['methods'][meth]['best_on_same_Dev']['dice'],meth+' diagnostic maximum')
check(frame('train_crop_coverage/images_110.csv').shape[0]==110,'Train110 retained')
check(frame('train_crop_coverage/images_110.csv').crops.sum()==8000,'8000 original crops')
for stage,pattern in [('train_crop_coverage','coverage_[0-9]*.png'),('spatial_candidate','comparison_[0-9]*.png')]:
 check(len(list((ROOT/stage).glob(pattern)))==110,stage+' all110 images')

n=frame('nl_terminal_audit/near_path_reanalysis_20261103_per_image.csv')
check(len(n)==112,'NL/N0 28x2x2')
for (meth,seed),g in n.groupby(['method','seed']):
 check(len(g)==28 and g.near_path_d1.notna().sum()==18,'18 valid /10 NA '+meth+str(seed))
 check(g.n_paths.sum()==174,'174 paths '+meth+str(seed))
 close(g.local_fp_pixels/g.near_area.replace(0,np.nan),g.local_fp_area,'local FP density')
allseed=frame('nl_terminal_audit/historical_near_path_20261103_all_seeds_table.csv')
for (meth,seed),g in n.groupby(['method','seed']):
 ref=allseed[(allseed.method==meth)&(allseed.seed==seed)].iloc[0]
 for metric in ['near_path_d1','dice','precision','recall','fp_density','local_fp_area']:close(g[metric].mean(),ref[metric],'NL macro '+meth+str(seed)+metric)
check(data('nl_terminal_audit/SOL_REVIEW.json')['scientific_success'] is False,'NL negative retained')

for stage,key in [('b1_b2_fp','images'),('nl_terminal_audit','cases')]:
 a=data(stage+'/public_atlas/manifest.json')
 check(a['RGB_read'] is False and a['Test_read'] is False and a['model_forwards']==0,stage+' CPU-only')
 for x in a[key]:check(hashlib.sha256((ROOT/stage/'public_atlas'/x['path']).read_bytes()).hexdigest()==x['sha256'],'public atlas SHA')
 if key=='images':check(len(a[key])==87 and a['dev_images']==28 and a['count_checks']==84,'B1 atlas completeness')
 else:check(len(a[key])==6 and a['metric_checks']==96,'NL six fixed cases')

# Validate actual local links; historical private source links are explicitly text.
for p in list(ROOT.rglob('*.md'))+list(ROOT.rglob('*.html'))+list((REPO/'docs/research/evidence_review_20261108').glob('*.md')):
 text=p.read_text(encoding='utf-8')
 links=re.findall(r'\]\(([^)]+)\)',text) if p.suffix=='.md' else re.findall(r'(?:href|src)=[\"\x27]([^\"\x27]+)',text)
 for link in links:
  link=unquote(link.strip('<>'))
  if re.match(r'^(https?://|mailto:|#)',link):continue
  check((p.parent/link.split('#')[0]).exists(),'broken link '+str(p.relative_to(REPO))+' -> '+link)
for p in ROOT.rglob('*'):
 if not p.is_file():continue
 check(p.suffix.lower() not in ['.pptx','.docx','.pdf','.npy','.npz','.pt','.pth','.ckpt','.zip'],'excluded artifact '+p.name)
 if p.suffix not in ['.png']:
  text=p.read_text(encoding='utf-8')
  check(not re.search(r'-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}',text),'secret scan '+p.name)
fm=ROOT/'file_manifest.json'
if fm.exists():
 for x in json.loads(fm.read_text(encoding='utf-8')):
  check(hashlib.sha256((REPO/x['path']).read_bytes()).hexdigest()==x['sha256'],'package SHA '+x['path'])
print(json.dumps({'status':'PASS','checks':checks,'new_training':0,'model_forwards':0,'Test_access':False,'historical_CIs_regenerated':False,'scope':'public files, arithmetic, complete denominators, figures and links'},ensure_ascii=False,indent=2))
