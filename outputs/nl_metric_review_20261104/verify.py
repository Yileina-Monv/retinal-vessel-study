"""Recompute exported numeric tables only; no image/model/data access."""
from pathlib import Path
import csv,json,hashlib,statistics,collections,math
import numpy as np
R=Path(__file__).resolve().parent
checks=0
def eq(a,b):
 global checks
 assert abs(float(a)-float(b))<1e-11,(a,b)
 checks+=1
def rows(n):
 with (R/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
manifest=json.loads((R/'file_manifest.json').read_text())
for name,h in manifest.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
ims=rows('historical_near_path_20261103_per_image.csv');ps=rows('historical_near_path_20261103_per_path.csv')
assert len(ims)==280 and len(ps)==1740
lookup={(r['filename'],int(r['seed']),r['method']):r for r in ims};assert len(lookup)==280
group=collections.defaultdict(list)
for r in ps:
 for d in [0,1]:eq(int(r[f'near_hits_d{d}'])/int(r['near_points']),r[f'near_d{d}'])
 group[r['filename'],int(r['seed']),r['method']].append(r)
for key,rs in group.items():
 for d in [0,1]:eq(statistics.mean(float(r[f'near_d{d}']) for r in rs),lookup[key][f'near_path_d{d}'])
for table in rows('historical_near_path_20261103_common_seeds_table.csv'):
 for k in ['near_path_d1','near_path_d0','near_edge_coverage','near_whole_missed_fraction','near_thin_recall_d1','dice','precision','recall','fp_density','local_fp_area','local_fp_background']:
  vs=[float(r[k]) for r in ims if r['method']==table['method'] and r['seed'] in ['2027','2028'] and r[k]!='']
  eq(statistics.mean(vs),table[k])
for e in rows('historical_near_path_20261103_effects.csv'):
 seeds=([2026,2027,2028] if e['comparison']=='TW-T0' else [2027,2028]) if e['seed']=='within_image_seed_average' else [int(e['seed'])]
 ids=sorted({r['filename'] for r in ims if r[e['metric']]!=''})
 bv=[statistics.mean(float(lookup[n,s,e['baseline']][e['metric']]) for s in seeds) for n in ids]
 cv=[statistics.mean(float(lookup[n,s,e['candidate']][e['metric']]) for s in seeds) for n in ids]
 delta=np.asarray(cv)-bv;eq(statistics.mean(bv),e['baseline_mean']);eq(statistics.mean(cv),e['candidate_mean']);eq(delta.mean(),e['delta'])
 assert len(ids)==int(e['n_images'])
 ix=np.random.default_rng(20261016).integers(0,len(ids),size=(1000,len(ids)))
 ci=np.quantile(delta[ix].mean(axis=1),[.025,.975]);eq(ci[0],e['CI_low']);eq(ci[1],e['CI_high'])
den=rows('metric_review_20261103_denominators.csv')
for w in rows('metric_review_20261103_width_summary.csv'):
 rs=[r for r in den if r['width_native_proxy']==w['width']];assert len(rs)==28
 eq(sum(int(r['near_width_points'])>0 for r in rs),w['valid_images'])
 eq(sum(int(r['near_width_points'])==0 for r in rs),w['NA_images'])
 eq(sum(int(r['near_width_points']) for r in rs),w['targets'])
 for r in rs:assert (r['NA']=='True')==(int(r['near_width_points'])==0)
nl=json.loads((R/'nl_fixed_endpoint.json').read_text())
assert nl['status']=='COMPLETE' and not nl['primary_signal'] and not nl['independent_validation']
for seed,rs in nl['per_image'].items():
 assert len(rs)==28
 for method in ['N0','NL']:
  for k,stored in nl['means'][seed][method].items():
   vs=[r[method][k] for r in rs.values() if r[method].get(k) is not None]
   if vs:eq(statistics.mean(vs),stored)
 for k,e in nl['per_seed'][seed].items():
  dv=[r['NL'][k]-r['N0'][k] for r in rs.values() if r['NL'].get(k) is not None and r['N0'].get(k) is not None]
  if dv:eq(statistics.mean(dv),e['estimate']);eq(len(dv),e['n_images'])
for k,e in nl['within_image_seed_average'].items():
 ds=[]
 for n in sorted(nl['per_image']['2027']):
  vals=[]
  for s in ['2027','2028']:
   r=nl['per_image'][s][n]
   if r['N0'].get(k) is not None and r['NL'].get(k) is not None:vals.append(r['NL'][k]-r['N0'][k])
  if vals:assert len(vals)==2;ds.append(statistics.mean(vals))
 if ds:eq(statistics.mean(ds),e['estimate']);eq(len(ds),e['n_images'])
fp={}
for r in ims:
 if r['method'] in ['N0','T0'] and r['seed'] in ['2027','2028'] and r['local_fp_area']!='':fp.setdefault(r['filename'],{}).setdefault(r['method'],[]).append(float(r['local_fp_area']))
ds={n:statistics.mean(v['N0'])-statistics.mean(v['T0']) for n,v in fp.items()}
case=json.loads((R/'fp_case_counts.json').read_text());assert max(ds,key=ds.get)==case['filename']
eq(statistics.mean(ds.values())*100,.43017229604896057)
eq(statistics.mean(v for n,v in ds.items() if n!=case['filename'])*100,.0660684490692493)
for method,total in [('T0',23815),('N0',22173)]:
 eq(sum(int(r['local_fp_pixels']) for r in ims if r['method']==method and r['seed'] in ['2027','2028'] and r['local_fp_pixels']!=''),total)
assert sum(d>0 for d in ds.values())==11 and sum(d<0 for d in ds.values())==7
assert case['near_area']==1284 and case['GT_pixels']==94
for r in case['counts']:assert r['TP']+r['FN']==94
print(json.dumps({'status':'PASS','numeric_comparisons':checks,'images':280,'paths':1740,'historical_bootstrap_effects':130,'NL_old_intervals':'reported from previously audited A7 output; not recomputed here','scope':'numeric tables only; no image/model access'},ensure_ascii=False))
