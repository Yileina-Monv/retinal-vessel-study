"""DRAFT Train-only CPU target-definition study. No model or training entry."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
from pathlib import Path
ROOT=Path('${RETINA_PROJECT_ROOT}')
CODE=ROOT/'code/retinal-vessel-study'
sys.path.insert(0,str(CODE))
import csv,json,hashlib,math,collections,time,html
import numpy as np
from scipy.ndimage import distance_transform_edt,label as cc_label
from scipy.spatial.distance import cdist
from PIL import Image
import matplotlib.pyplot as plt
from retinal_m0.maples_thin_skeleton_bce_cpu_gate import load_rgb_v_fov,s3_target
EXP=ROOT/'experiments/full_fov_thin_skeleton_bce_20261019'
OUT=EXP/'target_definition_review_v1/attempt_02'
DOC=ROOT/'docs/full_fov_thin_skeleton_bce_20261019'
OLD=EXP/'implementation_gate'
DRAFT='DRAFT_CPU_ONLY_NOT_FOR_TRAINING'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def js(name,obj):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise RuntimeError('Refuse overwrite: '+str(p))
 p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def table(name,rows):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise RuntimeError('Refuse overwrite: '+str(p))
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def distribution(a):
 a=np.asarray(a,dtype=float)
 return {'n':len(a),'min':float(a.min()) if len(a) else None,
 'q25':float(np.quantile(a,.25)) if len(a) else None,'median':float(np.median(a)) if len(a) else None,
 'q75':float(np.quantile(a,.75)) if len(a) else None,'max':float(a.max()) if len(a) else None,
 'zero_count':int((a==0).sum())}

def graph(sk):
 points=[tuple(map(int,q)) for q in np.argwhere(sk)];present=set(points);adj={}
 for y,x in points:
  ns=[]
  for dy in (-1,0,1):
   for dx in (-1,0,1):
    if not(dy or dx):continue
    q=(y+dy,x+dx)
    if q not in present:continue
    if dy and dx and ((y+dy,x) in present or (y,x+dx) in present):continue
    ns.append(q)
  adj[(y,x)]=sorted(ns)
 junctions={p for p in points if len(adj[p])>=3};jid={};nodes={}
 for p in points:
  if p not in junctions or p in jid:continue
  k=len(nodes)+1;stack=[p];jid[p]=k;group=[]
  while stack:
   q=stack.pop();group.append(q)
   for z in adj[q]:
    if z in junctions and z not in jid:jid[z]=k;stack.append(z)
  ports=sum(1 for q in group for z in adj[q] if z not in junctions)
  nodes[k]={'pixels':len(group),'ports':ports,'first':min(group)}
 def edge(a,b):return (a,b) if a<b else (b,a)
 internal={edge(p,q) for p in junctions for q in adj[p] if q in junctions}
 seen=set(internal);paths=[]
 for p in points:
  if len(adj[p])==2:continue
  if not adj[p]:paths.append({'points':[p],'closed':False});continue
  for q in adj[p]:
   e=edge(p,q)
   if e in seen:continue
   seen.add(e);path=[p,q];prev=p;cur=q
   while len(adj[cur])==2:
    nxt=next(z for z in adj[cur] if z!=prev);e=edge(cur,nxt)
    assert e not in seen
    seen.add(e);path.append(nxt);prev,cur=cur,nxt
   closed=path[-1]==path[0]
   if closed:path=path[:-1]
   closed=closed or (p in jid and cur in jid and jid[p]==jid[cur])
   paths.append({'points':path,'closed':closed})
 for p in points:
  for q in adj[p]:
   e=edge(p,q)
   if e in seen:continue
   seen.add(e);path=[p,q];prev=p;cur=q
   while cur!=p:
    nxt=[z for z in adj[cur] if z!=prev and edge(cur,z) not in seen]
    assert nxt
    z=nxt[0];seen.add(edge(cur,z));path.append(z);prev,cur=cur,z
   paths.append({'points':path[:-1],'closed':True})
 assert len(seen)==sum(map(len,adj.values()))//2
 covered=set(junctions)
 for p in paths:covered.update(p['points'])
 assert covered==present
 return adj,jid,nodes,paths,len(internal)

def graph_checks():
 def fixture(coords):
  a=np.zeros((25,25),bool)
  for p in coords:a[p]=1
  return graph(a)
 a=fixture([(12,x) for x in range(3,20)])
 assert len(a[3])==1 and len(a[3][0]['points'])==17
 a=fixture([(12,x) for x in range(3,20)]+[(y,12) for y in range(3,12)])
 assert len(a[3])==3 and list(a[2].values())[0]['ports']==3
 a=fixture([(12,x) for x in range(3,20)]+[(y,12) for y in range(3,20)])
 assert len(a[3])==4 and list(a[2].values())[0]['ports']==4
 a=fixture([(12,x) for x in range(3,9)]+[(12,x) for x in range(11,19)])
 assert len(a[3])==2
 a=fixture([(y,x) for y in range(5,15) for x in range(5,15) if y in(5,14) or x in(5,14)])
 assert len(a[3])==1 and a[3][0]['closed']
 a=fixture([(5,5)])
 assert len(a[3])==1 and len(a[3][0]['points'])==1
 return {'straight':True,'T_three_ports':True,'X_four_ports_not_anatomy':True,
 'gap_not_bridged':True,'loop_retained':True,'isolated_retained':True,
 'every_real_image_pixels_and_edges_accounted_for':True}

def characterize(row,sk,r,f,scale):
 adj,jid,nodes,paths,internal=graph(sk)
 cc,ncomp=cc_label(sk,structure=np.ones((3,3),np.uint8))
 boundary=distance_transform_edt(np.pad(f,1))[1:-1,1:-1]
 ends=sorted(p for p in adj if len(adj[p])==1);gap={}
 if ends:
  a=np.asarray(ends);d=cdist(a,a)*scale;labs=cc[a[:,0],a[:,1]]
  d[labs[:,None]==labs[None,:]]=np.inf
  for i,p in enumerate(ends):
   j=int(np.argmin(d[i]));gap[p]=float(d[i,j]) if np.isfinite(d[i,j]) else None
 out=[]
 for i,path in enumerate(paths):
  pp=path['points'];a=np.asarray(pp);yy,xx=a.T
  deg=[len(adj[pp[0]]),len(adj[pp[-1]])];jends=[jid.get(pp[0],0),jid.get(pp[-1],0)]
  if path['closed']:kind='loop_or_cycle'
  elif len(pp)==1:kind='isolated_pixel'
  elif deg==[1,1]:kind='endpoint_to_endpoint_fragment'
  elif 1 in deg:kind='endpoint_to_junction'
  else:kind='junction_to_junction'
  arc=np.r_[0.,np.cumsum(np.sqrt(np.sum(np.diff(a,axis=0)**2,axis=1)))]
  length=float(arc[-1])
  if path['closed'] and len(pp)>1 and pp[0] in adj[pp[-1]]:length+=math.dist(pp[0],pp[-1])
  widths=2*r[yy,xx].astype(float);body=np.array([p not in jid for p in pp])
  wb=widths[body]*scale;med=float(np.median(wb)) if body.any() else None
  cand=bool(not path['closed'] and len(pp)>1 and length*scale>=20 and med is not None and med<=6)
  n3=int((widths<=3).sum());n6=int((widths<=6).sum());nr=int((widths*scale<=6).sum())
  endat=[p for p in (pp[0],pp[-1]) if len(adj[p])==1]
  mindist=min((gap[p] for p in endat if gap[p] is not None),default=None)
  tip=False
  if n3 and endat and length>0:
   ds=[]
   if deg[0]==1:ds.append(arc)
   if deg[1]==1:ds.append(arc[-1]-arc)
   tip=bool(np.all(np.min(np.stack(ds),axis=0)[widths<=3]<=.1*length) and n3/len(pp)<=.2)
  rec={'filename':row['filename'],'shape':str(tuple(sk.shape)),'branch_id':i,'kind':kind,'n_points':len(pp),
  'length_native':length,'length_reference':length*scale,'start_y':pp[0][0],'start_x':pp[0][1],
  'end_y':pp[-1][0],'end_x':pp[-1][1],'start_degree':deg[0],'end_degree':deg[1],
  'start_junction_cluster':jends[0],'end_junction_cluster':jends[1],
  'width_native_min':float(widths.min()),'width_native_median':float(np.median(widths)),
  'width_native_max':float(widths.max()),'body_width_reference_median':med,
  'body_width_reference_q90':float(np.quantile(wb,.9)) if body.any() else None,
  'S3_points':n3,'S6_native_points':n6,'S6_reference_points':nr,
  'S3_fraction':n3/len(pp),'S6_native_fraction':n6/len(pp),'S6_reference_fraction':nr/len(pp),
  'candidate_B2':cand,'B2_fraction':1. if cand else 0.,'B1_reference_fraction':nr/len(pp) if cand else 0.,
  'S3_tip_only_10pct':tip,'boundary_contact_2refpx':bool(float(boundary[yy,xx].min())*scale<=2),
  'junction_complex_GT_only':any(nodes[k]['ports']!=3 or nodes[k]['pixels']>=8 for k in jends if k),
  'junction_four_or_more_ports':any(nodes[k]['ports']>=4 for k in jends if k),
  'nearest_other_component_endpoint_distance_reference':mindist,
  'possible_gap_10refpx':bool(mindist is not None and mindist<=10),
  'anatomical_or_visible_branch_confirmed':False,'human_review_status':'UNREVIEWED','candidate_status':DRAFT}
  # Strict target-edge coverage: isolated selected points cover zero path length.
  closing=math.dist(pp[0],pp[-1]) if path['closed'] and len(pp)>1 and pp[0] in adj[pp[-1]] else 0.
  for name,mask in [('S3',widths<=3),('S6_native',widths<=6),('S6_reference',widths*scale<=6)]:
   covered=float(np.diff(arc)[mask[:-1]&mask[1:]].sum())
   if closing and mask[0] and mask[-1]:covered+=closing
   rec[name+'_edge_covered_length_reference']=covered*scale
   rec[name+'_edge_coverage_fraction']=covered/length if length else None
  out.append((rec,{'coords':a,'width_native':widths,'arc_native':arc,'body':body}))
 return out,{'graph_components_8n':int(ncomp),'junction_clusters':len(nodes),'junction_internal_edges_contracted':internal,
 'graph_endpoints':len(ends),'junction_4plus_port_clusters':sum(n['ports']>=4 for n in nodes.values()),
 'junction_cluster_pixels':len(jid),'graph_paths':len(paths)},nodes

STRATA=['candidate_S3_zero','candidate_S3_tip_only','candidate_short_20_60','candidate_long_ge100',
 'candidate_junction_complex','fine_boundary_contact','fine_possible_gap','short_fine_under20','fine_loop_or_cycle']

def strata(r):
 c=r['candidate_B2'];L=r['length_reference'];med=r['body_width_reference_median'];fine=med is not None and med<=6
 t=[]
 if c and r['S3_points']==0:t.append(STRATA[0])
 if c and r['S3_tip_only_10pct']:t.append(STRATA[1])
 if c and 20<=L<60:t.append(STRATA[2])
 if c and L>=100:t.append(STRATA[3])
 if c and r['junction_complex_GT_only']:t.append(STRATA[4])
 if fine and r['n_points']>=5 and r['boundary_contact_2refpx']:t.append(STRATA[5])
 if fine and r['n_points']>=5 and r['possible_gap_10refpx']:t.append(STRATA[6])
 if fine and r['n_points']>=2 and L<20 and r['kind']!='loop_or_cycle':t.append(STRATA[7])
 if fine and r['kind']=='loop_or_cycle' and r['n_points']>=3:t.append(STRATA[8])
 return t

def select(records):
 cases=[];sels=[];used=set()
 for size in sorted(set(r['shape'] for r in records)):
  for tag in STRATA:
   pool=sorted([r for r in records if r['shape']==size and tag in strata(r)],key=lambda r:(r['filename'],r['branch_id']))
   unused=[r for r in pool if (r['filename'],r['branch_id']) not in used]
   if unused:
    r=unused[(len(unused)-1)//2];used.add((r['filename'],r['branch_id']));cid='CASE_'+str(len(cases)+1).zfill(2)
    cases.append({'case_id':cid,'stratum':tag,**r})
    sels.append({'shape':size,'stratum':tag,'pool_n':len(pool),'unused_pool_n':len(unused),
    'case_id':cid,'filename':r['filename'],'branch_id':r['branch_id'],'status':'selected_lower_median_unused'})
   else:sels.append({'shape':size,'stratum':tag,'pool_n':len(pool),'unused_pool_n':0,'case_id':'',
   'filename':'','branch_id':'','status':'unavailable_no_replacement' if not pool else 'already_covered_no_replacement'})
 return cases,sels

def crop_counts(row,masks,v,f):
 y,x=int(row['top']),int(row['left']);sl=np.s_[y:y+512,x:x+512]
 nf=int(f[sl].sum());nv=int((v[sl]&f[sl]).sum())
 assert nf==int(row['N_FOV']) and int(v[sl].sum())==int(row['N_positive_V'])
 out={k:row[k] for k in ('step','slot','draw','filename','top','left','hflip','vflip','ccw_rot90_quarters')}
 out.update({'N_FOV':nf,'N_positive_V':int(v[sl].sum()),'N_positive_V_in_FOV':nv,'input_hash_match_reused_C0':row['input_sha256_matches_frozen_T0']})
 A=nv+1e-6;D=.5*nf+nv+1e-6
 gp=-.5/nf+(A-2*D)/(4*D*D);gn=.5/nf+A/(4*D*D)
 b1=(nv*abs(gp)+(nf-nv)*abs(gn))/4;b2=math.sqrt(nv*gp*gp+(nf-nv)*gn*gn)/4
 for name,m in masks.items():
  n=int(m[sl].sum());out['N_'+name]=n
  l1=.1/8 if n else 0.;l2=.1/(8*math.sqrt(n)) if n else 0.
  out[name+'_z0_aux_gradient_L1']=l1;out[name+'_z0_aux_gradient_L2']=l2
  out[name+'_z0_aux_to_A0_L1']=l1/b1;out[name+'_z0_aux_to_A0_L2']=l2/b2
 assert out['N_S3']==int(row['N_S3'])
 # Target generation precedes cropping; apply exactly the bound discrete geometry.
 def augment(a):
  if int(row['hflip']):a=np.flip(a,axis=1)
  if int(row['vflip']):a=np.flip(a,axis=0)
  return np.rot90(a,int(row['ccw_rot90_quarters']))
 av=augment(v[sl]);af=augment(f[sl])
 for name in ('B1_reference','B2'):
  am=augment(masks[name][sl])
  assert int(am.sum())==out['N_'+name] and not np.any(am&~(av&af))
 out['synchronous_target_augmentation_CPU_checked']=True
 return out

def gradient_checks():
 import torch
 from retinal_m0.maples_thin_skeleton_bce_cpu_gate import thin_bce_loss
 from retinal_m0.maples_core import weighted_masked_loss
 z0=torch.linspace(-2,2,4*9*9).reshape(4,1,9,9)
 v=torch.zeros_like(z0);v[:,:,2:7,2:7]=1;f=torch.ones_like(v);a=torch.zeros_like(v);b=torch.zeros_like(v)
 a[0,0,3,3]=1;a[1,0,3:5,3]=1;b[0,0,2:6,3]=1;b[1,0,2:6,3]=1;b[2,0,2:6,3]=1
 checks={}
 for name,s in [('B1',a),('B2',b),('empty',torch.zeros_like(v))]:
  z=z0.clone().requires_grad_(True);old=weighted_masked_loss(z,v,f,f,epsilon=1e-6)
  new0=thin_bce_loss(z,v,f,s,lam=0.);go,=torch.autograd.grad(old,z,retain_graph=True)
  g0,=torch.autograd.grad(new0,z,retain_graph=True);assert torch.equal(old,new0) and torch.equal(go,g0)
  new=thin_bce_loss(z,v,f,s,lam=.1);g,=torch.autograd.grad(new,z)
  n=s.sum((1,2,3));analytic=.1*s*(z.detach().sigmoid()-1)/(4*n.clamp_min(1)[:,None,None,None])
  assert torch.allclose(g-go,analytic,rtol=1e-5,atol=1e-7)
  assert float(analytic.abs().sum())<=.1+1e-6
  checks[name]={'lambda0_value_gradient_bitwise_equal':True,'analytic_gradient_pass':True,
  'aux_logits_gradient_L1':float(analytic.abs().sum())}
 checks['actual_model_gradients']='NOT_MEASURED_NO_MODEL_FORWARD_AUTHORIZED'
 return checks

def main():
 started=time.time()
 assert OUT.is_dir() and not (OUT/'summary.json').exists()
 proto=json.loads((DOC/'protocol.json').read_text());manifest=json.loads((DOC/'train_manifest.json').read_text())
 oldcounts=list(csv.DictReader((OLD/'train_target_counts.csv').open()));oldby={r['filename']:r for r in oldcounts}
 crops=list(csv.DictReader((OLD/'crop_plan_and_counts.csv').open()))
 assert len(manifest['rows'])==110 and len(crops)==8000
 assert all(r['trial_split']=='train' for r in manifest['rows'])
 assert [r['filename'] for r in manifest['rows']]==proto['controls']['base_config']['train_sample_filenames']
 cropby=collections.defaultdict(list)
 binding=json.loads((OLD/'execution_binding.json').read_text())
 pairing_path=Path(proto['sources']['baseline_run'])/'pairing.jsonl'
 assert sha(pairing_path)==binding['baseline_pairing_sha256']
 pairs=[json.loads(line) for line in pairing_path.read_text().splitlines() if line.strip()]
 assert len(pairs)==2000
 for i,r in enumerate(crops):
  assert int(r['draw'])==i and int(r['step'])==i//4+1 and int(r['slot'])==i%4
  assert r['filename']==manifest['rows'][i%110]['filename'] and r['input_sha256_matches_frozen_T0']=='True'
  p=pairs[i//4];slot=i%4
  assert p['draws'][slot]==i and p['filenames'][slot]==r['filename']
  assert [int(z) for z in p['geometry'][slot]]==[int(r[k]) for k in ('top','left','hflip','vflip','ccw_rot90_quarters')]
  cropby[r['filename']].append(r)
 diams={r['filename']:2*math.sqrt(int(r['FOV_pixels'])/math.pi) for r in oldcounts}
 dref=float(np.median(list(diams.values())))
 definition={'status':DRAFT,'split':'110 frozen Train only','no_Dev_Test_probability_checkpoint_loaded':True,
 'skeleton':'bound full-native Zhang(V & FOV); no pruning, dilation, closing or repair',
 'graph':'8-neighbor; remove redundant diagonal edges only if axial 2-step route exists; no skeleton pixels removed',
 'junctions':'connected degree>=3 nodes; contract internal edges only, no radius merging; path stops at cluster boundary',
 'paths':'maximal degree2 chains between endpoints/junction boundary anchors; cycles, same-cluster loops, short paths and isolates retained',
 'crossings':'4+ ports is crossing/complex-junction suspicion only; no anatomical pairing across node',
 'width_native':'2 * padded native full-image EDT(V & FOV), no crop-local recomputation',
 'scale':'D=2*sqrt(FOV area/pi); alpha=median_Train_D/D; width_ref=width_native*alpha; length_ref=geodesic_native*alpha',
 'D_reference_Train_median':dref,'scale_limit':'FOV-relative proxy, NOT physical calibre; camera FOV angle/anatomy/annotation effects unresolved',
 'candidate_B2':'OPEN path length_ref>=20 and median(width_ref at non-junction points)<=6; whole path incl boundary anchors, excluding junction-core expansion',
 'candidate_B1':'same B2 branch set intersect pointwise width_ref<=6, called S6_reference; not all global/native S6',
 'parameter_status':'20 and 6 are unapproved semantic pilot choices, not tuned with model output or Dev scores',
 'S3_S6_native':'original native <=3/<=6 masks retained unchanged for explicit comparison',
 'short':'flag length_ref<20, not automatically biological spur; retain audit and atlas',
 'boundary':'full-path minimum FOV-EDT*alpha<=2 flag; no extrapolation or deletion',
 'gap':'nearest other-component endpoint<=10 ref-px is suspicion only; NEVER bridge or repair',
 'flags':'boundary/complex-junction/disconnected-fragment flags retained in candidate counts, not automatically approved',
 'atlas':'3 sizes x9 declared strata; lower-median lexicographic unused filename+branch_id; absent stratum not replaced',
 'gradient':'CPU logits-only synthetic checks; z=0 analytic diagnostics on actual crop target counts; no model forward',
 'formal_rules':'old freeze unchanged; this output is not a training/evaluation protocol'}
 js('candidate_definition_DRAFT.json',definition)
 protected=[DOC/n for n in ['protocol.json','train_manifest.json','source_manifest.json','dev_reference_manifest.json','freeze_receipt.json']]
 protected+=[OLD/n for n in ['verification.json','execution_binding.json','train_target_counts.csv','crop_plan_and_counts.csv','manual_review.csv']]
 protected+=[CODE/'retinal_m0/maples_core.py',CODE/'retinal_m0/maples_thin_skeleton_bce_cpu_gate.py']
 protected+=[pairing_path]
 protected+=[EXP/'targets/train'/(r['filename']+'.npz') for r in manifest['rows']]
 protected+=[Path(r[k]['path']) for r in manifest['rows'] for k in ('RGB','V')]
 before={str(p):sha(p) for p in protected}
 assert before[str(DOC/'protocol.json')]=='429fe51a8117df85116e8409ee57e77bd3a949c2773fa6bd2cfc2b92df8aa509'
 fixtures=graph_checks();grads=gradient_checks()
 records=[];images=[];cropout=[];junctions=[]
 (OUT/'path_data').mkdir(exist_ok=False);(OUT/'candidate_masks_NOT_FOR_TRAINING').mkdir(exist_ok=False)
 for idx,row in enumerate(manifest['rows']):
  name=row['filename'];rgb,v,f=load_rgb_v_fov(row,proto['controls']['base_config']['data_root'])
  g,sk,r,s3=s3_target(v,f)
  with np.load(EXP/'targets/train'/(name+'.npz'),allow_pickle=False) as z:
   assert np.array_equal(z['s_train'].astype(bool),s3)
  assert int(f.sum())==int(oldby[name]['FOV_pixels'])
  scale=dref/diams[name];paths,gs,nodes=characterize(row,sk,r,f,scale)
  b2=np.zeros_like(sk);coords=[];offs=[0];wn=[];arc=[];body=[]
  for rec,a in paths:
   if rec['candidate_B2']:yy,xx=a['coords'].T;b2[yy,xx]=1
   records.append(rec);coords.append(a['coords']);wn.append(a['width_native']);arc.append(a['arc_native']);body.append(a['body'])
   offs.append(offs[-1]+len(a['coords']))
  s6=sk&(r<=3);s6r=sk&(2*r.astype(float)*scale<=6);b1=b2&s6r
  assert not np.any(b1&~b2) and not np.any(b2&~sk) and not np.any(b2&~g)
  np.savez_compressed(OUT/'path_data'/(name+'.npz'),coords=np.concatenate(coords).astype(np.int32),
  offsets=np.asarray(offs,np.int32),width_native=np.concatenate(wn).astype(np.float32),
  arc_native=np.concatenate(arc).astype(np.float32),body=np.concatenate(body).astype(np.uint8))
  np.savez_compressed(OUT/'candidate_masks_NOT_FOR_TRAINING'/(name+'.npz'),B1_reference=b1.astype(np.uint8),B2_candidate=b2.astype(np.uint8))
  masks={'S3':s3,'S6_native':s6,'S6_reference':s6r,'B1_reference':b1,'B2':b2}
  cand=[a for a,_ in paths if a['candidate_B2']]
  images.append({'filename':name,'shape':str(tuple(sk.shape)),'height':sk.shape[0],'width':sk.shape[1],
  'FOV_pixels':int(f.sum()),'FOV_equivalent_diameter_native':diams[name],'scale_to_reference':scale,
  'native_cutoff_corresponding_to_ref6':6/scale,'full_skeleton_points':int(sk.sum()),
  **{k+'_points':int(m.sum()) for k,m in masks.items()},'candidate_paths':len(cand),
  'candidate_S3_zero_paths':sum(x['S3_points']==0 for x in cand),
  'candidate_boundary_paths':sum(x['boundary_contact_2refpx'] for x in cand),
  'candidate_complex_junction_paths':sum(x['junction_complex_GT_only'] for x in cand),
  'candidate_gap_flag_paths':sum(x['possible_gap_10refpx'] for x in cand),**gs})
  for k,n in nodes.items():junctions.append({'filename':name,'cluster_id':k,'pixels':n['pixels'],'ports':n['ports'],
  'first_y':n['first'][0],'first_x':n['first'][1],'confirmed_anatomical_crossing':False})
  for cr in cropby[name]:cropout.append(crop_counts(cr,masks,v,f))
  if (idx+1)%10==0:print('CPU_PROGRESS',idx+1,flush=True)
 cropout.sort(key=lambda r:int(r['draw']))
 table('branches_all_Train.csv',records);table('images_110.csv',images)
 table('junction_clusters.csv',junctions);table('crops_8000_and_analytic_gradients.csv',cropout)
 cases,sels=select(records);table('atlas_selection.csv',sels);table('atlas_cases.csv',cases)
 human=[{'case_id':c['case_id'],'filename':c['filename'],'branch_id':c['branch_id'],'stratum':c['stratum'],
 'reviewer_name':'','reviewed_at':'','GT_path_matches_intended_branch':'','RGB_visibility':'','visually_fine_branch':'',
 'junction_or_crossing_uncertainty':'','GT_gap_or_artifact_uncertainty':'','width_scale_definition_acceptable':'',
 'B1_B2_target_alignment':'','accept_pause_uncertain':'','notes':''} for c in cases]
 table('human_review_BLANK.csv',human)
 render_atlas(cases,manifest,proto,dref)
 summary=make_summary(images,records,cropout,cases,dref)
 after={p:sha(Path(p)) for p in before};assert before==after
 js('protected_sources_sha256.json',before);js('summary.json',summary)
 write_report(definition,summary)
 verify={'status':'CPU_COMPLETE_AWAITING_HUMAN_AND_USER_DECISION','candidate_status':DRAFT,
 'graph_fixtures':fixtures,'gradient_fixtures':grads,'protected_files_unchanged':len(before),
 'train_images':len(images),'crop_rows':len(cropout),'branch_rows':len(records),'atlas_cases':len(cases),
 'human_signed':0,'original_protocol_unchanged':True,'GPU_updates':0,'model_inference_calls':0,
 'Dev_images_or_probabilities_read':0,'official_Test_accessed':False,
 'source_script_sha256':sha(Path(__file__)),'elapsed_seconds':time.time()-started}
 js('verification.json',verify)
 files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and not p.name.startswith('STATUS.')}
 js('artifact_sha256.json',files)
 print('COMPLETE',json.dumps({'summary':summary,'verification':verify}),flush=True)


def make_summary(images,records,crops,cases,dref):
 masks=['S3','S6_native','S6_reference','B1_reference','B2']
 def imagegroup(rows):
  total=sum(r['full_skeleton_points'] for r in rows)
  return {'images':len(rows),'full_skeleton_points':total,
   'FOV_diameter_native':distribution([r['FOV_equivalent_diameter_native'] for r in rows]),
   'native_cutoff_for_ref6':distribution([r['native_cutoff_corresponding_to_ref6'] for r in rows]),
   'points':{k:sum(r[k+'_points'] for r in rows) for k in masks},
   'fraction_of_full_skeleton':{k:sum(r[k+'_points'] for r in rows)/total for k in masks},
   'candidate_paths':sum(r['candidate_paths'] for r in rows)}
 def pathgroup(rows):
  return {'n':len(rows),'S3_zero':sum(r['S3_points']==0 for r in rows),
   'S3_tip_only':sum(r['S3_tip_only_10pct'] for r in rows),
   'S3_ge80pct':sum(r['S3_fraction']>=.8 for r in rows),
   'S6_native_ge80pct':sum(r['S6_native_fraction']>=.8 for r in rows),
   'S6_reference_ge80pct':sum(r['S6_reference_fraction']>=.8 for r in rows),
   'point_macro_coverage':{k:float(np.mean([r[k+'_fraction'] for r in rows])) if rows else None for k in ['S3','S6_native','S6_reference']},
   'edge_length_macro_coverage':{k:float(np.mean([r[k+'_edge_coverage_fraction'] for r in rows if r[k+'_edge_coverage_fraction'] is not None])) if rows else None for k in ['S3','S6_native','S6_reference']},
   'length_reference':distribution([r['length_reference'] for r in rows]),
   'boundary_flag':sum(r['boundary_contact_2refpx'] for r in rows),
   'complex_junction_flag':sum(r['junction_complex_GT_only'] for r in rows),
   'four_port_flag':sum(r['junction_four_or_more_ports'] for r in rows),
   'gap_suspicion_flag':sum(r['possible_gap_10refpx'] for r in rows),
   'endpoint_to_endpoint_fragments':sum(r['kind']=='endpoint_to_endpoint_fragment' for r in rows)}
 cand=[r for r in records if r['candidate_B2']]
 return {'status':DRAFT,'reference_diameter':dref,'all_Train':imagegroup(images),
 'by_native_size':{s:imagegroup([r for r in images if r['shape']==s]) for s in sorted(set(r['shape'] for r in images))},
 'all_paths_n':len(records),'path_kinds':dict(collections.Counter(r['kind'] for r in records)),
 'candidate_paths':pathgroup(cand),
 'candidate_paths_by_size':{s:pathgroup([r for r in cand if r['shape']==s]) for s in sorted(set(r['shape'] for r in images))},
 'crops':{k:distribution([r['N_'+k] for r in crops]) for k in masks},
 'B2_nonempty_B1_empty_crops':sum(r['N_B2']>0 and r['N_B1_reference']==0 for r in crops),
 'z0_hypothetical_gradient_ratios':{k:{n:distribution([r[k+'_z0_aux_to_A0_'+n] for r in crops]) for n in ['L1','L2']} for k in ['B1_reference','B2']},
 'actual_model_gradient_status':'NOT_MEASURED_NO_MODEL_FORWARD',
 'atlas_cases':len(cases),'human_signed':0,'all_selected_available_cases_require_real_review':True,
 'branch_rows_are_GT_graph_proxies_not_visually_confirmed_vessels':True}

def render_atlas(cases,manifest,proto,dref):
 atlas=OUT/'atlas';atlas.mkdir(exist_ok=False)
 byname={r['filename']:r for r in manifest['rows']}
 pages=[];review=['# 完整细分支候选图册（仅 CPU 定义研究，未冻结）','',
 '原 S3 H1 不放行，所有 GPU 工作暂停。本图册不是训练许可。','',
 'S3/S6_native 沿用原生像素定义；S6_reference 是按 Train FOV 等效直径归一化后的候选，不是原 S6 的改名。','',
 '青色路径是 GT 图路径，不代表已确认解剖分支。点覆盖和相邻选中点的弧长覆盖分别报告。','',
 '请逐例填写 ../human_review_BLANK.csv 的真实姓名、时间及判断；允许暂停/未知，不补造人工通过。','',
 '选择规则：三个尺寸各九类，按 (filename,branch_id) 排序，在未选案例中取下中位数；空类不替换。','',
 '审核重点：路径是否完整对应目标分支；图像可见性；视觉细度；交叉/分叉/边界/标注断口疑问；尺度规则；B1/B2 是否符合研究目的。','']
 for c in cases:
  row=byname[c['filename']]
  rgb,v,f=load_rgb_v_fov(row,proto['controls']['base_config']['data_root'])
  with np.load(OUT/'path_data'/(c['filename']+'.npz'),allow_pickle=False) as z:
   i=c['branch_id'];start,end=map(int,z['offsets'][i:i+2])
   a=z['coords'][start:end];wn=z['width_native'][start:end].astype(float)
   arc=z['arc_native'][start:end].astype(float)
  scale=dref/(2*math.sqrt(float(f.sum())/math.pi));wr=wn*scale
  yy,xx=a.T;y0=max(0,int(yy.min())-32);y1=min(v.shape[0],int(yy.max())+33)
  x0=max(0,int(xx.min())-32);x1=min(v.shape[1],int(xx.max())+33)
  selected={'S3':wn<=3,'S6_native':wn<=6,'S6_reference':wr<=6,
            'B1_reference':(wr<=6)&c['candidate_B2'],'B2':np.full(len(a),c['candidate_B2'],bool)}
  sub=atlas/c['case_id'];sub.mkdir(exist_ok=False)
  Image.fromarray(rgb[y0:y1,x0:x1]).save(sub/'RGB_crop_native.png')
  Image.fromarray(v[y0:y1,x0:x1].astype(np.uint8)*255).save(sub/'GT_crop_native.png')
  rows=[{'k':i,'native_y':int(p[0]),'native_x':int(p[1]),'arc_native':float(arc[i]),
   'arc_reference':float(arc[i]*scale),'width_native':float(wn[i]),'width_reference':float(wr[i]),
   **{k:bool(m[i]) for k,m in selected.items()}} for i,p in enumerate(a)]
  table('atlas/'+c['case_id']+'/path_points_and_widths.csv',rows)
  fig,axes=plt.subplots(3,3,figsize=(17,14),constrained_layout=True)
  fig.suptitle(c['case_id']+' | '+c['stratum']+'\n'+c['filename']+' | branch '+str(c['branch_id'])+
  ' | '+c['kind']+' | DRAFT / NOT FOR TRAINING',fontsize=12)
  ax=axes[0,0];ax.imshow(rgb,interpolation='nearest')
  from matplotlib.patches import Rectangle
  ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='red',linewidth=1))
  ax.set_title('Full Train RGB / branch bbox');ax.axis('off')
  axes[0,1].imshow(rgb[y0:y1,x0:x1],interpolation='nearest')
  axes[0,1].set_title('Unmodified RGB native crop');axes[0,1].axis('off')
  positions=[(0,2,'PATH','#00cfff'),(1,0,'S3','#ffd400'),(1,1,'S6_native','#ff8c00'),
             (1,2,'S6_reference','#f45bff'),(2,0,'B1_reference','#33e377'),(2,1,'B2','#00cfff')]
  for ay,axid,name,color in positions:
   ax=axes[ay,axid];ax.imshow(v[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
   ax.plot(xx-x0,yy-y0,color='#777777',linewidth=1)
   mask=np.ones(len(a),bool) if name=='PATH' else selected[name]
   overlay=np.zeros((y1-y0,x1-x0,4),float)
   import matplotlib.colors as mc
   overlay[yy[mask]-y0,xx[mask]-x0]=(*mc.to_rgb(color),1)
   ax.imshow(overlay,interpolation='nearest')
   ax.plot(xx[0]-x0,yy[0]-y0,'o',markersize=6,markerfacecolor='none',markeredgecolor='red')
   ax.plot(xx[-1]-x0,yy[-1]-y0,'o',markersize=6,markerfacecolor='none',markeredgecolor='blue')
   label=name+' '+str(int(mask.sum()))+'/'+str(len(a))+' points'
   if name in ('S3','S6_native','S6_reference'):
    frac=c[name+'_edge_coverage_fraction']
    label+=' | edge length '+(str(round(100*frac,1))+'%' if frac is not None else 'NA')
   if name=='B2' and not c['candidate_B2']:label+=' / AUDIT ONLY, NOT SELECTED'
   ax.set_title(label,fontsize=10);ax.axis('off')
  ax=axes[2,2];ax.plot(arc*scale,wn,label='native width',color='#bb6b00',linewidth=1)
  ax.plot(arc*scale,wr,label='reference width',color='#b600b6',linewidth=1)
  ax.axhline(3,color='#999900',linestyle=':',label='native S3 cutoff')
  ax.axhline(6,color='#888888',linestyle='--',label='6 cutoff in each width unit')
  ax.set_xlabel('Arc length in reference pixels');ax.set_ylabel('2*EDT (native / reference px)')
  ax.set_title('Native coordinates + exact values: CSV');ax.legend(fontsize=7)
  fig.savefig(atlas/(c['case_id']+'.png'),dpi=120);plt.close(fig)
  note={'case_id':c['case_id'],'filename':c['filename'],'branch_id':c['branch_id'],
  'native_bbox':[y0,x0,y1,x1],'scale_to_reference':scale,'case_record':c,
  'display_only':'Figure rasterization/scaling is display only. RGB_crop_native and coordinate CSV preserve native pixels.'}
  js('atlas/'+c['case_id']+'/case.json',note)
  pages.append('<li><a href=\"'+c['case_id']+'.png\">'+html.escape(c['case_id']+' '+c['stratum'])+'</a> | '+
  html.escape(c['filename'])+' | <a href=\"'+c['case_id']+'/path_points_and_widths.csv\">native widths CSV</a></li>')
  review+=['## '+c['case_id']+'：'+c['stratum'],'',c['filename']+'；branch_id='+str(c['branch_id'])+
  '；原生 bbox='+str([y0,x0,y1,x1])+'；候选 B2='+str(c['candidate_B2']),'',
  '!['+c['case_id']+']('+c['case_id']+'.png)','',
  '[原生坐标与宽度 CSV]('+c['case_id']+'/path_points_and_widths.csv) | [原生 RGB]('+c['case_id']+'/RGB_crop_native.png)','']
 (atlas/'index.html').write_text('<!doctype html><meta charset=\"utf-8\"><title>Train branch target review DRAFT</title>'+
 '<h1>DRAFT CPU ONLY / NOT FOR TRAINING</h1><p>Human review pending. No model inference.</p><ul>'+
 ''.join(pages)+'</ul>',encoding='utf-8')
 (atlas/'REVIEW.md').write_text('\n'.join(review)+'\n',encoding='utf-8')

def write_report(definition,s):
 text=['# 完整细分支目标 CPU 定义研究与受控消融提案','',
 '**状态：DRAFT，待用户确认；原 S3 H1 不放行；无新 GPU 许可。**','',
 '本轮只读取固定110张Train原图/GT、既有C0来源及裁块记录，不读取Dev图像/概率、模型权重或Test。',
 '原冻结协议与正式结论不变。原生S3/S6与新参考尺度S6分开命名。','',
 '## 定义与边界','',
 '先在完整原生V∩FOV上建立原Zhang骨架和padding EDT。只删除有轴向两步替代路线的冗余对角图边，不删除骨架点。',
 '度1为GT端点；相邻度≥3像素合为分叉簇。最大度2链在端点/分叉簇边界停止；不跨簇猜测解剖走向。',
 '四端口及复杂节点只标疑问；短段、孤点、环路和端点对端点片段全部保留审计，不把它们自动诊断为伪支。',
 'FOV接触及不同连通域端点接近只作疑问标志，不外推、不补连GT断口、不人工修标签。','',
 'FOV等效直径D=2sqrt(area/pi)，参考直径取110张Train的中位数。width_ref=2EDT×Dref/D，length_ref=弧长×Dref/D。',
 '这是相对视野尺度代理，不是微米管径，不能消除拍摄角度、个体血管和标注厚度差异。',
 '本次候选：开放路径length_ref≥20且非分叉簇路径点的width_ref中位数≤6。20/6是语义试案，未根据模型结果选取，未获批冻结。',
 'B2选择整条候选路径（含端部锚点，不扩展分叉簇内部）；B1是同一集合与S6_reference的交集。原生S6单独保留，不冒充B1。','',
 '## 本轮统计','',
 '详细110图、所有图路径、8000实际Crop、分叉簇和逐例原生宽度均在同目录CSV/NPZ。所有路径仍未人工核验。',
 '本轮汇总如下（非正式评价终点）：','',
 json.dumps(s,ensure_ascii=False,indent=2),'',
 '## 人工审核','',
 'atlas/REVIEW.md为图册，human_review_BLANK.csv为真实人工表。三个尺寸各九类，缺失类保留而不有利替补。',
 '当前human_signed=0。不得把AI查看图册当作人工签署。需确认路径语义、可见性、视觉细度、尺度与GT局限。',
 '所有可用选例真实审核，系统性定义不匹配未解决则不得冻结。允许未知和否决，不删除异常来通过。','',
 '## B1/B2最小受控提案（无训练授权）','',
 'A0复用原T0 BCE+Dice，保持全FOV整体监督。B1/B2都为A0+0.1×每Crop自身目标数归一化的正样本BCE。',
 '空目标辅助0，仍留在Batch4均值；不能合并全Batch像素分母。两组共用同一候选分支集合，只改变分支内监督位置。',
 '同M0、seed2026、原8000draw与全部增强、Batch/LR/优化器/AMP/预算；各固定2000步，无lambda扫描或择Checkpoint。',
 '本轮只完成CPU公式/合成logits梯度与z=0计数诊断，实际模型梯度和跨进程恢复没有测量。',
 '任何GPU门槛及两组训练都须用户另批。普通V正样本辅助项为后续必要机制对照，尚未授权。',
 '相同lambda不等于真实模型梯度相同；B2更密集位置和非空Crop变化需全量报告，不能把差异直接归因拓扑机制。','',
 '## 评价候选，仍待单独冻结','',
 '完整候选分支覆盖同时报告逐图/逐分支、点与弧长权重，固定0.5、d0与原d1；保留原S3和病灶邻域/NA口径。',
 'Dice、Precision、FP/FOV、全图Recall、原远处Recall及粗血管召回均为不可忽略报告项。粗血管可预提body width_ref>6的明确分层，但边界及主保护定义须用户确认，不能临时挑有利分层。',
 '分支覆盖不等于连通性：d1点匹配可能跨过短缺口，BCE也没有显式路径约束。最长未覆盖弧长只是覆盖间隔。',
 '真实连通代理需在固定GT走廊内用未膨胀二值预测检查两端连通，并冻结锚点/交叉/NA规则；未核验前不宣称连接恢复。',
 'Dev已重复使用，未来结果均是探索。正式三seed和独立验证须另议；不访问Test。','',
 '## 停止与验收','',
 '原数据/冻结包变化、目标越界、路径/边计数遗漏、尺度或标签语义系统异常即停止，不进入GPU。',
 '本阶段工程完成仅指CPU证据与候选图册可复核、保护输入未变；不等于人工通过、目标获批或研究成功。']
 (OUT/'report_DRAFT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

if __name__=='__main__':
 main()
