#!/usr/bin/env python3
"""CPU-only analysis of frozen, pre-existing MAPLES probability maps."""
import os,sys,json,csv,hashlib,time,platform,argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
R=Path('${RETINA_PROJECT_ROOT}');D=R/'docs/plan_b_error_analysis_20261019';O=R/'experiments/plan_b_error_analysis_20261019';T=np.float32(.5)
PS='5822922a71e2f05ccc15fb68bde4a370c14738ebca5757ab6a93811b744b0b26';MS='4a5f4b0f6b2217ae23f7ab47aad7ea293620897c8d80ed906cd84b2a6ce72fd7'
MET=('dice','precision','recall','fp_density','near_pixel_recall','near_thin_recall_d1','far_recall')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def J(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def writej(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def C(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for x in rows for k in x))
 with p.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
def mask(p):
 a=np.asarray(Image.open(p))
 if a.ndim==3:a=a[...,0]
 if not np.isin(a,[0,1,255]).all():raise ValueError('nonbinary '+str(p))
 return a!=0
def ratio(a,b):return {'value':float(a/b) if b else None,'numerator':int(a),'denominator':int(b),'NA_reason':None if b else 'zero_denominator'}
def bind():
 if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA_VISIBLE_DEVICES must be empty')
 import torch
 if torch.cuda.is_available() or torch.cuda.device_count():raise RuntimeError('CUDA visible')
 pf=D/'protocol.json';mf=D/'source_manifest.json'
 if sha(pf)!=PS or sha(mf)!=MS:raise RuntimeError('protocol/manifest hash mismatch')
 p=J(pf);m=J(mf)
 if p['status']!='FROZEN_USER_APPROVED_BEFORE_ANALYSIS':raise RuntimeError('protocol is not frozen')
 if (len(m['conditions']),len(m['images']),len(m['point_catalog']),len(m['protected_files']))!=(11,28,77,565):raise RuntimeError('manifest inventory mismatch')
 bad=[]
 for i,(f,h) in enumerate(m['protected_files'].items(),1):
  if not Path(f).is_file() or sha(f)!=h:bad.append(f)
  if i%100==0:print('protected hash',i,flush=True)
 if bad:raise RuntimeError('protected hash mismatch '+repr(bad[:10]))
 names=[x['filename'] for x in m['images']];total=0
 for c in m['conditions']:
  if sha(c['metrics'])!=c['metrics_sha256']:raise RuntimeError('metrics hash '+c['condition_id'])
  ev=J(c['metrics'])
  if ev['threshold']!=.5 or ev['checkpoint_sha256']!=c['checkpoint_sha256_in_evaluation']:raise RuntimeError('metric lineage mismatch')
  if [x['filename'] for x in ev['images']]!=names or len(c['predictions'])!=28:raise RuntimeError('condition Dev inventory mismatch')
  for x in c['predictions']:
   if sha(x['path'])!=x['sha256']:raise RuntimeError('map hash '+x['path'])
   a=np.load(x['path'],mmap_mode='r',allow_pickle=False)
   if a.dtype!=np.float32 or list(a.shape)!=x['shape'] or list(a.shape)!=next(i['native_shape'] for i in m['images'] if i['filename']==x['filename']) or not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise RuntimeError('map content mismatch '+x['path'])
   total+=1
 if total!=308 or sum(i['primary_denominator'] for i in m['images'])!=77 or sum(i['primary_NA'] for i in m['images'])!=19:raise RuntimeError('frozen row/NA inventory mismatch')
 a=np.zeros((3,3),np.float32);a[1,2]=.5;P=a>=T
 assert P[1,2] and not P[1,1] and not P[0,0]
 assert not np.zeros((3,3),bool).any()
 f=np.ones((3,3),bool);u=np.zeros_like(f);u[0,0]=1;f[0,1]=0;rr=[np.zeros_like(f) for _ in range(4)]
 for y,x,k in [(0,0,3),(0,2,0),(1,0,1),(1,1,0),(1,2,1),(2,0,2),(2,1,2),(2,2,2)]:rr[k][y,x]=1
 assert np.array_equal(np.logical_or.reduce(rr),f)
 lab,n=ndimage.label(np.eye(3,dtype=bool),structure=np.ones((3,3),np.uint8));assert n==1
 O.mkdir(parents=True,exist_ok=True);(O/'validation').mkdir(exist_ok=True);libs={}
 for n in ['numpy','PIL','scipy','matplotlib','cv2']:
  try:q=__import__(n);libs[n]=getattr(q,'__version__','ok')
  except:pass
 binding={'protocol_id':p['protocol_id'],'protocol_sha256':PS,'source_manifest_sha256':MS,'implementation_path':str(Path(__file__).resolve()),'implementation_sha256':sha(__file__),'python':sys.version,'platform':platform.platform(),'libraries':libs,'CUDA_VISIBLE_DEVICES':'','torch_cuda_available':False,'analysis_mode':'CPU saved arrays only; no model construction/loading/inference/training/Test access','conditions':11,'prediction_maps':total,'protected_hashes_checked':565,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 writej(O/'execution_binding.json',binding)
 reuse={'status':'read_only_reuse_verified','protocol_sha256':PS,'manifest_sha256':MS,'protected_files':m['protected_files'],'conditions':[{'condition_id':c['condition_id'],'phase':c['phase'],'method':c['method'],'seed':c['seed'],'step':c['step'],'metrics':c['metrics'],'metrics_sha256':c['metrics_sha256'],'checkpoint_sha256':c['checkpoint_sha256_in_evaluation']} for c in m['conditions']],'probability_maps':[{'condition_id':c['condition_id'],'filename':x['filename'],'path':x['path'],'sha256':x['sha256'],'shape':x['shape'],'dtype':'float32'} for c in m['conditions'] for x in c['predictions']],'images':m['images'],'point_catalog':m['point_catalog']}
 writej(O/'reuse_manifest.json',reuse)
 v={'status':'passed','all_passed':True,'phase':'pre-analysis','checks':{'protocol_manifest_hashes':True,'565_protected_hashes':True,'308_maps_float32_native_finite_range':True,'11x28x77_inventory_9valid_19NA':True,'checkpoint_lineage_metrics':True,'CPU_synthetic_checks':True,'CUDA_hidden':True},'source_hashes_checked':565,'probability_maps_checked':total,'training_updates':0,'new_model_inference':0,'official_test_accessed':False,'execution_binding_sha256':sha(O/'execution_binding.json'),'reuse_manifest_sha256':sha(O/'reuse_manifest.json')}
 writej(O/'validation/verification.json',v);print('GATE PASS',v['execution_binding_sha256'])
def box(y,x,h,w,n):
 x0=max(0,x-n//2);y0=max(0,y-n//2);x1=min(w,x0+n);y1=min(h,y0+n);return [max(0,x1-n),max(0,y1-n),x1,y1]
def ov(rgb,gt,p,f,h,u):
 z=rgb.astype(np.float32).copy()
 for mk,col in [(p&gt&f,(0,230,70)),(p&~gt&f,(255,20,20)),(gt&~p&f,(50,100,255)),(h&f,(255,185,0)),(u&f,(0,220,255)),(~f,(0,0,0))]:z[mk]=z[mk]*.34+np.asarray(col)*.66
 return Image.fromarray(np.uint8(np.clip(z,0,255)))
def analyze():
 v=J(O/'validation/verification.json')
 if not v.get('all_passed') or sha(O/'execution_binding.json')!=v['execution_binding_sha256']:raise RuntimeError('gate/binding not passed')
 protocol=J(D/'protocol.json');m=J(D/'source_manifest.json');ims=m['images'];conds=m['conditions'];cat=m['point_catalog'];names=[i['filename'] for i in ims];out=O/'analysis';fig=O/'figures';fig.mkdir(parents=True,exist_ok=True)
 cache={};pt={};metrics=[];regions=[];components=[];recon=[];imrows=[];preds={}
 for im in ims:
  n=im['filename'];src=im['inputs'];rgb=np.asarray(Image.open(src['RGB']['path']).convert('RGB'));gt=mask(src['GT']['path']);f=mask(src['FOV']['path']);h=mask(src['H_path']['path'])
  with np.load(src['regions']['path'],allow_pickle=False) as b:r={k:b[k].astype(bool) for k in ['internal','near','far','thin']};u=b['U'].astype(bool)
  if rgb.shape[:2]!=tuple(im['native_shape']) or any(a.shape!=gt.shape for a in [f,h,u,*r.values()]):raise RuntimeError('native reference shape '+n)
  if not np.array_equal(r['internal']|r['near']|r['far'],f&~u):raise RuntimeError('Omega partition '+n)
  cache[n]=(rgb,gt,f,h,u,r);imrows.append({'filename':n,'native_shape':json.dumps(im['native_shape']),'primary_denominator':im['primary_denominator'],'primary_NA':im['primary_NA'],'RGB':src['RGB']['path'],'GT':src['GT']['path'],'FOV':src['FOV']['path'],'regions':src['regions']['path'],'H':src['H_path']['path'],'U_bright':src['U_bright_path']['path'],'U_red':src['U_red_path']['path'],'input_hashes':json.dumps({k:z['sha256'] for k,z in src.items()})})
  pointlist=[q for q in cat if q['filename']==n]
  for c in conds:
   x=next(q for q in c['predictions'] if q['filename']==n);prob=np.load(x['path'],mmap_mode='r',allow_pickle=False);P=(prob>=T)&f;omega=f&~u;support=P&omega;tp=P&gt&f;fp=P&~gt&f;fn=gt&~P&f
   ct={'TP':int(tp.sum()),'FP':int(fp.sum()),'FN':int(fn.sum()),'FOV_pixels':int(f.sum()),'GT_vessel_pixels':int((gt&f).sum())}
   vv={'dice':ratio(2*ct['TP'],2*ct['TP']+ct['FP']+ct['FN']),'precision':ratio(ct['TP'],ct['TP']+ct['FP']),'recall':ratio(ct['TP'],ct['TP']+ct['FN']),'fp_density':ratio(ct['FP'],ct['FOV_pixels']),'near_pixel_recall':ratio(int((P&gt&r['near']).sum()),int((gt&r['near']).sum())),'far_recall':ratio(int((P&gt&r['far']).sum()),int((gt&r['far']).sum()))}
   dmap,imap=ndimage.distance_transform_edt(~support,return_indices=True) if support.any() else (np.full(gt.shape,np.inf),None)
   for q in pointlist:
    y,xp=int(q['y']),int(q['x']);coords=[(y,xp),(y-1,xp),(y+1,xp),(y,xp-1),(y,xp+1)];coords=[(a,b) for a,b in coords if 0<=a<gt.shape[0] and 0<=b<gt.shape[1] and omega[a,b]]
    scores=[(float(prob[a,b]),a,b) for a,b in coords];mx=max(z[0] for z in scores);wy,wx=min((a,b) for z,a,b in scores if z==mx);hit=mx>=.5;dd=float(dmap[y,xp]) if support.any() and np.isfinite(dmap[y,xp]) else None;nearxy=(int(imap[0,y,xp]),int(imap[1,y,xp])) if dd is not None else None
    pt[c['condition_id'],q['point_id']]={'condition_id':c['condition_id'],'phase':c['phase'],'method':c['method'],'seed':c['seed'],'step':c['step'],'filename':n,'point_id':q['point_id'],'native_y':y,'native_x':xp,'image_primary_denominator':im['primary_denominator'],'center_probability':float(prob[y,xp]),'center_hit_0p5':bool(prob[y,xp]>=.5),'d1_local_max_probability':mx,'d1_margin_to_0p5':mx-.5,'d1_witness_y':wy,'d1_witness_x':wx,'d1_hit':hit,'point_FN_d1':not hit,'score_band':'covered' if hit else ('threshold_adjacent_score_miss' if mx>=.45 else 'below_near_band_score_miss'),'nearest_supported_prediction_distance_px':dd,'nearest_supported_prediction_y':nearxy[0] if nearxy else None,'nearest_supported_prediction_x':nearxy[1] if nearxy else None,'distance_NA_reason':'empty_prediction_support_in_Omega' if dd is None else None,'visual_review_record_id':'point_'+q['point_id']}
   vv['near_thin_recall_d1']=ratio(sum(pt[c['condition_id'],q['point_id']]['d1_hit'] for q in pointlist),len(pointlist));ev=J(c['metrics']);sr=next(z for z in ev['images'] if z['filename']==n)
   for k,z in vv.items():
    if (z['numerator'],z['denominator'])!=(sr[k]['numerator'],sr[k]['denominator']):raise RuntimeError('metric mismatch '+c['condition_id']+'/'+n+'/'+k)
    recon.append({'condition_id':c['condition_id'],'filename':n,'metric':k,'numerator':z['numerator'],'denominator':z['denominator'],'value':z['value'],'source_value':sr[k]['value']})
   sk={'TP':'tp','FP':'fp','FN':'fn','FOV_pixels':'fov_pixels','GT_vessel_pixels':'vessel_pixels'}
   for k,z in ct.items():
    if z!=sr['counts'][sk[k]]:raise RuntimeError('count mismatch '+c['condition_id']+'/'+n+'/'+k)
   row={'condition_id':c['condition_id'],'phase':c['phase'],'method':c['method'],'seed':c['seed'],'step':c['step'],'filename':n,**ct}
   for k,z in vv.items():row.update({k:z['value'],k+'_numerator':z['numerator'],k+'_denominator':z['denominator'],k+'_NA_reason':z['NA_reason']})
   metrics.append(row);preds[c['condition_id'],n]=(P,fp)
   for name,rm in {'internal':r['internal'],'near':r['near'],'far':r['far'],'U_in_FOV':u&f}.items():
    regions.append({'condition_id':c['condition_id'],'filename':n,'region':name,'FOV_region_pixels':int(rm.sum()),'GT_vessel_pixels':int((gt&rm).sum()),'TP':int((P&gt&rm).sum()),'FP':int((P&~gt&rm).sum()),'FN':int((gt&~P&rm).sum()),'FP_over_full_FOV':float((P&~gt&rm).sum()/f.sum()) if f.sum() else None,'FP_over_region_area':float((P&~gt&rm).sum()/rm.sum()) if rm.sum() else None,'conditional_rate_NA_reason':None if rm.sum() else 'zero_region_area'})
   last=regions[-4:]
   for k,z in [('TP','TP'),('FP','FP'),('FN','FN'),('FOV_region_pixels','FOV_pixels')]:
    if sum(y[k] for y in last)!=ct[z]:raise RuntimeError('region reconcile '+c['condition_id']+'/'+n)
   lab,_=ndimage.label(fp,structure=np.ones((3,3),np.uint8));tmp=[]
   for ix,sl in enumerate(ndimage.find_objects(lab),1):
    if sl is None:continue
    yy,xx=np.where(lab[sl]==ix);ys=yy+sl[0].start;xs=xx+sl[1].start;first=min(zip(ys.tolist(),xs.tolist()));tmp.append((first,ys,xs))
   tmp.sort(key=lambda z:z[0])
   if sum(len(z[1]) for z in tmp)!=int(fp.sum()):raise RuntimeError('component pixels do not sum to FP mask')
   for ix,(first,ys,xs) in enumerate(tmp,1):
    components.append({'condition_id':c['condition_id'],'filename':n,'component_id':f'cc{ix:05d}','pixel_area':len(ys),'bbox_x0':int(xs.min()),'bbox_y0':int(ys.min()),'bbox_x1':int(xs.max()+1),'bbox_y1':int(ys.max()+1),'centroid_x':float(xs.mean()),'centroid_y':float(ys.mean()),'first_y':first[0],'first_x':first[1],'pixels_internal':int(r['internal'][ys,xs].sum()),'pixels_near':int(r['near'][ys,xs].sum()),'pixels_far':int(r['far'][ys,xs].sum()),'pixels_U_in_FOV':int((u&f)[ys,xs].sum())})
 assert (len(metrics),len(pt),len(regions),len(recon))==(308,847,1232,2156)
 C(out/'image_manifest.csv',imrows);C(out/'point_catalog.csv',[{'point_id':q['point_id'],'filename':q['filename'],'native_y':q['y'],'native_x':q['x'],'local_order':q['local_order'],'image_primary_denominator':q['image_primary_denominator']} for q in cat])
 C(out/'per_point_observations.csv',list(pt.values()));C(out/'per_condition_image_metrics.csv',metrics);C(out/'fp_by_region.csv',regions);C(out/'fp_components.csv',components);C(out/'original_metric_reconciliation.csv',recon)
 formal=[c for c in conds if c['phase']=='formal'];cl=[c for c in conds if c['phase']=='class_split_exploratory'];bu=[c for c in conds if c['phase']=='budget_exploratory'];rec=[];pair=[];cpair=[];bpair=[]
 for q in cat:
  pid=q['point_id']
  for method,cs in [('T0',[c for c in formal if c['method']=='T0']),('TW',[c for c in formal if c['method']=='TW']),('2_1',cl)]:
   hs=[pt[c['condition_id'],pid]['d1_hit'] for c in cs];rec.append({'point_id':pid,'filename':q['filename'],'method':method,'block':'class_split_exploratory' if method=='2_1' else 'formal','missed_seeds':sum(not z for z in hs),'all_three_missed':sum(not z for z in hs)==3,'any_seed_missed':any(not z for z in hs),'between_seed_disagreement':len(set(hs))>1,'hit_seeds':sum(hs)})
  for seed in (2026,2027,2028):
   a=next(c for c in formal if c['method']=='T0' and c['seed']==seed);b=next(c for c in formal if c['method']=='TW' and c['seed']==seed);h0=pt[a['condition_id'],pid]['d1_hit'];h1=pt[b['condition_id'],pid]['d1_hit']
   pair.append({'point_id':pid,'filename':q['filename'],'seed':seed,'T0_hit':h0,'TW_hit':h1,'change':'recovery' if h1 and not h0 else 'loss' if h0 and not h1 else 'tie_hit' if h0 else 'tie_miss'})
   z=next(c for c in cl if c['seed']==seed);hc=pt[z['condition_id'],pid]['d1_hit']
   for meth in ('T0','TW'):
    b0=next(c for c in formal if c['method']==meth and c['seed']==seed);hb=pt[b0['condition_id'],pid]['d1_hit'];cpair.append({'point_id':pid,'filename':q['filename'],'seed':seed,'contrast':'2_1_minus_'+meth,'base_hit':hb,'2_1_hit':hc,'change':'recovery' if hc and not hb else 'loss' if hb and not hc else 'tie_hit' if hb else 'tie_miss'})
  a=next(c for c in formal if c['method']=='T0' and c['seed']==2026);b=next(c for c in formal if c['method']=='TW' and c['seed']==2026);a4=next(c for c in bu if c['method']=='T0');b4=next(c for c in bu if c['method']=='TW')
  x0=int(pt[a['condition_id'],pid]['d1_hit']);x1=int(pt[b['condition_id'],pid]['d1_hit']);x2=int(pt[a4['condition_id'],pid]['d1_hit']);x3=int(pt[b4['condition_id'],pid]['d1_hit'])
  bpair.append({'point_id':pid,'filename':q['filename'],'T0_2000_hit':x0,'TW_2000_hit':x1,'T0_4000_hit':x2,'TW_4000_hit':x3,'T0_temporal_change':x2-x0,'TW_temporal_change':x3-x1,'TW_minus_T0_2000':x1-x0,'TW_minus_T0_4000':x3-x2,'interaction':(x3-x2)-(x1-x0)})
 C(out/'point_recurrence.csv',rec);C(out/'formal_paired_point_changes.csv',pair);C(out/'class_split_paired_point_changes.csv',cpair);C(out/'budget_point_changes.csv',bpair)
 imfig=[]
 for n in names:
  rgb,gt,f,h,u,r=cache[n];H,W=gt.shape;pw=400;ph=int(pw*H/W);can=Image.new('RGB',(1600,(ph+22)*4),'white');dr=ImageDraw.Draw(can)
  panels=[('RGB',None),('GT',None)]+[(c['condition_id'],c) for c in conds]
  for ix,(label,c) in enumerate(panels):
   x=ix%4*pw;y=ix//4*(ph+22);dr.text((x+2,y+2),label[:42],fill='black')
   tile=Image.fromarray(rgb) if c is None and label=='RGB' else Image.fromarray(gt.astype('uint8')*255).convert('RGB') if c is None else ov(rgb,gt,preds[c['condition_id'],n][0],f,h,u)
   tile=tile.resize((pw,ph),Image.Resampling.BILINEAR if c is None else Image.Resampling.NEAREST);can.paste(tile,(x,y+20))
  file=fig/f'image_{n}.jpg';can.save(file,quality=88);imfig.append('../figures/'+file.name)
 ptfig=[]
 for page in range((len(cat)+11)//12):
  batch=cat[page*12:(page+1)*12];cw=116;rh=148;can=Image.new('RGB',(13*cw,len(batch)*rh),'white');dr=ImageDraw.Draw(can)
  for ri,q in enumerate(batch):
   n=q['filename'];rgb,gt,f,h,u,r=cache[n];x0,y0,x1,y1=box(q['y'],q['x'],gt.shape[0],gt.shape[1],128)
   panels=[('RGB',None),('GT',None)]+[(c['condition_id'],c) for c in conds]
   for ci,(label,c) in enumerate(panels):
    xx=ci*cw;yy=ri*rh;dr.text((xx+1,yy+1),label[:17],fill='black')
    if c is None:tile=Image.fromarray(rgb[y0:y1,x0:x1]) if label=='RGB' else Image.fromarray(gt[y0:y1,x0:x1].astype('uint8')*255).convert('RGB')
    else:
     tile=ov(rgb[y0:y1,x0:x1],gt[y0:y1,x0:x1],preds[c['condition_id'],n][0][y0:y1,x0:x1],f[y0:y1,x0:x1],h[y0:y1,x0:x1],u[y0:y1,x0:x1]);z=pt[c['condition_id'],q['point_id']];dr.text((xx+1,yy+rh-18),f"{z['d1_local_max_probability']:.2f} {'H' if z['d1_hit'] else 'M'}",fill='black')
    can.paste(tile.resize((cw,128),Image.Resampling.NEAREST),(xx,yy+18))
  file=fig/f'points_page_{page+1:02d}.png';can.save(file);ptfig.append('../figures/'+file.name)
 review=[]
 for q in cat:
  n=q['filename'];pg=cat.index(q)//12+1;review.append({'case_id':'point_'+q['point_id'],'case_type':'primary_point','filename':n,'point_id_or_NA':q['point_id'],'conditions_actually_viewed':'','reviewer_type':'not_assessed','reviewer_identifier':'','reviewed_at':'','figure_paths_and_native_bbox':json.dumps({'figure':f'figures/points_page_{pg:02d}.png','native_bbox':box(q['y'],q['x'],*cache[n][1].shape,128)}),'category':'not_assessed','assessment':'not_assessed','visible_evidence':'','alternative_explanation':'','limitation_or_uncertainty':'Actual AI inspection pending; no human confirmation claimed.','needs_human_confirmation':'yes'})
 for n in names:review.append({'case_id':'image_FP_'+n,'case_type':'whole_image_FP','filename':n,'point_id_or_NA':'NA','conditions_actually_viewed':'','reviewer_type':'not_assessed','reviewer_identifier':'','reviewed_at':'','figure_paths_and_native_bbox':json.dumps({'figure':f'figures/image_{n}.jpg','native_bbox':[0,0,cache[n][1].shape[1],cache[n][1].shape[0]]}),'category':'not_assessed','assessment':'not_assessed','visible_evidence':'','alternative_explanation':'','limitation_or_uncertainty':'FP is label-relative and does not establish biological absence.','needs_human_confirmation':'yes'})
 C(out/'visual_review_log.csv',review)
 summaries={}
 for c in conds:
  rr=[x for x in metrics if x['condition_id']==c['condition_id']];summaries[c['condition_id']]={}
  for k in MET:
   z=[x[k] for x in rr if x[k] is not None];summaries[c['condition_id']][k]={'mean':float(np.mean(z)) if z else None,'n':len(z),'NA':28-len(z)}
 fps={c['condition_id']:{'FP_pixels':sum(x['FP'] for x in metrics if x['condition_id']==c['condition_id']),'mean_FP_over_FOV':summaries[c['condition_id']]['fp_density']['mean'],'zero_FP_images':sum(x['FP']==0 for x in metrics if x['condition_id']==c['condition_id']),'FP_components':sum(x['condition_id']==c['condition_id'] for x in components)} for c in conds}
 sm={'protocol_id':protocol['protocol_id'],'counts':{'conditions':11,'probability_maps':308,'Dev_images':28,'primary_valid_images':9,'primary_positions':77,'primary_NA_images':19,'point_condition_rows':len(pt),'image_condition_rows':len(metrics),'NA_image_condition_rows':sum(x['near_thin_recall_d1_denominator']==0 for x in metrics),'region_rows':len(regions),'FP_components':len(components),'visual_review_cases':105},'metric_summaries':summaries,'FP_summary':fps,'evidence_levels':{'automatic_computation':'complete; all original per-image counts reconcile','AI_visual':'pending actual review','human':'not performed','unknown':['patient independence','microvessel label accuracy','causal mechanism','clinical meaning']},'training_updates':0,'new_model_inference':0,'official_test_accessed':False}
 writej(out/'summary.json',sm)
 selected={}
 def add(n,reason):selected.setdefault(n,[]).append(reason)
 for n in ['20051205_35276_0400_PP','20060411_61478_0200_PP','20060530_54390_0100_PP']:add(n,'previously influential, disclosed')
 primary=[]
 for n in names:
  z=[x['near_thin_recall_d1'] for x in metrics if x['filename']==n and x['phase']=='formal' and x['near_thin_recall_d1'] is not None]
  if z:primary.append((float(np.mean([1-x for x in z])),n))
 for _,n in sorted(primary,key=lambda z:(-z[0],z[1]))[:3]:add(n,'top formal mean primary miss fraction')
 fpr=[(float(np.mean([x['fp_density'] for x in metrics if x['filename']==n and x['phase']=='formal'])),n) for n in names]
 for _,n in sorted(fpr,key=lambda z:(-z[0],z[1]))[:3]:add(n,'top formal mean FP/FOV')
 C(out/'case_selection.csv',[{'filename':n,'selection_reasons':json.dumps(v,ensure_ascii=False),'figure_path':f'figures/image_{n}.jpg'} for n,v in sorted(selected.items())])
 html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Plan B 错误分析</title><style>body{font:15px system-ui;max-width:1200px;margin:28px auto;padding:0 18px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:5px}th{background:#eee}.note{background:#fff1d3;padding:12px}img{max-width:100%}</style><h1>Plan B：既有预测错误分析</h1><p class="note">自动计算完成；视觉复核尚未完成，真实人工复核未执行。所有概率图均为既有输出，阈值仍为0.5；AI意见会与人工判断明确分开。</p><p>11条件、308概率图、28张Dev、9张有效图、77个目标位置；847逐点条件、308逐图条件、209主指标NA、1232分区行、'+str(len(components))+'个8连通FP组件。</p><h2>七项固定指标（每图分子/分母见CSV）</h2><table><tr><th>条件</th><th>阶段</th><th>主 near-thin d1</th><th>Dice</th><th>Precision</th><th>FP/FOV</th><th>Recall</th><th>near pixel</th><th>far</th></tr>']
 for c in conds:
  z=summaries[c['condition_id']];cells=[c['condition_id'],c['phase']]+[('NA' if z[k]['mean'] is None else f"{z[k]['mean']:.6f}")+f" ({z[k]['n']}; NA {z[k]['NA']})" for k in MET]
  html.append('<tr>'+''.join('<td>'+str(x)+'</td>' for x in cells)+'</tr>')
 html.append('</table><p>NA按原定义保留；主指标按图宏平均。FP=P & ~GT & FOV（含U），FP/FOV分母为完整FOV。四区互斥覆盖FOV；8连通组件无面积过滤。预算结果只和同一seed 2026的2000步配对，不当作新种子。</p><h2>证据层级</h2><p>自动事实：像素差异、分数/距离、逐图指标、组件和复发。AI视觉判断不等于人工或临床确认。真实人工判断当前未执行。未知包括患者独立性、微血管标签有效性及因果机制。FP是相对GT差异，不证明血管为假；低概率不证明图像不可见；与GT不一致不证明标签错误。</p><h2>全部77个位置</h2><p>每行一个固定位置，RGB、GT及11个条件，裁剪128原生像素。</p>')
 for f in ptfig:html.append('<p><img src="'+f+'"></p>')
 html.append('<h2>全部28图误报图册</h2><p>含RGB、GT和11个固定阈值预测叠图。绿TP、红FP、蓝FN、橙H、青U。</p>')
 for f in imfig:html.append('<p><img src="'+f+'"></p>')
 html.append('<h2>逐图数据文件</h2><ul>')
 for f in sorted(out.glob('*.csv')):html.append('<li><a href="analysis/'+f.name+'">'+f.name+'</a></li>')
 html.append('</ul></html>');(O/'report.html').write_text('\n'.join(html),encoding='utf-8')
 print(json.dumps({'status':'quantitative_complete_visual_pending','counts':sm['counts']},ensure_ascii=False))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['bind','analyze']);a=ap.parse_args()
 bind() if a.mode=='bind' else analyze()
if __name__=='__main__':main()
