"""Same GT reference for every method; native counts and paired image statistics."""
from __future__ import annotations
import numpy as np
from retinal_m0.maples_branch_science_reference import valid_runs

def cross_dilate(mask):
    a=np.asarray(mask,bool);out=a.copy()
    out[1:]|=a[:-1];out[:-1]|=a[1:];out[:,1:]|=a[:,:-1];out[:,:-1]|=a[:,1:]
    return out

def ratio(num,den,reason='zero_denominator'):
    return dict(value=float(num/den) if den else None,numerator=int(num),denominator=int(den),NA_reason=None if den else reason)

def path_d1(paths,prediction,reference=None):
    hit=cross_dilate(prediction);values=[]
    for c in paths:
        c=np.asarray(c,np.int64)
        if c.ndim!=2 or c.shape[1]!=2:raise ValueError('path must be Nx2')
        if not len(c):continue
        if np.any(c<0) or np.any(c[:,0]>=hit.shape[0]) or np.any(c[:,1]>=hit.shape[1]):raise ValueError('out of bounds')
        c=np.unique(c,axis=0);values.append(float(hit[c[:,0],c[:,1]].mean()))
    return float(np.mean(values)) if values else None

def _longest(c,h,wanted):
    best=0.;points=0;start=0
    edges=np.linalg.norm(np.diff(c.astype(float),axis=0),axis=1);arc=np.r_[0.,np.cumsum(edges)]
    for i in range(len(c)+1):
        if i==len(c) or bool(h[i])!=wanted:
            if i>start:best=max(best,float(arc[i-1]-arc[start]));points=max(points,i-start)
            start=i+1
    return best,points

def evaluate_image(probability,arrays,method,threshold=.5):
    if method not in ('A0','B1','B2') or threshold!=.5:raise ValueError('frozen method/threshold required')
    p=np.asarray(probability);gt=np.asarray(arrays['GT'],bool);f=np.asarray(arrays['FOV'],bool);omega=np.asarray(arrays['Omega'],bool)
    if p.shape!=gt.shape or f.shape!=gt.shape or omega.shape!=gt.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('invalid aligned probability/GT')
    pred=(p>=threshold)&f;regional=pred&omega;hit=cross_dilate(regional)
    tp=int((pred&gt&f).sum());fp=int((pred&~gt&f).sum());fn=int((~pred&gt&f).sum())
    result=dict(dice=ratio(2*tp,2*tp+fp+fn),precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),fp_density=ratio(fp,int(f.sum())))
    coords=np.asarray(arrays['coords'],np.int64);offs=np.asarray(arrays['offsets'],np.int64);chosen=np.asarray(arrays['candidate_B2'],bool)
    b1points=np.asarray(arrays['B1_selected_points'],bool);union=np.zeros_like(f);support=np.zeros_like(f);records=[];excluded=[]
    for i,(a,b) in enumerate(zip(offs[:-1],offs[1:])):
        c=coords[a:b];keep=omega[c[:,0],c[:,1]];valid=np.unique(c[keep],axis=0)
        if not chosen[i]:
            n=len(valid);excluded.append(dict(branch_id=i,denominator=n,NA=n==0,
              d1_numerator=int(hit[valid[:,0],valid[:,1]].sum()) if n else 0,
              d0_numerator=int(regional[valid[:,0],valid[:,1]].sum()) if n else 0))
            continue
        union[c[:,0],c[:,1]]=True;sub=c[b1points[a:b]];support[sub[:,0],sub[:,1]]=True;den=len(valid)
        if not den:records.append(dict(branch_id=i,denominator=0,NA=True));continue
        n1=int(hit[valid[:,0],valid[:,1]].sum());n0=int(regional[valid[:,0],valid[:,1]].sum())
        arcden=arcnum=covered=0.;maxhit=maxmiss=0.;maxhitpoints=maxmisspoints=0
        for run in valid_runs(c,omega):
            h=hit[run[:,0],run[:,1]];lengths=np.linalg.norm(np.diff(run.astype(float),axis=0),axis=1)
            arcden+=float(lengths.sum());arcnum+=float((lengths*(h[:-1].astype(float)+h[1:].astype(float))/2).sum())
            covered+=float((lengths*(h[:-1]&h[1:])).sum())
            x,n=_longest(run,h,True);maxhit=max(maxhit,x);maxhitpoints=max(maxhitpoints,n)
            x,n=_longest(run,h,False);maxmiss=max(maxmiss,x);maxmisspoints=max(maxmisspoints,n)
        records.append(dict(branch_id=i,denominator=den,NA=False,d1_numerator=n1,d0_numerator=n0,d1=n1/den,d0=n0/den,
          valid_arc_length=arcden,hit_arc_length=arcnum,covered_edge_length=covered,arc_recall=arcnum/arcden if arcden else None,
          edge_coverage=covered/arcden if arcden else None,longest_hit=maxhit,longest_miss=maxmiss,
          longest_hit_points=maxhitpoints,longest_miss_points=maxmisspoints,whole_missed_d1=n1==0,whole_missed_d0=n0==0))
    valid=[r for r in records if not r['NA']]
    def mean(k):
        v=[r[k] for r in valid if r[k] is not None];return float(np.mean(v)) if v else None
    den=sum(r['denominator'] for r in valid)
    result.update(candidate_path_d1=ratio(sum(r['d1_numerator'] for r in valid),den),
      candidate_path_d0=ratio(sum(r['d0_numerator'] for r in valid),den),candidate_path_d1_macro=mean('d1'),candidate_path_d0_macro=mean('d0'),
      arc_recall_d1_macro=mean('arc_recall'),edge_coverage_d1_macro=mean('edge_coverage'),
      whole_missed_d1=ratio(sum(r['whole_missed_d1'] for r in valid),len(valid)),
      whole_missed_d0=ratio(sum(r['whole_missed_d0'] for r in valid),len(valid)),
      longest_hit_mean=mean('longest_hit'),longest_miss_mean=mean('longest_miss'),
      valid_candidate_paths=len(valid),path_records=records,excluded_path_records=excluded,fov_pixels=int(f.sum()))
    for name,mask in [('candidate_unique',union&omega),('B1_supported',support&omega),('B2_added',union&~support&omega),
                       ('coarse_vessel',np.asarray(arrays['coarse_vessel_reference'],bool)),
                       ('full_S3',np.asarray(arrays.get('full_S3',np.zeros_like(f)),bool)&omega)]:
        for d,h in [('d0',regional),('d1',hit)]:result[name+'_'+d]=ratio(int((mask&h).sum()),int(mask.sum()))
    far=np.asarray(arrays.get('far',np.zeros_like(f)),bool)&omega
    result['far_recall']=ratio(int((regional&gt&far).sum()),int((gt&far).sum()))
    near=np.asarray(arrays.get('near',np.zeros_like(f)),bool)&omega
    result['near_pixel_recall']=ratio(int((regional&gt&near).sum()),int((gt&near).sum()))
    thin=np.asarray(arrays.get('thin',np.zeros_like(f)),bool)&near
    result['near_thin_recall_d0']=ratio(int((thin&regional).sum()),int(thin.sum()))
    result['near_thin_recall_d1']=ratio(int((thin&hit).sum()),int(thin.sum()))
    al=sum(r['valid_arc_length'] for r in valid)
    result['arc_recall_d1_pooled']=sum(r['hit_arc_length'] for r in valid)/al if al else None
    result['edge_coverage_d1_pooled']=sum(r['covered_edge_length'] for r in valid)/al if al else None
    return result

def metric_values(record):
    excluded={'path_records','excluded_path_records','valid_candidate_paths','fov_pixels'}
    return {k:(v['value'] if isinstance(v,dict) else v) for k,v in record.items() if k not in excluded}

def path_strata(all_records):
    """GT-only descriptive strata, including excluded short/complex paths.

    No new optimized length bins: exact frozen reference lengths are retained.
    These diagnostics never replace the candidate-only primary valid set.
    """
    families={k:{} for k in ['length_reference_exact','short_branch','complex_flags','path_kind']}
    flags=['junction_complex_GT_only','junction_four_or_more_ports','boundary_contact_2refpx','possible_gap_10refpx']
    n_total=len(all_records['A0'])
    for method,images in all_records.items():
        for im in images:
            for p in im['path_records']+im['excluded_path_records']:
                length=float(p['length_reference'])
                keys=[('length_reference_exact',str(length)),('short_branch','short_lt20' if length<20 else 'not_short_ge20'),('path_kind',p.get('kind','unknown'))]
                keys += [('complex_flags',key+'='+str(bool(p.get(key,False))).lower()) for key in flags]
                for family,key in keys:families[family].setdefault(key,{}).setdefault(method,{}).setdefault(im['filename'],[]).append(p)
    result={}
    for family,groups in families.items():
        result[family]={}
        for group,methods in groups.items():
            summaries={};per_method={}
            for method in ['A0','B1','B2']:
                by_image=methods.get(method,{});all_paths=[p for pp in by_image.values() for p in pp];valid=[p for p in all_paths if p['denominator']>0]
                images={}
                for name,pp in by_image.items():
                    vv=[p for p in pp if p['denominator']>0]
                    if vv:images[name]=dict(d1=float(np.mean([p['d1_numerator']/p['denominator'] for p in vv])),d0=float(np.mean([p['d0_numerator']/p['denominator'] for p in vv])),n_paths=len(vv))
                per_method[method]=images
                summaries[method]=dict(n_images=len(images),NA_images=n_total-len(images),n_paths=len(valid),NA_paths=len(all_paths)-len(valid),
                  candidate_paths=sum(bool(p.get('candidate_B2')) for p in all_paths),excluded_paths=sum(not bool(p.get('candidate_B2')) for p in all_paths),
                  position_denominator=sum(p['denominator'] for p in valid),d1_numerator=sum(p['d1_numerator'] for p in valid),d0_numerator=sum(p['d0_numerator'] for p in valid),
                  path_d1_macro=float(np.mean([v['d1'] for v in images.values()])) if images else None,path_d0_macro=float(np.mean([v['d0'] for v in images.values()])) if images else None)
            ids=set(per_method['A0'])
            if any(set(per_method[m])!=ids for m in ['B1','B2']):raise ValueError('GT-only stratum validity changed across methods')
            comparisons={}
            for label in ['B2-A0','B1-A0','B2-B1']:
                left,right=label.split('-');changes=[dict(filename=n,delta=per_method[left][n]['d1']-per_method[right][n]['d1'],delta_d0=per_method[left][n]['d0']-per_method[right][n]['d0']) for n in sorted(ids)]
                comparisons[label]=dict(estimate=float(np.mean([r['delta'] for r in changes])) if changes else None,n_images=len(changes),rows=changes)
            result[family][group]=dict(**summaries,comparisons=comparisons)
    result['scope']='descriptive GT strata including excluded paths; exact lengths and existing20 threshold/flags; no new inference or confirmatory endpoint'
    return result

def change_summary(rows,metric):
    direction=-1 if metric in ('fp_density','whole_missed_d1','whole_missed_d0','longest_miss_mean') else 1
    labeled=[]
    for r in rows:
        x=r['delta']*direction
        labeled.append(dict(r,judgment='improved' if x>0 else 'declined' if x<0 else 'unchanged'))
    return dict(improved=sum(r['judgment']=='improved' for r in labeled),declined=sum(r['judgment']=='declined' for r in labeled),
      unchanged=sum(r['judgment']=='unchanged' for r in labeled),favorable_direction='higher' if direction==1 else 'lower',rows=labeled)

def bootstrap_indices(seed,n,reps=1000):
    if type(seed)is not int or type(n)is not int or type(reps)is not int or n<=0 or reps<=0:raise ValueError('invalid bootstrap dimensions')
    return np.random.Generator(np.random.PCG64(seed)).integers(0,n,size=(reps,n),dtype=np.int64)

def paired_bootstrap(a,b,indices):
    """Return b-a; callers pass reference first and candidate second."""
    a=np.asarray(a,float);b=np.asarray(b,float);ix=np.asarray(indices,np.int64)
    if a.ndim!=1 or b.shape!=a.shape or ix.ndim!=2 or ix.shape[1]!=len(a) or not len(a):raise ValueError('invalid paired vectors')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(ix<0) or np.any(ix>=len(a)):raise ValueError('invalid paired values/indices')
    estimates=(b-a)[ix].mean(axis=1)
    return dict(estimate=float((b-a).mean()),lower=float(np.quantile(estimates,.025,method='linear')),
      upper=float(np.quantile(estimates,.975,method='linear')),bootstrap_estimates=estimates.tolist(),n_images=len(a))

def paired_analysis(rows,comparisons=('B2-A0','B1-A0','B2-B1'),seed=20261016,reps=1000):
    if len({r['filename'] for r in rows})!=len(rows):raise ValueError('duplicate image IDs')
    metrics=set().union(*(set(r[m]) for r in rows for m in ('A0','B1','B2')))
    cache={};results={};validity={}
    for metric in sorted(metrics):
        keep=[]
        for r in rows:
            if any(metric not in r[m] for m in ('A0','B1','B2')):raise ValueError('method metric missing: '+metric)
            defined=r.get('fixed_valid_metrics',{}).get(metric)
            if defined is None:defined=r['fixed_primary_valid'] if metric.startswith(('candidate_','arc_','edge_','whole_missed','longest_')) else True
            if defined:keep.append(r)
        validity[metric]=[r['filename'] for r in keep]
        if keep and len(keep) not in cache:cache[len(keep)]=bootstrap_indices(seed,len(keep),reps)
        for label in comparisons:
            left,right=label.split('-');results.setdefault(label,{})
            if not keep:results[label][metric]=dict(estimate=None,lower=None,upper=None,n_images=0,NA_reason='GT_fixed_zero_valid_images');continue
            lefts=[r[left][metric] for r in keep];rights=[r[right][metric] for r in keep]
            if any(x is None for x in lefts+rights):raise ValueError('unexpected method NA on fixed valid set: '+metric)
            results[label][metric]=paired_bootstrap(rights,lefts,cache[len(keep)])
    primary_ids=[r['filename'] for r in rows if r['fixed_primary_valid']]
    return dict(fixed_primary_n=len(primary_ids),fixed_primary_ids=primary_ids,fixed_metric_ids=validity,
      bootstrap_indices_by_n={str(n):ix.tolist() for n,ix in cache.items()},comparisons=results,
      primary_contrast='B2-A0',seed_SD=None)
