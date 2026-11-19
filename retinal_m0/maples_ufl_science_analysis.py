"""Pure analysis for D; old metrics unchanged, ordinary Dice is sole primary."""
from __future__ import annotations
import numpy as np
from skimage.morphology import skeletonize


def dice_readout(analysis):
    """Prespecified UFL-A0 interpretation, not a checkpoint/threshold selector."""
    d=analysis['dice']['UFL-A0']
    if any(d.get(k) is None or not np.isfinite(d[k]) for k in ('estimate','lower','upper')):
        raise ValueError('ordinary Dice primary must be defined on all fixed Dev28')
    rules={'precision':lambda x:x>=0,'fp_density':lambda x:x<=0,
           'far_recall':lambda x:x>=0,'recall':lambda x:x>=0,
           'coarse_vessel_d0':lambda x:x>=0,'hard_cldice':lambda x:x>=0}
    means={m:analysis.get(m,{}).get('UFL-A0',{}).get('estimate') for m in rules}
    nonadverse=all(v is not None and np.isfinite(v) and rules[m](v) for m,v in means.items())
    strict={m:analysis.get(m,{}).get('UFL-A0',{}).get('upper' if m=='fp_density' else 'lower')
            for m in ('precision','far_recall','fp_density')}
    strict_ok=all(v is not None and np.isfinite(v) and (v<=0 if m=='fp_density' else v>=0) for m,v in strict.items())
    positive=d['estimate']>0
    interval_positive=positive and d['lower']>0
    eligible=interval_positive and nonadverse and strict_ok
    classification=('NO_DICE_IMPROVEMENT_SIGNAL' if not positive else
                    'DICE_EVIDENCE_INSUFFICIENT' if not interval_positive else
                    'DICE_SIGNAL_WITH_PROTECTION_COST' if not eligible else
                    'DEV_SINGLE_SEED_SIGNAL_WITH_GUARDS')
    return {'primary_metric':'dice','primary_mean_differences':{'dice':d['estimate']},
            'dice_95pct_interval':[d['lower'],d['upper']],
            'dice_positive_mean':bool(positive),'dice_positive_95pct':bool(interval_positive),
            'mean_guard_differences':means,'mean_guards_nonadverse':bool(nonadverse),
            'strict_guard_intervals':strict,'strict_guard_intervals_pass':bool(strict_ok),
            'eligible_to_request_additional_seeds':bool(eligible),
            'classification':classification,'independent_validation':False,
            'automatic_seed_expansion':False,
            'scope':'repeated-use Dev28, seed2026 only; image bootstrap does not measure seed stability'}


def hard_cldice(prediction, ground_truth, fov, *, uncertain=None):
    p=np.asarray(prediction,dtype=bool); g=np.asarray(ground_truth,dtype=bool); f=np.asarray(fov,dtype=bool)
    if p.ndim != 2 or p.shape != g.shape or p.shape != f.shape: raise ValueError('aligned 2D prediction, GT and FOV required')
    if uncertain is not None and np.asarray(uncertain).shape != p.shape: raise ValueError('uncertainty array shape mismatch')
    if not f.any(): raise ValueError('empty_fov')
    # U is intentionally retained: the frozen diagnostic defines full FOV evaluation including U.
    pm=p & f; gm=g & f
    sp=skeletonize(pm,method='zhang'); sg=skeletonize(gm,method='zhang')
    npred=int(sp.sum()); ngt=int(sg.sum()); pred_hit=int((sp & gm).sum()); gt_hit=int((sg & pm).sum())
    precision=None if npred==0 else pred_hit/npred
    sensitivity=None if ngt==0 else gt_hit/ngt
    if npred==0 and ngt==0: score=1.0
    elif npred==0 or ngt==0: score=0.0
    elif precision is None or sensitivity is None or precision+sensitivity==0: score=0.0
    else: score=2*precision*sensitivity/(precision+sensitivity)
    return {'topology_precision':precision,'topology_sensitivity':sensitivity,'cldice':float(score),
            'pred_skeleton_count':npred,'gt_skeleton_count':ngt,'pred_hit_count':pred_hit,'gt_hit_count':gt_hit,
            'NA_reason':{'topology_precision':'zero_denominator' if npred==0 else None,
                         'topology_sensitivity':'zero_denominator' if ngt==0 else None}}


def bootstrap_indices(n, *, seed=20261016, replicates=1000):
    if type(n) is not int or n < 1 or type(replicates) is not int or replicates < 1:
        raise ValueError('positive fixed image count and replicate count required')
    rng=np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0,n,size=(replicates,n))


def paired_bootstrap(left, right, *, indices=None, seed=20261016, replicates=1000):
    # Inputs must already use the frozen GT-valid image IDs. Never drop images by prediction NA.
    a=np.asarray(left,dtype=object); b=np.asarray(right,dtype=object)
    if a.ndim!=1 or a.shape!=b.shape: raise ValueError('paired one-dimensional arrays required')
    if any(x is None for x in a.tolist()+b.tolist()): raise ValueError('prediction NA on fixed image set; stop analysis')
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('nonfinite prediction metric on fixed image set; stop analysis')
    n=len(a)
    if n==0: raise ValueError('empty fixed image set; preserve as NA at caller')
    ix=bootstrap_indices(n,seed=seed,replicates=replicates) if indices is None else np.asarray(indices,dtype=np.int64)
    if ix.shape!=(replicates,n) or np.any(ix<0) or np.any(ix>=n): raise ValueError('bootstrap index set does not match fixed n')
    estimates=(a[ix]-b[ix]).mean(axis=1)
    return {'estimate':float((a-b).mean()),'lower':float(np.quantile(estimates,.025,method='linear')),
            'upper':float(np.quantile(estimates,.975,method='linear')),'n_images':n,'indices':ix.tolist(),
            'estimates':estimates.tolist(),'seed':seed,'generator':'PCG64','replicates':replicates}


def paired_bootstrap_many(rows, metrics, *, fixed_ids, contrasts=(('UFL','A0'),), seed=20261016, replicates=1000):
    if not fixed_ids: return {'fixed_ids':[],'n_images':0,'results':{m:{'NA_reason':'GT_fixed_zero_valid_images'} for m in metrics}}
    byid={r['filename']:r for r in rows}
    if set(fixed_ids)-set(byid): raise ValueError('missing row from fixed GT image set')
    n=len(fixed_ids); indices=bootstrap_indices(n,seed=seed,replicates=replicates)
    result={};
    for metric in metrics:
        result[metric]={}
        for left,right in contrasts:
            try:
                a=[byid[name][left][metric] for name in fixed_ids]
                b=[byid[name][right][metric] for name in fixed_ids]
            except KeyError as exc: raise ValueError(f'missing fixed-image method/metric value: {exc}') from exc
            result[metric][f'{left}-{right}']=paired_bootstrap(a,b,indices=indices,seed=seed,replicates=replicates)
    return {'fixed_ids':list(fixed_ids),'n_images':n,'bootstrap_indices':indices.tolist(),'results':result}



def fixed_gt_metric_validity(arrays, refrow):
    """Validity is determined from frozen GT regions, never prediction or method."""
    f=np.asarray(arrays['FOV'],bool); g=np.asarray(arrays['GT'],bool); o=np.asarray(arrays['Omega'],bool)
    near=np.asarray(arrays['near'],bool)&o; far=np.asarray(arrays['far'],bool)&o
    thin=np.asarray(arrays['thin'],bool)&near; path=int(refrow['B2_valid_paths'])>0
    valid={'dice':bool(f.any()),'precision':bool(f.any()),'recall':bool((g&f).any()),'fp_density':bool(f.any()),
           'far_recall':bool((g&far).any()),'near_pixel_recall':bool((g&near).any()),
           'near_thin_recall_d0':bool(thin.any()),'near_thin_recall_d1':bool(thin.any()),'hard_cldice':bool(f.any())}
    for key in ['candidate_path_d0','candidate_path_d1','candidate_path_d0_macro','candidate_path_d1_macro','candidate_unique_d0','candidate_unique_d1',
                'arc_recall_d1_macro','arc_recall_d1_pooled','edge_coverage_d1_macro','edge_coverage_d1_pooled','whole_missed_d0','whole_missed_d1','longest_hit_mean','longest_miss_mean',
                'B1_supported_d0','B1_supported_d1','B2_added_d0','B2_added_d1']:
        valid[key]=path
    for key,mask in [('full_S3',np.asarray(arrays['full_S3'],bool)&o),('coarse_vessel',np.asarray(arrays['coarse_vessel_reference'],bool))]:
        for d in ('d0','d1'): valid[key+'_'+d]=bool(mask.any())
    return valid


def descriptive_strata(rows, refs):
    """Same frozen exact lengths, <20 short cutoff, path kinds/flags, and image sizes."""
    methods=('UFL','A0','B1','B2'); families={}; native={}
    flags=('junction_complex_GT_only','junction_four_or_more_ports','boundary_contact_2refpx','possible_gap_10refpx')
    for row,ref in zip(rows,refs):
        native.setdefault(str(tuple(ref['native_shape'])),[]).append(row)
        metadata={p['branch_id']:p for p in ref['paths']}
        for method in methods:
            rr=row['full_records'][method]
            for role in ('path_records','excluded_path_records'):
                for path in rr[role]:
                    meta=metadata[path['branch_id']];path.update({k:v for k,v in meta.items() if k not in path})
                    length=float(meta['length_reference'])
                    keys=[('length_reference_exact',str(length)),('short_branch','short_lt20' if length<20 else 'not_short_ge20'),('path_kind',meta.get('kind','unknown'))]
                    keys += [('complex_flags',k+'='+str(bool(meta.get(k,False))).lower()) for k in flags]
                    for family,group in keys:families.setdefault(family,{}).setdefault(group,{}).setdefault(method,{}).setdefault(row['filename'],[]).append(path)
    out={}
    for family,groups in families.items():
        out[family]={}
        for group,by_method in groups.items():
            result={}; image_values={}
            for method in methods:
                ims={}; all_paths=[p for ps in by_method.get(method,{}).values() for p in ps]
                for name,pp in by_method.get(method,{}).items():
                    pp=[p for p in pp if p['denominator']>0]
                    if pp:ims[name]={'d1':float(np.mean([p['d1_numerator']/p['denominator'] for p in pp])),'d0':float(np.mean([p['d0_numerator']/p['denominator'] for p in pp])),'n_paths':len(pp)}
                vv=[p for p in all_paths if p['denominator']>0]; image_values[method]=ims
                result[method]={'n_images':len(ims),'NA_images':28-len(ims),'n_paths':len(vv),'NA_paths':len(all_paths)-len(vv),
                    'candidate_paths':sum(bool(p['candidate_B2']) for p in all_paths),'excluded_paths':sum(not bool(p['candidate_B2']) for p in all_paths),
                    'position_denominator':sum(p['denominator'] for p in vv),'d1_numerator':sum(p['d1_numerator'] for p in vv),'d0_numerator':sum(p['d0_numerator'] for p in vv),
                    'path_d1_macro':float(np.mean([v['d1'] for v in ims.values()])) if ims else None,'per_image':ims}
            ids=sorted(image_values['UFL'])
            if any(sorted(image_values[m])!=ids for m in methods):raise ValueError('GT stratum validity differs by method')
            result['comparisons']={}
            for right in ('A0','B1','B2'):
                deltas=[{'filename':name,'delta':image_values['UFL'][name]['d1']-image_values[right][name]['d1'],'delta_d0':image_values['UFL'][name]['d0']-image_values[right][name]['d0']} for name in ids]
                result['comparisons']['UFL-'+right]={'estimate':float(np.mean([x['delta'] for x in deltas])) if deltas else None,'n_images':len(ids),'rows':deltas}
            out[family][group]=result
    sizes={}
    for size,ims in native.items():
        sizes[size]={m:{'n_images':len(ims),'candidate_path_d1_macro':float(np.mean([r['methods'][m]['candidate_path_d1_macro'] for r in ims]))} for m in methods}
    return out,sizes
