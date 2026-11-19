"""Opt-in A15.1 proposed protocol helpers; NOT automatic protocol approval."""
import math
import numpy as np
from retinal_m1.maps import targets
from retinal_m0.maples_evaluate import make_regions,mask

def proposed_references(v,f,h,bright,red,*,diameter_max):
    f=np.asarray(f);v=mask(v,f.shape,'V');f=mask(f,f.shape,'FOV')
    if f.ndim!=2 or diameter_max not in (2,3,4,5):raise ValueError('Native2D and explicit diameter cap2/3/4/5 required')
    u=mask(bright,f.shape,'BrightUncertains')|mask(red,f.shape,'RedUncertains')
    radius=max(1,math.ceil(f.shape[1]/100))
    regions=make_regions(f,h,u,radius=radius,kernel='euclidean_disk')
    # Compute topology/EDT on full native V&FOV BEFORE subtracting U/near.
    maps=targets(v,f);diameter=2.*maps['skeleton_radius']
    thin=maps['skeleton']&(diameter<=diameter_max)
    return dict(U=u,thin=thin,**regions),dict(status='PROPOSED_not_frozen',width_definition='2*EDT(V&FOV) at Zhang skeleton, native pixel-center diameter proxy',diameter_max=diameter_max,radius=radius,radius_rule='max(1,ceil(native_RGB_width/100))',U_sources=['BrightUncertains','RedUncertains'],matching_support='omega')

def select_checkpoint(candidates,z0):
    """Proposed strict zero-degradation point-estimate gates, then primary max."""
    keys=('precision','fp_density','far_recall','near_thin_recall_d1')
    def valid(r):return all(r.get(k) is not None and np.isfinite(r[k]) and 0<=r[k]<=1 for k in keys) and r.get('primary_n',0)>0
    if not valid(z0):raise ValueError('Current same-protocol Z0 baseline and nonempty fixed primary targets required')
    eligible=[]
    for r in candidates:
        if r['step']<=0 or not valid(r):continue
        if r.get('reference_id')!=z0.get('reference_id') or not r.get('reference_id') or r.get('primary_ids')!=z0.get('primary_ids') or not r.get('primary_ids'):raise ValueError('Fixed common reference/eligible image identities differ')
        if r['precision']>=z0['precision'] and r['fp_density']<=z0['fp_density'] and r['far_recall']>=z0['far_recall']:eligible.append(r)
    return min(eligible,key=lambda r:(-r['near_thin_recall_d1'],r['step'])) if eligible else None

def matched_threshold_choice(candidates,target_fp_density):
    if not np.isfinite(target_fp_density) or not 0<=target_fp_density<=1:raise ValueError('Invalid common FP/FOV target')
    valid=[r for r in candidates if np.isfinite(r['threshold']) and 0<=r['threshold']<=1 and r['fp_density'] is not None and np.isfinite(r['fp_density']) and 0<=r['fp_density']<=target_fp_density]
    return min(valid,key=lambda r:r['threshold']) if valid else None
