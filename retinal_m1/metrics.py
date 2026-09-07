"""Whole-image diagnostics, per-image denominators and explicit undefined values."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from sklearn.metrics import average_precision_score
from retinal_m1.maps import interior

def score(probability,label,fov,thin,*,threshold=.5,include_ap=True):
    p=np.asarray(probability);y=np.asarray(label,bool);f=np.asarray(fov,bool);thin=np.asarray(thin,bool)
    if p.ndim!=2 or not p.shape==y.shape==f.shape==thin.shape:
        raise ValueError('Matching 2D arrays required')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not 0<=threshold<=1:
        raise ValueError('Invalid probabilities or threshold')
    if np.any(thin & ~(y&f)):raise ValueError('Thin target outside label/FOV')
    y=y&f;pred=(p>=threshold)&f;tp=int((y&pred).sum());ny=int(y.sum());npred=int(pred.sum())
    sy=skeletonize(y,method='zhang');sp=skeletonize(pred,method='zhang')
    tprec=float(y[sp].mean()) if sp.any() else None
    tsens=float(pred[sy].mean()) if sy.any() else None
    cl=(None if not ny and not npred else 0.) if tprec is None or tsens is None else (2*tprec*tsens/(tprec+tsens) if tprec+tsens else 0.)
    # No prediction means infinite distance; scipy EDT on all-ones is not this case.
    distance=distance_transform_edt(~pred) if pred.any() else np.full(p.shape,np.inf)
    result=dict(dice=2*tp/(ny+npred) if ny+npred else None,
        precision=tp/npred if npred else None,recall=tp/ny if ny else None,cldice=cl,
        average_precision=float(average_precision_score(y[f],p[f])) if include_ap and ny else None,
        fov_pixels=int(f.sum()),label_pixels=ny,predicted_pixels=npred,thin_pixels=int(thin.sum()))
    for delta in (0,1,2):
        result[f'thin_recall_d{delta}']=float((distance[thin]<=delta).mean()) if thin.any() else None
    inside=thin&interior(f)
    result['thin_recall_d1_interior_sensitivity']=float((distance[inside]<=1).mean()) if inside.any() else None
    result['thin_interior_pixels']=int(inside.sum())
    return result

def aggregate(rows):
    keys=('dice','precision','recall','cldice','average_precision','thin_recall_d0','thin_recall_d1','thin_recall_d2','thin_recall_d1_interior_sensitivity')
    def group(items):
        result={'images':len(items)}
        for k in keys:
            values=[r[k] for r in items if r.get(k) is not None]
            result[k]=dict(mean=float(np.mean(values)) if values else None,defined=len(values),undefined=len(items)-len(values))
        return result
    return {'all':group(rows),'disease':{d:group([r for r in rows if r['disease']==d]) for d in ('A','D','G','N')}}
