"""Native MAPLES evaluation. No training and no implicit experimental definitions."""
from __future__ import annotations
import argparse,csv,hashlib,json,re
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from scipy.ndimage import binary_dilation,distance_transform_edt
from retinal_m0.objectives import binary_metrics
from retinal_m1.maps import targets

METRICS=('dice','precision','recall','fp_density','near_pixel_recall','near_thin_recall_d1','far_recall')

def ratio(num,den,reason=None):
    return {'value':float(num/den) if den else None,'numerator':int(num),'denominator':int(den),'NA_reason':None if den else (reason or 'zero_denominator')}

def mask(value,shape,name):
    a=np.asarray(value)
    if a.shape!=shape or not np.isin(a,[0,1,False,True]).all():raise ValueError('Invalid binary mask: '+name)
    return a.astype(bool)

def make_regions(fov,h,u,*,radius,kernel):
    f=np.asarray(fov,bool);h=mask(h,f.shape,'H');u=mask(u,f.shape,'U')
    if type(radius) is not int or radius<1 or kernel not in ('euclidean_disk','square'):raise ValueError('Explicit positive radius and kernel required')
    yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
    structure=(xx*xx+yy*yy<=radius*radius) if kernel=='euclidean_disk' else np.ones((2*radius+1,)*2,bool)
    dilated=binary_dilation(h,structure=structure);omega=f&~u
    return dict(internal=h&omega,near=dilated&~h&omega,far=omega&~dilated)

def thin_reference(label,fov,*,radius_max):
    if radius_max is None or not np.isfinite(radius_max) or radius_max<=0:raise ValueError('Explicit thin radius threshold required; not inherited from FIVES')
    maps=targets(label,fov)
    return maps['skeleton']&(maps['skeleton_radius']<=radius_max)

def evaluate_arrays(probability,label,fov,*,threshold,u=None,regions=None,thin=None,matching_support=None):
    p=np.asarray(probability);y=np.asarray(label);f=np.asarray(fov)
    if p.ndim!=2 or p.shape!=y.shape or p.shape!=f.shape:raise ValueError('Matching native 2D probability/label/FOV required; no resizing')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Probability/threshold outside [0,1]')
    y=mask(y,p.shape,'V');f=mask(f,p.shape,'FOV')
    base=binary_metrics(torch.from_numpy(np.ascontiguousarray(p[f])),torch.from_numpy(np.ascontiguousarray(y[f])),threshold)
    tp,fp,fn=base['tp'],base['fp'],base['fn'];pred=(p>=threshold)&f
    z={'threshold':float(threshold),'counts':dict(tp=tp,fp=fp,fn=fn,fov_pixels=int(f.sum()),vessel_pixels=int((y&f).sum())),
       'dice':ratio(2*tp,2*tp+fp+fn),'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),'fp_density':ratio(fp,int(f.sum()))}
    for k in ('near_pixel_recall','near_thin_recall_d1','far_recall'):z[k]=ratio(0,0,'region_definition_not_provided')
    if regions is None:return z
    if u is None:raise ValueError('Explicit U mask required for regional evaluation')
    u=mask(u,p.shape,'U');omega=f&~u
    if set(regions)!={'internal','near','far'}:raise ValueError('Three explicit region masks required')
    r={k:mask(v,p.shape,k) for k,v in regions.items()}
    coverage=sum(v.astype(np.uint8) for v in r.values())
    if not np.array_equal(coverage,omega.astype(np.uint8)):raise ValueError('Regions must partition FOV minus U exactly')
    near,far=r['near'],r['far']
    z['regional_counts']={k:int(v.sum()) for k,v in r.items()};z['regional_counts']['U_in_FOV']=int((u&f).sum())
    z['near_pixel_recall']=ratio(int((pred&y&near).sum()),int((y&near).sum()))
    z['far_recall']=ratio(int((pred&y&far).sum()),int((y&far).sum()))
    if thin is not None:
        thin=mask(thin,p.shape,'thin_reference')
        if np.any(thin&~(y&f)):raise ValueError('Reference thin target must be subset of V and FOV')
        if matching_support not in ('omega','near'):raise ValueError('Specify d1 matching support: omega or near')
        supported=pred&(omega if matching_support=='omega' else near)
        distance=distance_transform_edt(~supported) if supported.any() else np.full(p.shape,np.inf)
        target=thin&near
        z['near_thin_recall_d1']=ratio(int((target&(distance<=1.)).sum()),int(target.sum()))
        z['d1_matching_support']=matching_support
    else:z['near_thin_recall_d1']=ratio(0,0,'thin_definition_not_provided')
    return z

def aggregate(rows):
    result={}
    for k in METRICS:
        values=[row[k]['value'] for row in rows if row[k]['value'] is not None]
        result[k]={'mean':float(np.mean(values)) if values else None,'n':len(values),'NA':len(rows)-len(values),'images':len(rows)}
    return result

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_mask(path):
    if str(path).endswith('.npy'):return np.load(path,allow_pickle=False)
    with Image.open(path) as im:a=np.asarray(im)
    if a.ndim!=2 or not (np.isin(a,[0,1]).all() or np.isin(a,[0,255]).all()):raise ValueError('Invalid binary reference: '+str(path))
    return a!=0

def validate_dev_membership(rows):
    root=Path('${RETINA_PROJECT_ROOT}/data/maples_handoff_20261008')
    native_manifest=root/'data_manifest.csv'
    if sha(native_manifest)!='238f897d45f8d6b8e69693b663e80aa7979d750db1aff960f126464427421f90':raise ValueError('Frozen MAPLES manifest changed')
    with native_manifest.open(encoding='utf-8-sig') as f:native={r['filename']:r for r in csv.DictReader(f) if r['trial_split']=='dev'}
    if len(rows)!=28 or {r['filename'] for r in rows}!=set(native):raise ValueError('Membership differs from frozen dev28')
    for r in rows:
        n=native[r['filename']]
        if r.get('split')!='dev' or r.get('source_split')!='train' or r.get('native_shape')!=[int(n['height']),int(n['width'])] or r.get('vessel_path_sha256')!=n['label_vessels_sha256'] or Path(r.get('vessel_path','')).resolve()!=(root/n['label_vessels']).resolve():
            raise ValueError('Dev reference does not match original handoff: '+r['filename'])

def evaluate_manifest(manifest,probability_dir,output,*,threshold):
    doc=json.loads(Path(manifest).read_text());rows=doc['images'];output=Path(output)
    if doc['scope'] not in ('MAPLES_dev28','synthetic_fixture'):raise ValueError('Only dev or independent fixtures are allowed')
    if doc['scope']=='MAPLES_dev28':validate_dev_membership(rows)
    names=[r['filename'] for r in rows]
    if len(names)!=len(set(names)) or not all(re.fullmatch('[A-Za-z0-9_]+',name) for name in names):raise ValueError('Unique safe identities required')
    if output.exists():raise ValueError('Use a fresh evaluation output directory')
    results=[]
    for row in rows:
        inputs={k:Path(row[k]) for k in ('vessel_path','fov_path')}
        for role,path in inputs.items():
            if sha(path)!=row[role+'_sha256']:raise ValueError('Reference changed: '+str(path))
        ppath=Path(probability_dir)/(row['filename']+'.npy');p=np.load(ppath,allow_pickle=False)
        if list(p.shape)!=row['native_shape']:raise ValueError('Probability is not native size')
        kwargs={}
        if row.get('region_bundle'):
            if sha(row['region_bundle'])!=row['region_bundle_sha256']:raise ValueError('Regional references changed')
            if not doc.get('definition_status') or not doc.get('regional_definition'):raise ValueError('Regional bundle requires explicit provenance')
            with np.load(row['region_bundle'],allow_pickle=False) as b:
                kwargs=dict(u=b['U'],regions={k:b[k] for k in ('internal','near','far')},thin=b['thin'],matching_support=doc['regional_definition']['matching_support'])
        result=evaluate_arrays(p,load_mask(inputs['vessel_path']),load_mask(inputs['fov_path']),threshold=threshold,**kwargs)
        result.update(filename=row['filename'],probability_sha256=sha(ppath));results.append(result)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(evaluator_sha256=sha(__file__),prediction_provenance=doc.get('prediction_provenance','not declared'),scope=doc['scope'],definition_status=doc.get('definition_status','regional_definitions_pending'),regional_definition=doc.get('regional_definition'),threshold=float(threshold),threshold_status='explicit_invocation_value_not_automatically_formal',manifest_sha256=sha(manifest),images=results,aggregate=aggregate(results))
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    with (output/'per_image.csv').open('w',newline='') as f:
        fields=['filename']+[k+s for k in METRICS for s in ('','_numerator','_denominator','_NA_reason')];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in results:
            flat={'filename':row['filename']}
            for k in METRICS:
                flat[k]=row[k]['value'] if row[k]['value'] is not None else 'NA'
                for suffix,src in [('_numerator','numerator'),('_denominator','denominator'),('_NA_reason','NA_reason')]:flat[k+suffix]=row[k][src]
            writer.writerow(flat)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--probability-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--threshold',required=True,type=float)
    a=p.parse_args();r=evaluate_manifest(a.manifest,a.probability_dir,a.output,threshold=a.threshold);print(json.dumps(r['aggregate'],indent=2))
if __name__=='__main__':main()
