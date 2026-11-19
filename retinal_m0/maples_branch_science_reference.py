"""GT-only reference construction for the frozen scientific B1/B2 protocol."""
from __future__ import annotations
import ast
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import label as cc_label, distance_transform_edt
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

ROOT=Path('${RETINA_PROJECT_ROOT}')
CODE=ROOT/'code/retinal-vessel-study'
PROTOCOL=ROOT/'docs/full_fov_thin_skeleton_bce_20261019/branch_review_20261021/scientific_b1_b2_2000_v1_20261022.json'
SPLIT=ROOT/'data/maples_handoff_20261008/split_record.json'
DEV_FOV=ROOT/'experiments/a15_1_maples_dev_evaluation_20261015/dev'
DEV_REGIONS=ROOT/'experiments/a15_2_z0_t0_calibration_20261015/references'
GRAPH_CODE=ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/review_definition_v2.py'
GT_DIR=ROOT/'data/maples_handoff_20261008/labels/Vessels'
UNCERTAINTY_PROTOCOL_SHA='72fb5a23d11ac1d2750aeb0aa9addf7dfa4f9753bd10ebc21233b7d0b886dfbf'
ALPHA_NUMERATOR=911.1106643779144
WIDTH_LIMIT=6.0
LENGTH_LIMIT=20.0

def graph_functions():
    """Load only the two frozen pure function ASTs; never execute preparation main/imports."""
    protocol=json.loads(PROTOCOL.read_text())
    if file_sha(PROTOCOL)!=UNCERTAINTY_PROTOCOL_SHA or file_sha(GRAPH_CODE)!=protocol['evidence_anchors_sha256'][str(GRAPH_CODE)]:
        raise RuntimeError('protocol/graph source drift')
    tree=ast.parse(GRAPH_CODE.read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('graph','characterize')]
    if len(funcs)!=2:raise RuntimeError('frozen pure graph functions missing')
    ns=dict(np=np,math=math,collections=collections,cc_label=cc_label,cdist=cdist,
            distance_transform_edt=distance_transform_edt,DRAFT='DRAFT_CPU_ONLY_NOT_FOR_TRAINING')
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(GRAPH_CODE),'exec'),ns)
    return ns['characterize']

def native_paths(gt,fov,alpha,filename,characterize=None):
    g=np.asarray(gt,bool)&np.asarray(fov,bool)
    sk=skeletonize(g,method='zhang')
    radius=distance_transform_edt(np.pad(g,1))[1:-1,1:-1].astype(np.float32)
    rows,summary,_=(characterize or graph_functions())({'filename':filename},sk,radius,fov,alpha)
    paths=[dict(a,closed=rec['kind']=='loop_or_cycle',legacy_record=rec) for rec,a in rows]
    return paths,summary,sk,radius


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def valid_runs(coords, valid):
    coords=np.asarray(coords,dtype=np.int64); valid=np.asarray(valid,dtype=bool)
    if coords.ndim!=2 or coords.shape[1]!=2 or valid.ndim!=2: raise ValueError('invalid path or valid mask')
    if len(coords)==0: return []
    if np.any(coords<0) or np.any(coords[:,0]>=valid.shape[0]) or np.any(coords[:,1]>=valid.shape[1]): raise ValueError('path coordinate out of bounds')
    keep=valid[coords[:,0],coords[:,1]]
    starts=np.flatnonzero(keep & np.r_[True,~keep[:-1]])
    ends=np.flatnonzero(keep & np.r_[~keep[1:],True])+1
    return [np.ascontiguousarray(coords[a:b],dtype=np.int32) for a,b in zip(starts,ends)]


def _check_binary(a,name):
    a=np.asarray(a)
    if a.ndim!=2 or not np.isin(a,(0,1,False,True)).all(): raise ValueError(name+' must be binary 2D')
    return a.astype(bool,copy=False)


def build_image_reference(gt,fov,uncertain,paths,*,alpha):
    """Build candidate references and per-path Omega denominators from GT paths."""
    gt=_check_binary(gt,'GT'); fov=_check_binary(fov,'FOV'); uncertain=_check_binary(uncertain,'U')
    if gt.shape!=fov.shape or gt.shape!=uncertain.shape or not fov.any(): raise ValueError('aligned nonempty GT/FOV/U required')
    if not math.isfinite(float(alpha)) or alpha<=0: raise ValueError('invalid fixed Train scale')
    omega=fov & ~uncertain
    b1=np.zeros(gt.shape,bool); b2=np.zeros(gt.shape,bool); path_records=[]
    flat_coords=[]; flat_width=[]; flat_arc=[]; flat_body=[]; offsets=[0]; candidates=[]; b1_point=[]
    for index,path in enumerate(paths):
        coords=np.asarray(path['coords'],dtype=np.int32)
        width=np.asarray(path['width_native'],dtype=np.float64); arc=np.asarray(path['arc_native'],dtype=np.float64)
        body=np.asarray(path['body'],dtype=bool)
        if coords.ndim!=2 or coords.shape[1]!=2 or len(coords)<1 or any(len(x)!=len(coords) for x in (width,arc,body)): raise ValueError('malformed path arrays')
        if not np.isfinite(width).all() or not np.isfinite(arc).all() or np.any(width<0) or (len(arc)>1 and np.any(np.diff(arc)<0)): raise ValueError('invalid path geometry')
        if np.any(coords[:,0]<0) or np.any(coords[:,0]>=gt.shape[0]) or np.any(coords[:,1]<0) or np.any(coords[:,1]>=gt.shape[1]): raise ValueError('path outside image')
        if np.any(~gt[coords[:,0],coords[:,1]]) or np.any(~fov[coords[:,0],coords[:,1]]): raise ValueError('GT path pixel outside GT/FOV')
        body_width=width[body]
        body_med=float(np.median(body_width*alpha)) if len(body_width) else None
        length=float((arc[-1]-arc[0])*alpha) if len(arc)>1 else 0.0
        closed=bool(path.get('closed',np.array_equal(coords[0],coords[-1])))
        is_open=not closed
        candidate=bool(is_open and len(coords)>1 and length>=LENGTH_LIMIT and body_med is not None and body_med<=WIDTH_LIMIT)
        legacy=path.get('legacy_record',{})
        if legacy and candidate!=bool(legacy['candidate_B2']):raise RuntimeError('legacy selection differs')
        point_b1=(width*alpha<=WIDTH_LIMIT) if candidate else np.zeros(len(coords),bool)
        if candidate:
            b2[coords[:,0],coords[:,1]]=True
            b1[coords[point_b1,0],coords[point_b1,1]]=True
        intervals=valid_runs(coords,omega) if candidate else []
        denom=len(np.unique(coords[omega[coords[:,0],coords[:,1]]],axis=0)) if candidate else 0
        b1_denom=len(np.unique(coords[point_b1 & omega[coords[:,0],coords[:,1]]],axis=0)) if candidate else 0
        path_records.append({**legacy,'branch_id':int(index),'n_points':int(len(coords)),'open':bool(is_open),
             'length_native':float(arc[-1]-arc[0]) if len(arc)>1 else 0.0,'length_reference':length,
             'body_width_reference_median':body_med,'candidate_B2':candidate,
             'B1_point_count':int(point_b1.sum()),'denominator':int(denom),'B1_denominator':int(b1_denom),'NA':bool(denom==0),
             'valid_intervals':int(len(intervals)),
              'valid_interval_lengths':[int(len(x)) for x in intervals]})
        flat_coords.append(coords); flat_width.append(width.astype(np.float32)); flat_arc.append(arc.astype(np.float32)); flat_body.append(body.astype(np.uint8))
        offsets.append(offsets[-1]+len(coords)); candidates.append(candidate); b1_point.extend(point_b1.tolist())
    if flat_coords:
        coords=np.concatenate(flat_coords); widths=np.concatenate(flat_width); arcs=np.concatenate(flat_arc); body_arr=np.concatenate(flat_body)
    else:
        coords=np.empty((0,2),np.int32); widths=np.empty(0,np.float32); arcs=np.empty(0,np.float32); body_arr=np.empty(0,np.uint8)
    skel=skeletonize(gt & fov,method='zhang')
    padded=np.pad((gt & fov).astype(bool),1,constant_values=False)
    dist=ndimage.distance_transform_edt(padded)[1:-1,1:-1].astype(np.float32)
    coarse=skel & ((2.0*dist.astype(float)*float(alpha))>WIDTH_LIMIT) & omega
    return {'shape':list(gt.shape),'alpha':float(alpha),'Omega':omega.astype(np.uint8),
        'B1_reference':b1.astype(np.uint8),'B2_reference':b2.astype(np.uint8),
        'coarse_vessel_reference':coarse.astype(np.uint8),'GT':gt.astype(np.uint8),
        'FOV':fov.astype(np.uint8),'U':uncertain.astype(np.uint8),
        'B1_unique_points':int(b1.sum()),'B2_unique_points':int(b2.sum()),
        'coarse_denominator':int(coarse.sum()),'paths':path_records,
        'path_arrays':{'coords':coords,'offsets':np.asarray(offsets,np.int32),
          'width_native':widths,'arc_native':arcs,'body':body_arr,
          'candidate_B2':np.asarray(candidates,np.uint8),
          'B1_selected_points':np.asarray(b1_point,np.uint8)}}


def _read_mask(path):
    a=np.asarray(Image.open(path).convert('L'))
    return a>0


def create_dev_references(output_dir):
    """Create and seal one GT-only reference package for the fixed 28 Dev IDs."""
    out=Path(output_dir).resolve()
    expected=(ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022').resolve()
    if out!=expected: raise ValueError('output directory differs from frozen protocol')
    refs=out/'references/dev'
    if (out/'SCIENCE_RELEASE.json').exists():raise RuntimeError('cannot regenerate released references')
    refs.mkdir(parents=True,exist_ok=True)
    if (refs/'manifest.json').exists():
        current=json.loads((refs/'manifest.json').read_text())
        if current.get('schema')==2 and current.get('reference_code_sha256')==file_sha(Path(__file__)):
            return current
        archive=refs/('manifest.before_sol_'+file_sha(refs/'manifest.json')[:12]+'.json')
        if not archive.exists():archive.write_bytes((refs/'manifest.json').read_bytes())
    protocol=json.loads(PROTOCOL.read_text())
    if file_sha(PROTOCOL)!=UNCERTAINTY_PROTOCOL_SHA: raise RuntimeError('frozen protocol hash mismatch')
    if not GRAPH_CODE.is_file(): raise FileNotFoundError('frozen GT graph implementation missing')
    characterize=graph_functions()
    split=json.loads(SPLIT.read_text()); names=split.get('dev_files',[])
    if len(names)!=28 or len(set(names))!=28: raise RuntimeError('fixed Dev ID list mismatch')
    original=ROOT/'docs/full_fov_thin_skeleton_bce_20261019/dev_reference_manifest.json'
    if file_sha(original)!=protocol['evidence_anchors_sha256'][str(original)]:raise RuntimeError('original Dev source manifest drift')
    original_rows=json.loads(original.read_text())['images'];names=[x['filename'] for x in original_rows]
    if len(names)!=28 or set(names)!=set(split['dev_files']):raise RuntimeError('Dev order/membership differs')
    byname={x['filename']:x for x in original_rows}
    records=[]
    for name in names:
        gt_path=GT_DIR/(name+'.png'); fov_path=DEV_FOV/name/'fov.png'; region_path=DEV_REGIONS/(name+'.npz')
        if not all(x.is_file() for x in (gt_path,fov_path,region_path)): raise FileNotFoundError(name)
        for role,path in [('GT',gt_path),('FOV',fov_path),('regions',region_path)]:
            expected=byname[name]['inputs'][role]
            if str(path)!=expected['path'] or file_sha(path)!=expected['sha256']:raise RuntimeError('Dev source drift:'+name+':'+role)
        gt=_read_mask(gt_path); fov=_read_mask(fov_path)
        with np.load(region_path,allow_pickle=False) as z:
            if 'U' not in z.files or 'far' not in z.files: raise RuntimeError('U/far reference missing: '+name)
            uncertain=np.asarray(z['U'],dtype=bool); far=np.asarray(z['far'],dtype=bool)
            old_regions={k:np.asarray(z[k],np.uint8) for k in ('internal','near','thin')}
        if gt.shape!=fov.shape or gt.shape!=uncertain.shape or gt.shape!=far.shape: raise ValueError('Dev mask dimensions differ: '+name)
        fov_area=int(fov.sum()); D=2*math.sqrt(fov_area/math.pi); alpha=ALPHA_NUMERATOR/D
        paths,graph_summary,sk,radius=native_paths(gt,fov,alpha,name,characterize)
        result=build_image_reference(gt,fov,uncertain,paths,alpha=alpha)
        legacy_flags=[bool(p['legacy_record']['candidate_B2']) for p in paths]
        new_flags=[bool(rec['candidate_B2']) for rec in result['paths']]
        if legacy_flags!=new_flags: raise RuntimeError('frozen graph/characterization rule drift: '+name)
        arrays={k:v for k,v in result['path_arrays'].items()}
        arrays.update({'Omega':result['Omega'],'B1_reference':result['B1_reference'],'B2_reference':result['B2_reference'],
                       'coarse_vessel_reference':result['coarse_vessel_reference'],'GT':result['GT'],'FOV':result['FOV'],'U':result['U'],'far':far.astype(np.uint8)})
        arrays.update(old_regions)
        arrays['full_S3']=(sk&(radius<=np.float32(1.5))&result['Omega'].astype(bool)).astype(np.uint8)
        arr_path=refs/(name+'.sol_'+file_sha(Path(__file__))[:12]+'.npz')
        if arr_path.exists():
            with np.load(arr_path,allow_pickle=False) as old:
                if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):raise RuntimeError('partial reference differs; retain for review')
        else:np.savez_compressed(arr_path,**arrays)
        records.append({'filename':name,'native_shape':result['shape'],'fov_pixels':fov_area,'D_native':D,'alpha':alpha,
             'gt_sha256':file_sha(gt_path),'fov_sha256':file_sha(fov_path),'U_regions_sha256':file_sha(region_path),
             'GT_path_graph_code_sha256':file_sha(GRAPH_CODE),'reference_npz':str(arr_path),'reference_npz_sha256':file_sha(arr_path),
             'graph_summary':graph_summary,'B1_unique_points':result['B1_unique_points'],'B2_unique_points':result['B2_unique_points'],
             'coarse_vessel_denominator':result['coarse_denominator'],
             'candidate_path_count':sum(x['candidate_B2'] for x in result['paths']),
             'B1_path_denominator':sum(x['B1_denominator'] for x in result['paths'] if x['candidate_B2']),
             'B2_path_denominator':sum(x['denominator'] for x in result['paths'] if x['candidate_B2']),
             'B1_valid_paths':sum(x['candidate_B2'] and x['B1_denominator']>0 for x in result['paths']),
             'B2_valid_paths':sum(x['candidate_B2'] and x['denominator']>0 for x in result['paths']),
             'B2_Omega_unique_points':int((arrays['B2_reference'].astype(bool)&arrays['Omega'].astype(bool)).sum()),
             'shared_anchor_extra_occurrences':sum(x['denominator'] for x in result['paths'])-int((arrays['B2_reference'].astype(bool)&arrays['Omega'].astype(bool)).sum()),
             'sources':{k:byname[name]['inputs'][k] for k in ('GT','FOV','regions')},'paths':result['paths']})
    manifest={'schema':2,'status':'SEALED_GT_ONLY_REFERENCE','protocol_path':str(PROTOCOL),
       'protocol_sha256':file_sha(PROTOCOL),'split_record_sha256':file_sha(SPLIT),
       'graph_code_path':str(GRAPH_CODE),'graph_code_sha256':file_sha(GRAPH_CODE),
       'fixed_dev_count':28,'dev_ids':names,'source_policy':'GT/FOV/U and GT-derived graph only; no RGB, probability, or model output opened',
       'path_rule':'frozen review_definition_v2 graph()/characterize() source; B2 full qualifying open path, B1 candidate points whose relative width <= 6',
       'alpha_rule':'911.1106643779144 / (2*sqrt(FOV_pixels/pi)); constant derived from fixed Train scale',
       'source_manifest_sha256':file_sha(original),'reference_code_sha256':file_sha(Path(__file__)),
       'images':records,'created_by':'maples_branch_science_reference.create_dev_references'}
    mpath=refs/'manifest.json'
    from retinal_m0.maples_g2_engineering import atomic_json
    atomic_json(mpath,manifest)
    return manifest
