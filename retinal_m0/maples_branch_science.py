"""CPU preparation and hard-gated runner for the frozen B1/B2 scientific task."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys
from pathlib import Path
import numpy as np

ROOT=Path('${RETINA_PROJECT_ROOT}')
CODE=ROOT/'code/retinal-vessel-study'
OUT=ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022'
PROTOCOL=ROOT/'docs/full_fov_thin_skeleton_bce_20261019/branch_review_20261021/scientific_b1_b2_2000_v1_20261022.json'
PROTOCOL_SHA='72fb5a23d11ac1d2750aeb0aa9addf7dfa4f9753bd10ebc21233b7d0b886dfbf'
A0=ROOT/'experiments/t0_tw_core_20261016/runs/seed_2026/T0'
SOURCE=ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/attempt_02'
STATUS=CODE/'STATUS.md'
A0_DEV=A0/'evaluation/step_002000'
DOC=ROOT/'docs/full_fov_thin_skeleton_bce_20261019'
GRAPH_CODE=SOURCE.parent/'review_definition_v2.py'
DEV_NAMES=[r['filename'] for r in json.loads((DOC/'dev_reference_manifest.json').read_text())['images']]
from retinal_m0.maples_g2_engineering import read_json,canonical_sha,lock,atomic_json
_ACTIVE=None

def environment_versions():
    import platform,torch,scipy,skimage,PIL
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda_build=torch.version.cuda,
      scipy=scipy.__version__,skimage=skimage.__version__,pillow=PIL.__version__)

def merge_source_pins(sources,additional):
    """An inherited expected digest is immutable; preparation cannot bless drift."""
    for path,digest in additional.items():
        if path in sources and sources[path]!=digest:raise RuntimeError('inherited source digest conflict:'+path)
    sources.update(additional)

def validate_config(cfg,method):
    baseline=read_json(A0/'config.json')
    allowed={'kind','phase','method','loss','lambda','safe_stop_seconds','protocol_sha256','execution_binding_id'}
    for key,value in baseline.items():
        if key not in allowed and (cfg.get(key)!=value or type(cfg.get(key))is not type(value)):raise RuntimeError('immutable training setting drift:'+key)
    if cfg.get('method')!=method or cfg.get('lambda')!=.03 or cfg.get('max_steps')!=2000 or cfg.get('safe_stop_seconds') is not None or cfg.get('protocol_sha256')!=PROTOCOL_SHA:raise RuntimeError('science configuration scope drift')
    return cfg


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.pending')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def validate_science_release(path,expected_protocol_sha,full=False):
    path=Path(path)
    if path.resolve()!=(OUT/'SCIENCE_RELEASE.json').resolve() or not path.is_file():raise RuntimeError('canonical science-only release missing')
    q=_assert_protocol();current=current_authorization()
    if current.get('state')!='RELEASED' or current.get('GPU_released') is not True or current.get('user_design_approved') is not True:raise RuntimeError('current science GPU authorization held')
    release=read_json(path)
    scope=dict(methods=['B1','B2'],seed=2026,**{'lambda':.03},updates_per_arm=2000,charged_update_cap=4000,official_Test_allowed=False,gpu1_allowed=False)
    if release.get('state')!='RELEASED' or release.get('phase')!='SCIENTIFIC_B1_B2_2000' or expected_protocol_sha!=PROTOCOL_SHA or release.get('protocol_sha256')!=PROTOCOL_SHA:raise RuntimeError('wrong science phase/protocol')
    for k,v in scope.items():
        if release.get(k)!=v:raise RuntimeError('science scope mismatch:'+k)
    if current.get('release_id')!=release.get('release_id') or current.get('release_file_sha256')!=file_sha(path) or current.get('protocol_sha256')!=PROTOCOL_SHA:raise RuntimeError('release/current STATUS mismatch')
    binding=read_json(OUT/'ready_binding.json');receipt=read_json(OUT/'cpu_review_receipt.json')
    bsha=canonical_sha(binding)
    if release.get('binding_sha256')!=bsha or current.get('binding_sha256')!=bsha or receipt.get('binding_sha256')!=bsha:raise RuntimeError('science execution binding mismatch')
    if release.get('ready_binding_file_sha256')!=file_sha(OUT/'ready_binding.json') or release.get('cpu_receipt_sha256')!=file_sha(OUT/'cpu_review_receipt.json') or not receipt.get('all_passed'):raise RuntimeError('CPU evidence/release mismatch')
    if receipt.get('status')!='CPU_VALIDATED_NOT_A_GPU_RELEASE' or binding.get('protocol_sha256')!=PROTOCOL_SHA or binding.get('phase')!='SCIENTIFIC_B1_B2_2000':raise RuntimeError('CPU receipt phase mismatch')
    required=set(q['evidence_anchors_sha256'])|set(q['A0_endpoint_probability_files_sha256'])
    required|={str(CODE/'retinal_m0'/n) for n in ['maples_branch_science.py','maples_branch_science_worker.py','maples_branch_science_reference.py','maples_branch_science_analysis.py']}
    required|={str(CODE/'tests/test_maples_branch_science.py'),str(OUT/'configs/B1.json'),str(OUT/'configs/B2.json'),str(OUT/'references/dev/manifest.json'),str(OUT/'pairing_8000_verification.json'),str(OUT/'a0_reuse_binding.json'),str(OUT/'train_target_reconciliation.json')}
    if not required.issubset(binding['sources']) or receipt.get('source_count')!=len(binding['sources']):raise RuntimeError('incomplete science source binding')
    if binding.get('environment')!=environment_versions():raise RuntimeError('bound software environment drift')
    for method in ['B1','B2']:validate_config(read_json(OUT/'configs'/(method+'.json')),method)
    for p,h in binding['sources'].items():
        pp=Path(p)
        if not pp.resolve().is_relative_to(ROOT.resolve()):raise RuntimeError('source path outside project')
        if full or pp.suffix=='.py' or pp==PROTOCOL or pp.parent==OUT/'configs':
            if file_sha(pp)!=h:raise RuntimeError('bound source drift:'+p)
    return binding



def _assert_protocol():
    if not PROTOCOL.is_file() or file_sha(PROTOCOL)!=PROTOCOL_SHA: raise RuntimeError('frozen scientific protocol missing or changed')
    d=json.loads(PROTOCOL.read_text())
    if d.get('protocol_id')!='P1_B1_B2_SCIENTIFIC_2000_V1_20261022': raise RuntimeError('protocol identity mismatch')
    return d


def _load_reference_manifest():
    from PIL import Image
    from retinal_m0.maples_branch_science_reference import file_sha,valid_runs
    from skimage.morphology import skeletonize
    from scipy.ndimage import distance_transform_edt
    q=_assert_protocol();path=OUT/'references/dev/manifest.json';m=read_json(path)
    refs=read_json(DOC/'dev_reference_manifest.json')['images']
    if m.get('schema')!=2 or m.get('protocol_sha256')!=PROTOCOL_SHA or m.get('dev_ids')!=DEV_NAMES or len(m['images'])!=28:
        raise RuntimeError('reference seal/protocol/order mismatch')
    if m['reference_code_sha256']!=file_sha(CODE/'retinal_m0/maples_branch_science_reference.py') or m['graph_code_sha256']!=q['evidence_anchors_sha256'][str(GRAPH_CODE)]:raise RuntimeError('reference source drift')
    for row,original in zip(m['images'],refs):
        if row['filename']!=original['filename']:raise RuntimeError('reference image order drift')
        for role in ('GT','FOV','regions'):
            if row['sources'][role]!=original['inputs'][role] or file_sha(row['sources'][role]['path'])!=row['sources'][role]['sha256']:raise RuntimeError('reference source file drift')
        p=Path(row['reference_npz'])
        if not p.resolve().is_relative_to((OUT/'references/dev').resolve()) or file_sha(p)!=row['reference_npz_sha256']:raise RuntimeError('reference artifact drift')
        with np.load(p,allow_pickle=False) as archive:
            z={k:archive[k] for k in archive.files}
            omega=z['Omega'].astype(bool);gt=z['GT'].astype(bool);f=z['FOV'].astype(bool);u=z['U'].astype(bool)
            if list(gt.shape)!=row['native_shape'] or not np.array_equal(omega,f&~u):raise RuntimeError('reference dimensions/Omega drift')
            for role,array in [('GT',gt),('FOV',f)]:
                with Image.open(row['sources'][role]['path']) as im:actual=np.asarray(im.convert('L'))>0
                if not np.array_equal(actual,array):raise RuntimeError('reference array differs from original source:'+role)
            with np.load(row['sources']['regions']['path'],allow_pickle=False) as old:
                for key in ('U','internal','near','far','thin'):
                    if not np.array_equal(old[key],z[key]):raise RuntimeError('old regional array changed:'+key)
            expected_alpha=911.1106643779144/(2*math.sqrt(int(f.sum())/math.pi))
            if not math.isclose(row['alpha'],expected_alpha,rel_tol=0,abs_tol=1e-12):raise RuntimeError('Train-derived scale mismatch')
            c=z['coords'].astype(np.int64);off=z['offsets'].astype(np.int64);chosen=z['candidate_B2'].astype(bool);s=z['B1_selected_points'].astype(bool)
            if len(off)!=len(row['paths'])+1 or len(chosen)!=len(row['paths']) or off[0]!=0 or off[-1]!=len(c) or np.any(np.diff(off)<=0) or len(s)!=len(c):raise RuntimeError('reference path offsets drift')
            if np.any(c<0) or np.any(c[:,0]>=gt.shape[0]) or np.any(c[:,1]>=gt.shape[1]) or np.any(~(gt&f)[c[:,0],c[:,1]]):raise RuntimeError('path exterior to GT/FOV')
            union1=np.zeros_like(f);union2=np.zeros_like(f);multip=np.zeros(gt.shape,np.int32);d1=d2=n1=n2=0
            for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
                x=c[a:b];r=row['paths'][i];k=omega[x[:,0],x[:,1]];sel=s[a:b]
                if bool(chosen[i])!=bool(r['candidate_B2']):raise RuntimeError('candidate flags mismatch')
                expected_selection=(z['width_native'][a:b].astype(float)*row['alpha']<=6)&chosen[i]
                if not np.array_equal(sel,expected_selection):raise RuntimeError('B1 support mismatch')
                body=z['body'][a:b].astype(bool);width=z['width_native'][a:b].astype(float)
                length=float(np.linalg.norm(np.diff(x.astype(float),axis=0),axis=1).sum())
                candidate=bool(r['open'] and len(x)>1 and length*row['alpha']>=20 and body.any() and np.median(width[body]*row['alpha'])<=6)
                if candidate!=bool(chosen[i]):raise RuntimeError('reference GT selection rule mismatch')
                denom=len(np.unique(x[k],axis=0)) if chosen[i] else 0
                den1=len(np.unique(x[k&sel],axis=0)) if chosen[i] else 0
                if denom!=r['denominator'] or den1!=r['B1_denominator']:raise RuntimeError('path denominator mismatch')
                if chosen[i]:
                    union2[x[:,0],x[:,1]]=1;union1[x[sel,0],x[sel,1]]=1
                    distinct=np.unique(x[k],axis=0)
                    if len(distinct):multip[distinct[:,0],distinct[:,1]]+=1
                d1+=den1;d2+=denom;n1+=den1>0;n2+=denom>0
            if not np.array_equal(union1,z['B1_reference']) or not np.array_equal(union2,z['B2_reference']):raise RuntimeError('target union mismatch')
            if (d1,d2,n1,n2)!=(row['B1_path_denominator'],row['B2_path_denominator'],row['B1_valid_paths'],row['B2_valid_paths']):raise RuntimeError('image denominator mismatch')
            sk=skeletonize(gt&f,method='zhang');rad=distance_transform_edt(np.pad(gt&f,1))[1:-1,1:-1].astype(np.float32)
            coarse=sk&(2*rad.astype(float)*row['alpha']>6)&omega
            if not np.array_equal(coarse,z['coarse_vessel_reference']) or int(coarse.sum())!=row['coarse_vessel_denominator']:raise RuntimeError('coarse reference mismatch')
            if not np.array_equal(sk&(rad<=np.float32(1.5))&omega,z['full_S3']):raise RuntimeError('S3 reference mismatch')
            if not np.array_equal(z['internal']+z['near']+z['far'],omega.astype(np.uint8)):raise RuntimeError('old regional partition drift')
            row['B2_Omega_unique_points']=int((union2&omega).sum())
            row['shared_anchor_extra_occurrences']=int(np.maximum(multip-1,0).sum())
    return m



def verify_a0_reuse():
    import torch
    from retinal_m0.state import state_digest
    from retinal_m0.maples_branch_science_worker import exact_equal
    m=_load_reference_manifest();q=_assert_protocol();summary=read_json(A0/'summary.json');runbinding=read_json(A0/'run_binding.json');config=read_json(A0/'config.json')
    if summary.get('status')!='passed' or summary.get('completed_steps')!=2000 or summary.get('next_global_draw')!=8000 or summary.get('seed')!=2026:raise RuntimeError('A0 incomplete')
    if summary['run_binding']!=runbinding or file_sha(A0/'config.json')!=runbinding['config_sha256'] or file_sha(A0/'config.json')!=q['evidence_anchors_sha256'][str(A0/'config.json')]:raise RuntimeError('A0 configuration provenance drift')
    milestone=A0/'step_002000.pt';last=A0/'last.pt';evaluation=read_json(A0_DEV/'metrics.json')
    if file_sha(last)!=summary['checkpoint_sha256'] or file_sha(milestone)!=evaluation['checkpoint_sha256'] or evaluation['run_binding']!=runbinding:raise RuntimeError('A0 endpoint file/probability binding drift')
    a=torch.load(milestone,map_location='cpu',weights_only=True);b=torch.load(last,map_location='cpu',weights_only=True)
    for k in ('model','optimizer','scaler','rng','step','next_global_draw'):
        if not exact_equal(a[k],b[k]):raise RuntimeError('A0 last and milestone states differ:'+k)
    for k in ('model','optimizer','scaler','rng'):
        if hashlib.sha256(state_digest(a[k]).encode()).hexdigest()!=summary[k+'_digest']:raise RuntimeError('A0 original state digest mismatch:'+k)
    if a['step']!=2000 or a['next_global_draw']!=8000 or a['binding']!=runbinding:raise RuntimeError('A0 saved cursor/binding drift')
    del a,b
    files={p.stem:p for p in (A0_DEV/'probabilities').glob('*.npy')}
    if set(files)!=set(DEV_NAMES) or len(files)!=28:raise RuntimeError('A0 probability membership drift')
    hashes=[]
    for row in m['images']:
        p=files[row['filename']];expected=q['A0_endpoint_probability_files_sha256'][str(p)]
        if file_sha(p)!=expected:raise RuntimeError('A0 frozen probability changed')
        x=np.load(p,mmap_mode='r',allow_pickle=False)
        if list(x.shape)!=row['native_shape'] or x.dtype!=np.float32 or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise RuntimeError('invalid A0 probability')
        hashes.append(dict(filename=row['filename'],path=str(p),sha256=expected))
    result=dict(status='A0_EXISTING_ENDPOINT_REUSE_VERIFIED_NO_INFERENCE',method='A0',completed_steps=2000,next_global_draw=8000,
      milestone_sha256=file_sha(milestone),last_sha256=file_sha(last),full_states_identical=True,
      summary_digest_algorithm='SHA256(state_digest(value).encode())',reference_manifest_sha256=file_sha(OUT/'references/dev/manifest.json'),
      probability_maps=hashes,new_model_forward_calls=0)
    write_json(OUT/'a0_reuse_binding.json',result);return result



def verify_train_targets():
    from PIL import Image
    from retinal_data.maples_fov import candidate_fov
    from retinal_m0.maples_core import _decode_label_image
    from retinal_m0.maples_branch_science_reference import native_paths,graph_functions,ALPHA_NUMERATOR
    q=_assert_protocol();manifest_path=DOC/'train_manifest.json'
    if file_sha(manifest_path)!=q['evidence_anchors_sha256'][str(manifest_path)]:raise RuntimeError('Train manifest drift')
    images=read_json(manifest_path)['rows'];artifacts=read_json(SOURCE/'artifact_sha256.json');characterize=graph_functions();records=[]
    if len(images)!=110 or [r['filename'] for r in images]!=q['training']['baseline_config']['train_sample_filenames']:raise RuntimeError('Train order drift')
    for row in images:
        name=row['filename']
        for role in ('RGB','V'):
            if file_sha(row[role]['path'])!=row[role]['sha256']:raise RuntimeError('Train source drift')
        with Image.open(row['RGB']['path']) as im:rgb=np.asarray(im.convert('RGB'),np.uint8)
        with Image.open(row['V']['path']) as im:v=_decode_label_image(im).astype(bool)
        f=candidate_fov(rgb).astype(bool);alpha=ALPHA_NUMERATOR/(2*math.sqrt(int(f.sum())/math.pi))
        paths,_,_,_=native_paths(v,f,alpha,name,characterize)
        b1=np.zeros_like(f);b2=np.zeros_like(f);coords=[];width=[];arc=[];body=[];offs=[0]
        for p in paths:
            c=p['coords'];w=p['width_native']
            if p['legacy_record']['candidate_B2']:
                b2[c[:,0],c[:,1]]=1;k=w*alpha<=6;b1[c[k,0],c[k,1]]=1
            coords.append(c);width.append(w);arc.append(p['arc_native']);body.append(p['body']);offs.append(offs[-1]+len(c))
        rel='candidate_masks_NOT_FOR_TRAINING/'+name+'.npz';mp=SOURCE/rel;gp=SOURCE/'path_data'/(name+'.npz')
        if file_sha(mp)!=artifacts[rel] or file_sha(gp)!=artifacts['path_data/'+name+'.npz']:raise RuntimeError('old target/path bytes changed')
        with np.load(mp,allow_pickle=False) as z:
            if not np.array_equal(b1,z['B1_reference']) or not np.array_equal(b2,z['B2_candidate']):raise RuntimeError('Train target reconstruction differs')
        joined=dict(coords=np.concatenate(coords).astype(np.int32),offsets=np.asarray(offs,np.int32),width_native=np.concatenate(width).astype(np.float32),arc_native=np.concatenate(arc).astype(np.float32),body=np.concatenate(body).astype(np.uint8))
        with np.load(gp,allow_pickle=False) as z:
            if any(not np.array_equal(joined[k],z[k]) for k in joined):raise RuntimeError('Train path arrays differ')
        records.append(dict(filename=name,mask_path=str(mp),mask_sha256=file_sha(mp),path_data_sha256=file_sha(gp),B1_points=int(b1.sum()),B2_points=int(b2.sum())))
        if len(records)%10==0:print('Train targets',len(records),flush=True)
    receipt=dict(status='ALL_110_TRAIN_TARGETS_AND_PATHS_EXACT',image_count=110,records=records,new_model_forward_calls=0)
    write_json(OUT/'train_target_reconciliation.json',receipt);return receipt



def verify_8000_pairs():
    from retinal_m0.maples_core import CorePatches
    from retinal_m0.maples_branch_bce import transform_native_target
    _assert_protocol();cfg=read_json(A0/'config.json');data=CorePatches(cfg,'T0')
    expected=[json.loads(x) for x in (A0/'pairing.jsonl').read_text().splitlines() if x.strip()]
    trainrows=[json.loads(x) for x in (A0/'train.jsonl').read_text().splitlines() if x.strip()]
    with (SOURCE/'crops_8000_and_analytic_gradients.csv').open() as f:crops=list(csv.DictReader(f))
    if len(expected)!=2000 or len(trainrows)!=2000 or len(crops)!=8000:raise RuntimeError('A0 complete stream missing')
    records=[];masks={};artifacts=read_json(SOURCE/'artifact_sha256.json')
    for draw,row in enumerate(crops):
        step=draw//4+1;slot=draw%4;b=expected[step-1];actual=data[draw]
        geom=actual['geometry'].tolist();name=actual['filename']
        if b['step']!=step or b['draws'][slot]!=draw or b['filenames'][slot]!=name or b['geometry'][slot]!=geom or b['input_sha256'][slot]!=actual['input_sha256']:raise RuntimeError('actual T0 RGB/V/FOV/geometry fingerprint drift:'+str(draw))
        if trainrows[step-1]['step']!=step:raise RuntimeError('A0 train log cursor drift')
        if int(row['draw'])!=draw or row['filename']!=name or [int(row[k]) for k in ('top','left','hflip','vflip','ccw_rot90_quarters')]!=geom:raise RuntimeError('old crop plan drift')
        if name not in masks:
            rel='candidate_masks_NOT_FOR_TRAINING/'+name+'.npz';p=SOURCE/rel
            if file_sha(p)!=artifacts[rel]:raise RuntimeError('mask source drift')
            with np.load(p,allow_pickle=False) as z:masks[name]=(z['B1_reference'].copy(),z['B2_candidate'].copy())
        targets={};counts={}
        for method,a,countname in [('B1',masks[name][0],'N_B1_reference'),('B2',masks[name][1],'N_B2')]:
            s=transform_native_target(a,top=geom[0],left=geom[1],size=512,hflip=bool(geom[2]),vflip=bool(geom[3]),quarter_turns=geom[4])
            if int(s.sum())!=int(row[countname]) or np.any(s.astype(bool)&~(actual['mask'][0].numpy().astype(bool)&actual['fov'][0].numpy().astype(bool))):raise RuntimeError('actual target count/containment mismatch')
            targets[method]=hashlib.sha256(s.tobytes()).hexdigest();counts[method+'_count']=int(s.sum())
        records.append(dict(step=step,slot=slot,draw=draw,filename=name,geometry=geom,input_sha256=actual['input_sha256'],target_sha256=targets,**counts))
        if (draw+1)%1000==0:print('Actual paired crops',draw+1,flush=True)
    receipt=dict(status='8000_ACTUAL_INPUTS_AND_TARGETS_EXACT',draw_count=8000,batches=2000,rows=records,model_forward_calls=0,source_artifact_sha256=file_sha(SOURCE/'artifact_sha256.json'))
    write_json(OUT/'pairing_8000_verification.json',receipt);return receipt



def run_cpu_preflight(tests_log=None):
    import re,torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
    if (OUT/'SCIENCE_RELEASE.json').exists():raise RuntimeError('do not rebuild released evidence')
    q=_assert_protocol();m=_load_reference_manifest()
    a0=read_json(OUT/'a0_reuse_binding.json');train=read_json(OUT/'train_target_reconciliation.json');pairs=read_json(OUT/'pairing_8000_verification.json')
    if not a0.get('full_states_identical') or train.get('image_count')!=110 or pairs.get('draw_count')!=8000 or pairs.get('status')!='8000_ACTUAL_INPUTS_AND_TARGETS_EXACT':raise RuntimeError('actual CPU data receipts incomplete')
    tests=Path(tests_log or OUT/'sol_repair_20261022/attempt_01/combined_tests.log').resolve()
    if not tests.is_relative_to(OUT.resolve()):raise RuntimeError('test log outside science phase')
    text=tests.read_text();n=re.findall(r'Ran (\d+) tests?',text)
    if not n or int(n[-1])<20 or not text.rstrip().endswith('OK'):raise RuntimeError('complete current CPU tests have not passed')
    configs=OUT/'configs';configs.mkdir(exist_ok=True)
    for method in ['B1','B2']:
        cfg=dict(q['training']['baseline_config']);cfg.update(kind='P1_BRANCH_SCIENTIFIC_V1',phase='single_seed_exploratory_fixed2000',method=method,loss='original_BCE_Dice_plus_ownN_positive_skeleton_BCE',**{'lambda':.03},safe_stop_seconds=None,protocol_sha256=PROTOCOL_SHA,execution_binding_id='branch_science_20261022')
        write_json(configs/(method+'.json'),cfg)
        validate_config(cfg,method)
    sources=dict(q['evidence_anchors_sha256']);merge_source_pins(sources,q['A0_endpoint_probability_files_sha256'])
    g2=read_json(SOURCE.parent/'g2_engineering_v3/lambda003_revision_v4/ready_binding.json')
    merge_source_pins(sources,g2['sources'])
    a0binding_path=A0.parents[2]/'execution_binding.json';old=read_json(a0binding_path)
    if file_sha(a0binding_path)!=read_json(A0/'run_binding.json')['execution_binding_sha256']:raise RuntimeError('original A0 execution binding drift')
    merge_source_pins(sources,{str(CODE/k):v for k,v in old['sources'].items()})
    merge_source_pins(sources,old['protected_source_hashes'])
    merge_source_pins(sources,{str(a0binding_path):file_sha(a0binding_path)})
    versions=environment_versions()
    for key in ('python','torch','numpy'):
        if versions[key]!=g2['versions'][key]:raise RuntimeError('original G2 software baseline drift:'+key)
    for row in read_json(DOC/'train_manifest.json')['rows']:
        for role in ('RGB','V'):merge_source_pins(sources,{row[role]['path']:row[role]['sha256']})
    artifacts=read_json(SOURCE/'artifact_sha256.json')
    for row in train['records']:
        for rel in ['candidate_masks_NOT_FOR_TRAINING/'+row['filename']+'.npz','path_data/'+row['filename']+'.npz']:merge_source_pins(sources,{str(SOURCE/rel):artifacts[rel]})
    for row in m['images']:
        merge_source_pins(sources,{row['reference_npz']:row['reference_npz_sha256']})
        for rec in row['sources'].values():merge_source_pins(sources,{rec['path']:rec['sha256']})
    original_refs=read_json(DOC/'dev_reference_manifest.json')['images']
    for row in original_refs:merge_source_pins(sources,{row['inputs']['RGB']['path']:row['inputs']['RGB']['sha256']})
    paths=[OUT/n for n in ['a0_reuse_binding.json','train_target_reconciliation.json','pairing_8000_verification.json','references/dev/manifest.json']]
    paths+=[CODE/'retinal_m0'/n for n in ['maples_branch_science.py','maples_branch_science_worker.py','maples_branch_science_reference.py','maples_branch_science_analysis.py','maples_g2_engineering.py','state.py','model.py','inference.py']]
    paths+=[CODE/'tests/test_maples_branch_science.py',A0/'summary.json',A0/'last.pt',tests,*configs.glob('*.json')]
    paths += [CODE/'retinal_data/maples_native.py',CODE/'retinal_data/maples_fov.py',ROOT/'data/maples_handoff_20261008/data_manifest.csv',ROOT/'data/maples_handoff_20261008/split_record.json',CODE/'configs/development_v1.json',CODE/'releases/development_v1/normalization.json']
    for path in paths:merge_source_pins(sources,{str(path):file_sha(path)})
    for path,h in sources.items():
        if file_sha(path)!=h:raise RuntimeError('preflight protected source drift:'+path)
    binding=dict(schema=1,phase='SCIENTIFIC_B1_B2_2000',protocol_sha256=PROTOCOL_SHA,methods=['B1','B2'],seed=2026,**{'lambda':.03},charged_update_cap=4000,environment=versions,sources=sources)
    write_json(OUT/'ready_binding.json',binding)
    receipt=dict(status='CPU_VALIDATED_NOT_A_GPU_RELEASE',all_passed=True,tests_passed=int(n[-1]),tests_log=str(tests),tests_log_sha256=file_sha(tests),binding_sha256=canonical_sha(binding),
      Dev_valid_images=sum(r['B2_valid_paths']>0 for r in m['images']),Dev_valid_paths=sum(r['B2_valid_paths'] for r in m['images']),Dev_path_position_occurrences=sum(r['B2_path_denominator'] for r in m['images']),
      source_count=len(sources),retinal_model_forward_calls=0,CUDA_initialized=False,GPU_released=False)
    write_json(OUT/'cpu_review_receipt.json',receipt);return receipt



def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for n in ['prepare-reference','verify-a0-reuse','verify-train-targets','verify-pairing','cpu-preflight','analyze-endpoint']:g.add_argument('--'+n,action='store_true')
    g.add_argument('--run',nargs='?',const='all',choices=['B1','B2','all']);g.add_argument('--launch',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--tests-log',type=Path);p.add_argument('--output',type=Path,default=OUT)
    a=p.parse_args()
    if a.output.resolve()!=OUT.resolve():raise SystemExit('output path frozen')
    if a.launch:result=launch_authorized(resume=a.resume)
    elif a.run:result=run_authorized(a.run,resume=a.resume)
    else:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        if a.prepare_reference:
            from retinal_m0.maples_branch_science_reference import create_dev_references
            result=create_dev_references(OUT);result={'status':result['status'],'images':len(result['images'])}
        elif a.verify_a0_reuse:result=verify_a0_reuse()
        elif a.verify_train_targets:result=verify_train_targets()
        elif a.verify_pairing:result=verify_8000_pairs();result={k:v for k,v in result.items() if k!='rows'}
        elif a.cpu_preflight:result=run_cpu_preflight(a.tests_log)
        else:result=analyze_endpoint()
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))



def current_authorization():
    s=STATUS.read_text();a='<!-- P1_SCIENTIFIC_2000_CURRENT -->';b='<!-- /P1_SCIENTIFIC_2000_CURRENT -->'
    if s.count(a)!=1 or s.count(b)!=1:raise RuntimeError('unique current science authorization required')
    return json.loads(s.split(a)[1].split(b)[0].strip())

def clean_pause_ready(state,runs):
    if state.get('compute_pending') is not None or any(t.get('pending') or t.get('retry_cursor') for t in state['tracks'].values()):return False
    # A signal during first-step setup must wait for that first complete commit.
    # Before setup (no output directory) it can stop without consuming any work.
    return all(not (Path(runs)/m).exists() or (Path(runs)/m/'last.pt').is_file() for m in ['B1','B2'])

def verify_active_session():
    import hashlib
    import time
    uuid='${GPU_UUID}'
    if _ACTIVE is None or _ACTIVE['pid']!=os.getpid() or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:raise RuntimeError('no parent-bound GPU0 session')
    launch=read_json(OUT/'science_launch.json')
    if launch['pid']!=os.getpid() or launch['token_sha256']!=hashlib.sha256(_ACTIVE['token'].encode()).hexdigest():raise RuntimeError('science lease/token mismatch')
    binding=validate_science_release(OUT/'SCIENCE_RELEASE.json',PROTOCOL_SHA,full=False)
    if _ACTIVE.get('stop_requested'):
        from retinal_m0.maples_branch_science_worker import BudgetLedger,CleanPause
        state=BudgetLedger(OUT/'science_budget.json',canonical_sha(binding)).snapshot()
        if clean_pause_ready(state,OUT/'runs'):raise CleanPause('requested clean boundary pause')
    if time.monotonic()-_ACTIVE.get('heartbeat_at',-1e9)>=30:
        write_json(OUT/'heartbeat.json',dict(status='RUNNING',pid=os.getpid(),time=time.time(),binding_sha256=canonical_sha(binding)))
        _ACTIVE['heartbeat_at']=time.monotonic()
    return binding

def launch_authorized(resume=False):
    import time
    validate_science_release(OUT/'SCIENCE_RELEASE.json',PROTOCOL_SHA,full=True)
    with lock(OUT/'LAUNCH_REQUEST.lock'):
        request=OUT/'launch_request.json'
        if request.exists():
            old=read_json(request)
            try:os.kill(old['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('previous launcher PID still exists; no duplicate launch')
            if not resume:raise RuntimeError('prior launch exists; explicit exact resume required')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='${GPU_UUID}',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
        logfile=OUT/('controller_'+str(time.time_ns())+'.log')
        command=[sys.executable,'-B','-m','retinal_m0.maples_branch_science','--run']+(['--resume'] if resume else [])
        with logfile.open('xb') as log:
            child=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(status='CONTROLLER_DISPATCHED_NOT_COMPLETION',pid=child.pid,log=str(logfile),time=time.time(),resume=resume)
        write_json(request,record);return record

def run_authorized(method='all',resume=False):
    global _ACTIVE
    import secrets,time,signal
    # Strict authorization precedes CUDA queries, locks, model construction and budget creation.
    binding=validate_science_release(OUT/'SCIENCE_RELEASE.json',PROTOCOL_SHA,full=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='${GPU_UUID}':raise RuntimeError('exact GPU0 mask required')
    from retinal_m0.maples_branch_science_worker import BudgetLedger,run_scientific_training
    from retinal_m0.maples_g1_runtime import probe_resources
    with lock(OUT/'SCIENCE_RUNTIME.lock'),lock(ROOT/'experiments/a14_gpu0.lock'):
        validate_science_release(OUT/'SCIENCE_RELEASE.json',PROTOCOL_SHA,full=True)
        resources=probe_resources(require_idle=True)
        if resources['free_disk_bytes']<20*1024**3:raise RuntimeError('less than20GiB free')
        ledger=BudgetLedger(OUT/'science_budget.json',canonical_sha(binding))
        prior=(OUT/'science_budget.json').exists() or (OUT/'science_launch.json').exists() or (OUT/'runs').exists()
        if prior and not resume:raise RuntimeError('existing scientific work: explicit exact resume required')
        if not prior and resume:raise RuntimeError('nothing to resume')
        if not prior:ledger.initialize_isolated()
        state=ledger.snapshot()
        if state['status']!='READY' or state.get('compute_pending') or any(t.get('pending') for t in state['tracks'].values()):raise RuntimeError('unsafe outstanding compute/update; no automatic recovery')
        token=secrets.token_hex(32);_ACTIVE=dict(pid=os.getpid(),token=token,stop_requested=False,heartbeat_at=-1e9)
        prior_signals={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        def request_pause(*_):
            if _ACTIVE is not None:_ACTIVE['stop_requested']=True
        for sig in prior_signals:signal.signal(sig,request_pause)
        write_json(OUT/'science_launch.json',dict(pid=os.getpid(),token_sha256=hashlib.sha256(token.encode()).hexdigest(),binding_sha256=canonical_sha(binding),resources=resources,started_at=time.time(),resume=resume))
        try:
            outputs=[run_scientific_training(m,OUT) for m in (['B1','B2'] if method=='all' else [method])]
            complete=all(ledger.snapshot()['tracks'].get(m,{}).get('verified_successes')==2000 for m in ['B1','B2'])
            result=dict(status='SCIENCE_MATRIX_COMPLETE' if complete else 'AUTHORIZED_ARM_COMPLETE',arms=outputs,ledger=ledger.snapshot())
            if complete:result['analysis']=analyze_endpoint()
            target=OUT/('completion.json' if complete else 'arm_progress.json')
            if not target.exists():write_json(target,result)
            write_json(OUT/'heartbeat.json',dict(status=result['status'],pid=os.getpid(),time=time.time()))
            return result
        except BaseException as e:
            from retinal_m0.maples_branch_science_worker import CleanPause
            if isinstance(e,CleanPause):
                result=dict(status='CLEAN_PAUSE_COMMITTED',ledger=ledger.snapshot(),pid=os.getpid(),time=time.time())
                write_json(OUT/'pause.json',result);write_json(OUT/'heartbeat.json',result);return result
            s=ledger.snapshot()
            for name,t in s['tracks'].items():
                if t.get('pending'):ledger.mark_crash_uncertain(name)
            failure=dict(status='FAILED',error=str(e),ledger=ledger.snapshot(),pid=os.getpid(),time=time.time())
            write_json(OUT/('runtime_failure_'+str(time.time_ns())+'.json'),failure)
            write_json(OUT/'runtime_failure.json',failure);write_json(OUT/'heartbeat.json',failure)
            raise
        finally:
            for sig,handler in prior_signals.items():signal.signal(sig,handler)
            _ACTIVE=None

def verify_completed_evaluation(method,step):
    dest=OUT/'runs'/method/'evaluation'/f'step_{step:06d}'
    mp=dest/'metrics.json'
    if not mp.exists():return False
    d=read_json(mp)
    if d.get('method')!=method or d.get('step')!=step or d.get('threshold')!=.5 or d.get('reference_manifest_sha256')!=file_sha(OUT/'references/dev/manifest.json'):raise RuntimeError('evaluation binding drift')
    if d.get('checkpoint_sha256')!=file_sha(OUT/'runs'/method/f'step_{step:06d}.pt'):raise RuntimeError('evaluation model drift')
    if [r['filename'] for r in d['images']]!=DEV_NAMES:raise RuntimeError('evaluation image set/order drift')
    for r in d['images']:
        if file_sha(dest/'probabilities'/(r['filename']+'.npy'))!=r['probability_sha256']:raise RuntimeError('completed probability drift')
    return True

def evaluate_checkpoint(model,method,step,binding):
    import time,torch
    from PIL import Image
    from retinal_m0.inference import tiled_probability
    from retinal_m0.maples_branch_science_analysis import evaluate_image
    verify_active_session();validate_science_release(OUT/'SCIENCE_RELEASE.json',PROTOCOL_SHA,full=True)
    if method not in ['B1','B2'] or step not in range(200,2001,200):raise RuntimeError('unplanned evaluation')
    reference=read_json(OUT/'references/dev/manifest.json');source={r['filename']:r for r in read_json(DOC/'dev_reference_manifest.json')['images']}
    checkpoint=OUT/'runs'/method/f'step_{step:06d}.pt';checkpoint_sha=file_sha(checkpoint)
    dest=OUT/'runs'/method/'evaluation'/f'step_{step:06d}';dest.mkdir(parents=True,exist_ok=True)
    pd=dest/'probabilities';pd.mkdir(exist_ok=True);mp=dest/'metrics.json'
    if verify_completed_evaluation(method,step):return read_json(mp)
    counters=OUT/'evaluation_budget.json'
    if not counters.exists():write_json(counters,dict(schema=1,slots={}))
    budget=read_json(counters);key=method+':'+str(step)
    allowed={m+':'+str(i) for m in ['B1','B2'] for i in range(200,2001,200)}
    if set(budget['slots'])-allowed:raise RuntimeError('unapproved evaluation slot')
    slot=budget['slots'].setdefault(key,dict(checkpoint_sha256=checkpoint_sha,images={}))
    if slot['checkpoint_sha256']!=checkpoint_sha:raise RuntimeError('evaluation ledger checkpoint drift')
    norm=read_json(CODE/'releases/development_v1/normalization.json');mu=torch.tensor(norm['mean'])[:,None,None];sd=torch.tensor(norm['std'])[:,None,None]
    was_training=model.training;model.eval();records=[]
    try:
        for row in reference['images']:
            verify_active_session();name=row['filename'];destp=pd/(name+'.npy');state=slot['images'].get(name)
            if state and state.get('status')=='COMPLETE':
                if file_sha(destp)!=state['probability_sha256']:raise RuntimeError('reused probability digest drift')
                prob=np.load(destp,allow_pickle=False)
            else:
                inp=source[name]['inputs']['RGB']
                if file_sha(inp['path'])!=inp['sha256']:raise RuntimeError('Dev RGB drift')
                with Image.open(inp['path']) as im:raw=np.asarray(im.convert('RGB')).copy()
                with np.load(row['reference_npz'],allow_pickle=False) as z:fov=z['FOV'].copy()
                image=(torch.from_numpy(raw.transpose(2,0,1).copy()).float()/255-mu)/sd
                image*=torch.from_numpy(fov[None]).float()
                state=slot['images'].setdefault(name,dict(status='INCOMPLETE',attempts=0,forward_calls=0,tiles=0))
                state['attempts']+=1;write_json(counters,budget)
                def before_forward(module,inputs):
                    verify_active_session();state['forward_calls']+=1;state['tiles']+=int(inputs[0].shape[0]);write_json(counters,budget)
                hook=model.register_forward_pre_hook(before_forward)
                try:p,tiles=tiled_probability(model,image,patch=512,stride=256,batch_size=4,device='cuda',amp=False)
                finally:hook.remove()
                prob=p.numpy().astype(np.float32);tmp=destp.with_name(destp.name+'.pending')
                with tmp.open('wb') as f:np.save(f,prob,allow_pickle=False);f.flush();os.fsync(f.fileno())
                os.replace(tmp,destp);state.update(status='COMPLETE',probability_sha256=file_sha(destp));write_json(counters,budget)
            with np.load(row['reference_npz'],allow_pickle=False) as z:metrics=evaluate_image(prob,{k:z[k] for k in z.files},method)
            records.append(dict(filename=name,probability_sha256=file_sha(destp),**metrics))
        result=dict(method=method,step=step,threshold=.5,checkpoint_sha256=checkpoint_sha,reference_manifest_sha256=file_sha(OUT/'references/dev/manifest.json'),images=records)
        write_json(mp,result);return result
    finally:model.train(was_training)

def analyze_endpoint():
    from retinal_m0.maples_branch_science_analysis import evaluate_image,metric_values,paired_analysis,change_summary,path_strata
    for method in ['B1','B2']:
        for step in range(200,2001,200):
            if not verify_completed_evaluation(method,step):raise RuntimeError('planned evaluation incomplete')
    m=_load_reference_manifest();rows=[];all_records={method:[] for method in ['A0','B1','B2']}
    for row in m['images']:
        name=row['filename']
        with np.load(row['reference_npz'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        sample=dict(filename=name,native_shape=row['native_shape'],fixed_primary_valid=row['B2_valid_paths']>0)
        full=evaluate_image(np.ones(row['native_shape'],np.float32),arrays,'A0')
        sample['fixed_valid_metrics']={k:v is not None for k,v in metric_values(full).items()}
        for method in ['A0','B1','B2']:
            p=(A0_DEV/'probabilities'/(name+'.npy')) if method=='A0' else (OUT/'runs'/method/'evaluation/step_002000/probabilities'/(name+'.npy'))
            if not p.exists():raise RuntimeError('terminal probability absent:'+method+':'+name)
            if method=='A0' and file_sha(p)!=_assert_protocol()['A0_endpoint_probability_files_sha256'][str(p)]:raise RuntimeError('A0 endpoint drift')
            rec=evaluate_image(np.load(p,allow_pickle=False),arrays,method);sample[method]=metric_values(rec)
            for path_record in rec['path_records']+rec['excluded_path_records']:
                metadata=row['paths'][path_record['branch_id']]
                path_record.update({k:v for k,v in metadata.items() if k not in path_record})
            all_records[method].append(dict(filename=name,native_shape=row['native_shape'],**rec))
        rows.append(sample)
    result=paired_analysis(rows);result['images']=rows
    result['absolute_means']={}
    result['per_image_changes']={}
    result['native_size_strata']={}
    for shape in sorted({tuple(r['native_shape']) for r in rows}):
        group=[r for r in rows if tuple(r['native_shape'])==shape and r['fixed_primary_valid']]
        result['native_size_strata'][str(shape)]={method:dict(n_images=len(group),candidate_path_d1_macro=float(np.mean([r[method]['candidate_path_d1_macro'] for r in group])) if group else None) for method in ['A0','B1','B2']}
    for method in ['A0','B1','B2']:
        result['absolute_means'][method]={}
        for metric,ids in result['fixed_metric_ids'].items():
            vv=[r[method][metric] for r in rows if r['filename'] in ids]
            result['absolute_means'][method][metric]=dict(mean=float(np.mean(vv)) if vv else None,n=len(vv),NA=len(rows)-len(vv))
    for label in ['B2-A0','B1-A0','B2-B1']:
        left,right=label.split('-');result['per_image_changes'][label]={}
        for metric,ids in result['fixed_metric_ids'].items():
            dd=[dict(filename=r['filename'],delta=r[left][metric]-r[right][metric]) for r in rows if r['filename'] in ids]
            result['per_image_changes'][label][metric]=change_summary(dd,metric)
    guards={}
    for method in ['B1','B2']:
        c=result['comparisons'][method+'-A0'];observed={}
        for metric,direction in [('precision',1),('far_recall',1),('fp_density',-1),('dice',1),('recall',1),('candidate_path_d0_macro',1),('coarse_vessel_d0',1)]:
            d=c[metric];observed[metric]=dict(mean_nonadverse=d['estimate'] is not None and direction*d['estimate']>=0,interval_nonadverse=d['lower'] is not None and (d['lower']>=0 if direction==1 else d['upper']<=0),**d)
        guards[method]=observed
    result['guards']=guards;result['limits']='single-seed/reused Dev; conditional image bootstrap; graph paths not medical anatomy; no Test'
    result['path_strata']=path_strata(all_records)
    main=result['comparisons']['B2-A0']['candidate_path_d1_macro']
    signal=main['estimate'] is not None and main['estimate']>0 and main['lower']>0
    point_safety=all(guards['B2'][k]['mean_nonadverse'] for k in ['precision','far_recall','fp_density','dice','recall','candidate_path_d0_macro','coarse_vessel_d0'])
    strict=all(guards['B2'][k]['interval_nonadverse'] for k in ['precision','far_recall','fp_density'])
    result['scientific_judgment']=dict(primary_conditional_signal=signal,point_safety_nonadverse=point_safety,
      original_guard_interval_support=strict,protected_exploratory_signal=signal and point_safety and strict,
      wording='未观察到主指标增益' if main['estimate'] is None or main['estimate']<=0 else '存在正向均值，区间证据不足' if not signal else '条件性主信号；仍须同时核对代价与保护',
      never_independent_validation=True)
    dest=OUT/'analysis';dest.mkdir(exist_ok=True);write_json(dest/'endpoint_analysis.json',result)
    for method,recs in all_records.items():write_json(dest/(method+'_per_image_and_path_counts.json'),recs)
    lines=['# B1/B2固定终点探索性结果','',result['limits'],'','|比较|主指标配对差|95%区间|有效图数|','|---|---:|---|---:|']
    for label in ['B2-A0','B1-A0','B2-B1']:
        x=result['comparisons'][label]['candidate_path_d1_macro'];lines.append('|'+label+'|'+str(x['estimate'])+'|'+str([x['lower'],x['upper']])+'|'+str(x['n_images'])+'|')
    lines+=['','|指标|A0|B1|B2|','|---|---:|---:|---:|']
    for metric in result['fixed_metric_ids']:
        lines.append('|'+metric+'|'+'|'.join(str(result['absolute_means'][m][metric]['mean']) for m in ['A0','B1','B2'])+'|')
    lines+=['','## 保护指标与实际代价','|方法|指标|配对差|95%区间|有效图数|均值未恶化|区间支持未恶化|','|---|---|---:|---|---:|---|---|']
    for method,metrics in guards.items():
        for metric,x in metrics.items():lines.append('|'+method+'|'+metric+'|'+str(x['estimate'])+'|'+str([x['lower'],x['upper']])+'|'+str(x['n_images'])+'|'+str(x['mean_nonadverse'])+'|'+str(x['interval_nonadverse'])+'|')
    lines+=['','## 逐图变化与不利病例','|比较|指标|改善/下降/不变|最不利图片及变化|','|---|---|---|---|']
    for label,metrics in result['per_image_changes'].items():
        for metric in ['candidate_path_d1_macro','precision','fp_density','far_recall','coarse_vessel_d0']:
            x=metrics[metric];direction=1 if x['favorable_direction']=='higher' else -1
            worst=sorted(x['rows'],key=lambda r:direction*r['delta'])[:3]
            lines.append('|'+label+'|'+metric+'|'+str([x['improved'],x['declined'],x['unchanged']])+'|'+'; '.join(r['filename']+': '+str(r['delta']) for r in worst)+'|')
    lines+=['','## 分层结果','原生尺寸层沿用主指标；路径长度逐一保留原值，不另行搜索切点。短支使用已冻结的20阈值，复杂/边界/疑似缺口沿原GT标记。分层是描述性诊断，保留排除路径和NA；不替换主比较、不视为新增独立验证。','|分层|组别|A0/B1/B2路径宏d1|有效图数|有效路径数|位置分母|','|---|---|---|---|---|---|']
    for family in ['short_branch','complex_flags','path_kind']:
        for group,values in result['path_strata'][family].items():
            lines.append('|'+family+'|'+group+'|'+'/'.join(str(values[m]['path_d1_macro']) for m in ['A0','B1','B2'])+'|'+str(values['A0']['n_images'])+'|'+str(values['A0']['n_paths'])+'|'+str(values['A0']['position_denominator'])+'|')
    lines+=['',result['scientific_judgment']['wording'],'受保护探索信号：'+str(result['scientific_judgment']['protected_exploratory_signal']),
      '全部绝对值、逐图/逐路径分子分母、区间与保护代价见同目录JSON。正负结果均保留；主信号不代表方法已独立验证。']
    (dest/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return dict(status='ENDPOINT_ANALYSIS_COMPLETE',path=str(dest/'endpoint_analysis.json'))

if __name__=='__main__':
    # Workers import the canonical module; its entry must own the live session.
    from retinal_m0.maples_branch_science import main as canonical_main
    canonical_main()
