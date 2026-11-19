"""CPU-only C0 preparation for the frozen thin-skeleton BCE protocol.

This module never loads a model or calls inference. It derives Train targets,
checks the frozen draw/evaluation records, and makes a human-review atlas.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation, convolve, label as cc_label, distance_transform_edt
from skimage.morphology import skeletonize

from retinal_data.maples_fov import candidate_fov
from retinal_data.maples_native import NativeMaplesDataset
from retinal_m0.maples_core import CorePatches, draw_geometry, weighted_masked_loss, _decode_label_image
from retinal_m0.maples_evaluate import evaluate_arrays, load_mask

ROOT = Path('${RETINA_PROJECT_ROOT}')
CODE = ROOT / 'code/retinal-vessel-study'
PROTO_DIR = ROOT / 'docs/full_fov_thin_skeleton_bce_20261019'
PROTO = PROTO_DIR / 'protocol.json'
PROTO_SHA = '429fe51a8117df85116e8409ee57e77bd3a949c2773fa6bd2cfc2b92df8aa509'
OUT = ROOT / 'experiments/full_fov_thin_skeleton_bce_20261019'
TRAIN_DIR = ROOT / 'experiments/t0_tw_core_20261016/runs/seed_2026/T0'
T0_PROB = TRAIN_DIR / 'evaluation/step_002000/probabilities'
PLANB = ROOT / 'docs/plan_b_error_analysis_20261019/source_manifest.json'
DEV_MANIFEST = ROOT / 'experiments/a15_2_z0_t0_calibration_20261015/dev_manifest.json'
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
EIGHT = np.ones((3, 3), dtype=np.uint8)
EIGHT[1, 1] = 0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path: Path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else []
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def s3_target(vessel: np.ndarray, fov: np.ndarray):
    if vessel.ndim != 2 or fov.shape != vessel.shape:
        raise ValueError('GT/FOV must be aligned native 2D arrays')
    if not np.isin(vessel, (0, 1, False, True)).all() or not np.isin(fov, (0, 1, False, True)).all():
        raise ValueError('GT/FOV must be binary')
    g = vessel.astype(bool) & fov.astype(bool)
    sk = skeletonize(g, method='zhang')
    radius = distance_transform_edt(np.pad(g, 1))[1:-1, 1:-1].astype(np.float32)
    s = sk & (radius <= np.float32(1.5))
    if np.any(s & ~g):
        raise AssertionError('S3 must be a subset of V&FOV')
    return g, sk, radius, s


def thin_bce_loss(logits, target, fov, thin, lam=0.1):
    """Frozen per-crop T0 loss plus normalized positive BCE on S3."""
    if logits.shape != target.shape or logits.shape != fov.shape or logits.shape != thin.shape or logits.ndim != 4:
        raise ValueError('Matching NCHW tensors required')
    if type(lam) not in (int, float) or not np.isfinite(lam) or lam < 0:
        raise ValueError('lambda must be finite and nonnegative')
    if lam == 0:
        # Exact bypass preserves the bound T0 arithmetic and gradient graph.
        return weighted_masked_loss(logits, target, fov, fov, epsilon=1e-6)
    z, y, f, s = logits.float(), target.float(), fov.float(), thin.float()
    if not all(torch.isfinite(x).all() for x in (z, y, f, s)):
        raise ValueError('Loss inputs must be finite')
    if not torch.all((y == 0) | (y == 1)) or not torch.all((f == 0) | (f == 1)) or not torch.all((s == 0) | (s == 1)):
        raise ValueError('GT/FOV/S3 must be binary')
    if torch.any(s > y * f):
        raise ValueError('S3 must be a subset of V&FOV')
    axes = (1, 2, 3)
    base_weight = f * f
    denom = base_weight.sum(axes)
    if torch.any(denom <= 0):
        raise ValueError('empty_fov: no resampling is permitted')
    bce_map = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction='none')
    base_bce = (bce_map * base_weight).sum(axes) / denom
    prob = z.sigmoid()
    dice = 1 - (2 * (prob * y * f).sum(axes) + 1e-6) / ((prob * f).sum(axes) + (y * f).sum(axes) + 1e-6)
    count = s.sum(axes)
    aux_map = torch.nn.functional.softplus(-z) * s
    aux = torch.where(count > 0, aux_map.sum(axes) / count.clamp_min(1), aux_map.sum(axes) * 0)
    # Empty-S crops remain present in this fixed batch mean.
    return (base_bce + dice + float(lam) * aux).mean()


def synthetic_checks():
    torch.manual_seed(7)
    z0 = torch.linspace(-2, 2, 4 * 1 * 4 * 4, dtype=torch.float32).reshape(4, 1, 4, 4)
    y = torch.zeros_like(z0); y[:, :, 1:3, 1:3] = 1
    f = torch.ones_like(z0)
    # Lambda zero must reproduce both value and logits gradient exactly.
    a = z0.clone().requires_grad_(True); b = z0.clone().requires_grad_(True)
    old = weighted_masked_loss(a, y, f, f, epsilon=1e-6)
    new = thin_bce_loss(b, y, f, torch.zeros_like(y), lam=0)
    ga, = torch.autograd.grad(old, a); gb, = torch.autograd.grad(new, b)
    assert torch.equal(old, new) and torch.equal(ga, gb)
    # Empty target: auxiliary term and its gradient are exactly zero.
    z = z0.clone().requires_grad_(True)
    empty = torch.zeros_like(y)
    aux_empty = (torch.nn.functional.softplus(-z) * empty).sum()
    ge, = torch.autograd.grad(aux_empty, z)
    assert torch.count_nonzero(ge).item() == 0
    # Single target, unequal crop counts: per-crop means, not pooled pixels.
    s = torch.zeros_like(y); s[0, 0, 1, 1] = 1; s[1, 0, 1, 1:3] = 1; s[1, 0, 2, 1] = 1; s[2, 0, 1:3, 1:3] = 1
    z = z0.clone().requires_grad_(True)
    loss = thin_bce_loss(z, y, f, s, lam=.1)
    auxgrad, = torch.autograd.grad(loss, z)
    analytic = .1 * s * (torch.sigmoid(z.detach()) - 1) / (4 * s.sum((1, 2, 3)).clamp_min(1)[:, None, None, None])
    # Subtract the base gradient before checking the auxiliary closed form.
    z2 = z0.clone().requires_grad_(True)
    base_loss = weighted_masked_loss(z2, y, f, f, epsilon=1e-6)
    basegrad, = torch.autograd.grad(base_loss, z2)
    observed = auxgrad - basegrad
    assert torch.allclose(observed, analytic, rtol=1e-5, atol=1e-7)
    assert float(observed.abs().sum()) <= .1 + 1e-6
    assert torch.isfinite(loss)
    try:
        thin_bce_loss(z0, y, torch.zeros_like(f), torch.zeros_like(s), lam=.1)
    except ValueError as e:
        assert 'empty_fov' in str(e)
    else:
        raise AssertionError('empty FOV must error without resampling')
    return {'passed': True, 'lambda0_value_gradient_bitwise_equal': True,
            'empty_S_aux_gradient_exact_zero': True, 'single_and_unequal_count_analytic_gradient': True,
            'all_fov_finite': True, 'aux_gradient_l1': float(observed.abs().sum()),
            'empty_fov_error_no_resampling': True}


def crop_bbox(y, x, shape, size=256):
    h, w = shape
    if h < size or w < size:
        raise ValueError('Image smaller than protocol crop')
    y0 = min(max(int(y) - size // 2, 0), h - size)
    x0 = min(max(int(x) - size // 2, 0), w - size)
    return [y0, x0, y0 + size, x0 + size]


def load_rgb_v_fov(row, data_root):
    rgbp, vp = Path(row['RGB']['path']), Path(row['V']['path'])
    if sha(rgbp) != row['RGB']['sha256'] or sha(vp) != row['V']['sha256']:
        raise RuntimeError('Frozen Train RGB/GT bytes changed: ' + row['filename'])
    with Image.open(rgbp) as im:
        im.load(); rgb = np.asarray(im.convert('RGB'), dtype=np.uint8)
    with Image.open(vp) as im:
        im.load(); v = _decode_label_image(im).astype(bool)
    fov = candidate_fov(rgb).astype(bool)
    if rgb.shape[:2] != tuple(row['native_shape']) or v.shape != fov.shape:
        raise RuntimeError('Native shape mismatch: ' + row['filename'])
    return rgb, v, fov


def bind_and_build_train(protocol, train_manifest, gate_dir):
    baseline_config = json.loads((TRAIN_DIR / 'config.json').read_text())
    native = NativeMaplesDataset(baseline_config['data_root'], split='train')
    if len(native.rows) != 110 or [r['filename'] for r in native.rows] != [r['filename'] for r in train_manifest['rows']]:
        raise RuntimeError('Train native loader membership/order differs')
    if baseline_config['seed'] != protocol['scope']['seed'] or baseline_config['patch'] != 512 or baseline_config['crops_per_image'] != 4:
        raise RuntimeError('Baseline settings do not match frozen crop plan')
    patches = CorePatches(baseline_config, 'T0')
    target_dir = OUT / 'targets/train'
    target_dir.mkdir(parents=True, exist_ok=False)
    image_cache = {}
    target_rows = []
    for idx, row in enumerate(train_manifest['rows']):
        rgb, v, fov = load_rgb_v_fov(row, baseline_config['data_root'])
        g, sk, radius, s = s3_target(v, fov)
        if tuple(rgb.shape[:2]) != tuple(row['native_shape']):
            raise RuntimeError('RGB source shape mismatch')
        # Informative U count only: U never enters target construction or masking.
        native_row = native.rows[idx]
        uncertain = np.zeros_like(fov)
        u_paths = []
        for key in ('label_brightuncertains', 'label_reduncertains'):
            path = Path(baseline_config['data_root']) / native_row[key]
            with Image.open(path) as im:
                im.load(); uncertain |= _decode_label_image(im).astype(bool)
            u_paths.append(sha(path))
        if uncertain.shape != s.shape:
            raise RuntimeError('U shape mismatch')
        # Verify this target matches the bound loader's original FOV and vessel arrays.
        raw2, mask2, fov2, _ = patches._full_arrays(idx)
        if not (np.array_equal(raw2, rgb) and np.array_equal(mask2.astype(bool), v) and np.array_equal(fov2.astype(bool), fov)):
            raise RuntimeError('CorePatches differs from independently loaded RGB/V/FOV')
        n_ep = convolve(sk.astype(np.uint8), EIGHT, mode='constant', cval=0) == 1
        target_path = target_dir / (row['filename'] + '.npz')
        np.savez_compressed(target_path, s_train=s.astype(np.uint8))
        image_cache[row['filename']] = (rgb, v, fov, g, sk, radius, s, n_ep, uncertain)
        target_rows.append({'filename': row['filename'], 'native_height': rgb.shape[0], 'native_width': rgb.shape[1],
            'V_pixels': int(v.sum()), 'FOV_pixels': int(fov.sum()), 'G_V_and_FOV_pixels': int(g.sum()),
            'full_zhang_skeleton_pixels': int(sk.sum()), 'S3_train_pixels': int(s.sum()),
            'S3_train_in_U_retained': int((s & uncertain).sum()), 'skeleton_endpoints_8n': int((sk & n_ep).sum()),
            'target_file': str(target_path.relative_to(OUT)), 'target_sha256': sha(target_path),
            'target_array_sha256': array_sha(s), 'RGB_file_sha256': row['RGB']['sha256'],
            'GT_file_sha256': row['V']['sha256'], 'FOV_array_sha256': array_sha(fov),
            'U_files_sha256_informative_only': ';'.join(u_paths)})
    if len(target_rows) != 110 or sum(r['S3_train_pixels'] for r in target_rows) <= 0:
        raise RuntimeError('Train S3 is empty or incomplete')

    pairing_path = TRAIN_DIR / 'pairing.jsonl'
    pair_rows = [json.loads(x) for x in pairing_path.read_text().splitlines() if x.strip()]
    if len(pair_rows) != 2000:
        raise RuntimeError('Frozen T0 pairing log does not contain 2000 updates')
    plan = []
    for step, p in enumerate(pair_rows, start=1):
        if p.get('step') != step or p.get('draws') != list(range((step - 1) * 4, step * 4)):
            raise RuntimeError('T0 draw schedule discontinuity at step ' + str(step))
        if not (len(p.get('filenames', [])) == len(p.get('geometry', [])) == len(p.get('input_sha256', [])) == 4):
            raise RuntimeError('Malformed paired draw row at step ' + str(step))
        for slot, draw in enumerate(p['draws']):
            idx = draw % 110
            row = train_manifest['rows'][idx]
            rgb, v, fov, g, sk, radius, s, end, uncertain = image_cache[row['filename']]
            geom = draw_geometry(2026, draw, *rgb.shape[:2], patch=512)
            geom_i = [int(geom[0]), int(geom[1]), int(geom[2]), int(geom[3]), int(geom[4])]
            if p['filenames'][slot] != row['filename'] or p['geometry'][slot] != geom_i:
                raise RuntimeError('Frozen draw filename/geometry mismatch at draw ' + str(draw))
            # Reconstruct the actual bound T0 input tensor hash, without model execution.
            item = patches[draw]
            if item['input_sha256'] != p['input_sha256'][slot]:
                raise RuntimeError('Frozen paired tensor hash mismatch at draw ' + str(draw))
            top, left = geom[0], geom[1]
            crop_f = fov[top:top + 512, left:left + 512]
            crop_v = v[top:top + 512, left:left + 512]
            crop_s = s[top:top + 512, left:left + 512]
            if not crop_f.any():
                raise RuntimeError('Frozen crop unexpectedly has empty FOV')
            if np.any(crop_s & ~crop_v) or np.any(crop_s & ~crop_f):
                raise RuntimeError('Transformed S3 is outside GT/FOV')
            plan.append({'step': step, 'slot': slot, 'draw': draw, 'filename': row['filename'],
                'native_height': rgb.shape[0], 'native_width': rgb.shape[1], 'top': top, 'left': left,
                'hflip': int(geom[2]), 'vflip': int(geom[3]), 'ccw_rot90_quarters': int(geom[4]),
                'input_sha256_matches_frozen_T0': True, 'N_FOV': int(crop_f.sum()),
                'N_positive_V': int(crop_v.sum()), 'N_S3': int(crop_s.sum()), 'empty_S3': int(not crop_s.any()),
                'S3_pixels_in_U_retained': int((crop_s & uncertain[top:top + 512, left:left + 512]).sum())})
    if len(plan) != 8000 or sum(not r['empty_S3'] for r in plan) == 0:
        raise RuntimeError('Crop plan incomplete or has zero nonempty S3 crops')
    return image_cache, target_rows, plan


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    return {'n': int(len(a)), 'min': int(a.min()) if len(a) else None,
            'q25': float(np.quantile(a, .25)) if len(a) else None,
            'median': float(np.quantile(a, .5)) if len(a) else None,
            'q75': float(np.quantile(a, .75)) if len(a) else None,
            'max': int(a.max()) if len(a) else None,
            'zero_count': int((a == 0).sum()), 'zero_fraction': float((a == 0).mean()) if len(a) else None}


def recalc_existing_evaluation(protocol, dev_ref, planb, gate_dir):
    olddoc = json.loads(DEV_MANIFEST.read_text())
    from retinal_m0.maples_evaluate import validate_dev_membership
    validate_dev_membership(olddoc['images'])
    old_eval = json.loads((TRAIN_DIR / 'evaluation/step_002000/metrics.json').read_text())
    condition = next(c for c in planb['conditions'] if c['phase'] == 'formal' and c['method'] == 'T0' and c['seed'] == 2026 and c['step'] == 2000)
    if sha(Path(condition['metrics'])) != condition['metrics_sha256'] or sha(Path(condition['metrics'])) != sha(TRAIN_DIR / 'evaluation/step_002000/metrics.json'):
        raise RuntimeError('Frozen T0 evaluation hash differs from source manifest')
    if old_eval['checkpoint_sha256'] != protocol['sources']['baseline_checkpoint_sha256'] or old_eval['threshold'] != .5:
        raise RuntimeError('Frozen T0 checkpoint/threshold binding mismatch')
    prediction = {r['filename']: r for r in condition['predictions']}
    stored = {r['filename']: r for r in old_eval['images']}
    if set(prediction) != set(stored) or len(dev_ref['images']) != 28:
        raise RuntimeError('Frozen prediction/evaluation cohort mismatch')
    results = []
    for row in dev_ref['images']:
        name = row['filename']; inp = row['inputs']; predrow = prediction[name]
        for role in ('RGB', 'GT', 'FOV', 'regions'):
            if sha(Path(inp[role]['path'])) != inp[role]['sha256']:
                raise RuntimeError(f'Frozen Dev {role} hash mismatch: {name}')
        ppath = Path(predrow['path'])
        if sha(ppath) != predrow['sha256'] or list(np.load(ppath, mmap_mode='r', allow_pickle=False).shape) != row['native_shape']:
            raise RuntimeError('Frozen T0 probability hash/shape mismatch: ' + name)
        rgb = np.asarray(Image.open(inp['RGB']['path']).convert('RGB'))
        v = load_mask(inp['GT']['path']); fov = load_mask(inp['FOV']['path'])
        g, sk, radius, s3 = s3_target(v, fov)
        with np.load(inp['regions']['path'], allow_pickle=False) as z:
            u = z['U'].astype(bool)
            regions = {k: z[k].astype(bool) for k in ('internal', 'near', 'far')}
            oldthin = z['thin'].astype(bool)
        if not np.array_equal(oldthin, s3):
            raise RuntimeError('Dev S3 differs from frozen thin reference: ' + name)
        omega = fov & ~u
        newden = int((s3 & omega).sum())
        if newden != row['new_full_domain_S3_Omega_denominator']:
            raise RuntimeError('New endpoint denominator mismatch: ' + name)
        prob = np.load(ppath, mmap_mode='r', allow_pickle=False)
        got = evaluate_arrays(prob, v, fov, threshold=.5, u=u, regions=regions, thin=s3, matching_support='omega')
        expected = stored[name]
        old_ok = True
        for metric in ('dice','precision','recall','fp_density','near_pixel_recall','near_thin_recall_d1','far_recall'):
            for field in ('numerator','denominator','NA_reason'):
                if got[metric][field] != expected[metric][field]:
                    old_ok = False
                    raise RuntimeError(f'Existing metric mismatch {name}/{metric}/{field}: {got[metric][field]} != {expected[metric][field]}')
        pred = np.asarray(prob >= .5) & omega
        hit = binary_dilation(pred, structure=CROSS)
        d0 = int((s3 & omega & pred).sum())
        d1 = int((s3 & omega & hit).sum())
        results.append({'filename': name, 'native_height': v.shape[0], 'native_width': v.shape[1],
            'new_full_domain_denominator': newden, 'new_d0_hits': d0, 'new_d1_hits': d1,
            'new_d0_recall': d0 / newden if newden else 'NA', 'new_d1_recall': d1 / newden if newden else 'NA',
            'expected_NA': newden == 0, 'old_7_metrics_exact_recalc': old_ok,
            'probability_sha256': predrow['sha256'],
            **{f'{m}_{field}': got[m][field] for m in ('dice','precision','recall','fp_density','near_pixel_recall','near_thin_recall_d1','far_recall') for field in ('numerator','denominator','NA_reason')}})
    if sum(r['new_full_domain_denominator'] for r in results) != 7291 or len(results) != 28:
        raise RuntimeError('New endpoint cohort total is not frozen 28/7291')
    if sum(r['near_thin_recall_d1_denominator'] for r in results) != 77 or sum(r['near_thin_recall_d1_denominator'] == 0 for r in results) != 19:
        raise RuntimeError('Original near endpoint 9/77/19NA changed')
    write_csv(gate_dir / 'existing_t0_recalculation.csv', results)
    return results, prediction


def selection_and_atlas(protocol, train_manifest, dev_ref, planb, image_cache, prediction, gate_dir):
    sel_rows = []
    cases = []
    train_groups = {}
    for row in train_manifest['rows']:
        rgb, v, fov, g, sk, radius, s, endpoint, uncertain = image_cache[row['filename']]
        size = f'{rgb.shape[0]}x{rgb.shape[1]}'
        for stratum, mask in [('endpoint', s & endpoint), ('nonendpoint', s & ~endpoint)]:
            group = train_groups.setdefault((size,stratum), [])
            group.extend((row['filename'],int(y),int(x)) for y,x in np.argwhere(mask))
    for (size,stratum),coords in sorted(train_groups.items()):
        coords.sort()
        if len(coords) < 2:
            sel_rows.append({'split':'train','stratum':f'{size}/{stratum}','status':'unavailable_empty_or_singleton',
                'candidate_count':len(coords),'rank_q25':None,'rank_q75':None,'case_id':None})
            continue
        chosen=[]; details=[]
        for qname,q in [('q25',.25),('q75',.75)]:
            rank=int(math.floor(q*(len(coords)-1)))
            name,y,x=coords[rank]
            rgb,v,fov,g,sk,radius,s,endpoint,uncertain=image_cache[name]
            cid=f'TRAIN_{size}_{stratum.upper()}_{qname.upper()}'
            rec=make_case(cid,'train',name,y,x,'target_definition',rgb,v,fov,sk,s,gate_dir)
            rec.update({'stratum':f'{size}/{stratum}','candidate_count':len(coords),'rank':rank,'rank_quantile':q})
            cases.append(rec); chosen.append(cid)
            details.append({'case_id':cid,'filename':name,'y':y,'x':x,'bbox':rec['bbox'],'rank':rank,'quantile':q})
        if len(set(chosen))!=2: raise RuntimeError('Train q25/q75 target cases are not distinct')
        sel_rows.append({'split':'train','stratum':f'{size}/{stratum}','status':'selected','candidate_count':len(coords),
            'rank_q25':int(math.floor(.25*(len(coords)-1))),'rank_q75':int(math.floor(.75*(len(coords)-1))),
            'case_id':';'.join(chosen),'case_details_json':json.dumps(details,ensure_ascii=False)})

    # Dev selection: only unanimous all-11 hit or all-11 miss S3 points, with no replacement.
    conditions = planb['conditions']
    dev_groups={}
    for image in dev_ref['images']:
        name = image['filename']; inp = image['inputs']
        v = load_mask(inp['GT']['path']); fov = load_mask(inp['FOV']['path'])
        _, sk, _, s3 = s3_target(v,fov)
        with np.load(inp['regions']['path'],allow_pickle=False) as z: omega=fov & ~z['U'].astype(bool)
        candidates = np.argwhere(s3 & omega)
        if not len(candidates): continue
        all_hits=[]; all_misses=[]
        for cond in conditions:
            pr=next(x for x in cond['predictions'] if x['filename']==name)
            ppath=Path(pr['path'])
            if sha(ppath)!=pr['sha256']: raise RuntimeError('Plan B prediction hash mismatch: '+name)
            pp=np.load(ppath,mmap_mode='r',allow_pickle=False)
            if pp.shape!=v.shape: raise RuntimeError('Plan B prediction shape mismatch: '+name)
            hit=binary_dilation((np.asarray(pp)>=.5)&omega,structure=CROSS)
            all_hits.append(hit[candidates[:,0],candidates[:,1]])
        hitall=np.all(np.stack(all_hits),axis=0); missall=~np.any(np.stack(all_hits),axis=0)
        size=f'{v.shape[0]}x{v.shape[1]}'
        devcache[name]=(np.asarray(Image.open(inp['RGB']['path']).convert('RGB')),v,fov,sk,s3,omega)
        for kind,ok in [('all11_hit',hitall),('all11_miss',missall)]:
            dev_groups.setdefault((size,kind),[]).extend((name,int(y),int(x)) for y,x in candidates[ok])
    for (size,kind),coords in sorted(dev_groups.items()):
        coords.sort()
        if not coords:
            sel_rows.append({'split':'dev','stratum':f'{size}/{kind}','status':'unavailable_no_unanimous_point',
                'candidate_count':0,'rank_median':None,'case_id':None})
            continue
        rank=(len(coords)-1)//2; name,y,x=coords[rank]
        cid=f'DEV_{size}_{kind.upper()}'
        rgb,vv,ff,fullsk,ss,om=devcache[name]
        rec=make_case(cid,'dev',name,y,x,'old_prediction_'+kind,rgb,vv,ff,fullsk,ss,gate_dir)
        rec.update({'stratum':f'{size}/{kind}','candidate_count':len(coords),'rank':rank,'rank_rule':'lexicographic lower median'})
        cases.append(rec)
        sel_rows.append({'split':'dev','stratum':f'{size}/{kind}','status':'selected','candidate_count':len(coords),
            'rank_median':rank,'case_id':cid,'filename':name,'y':y,'x':x})
    # FP/FOV extremes from the three formal T0 seeds, with deterministic largest FP component.
    t0conds=[c for c in conditions if c['phase']=='formal' and c['method']=='T0' and c['step']==2000]
    if len(t0conds)!=3: raise RuntimeError('Expected three formal T0 seed endpoints for FP case selection')
    fp_macro={}
    metrics_by={}
    for c in t0conds:
        mp=Path(c['metrics'])
        if sha(mp)!=c['metrics_sha256']: raise RuntimeError('T0 selection metric source changed')
        md=json.loads(mp.read_text())
        for r in md['images']:
            metrics_by.setdefault(r['filename'],[]).append(r['fp_density']['value'])
    for name,vals in metrics_by.items(): fp_macro[name]=float(np.mean(vals))
    ranked_high=sorted(fp_macro,key=lambda n:(-fp_macro[n],n))[:2]
    ranked_low=sorted(fp_macro,key=lambda n:(fp_macro[n],n))[:2]
    ranked=sorted(fp_macro,key=lambda n:(fp_macro[n],n))
    chosen=[('highest',n) for n in ranked_high]+[('lowest',n) for n in ranked_low]
    t0seed=next(c for c in t0conds if c['seed']==2026)
    t0pred={r['filename']:r for r in t0seed['predictions']}
    dev_by={r['filename']:r for r in dev_ref['images']}
    for direction,name in chosen:
        row=dev_by[name]; inp=row['inputs']; v=load_mask(inp['GT']['path']); fov=load_mask(inp['FOV']['path'])
        rgb=np.asarray(Image.open(inp['RGB']['path']).convert('RGB')); _,sk,_,s3=s3_target(v,fov)
        with np.load(inp['regions']['path'],allow_pickle=False) as z: omega=fov&~z['U'].astype(bool)
        pr=t0pred[name]; pp=np.load(pr['path'],mmap_mode='r',allow_pickle=False)
        if sha(Path(pr['path']))!=pr['sha256']: raise RuntimeError('FP prediction hash mismatch')
        false=(np.asarray(pp)>=.5)&fov&~v
        labels,ncomp=cc_label(false,structure=np.ones((3,3),dtype=np.uint8))
        if ncomp:
            sizes=np.bincount(labels.ravel())[1:]
            lab=int(np.argmax(sizes))+1
            yy,xx=np.where(labels==lab); y,x=int((yy.min()+yy.max())//2),int((xx.min()+xx.max())//2)
            bbox=[int(yy.min()),int(xx.min()),int(yy.max())+1,int(xx.max())+1]
            cid=f'FP_{direction.upper()}_{name}'
            rec=make_case(cid,'dev',name,y,x,'illustrative_largest_T0_2026_FP_component',rgb,v,fov,sk,s3,gate_dir)
            rec.update({'stratum':'fp_fov_extreme/'+direction,'fp_fov_three_seed_mean':fp_macro[name],
                'largest_component_pixels':int(sizes[lab-1]),'largest_component_bbox':bbox,'selection_note':'GT-relative pixel component; not biological FP adjudication'})
            rec['t0_prediction_path']=pr['path']; cases.append(rec)
            sel_rows.append({'split':'dev','stratum':f'fp_fov_extreme/{direction}','status':'selected','candidate_count':len(ranked),
                'case_id':cid,'filename':name,'y':y,'x':x,'fp_fov_three_seed_mean':fp_macro[name],
                'largest_component_pixels':int(sizes[lab-1]),'largest_component_bbox':json.dumps(bbox)})
        else:
            sel_rows.append({'split':'dev','stratum':f'fp_fov_extreme/{direction}','status':'unavailable_no_FP_component',
                'candidate_count':0,'case_id':None,'filename':name,'fp_fov_three_seed_mean':fp_macro[name]})
    if len(cases)>22: raise RuntimeError('Atlas exceeds protocol case maximum 22')
    review_rows=[]
    for c in cases:
        review_rows.append({'case_id':c['case_id'],'split':c['split'],'filename':c['filename'],'native_bbox':json.dumps(c['bbox']),
            'source_sha256s':json.dumps(c['source_sha256s'],sort_keys=True),'reviewer_type':'human','reviewer_name':'','reviewed_at':'',
            'definition_alignment':'','visible_vessel':'','artifact_or_label_limit':'','notes':'','overall_accept_or_pause':'',
            'verbatim_human_evidence_reference':''})
    write_csv(gate_dir/'manual_selection.csv',sel_rows)
    write_csv(gate_dir/'manual_review.csv',review_rows,fields=['case_id','split','filename','native_bbox','source_sha256s','reviewer_type','reviewer_name','reviewed_at','definition_alignment','visible_vessel','artifact_or_label_limit','notes','overall_accept_or_pause','verbatim_human_evidence_reference'])
    render_atlas(cases, prediction, planb, gate_dir)
    return cases,sel_rows


devcache={}


def make_case(case_id,split,name,y,x,purpose,rgb,v,fov,sk,s3,gate_dir):
    bbox=crop_bbox(y,x,v.shape,256)
    inp_hashes={}
    if split=='train':
        row=next(r for r in json.loads((PROTO_DIR/'train_manifest.json').read_text())['rows'] if r['filename']==name)
        inp_hashes={'RGB':row['RGB']['sha256'],'GT':row['V']['sha256'],'FOV_array':array_sha(fov),'S3_array':array_sha(s3)}
    else:
        row=next(r for r in json.loads((PROTO_DIR/'dev_reference_manifest.json').read_text())['images'] if r['filename']==name)
        inp_hashes={k:v0['sha256'] for k,v0 in row['inputs'].items()}
        inp_hashes.update({'FOV_array':array_sha(fov),'S3_array':array_sha(s3)})
    return {'case_id':case_id,'split':split,'filename':name,'y':int(y),'x':int(x),'bbox':bbox,
        'purpose':purpose,'source_sha256s':inp_hashes,'rgb':rgb,'v':v,'fov':fov,'sk':sk,'s3':s3}


def render_atlas(cases, t0_prediction, planb, gate_dir):
    atlas=gate_dir/'atlas'; atlas.mkdir(parents=True,exist_ok=True)
    conds=planb['conditions']; t0seed=next(c for c in conds if c['phase']=='formal' and c['method']=='T0' and c['seed']==2026)
    predmaps={r['filename']:r for r in t0seed['predictions']}
    html_rows=[]; markdown_rows=['# H1 真实人工审核图册','','> 本文是未作 AI 视觉判断的案例索引。请在 VS Code 使用 Markdown 预览查看图像，并在同目录的 `../manual_review.csv` 填写真实人工判断。红圈标出选例位置；11×放大框仅用于显示，原始数据像素未改变。']
    for case in cases:
        y,x=case['y'],case['x']; y0,x0,y1,x1=case['bbox']; rgb=case['rgb']; v=case['v']; f=case['fov']; sk=case['sk']; s=case['s3']
        fig,axes=plt.subplots(2,3,figsize=(18,11),constrained_layout=True)
        fig.suptitle(f"{case['case_id']} | {case['split'].upper()} | {case['filename']} | point (y={y}, x={x})\nRed ring marks the exact review location; 11× inset is display-only.",fontsize=15)
        crop=rgb[y0:y1,x0:x1]; point=(x-x0,y-y0)
        panels=[('Full image context — unmodified RGB','context',rgb),('Original RGB crop — 256 native pixels','rgb',crop),
                ('GT Mask — white is vessel label','mask',v[y0:y1,x0:x1]),('FOV — white is evaluation/training field','fov',f[y0:y1,x0:x1]),
                ('Full Zhang skeleton on V∩FOV — cyan overlay','skeleton',sk[y0:y1,x0:x1]),('Thin skeleton training target S3 — yellow overlay','s3',s[y0:y1,x0:x1])]
        for ax,(title,kind,image) in zip(axes.flat,panels):
            if kind=='context' or kind=='rgb': ax.imshow(image,interpolation='nearest')
            elif kind in ('mask','fov'): ax.imshow(image,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            elif kind=='skeleton':
                ax.imshow(v[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=1,alpha=.32,interpolation='nearest')
                overlay=np.ma.masked_where(~image,image)
                ax.imshow(overlay,cmap=ListedColormap(['#00ffff']),vmin=0,vmax=1,alpha=1,interpolation='nearest')
            elif kind=='s3':
                ax.imshow(v[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=1,alpha=.26,interpolation='nearest')
                overlay=np.ma.masked_where(~image,image)
                ax.imshow(overlay,cmap=ListedColormap(['#ffd400']),vmin=0,vmax=1,alpha=1,interpolation='nearest')
                zy0=max(0,point[1]-5); zx0=max(0,point[0]-5); zy1=min(image.shape[0],point[1]+6); zx1=min(image.shape[1],point[0]+6)
                inset=ax.inset_axes([.65,.03,.32,.32])
                inset.imshow(v[y0+zy0:y0+zy1,x0+zx0:x0+zx1],cmap='gray',vmin=0,vmax=1,alpha=.26,interpolation='nearest')
                inset.imshow(np.ma.masked_where(~image[zy0:zy1,zx0:zx1],image[zy0:zy1,zx0:zx1]),cmap=ListedColormap(['#ffd400']),vmin=0,vmax=1,interpolation='nearest')
                inset.add_patch(plt.Rectangle((point[0]-zx0-.5,point[1]-zy0-.5),1,1,fill=False,edgecolor='red',linewidth=1.5))
                inset.set_title('11× display zoom',fontsize=7); inset.set_xticks([]); inset.set_yticks([])
            if title.startswith('Full image'):
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((x0,y0),256,256,fill=False,edgecolor='red',linewidth=2))
            else:
                ax.plot(point[0],point[1],marker='o',markersize=14,markerfacecolor='none',markeredgecolor='red',markeredgewidth=2)
            ax.set_title(title,fontsize=11); ax.axis('off')
        out=atlas/f"{case['case_id']}.png"; fig.savefig(out,dpi=160); plt.close(fig)
        prediction_file=''
        if case['split']=='dev':
            # Separate 11-condition prediction sheet keeps each original probability source visible.
            fig,axes=plt.subplots(3,4,figsize=(16,11),constrained_layout=True)
            fig.suptitle(f"Existing predictions only — {case['filename']} | threshold 0.5 | no new inference",fontsize=14)
            axes=axes.flat
            with np.load(next(r for r in json.loads((PROTO_DIR/'dev_reference_manifest.json').read_text())['images'] if r['filename']==case['filename'])['inputs']['regions']['path'],allow_pickle=False) as z:
                omega=case['fov']&~z['U'].astype(bool)
            for i,cond in enumerate(conds):
                pred=next(r for r in cond['predictions'] if r['filename']==case['filename'])
                p=np.load(pred['path'],mmap_mode='r',allow_pickle=False)
                axes[i].imshow(((np.asarray(p)>=.5)&omega)[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                axes[i].plot(x-x0,y-y0,marker='o',markersize=12,markerfacecolor='none',markeredgecolor='red',markeredgewidth=2)
                axes[i].set_title(f"{cond['condition_id']}\nselected target: {'hit' if binary_dilation((np.asarray(p)>=.5)&omega,structure=CROSS)[y,x] else 'miss'}",fontsize=8)
                axes[i].axis('off')
            axes[11].imshow(s[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=1,interpolation='nearest'); axes[11].set_title('S3 target reference',fontsize=9); axes[11].axis('off')
            prediction_file=f"{case['case_id']}_existing_predictions.png"
            fig.savefig(atlas/prediction_file,dpi=150); plt.close(fig)
        if case.get('t0_prediction_path'):
            p=np.load(case['t0_prediction_path'],mmap_mode='r',allow_pickle=False)
            fp=(np.asarray(p)>=.5)&case['fov']&~case['v']
            fig,ax=plt.subplots(figsize=(7,7)); ax.imshow(case['rgb'][y0:y1,x0:x1]); ax.imshow(fp[y0:y1,x0:x1],cmap='Reds',alpha=.55,interpolation='nearest')
            ax.plot(x-x0,y-y0,marker='o',markersize=14,markerfacecolor='none',markeredgecolor='cyan',markeredgewidth=2)
            ax.set_title('Seed 2026 T0 label-relative FP component (red); cyan ring is selected location\nIllustrative pixel disagreement, not biological FP adjudication'); ax.axis('off')
            fpname=f"{case['case_id']}_fp_overlay.png"; fig.savefig(atlas/fpname,dpi=160,bbox_inches='tight'); plt.close(fig)
            prediction_file += ('; ' if prediction_file else '') + fpname
        cid=html.escape(case['case_id']); split=html.escape(case['split']); filename=html.escape(case['filename']); purpose=html.escape(case['purpose'])
        links=''
        if case['split']=='dev': links+='<p><a href="'+html.escape(prediction_file.split('; ')[0])+'">11 existing prediction panels</a></p>'
        if case.get('t0_prediction_path'): links+='<p><a href="'+html.escape(prediction_file.split('; ')[-1])+'">T0 FP overlay</a></p>'
        html_rows.append(f"<section id='{cid}'><h2>{cid}</h2><p>{split} / {filename}; native location (y={y}, x={x}); crop [y0,x0,y1,x1]={case['bbox']}; purpose={purpose}</p><img src='{html.escape(out.name)}' style='max-width:100%;height:auto' alt='Original RGB, GT mask, FOV, full skeleton and S3 target panels'>{links}</section>")
        markdown_rows += [f"## {case['case_id']}", '', f"- Split / image: {case['split']} / `{case['filename']}`", f"- Exact location: `(y={y}, x={x})`; native crop `[y0,x0,y1,x1]={case['bbox']}`", f"- Selection purpose: {case['purpose']}", '', f"![{case['case_id']}: RGB, GT mask, FOV, full skeleton and S3 training target]({out.name})", '']
        if case['split']=='dev':
            markdown_rows += [f"[Open the 11 existing prediction panels]({prediction_file.split('; ')[0]})", '']
        if case.get('t0_prediction_path'):
            markdown_rows += [f"[Open the seed-2026 T0 FP illustration]({prediction_file.split('; ')[-1]})", '']
    page='<!doctype html><html><head><meta charset="utf-8"><title>H1 Human Review Atlas</title><style>body{font-family:Arial,sans-serif;max-width:1500px;margin:2em auto;padding:0 1em}section{border-top:2px solid #888;padding-top:1em;margin-top:2em}img{border:1px solid #555}.note{background:#fff3cd;padding:1em}</style></head><body><h1>H1 真实人工审核图册</h1><div class="note"><b>请研究者本人判断并填写空白审核表。</b>本图册不含 AI 判断或结论；图像和掩膜仅供核对冻结目标与既有预测。红圈为选例位置，缩放只用于显示。Train 的 S3 是唯一拟用的辅助训练目标；U 不从训练目标中扣除。</div><p>审核记录：manual_review.csv。每例需记录真实审核人、时间、定义对齐、可见性/标签局限、总体 Accept/Pause 及可引用的人工证据。不要把未确定的可见血管外观强行判成确定。</p>{''.join(html_rows)}</body></html>'
    (atlas/'index.html').write_text(page,encoding='utf-8')
    (atlas/'REVIEW.md').write_text('\n'.join(markdown_rows)+'\n',encoding='utf-8')


def run():
    if OUT.exists(): raise RuntimeError('Output path already exists; never overwrite an existing attempt')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('C0 must run with CUDA disabled')
    if torch.cuda.is_initialized(): raise RuntimeError('Unexpected CUDA initialization')
    if sha(PROTO)!=PROTO_SHA: raise RuntimeError('Frozen protocol SHA mismatch')
    protocol=json.loads(PROTO.read_text()); train_manifest=json.loads((PROTO_DIR/'train_manifest.json').read_text()); dev_ref=json.loads((PROTO_DIR/'dev_reference_manifest.json').read_text()); planb=json.loads(PLANB.read_text())
    if sha(TRAIN_DIR/'config.json')!=protocol['controls']['base_config_sha256']:
        raise RuntimeError('Frozen baseline training config hash mismatch')
    OUT.mkdir(parents=True,exist_ok=False); gate_dir=OUT/'implementation_gate'; gate_dir.mkdir()
    write_json(gate_dir/'run_started.json',{'protocol_sha256':PROTO_SHA,'started_unix':time.time(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_updates':0,'model_forward_calls':0,'official_test_accessed':False})
    synthetic=synthetic_checks()
    image_cache,target_rows,plan=bind_and_build_train(protocol,train_manifest,gate_dir)
    write_csv(gate_dir/'train_target_counts.csv',target_rows)
    write_csv(gate_dir/'crop_plan_and_counts.csv',plan)
    by_image={}
    by_size={}
    for r in plan:
        by_image.setdefault(r['filename'],[]).append(r)
        by_size.setdefault(f"{r['native_height']}x{r['native_width']}",[]).append(r)
    crop_summary={'all_8000':{k:distribution([r[k] for r in plan]) for k in ('N_FOV','N_positive_V','N_S3')},
        'empty_S3_crops':sum(r['empty_S3'] for r in plan),'empty_S3_fraction':float(np.mean([r['empty_S3'] for r in plan])),
        'per_image':{n:{'draws':len(rr),'empty_S3_crops':sum(r['empty_S3'] for r in rr),'empty_S3_fraction':float(np.mean([r['empty_S3'] for r in rr])),
            'N_S3_distribution':distribution([r['N_S3'] for r in rr])} for n,rr in by_image.items()},
        'per_native_size':{sz:{'draws':len(rr),'empty_S3_crops':sum(r['empty_S3'] for r in rr),'empty_S3_fraction':float(np.mean([r['empty_S3'] for r in rr])),
            'N_S3_distribution':distribution([r['N_S3'] for r in rr])} for sz,rr in by_size.items()}}
    write_json(gate_dir/'crop_summary.json',crop_summary)
    evaluation,prediction=recalc_existing_evaluation(protocol,dev_ref,planb,gate_dir)
    cases,selections=selection_and_atlas(protocol,train_manifest,dev_ref,planb,image_cache,prediction,gate_dir)
    target_total=sum(r['S3_train_pixels'] for r in target_rows)
    allpassed=bool(synthetic['passed'] and len(target_rows)==110 and len(plan)==8000 and target_total>0 and
        len(evaluation)==28 and all(r['old_7_metrics_exact_recalc'] for r in evaluation) and
        sum(r['new_full_domain_denominator'] for r in evaluation)==7291 and len(cases)<=22)
    binding={'protocol_sha256':PROTO_SHA,'train_manifest_sha256':sha(PROTO_DIR/'train_manifest.json'),
        'dev_reference_manifest_sha256':sha(PROTO_DIR/'dev_reference_manifest.json'),'source_manifest_sha256':sha(PROTO_DIR/'source_manifest.json'),
        'baseline_config_sha256':sha(TRAIN_DIR/'config.json'),'baseline_pairing_sha256':sha(TRAIN_DIR/'pairing.jsonl'),
        'baseline_endpoint_metrics_sha256':sha(TRAIN_DIR/'evaluation/step_002000/metrics.json'),
        'baseline_checkpoint_sha256':protocol['sources']['baseline_checkpoint_sha256'],
        'entry_sha256':sha(Path(__file__)),'scope':{'train_images':110,'train_draws':8000,'baseline_metric_images':28,'new_endpoint_points':7291,'human_cases_max':22},
        'environment':{'python':sys.version,'numpy':np.__version__,'torch':str(torch.__version__),
            'device':'cpu_only','model_inference':0,'gpu_updates':0,'official_test_accessed':False}}
    write_json(gate_dir/'execution_binding.json',binding)
    verification={'status':'C0_CPU_READY_H1_REQUIRED' if allpassed else 'C0_BLOCKED','all_cpu_checks_passed':allpassed,
        'protocol_sha256':PROTO_SHA,'synthetic_loss_gradient_checks':synthetic,'train_images':len(target_rows),
        'train_S3_total_pixels':target_total,'train_images_with_zero_S3':sum(r['S3_train_pixels']==0 for r in target_rows),
        'train_draws':len(plan),'draws_matching_frozen_T0_inputs_and_geometry':sum(r['input_sha256_matches_frozen_T0'] for r in plan),
        'empty_S3_crops':crop_summary['empty_S3_crops'],'empty_S3_crop_fraction':crop_summary['empty_S3_fraction'],
        'old_evaluation_images_recalculated':len(evaluation),'old_metric_rows_exact':sum(r['old_7_metrics_exact_recalc'] for r in evaluation),
        'new_full_domain_S3_Omega_points':sum(r['new_full_domain_denominator'] for r in evaluation),
        'new_endpoint_NA_images':sum(r['new_full_domain_denominator']==0 for r in evaluation),
        'old_near_valid_images':9,'old_near_points':77,'old_near_NA_images':19,'manual_cases_available':len(cases),
        'manual_review_status':'NOT_PERFORMED; blank human review form generated','H1_passed':False,
        'GPU_updates':0,'model_inference_calls':0,'official_test_accessed':False,
        'limitations':['No real researcher review is recorded.','Atlas selections are deterministic and pre-candidate; no AI visual judgments were made.'],
        'evidence_paths':{'targets':'targets/train/','crop_plan':'implementation_gate/crop_plan_and_counts.csv','crop_summary':'implementation_gate/crop_summary.json',
            'evaluation_recalc':'implementation_gate/existing_t0_recalculation.csv','selection':'implementation_gate/manual_selection.csv',
            'review_form':'implementation_gate/manual_review.csv','atlas':'implementation_gate/atlas/index.html','binding':'implementation_gate/execution_binding.json'}}
    write_json(gate_dir/'verification.json',verification)
    write_json(gate_dir/'run_completed.json',{'completed_unix':time.time(),'gpu_updates':0,'model_forward_calls':0,'official_test_accessed':False})
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    if not allpassed: raise SystemExit(2)


def refresh_atlas():
    """Re-render only the already-selected human atlas after a display-only fix."""
    if not OUT.is_dir() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise RuntimeError('Atlas refresh requires the existing C0 output and CPU-only process')
    protocol=json.loads(PROTO.read_text()); train=json.loads((PROTO_DIR/'train_manifest.json').read_text())
    dev=json.loads((PROTO_DIR/'dev_reference_manifest.json').read_text()); planb=json.loads(PLANB.read_text())
    gate=OUT/'implementation_gate'; cache={}
    target_dir=OUT/'targets/train'
    for row in train['rows']:
        rgb,v,fov=load_rgb_v_fov(row,protocol['controls']['base_config']['data_root'])
        g,sk,radius,s3=s3_target(v,fov)
        with np.load(target_dir/(row['filename']+'.npz'),allow_pickle=False) as z:
            saved=z['s_train'].astype(bool)
        if not np.array_equal(saved,s3):
            raise RuntimeError('Saved Train target differs during atlas refresh: '+row['filename'])
        endpoint=convolve(sk.astype(np.uint8),EIGHT,mode='constant',cval=0)==1
        cache[row['filename']]=(rgb,v,fov,g,sk,radius,s3,endpoint,np.zeros_like(fov))
    cases,selection=selection_and_atlas(protocol,train,dev,planb,cache,{},gate)
    binding=json.loads((gate/'execution_binding.json').read_text()); binding['entry_sha256']=sha(Path(__file__))
    write_json(gate/'execution_binding.json',binding)
    verification=json.loads((gate/'verification.json').read_text())
    atlas=gate/'atlas'
    verification['manual_cases_available']=len(cases)
    verification['atlas_sha256s']={p.name:sha(p) for p in sorted(atlas.iterdir()) if p.is_file()}
    verification['execution_binding_sha256']=sha(gate/'execution_binding.json')
    verification['atlas_display_revision']='colored exact S3 pixels with 11x inset; pixels unchanged'
    write_json(gate/'verification.json',verification)
    print(json.dumps({'cases':len(cases),'atlas_files':len(verification['atlas_sha256s']),
        'review_rows':sum(1 for _ in (gate/'manual_review.csv').open(encoding='utf-8'))-1,
        'human_review_status':verification['manual_review_status'],'GPU_updates':0,'model_inference_calls':0},ensure_ascii=False,indent=2))


if __name__=='__main__':
    if '--refresh-atlas' in sys.argv:
        refresh_atlas()
    else:
        run()
