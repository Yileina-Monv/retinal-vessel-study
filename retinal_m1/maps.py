"""Whole-image, FOV-clipped targets. No pruning or crop-local skeletonization."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion
from skimage.morphology import skeletonize, disk
from retinal_data.fives import decode_label
from retinal_release.common import ROOT, read, write, digest, safe

DEST = 'releases/m1_pretrial_v1'

def targets(label, fov):
    label, fov = np.asarray(label, bool), np.asarray(fov, bool)
    if label.ndim != 2 or label.shape != fov.shape:
        raise ValueError('Matching 2D masks required')
    foreground = label & fov
    skeleton = skeletonize(foreground, method='zhang')
    # Explicit background padding also defines EDT for all-foreground fixtures.
    distance = distance_transform_edt(np.pad(foreground, 1))[1:-1, 1:-1].astype(np.float32)
    tube = binary_dilation(skeleton, structure=disk(2)) & foreground
    skeleton_radius = np.where(skeleton, distance, 0).astype(np.float32)
    tube_radius = np.zeros(label.shape, np.float32)
    if skeleton.any():
        nearest = distance_transform_edt(~skeleton, return_distances=False, return_indices=True)
        tube_radius[tube] = skeleton_radius[tuple(nearest[:, tube])]
    return dict(skeleton=skeleton, tube=tube, skeleton_radius=skeleton_radius, tube_radius=tube_radius)

def interior(fov):
    return binary_erosion(np.asarray(fov, bool), structure=disk(2), border_value=0)

def build_one(args):
    root, row = args
    root = Path(root)
    for role in ('label', 'fov'):
        if digest(safe(root, row[role+'_path'])) != row[role+'_sha256']:
            raise ValueError('Source mismatch: '+row['key'])
    label = decode_label(safe(root, row['label_path']).read_bytes())
    with Image.open(safe(root, row['fov_path'])) as im:
        fov = np.asarray(im) > 0
    arrays = targets(label, fov)
    path = root/DEST/'maps'/(Path(row['filename']).stem+'.npz')
    # Exclusive output: an interrupted build uses a fresh version or explicit cleanup.
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    s = arrays['skeleton']
    return dict(key=row['key'], split=row['trial_split'], path=path.relative_to(root).as_posix(),
        sha256=digest(path), label_sha256=row['label_sha256'], fov_sha256=row['fov_sha256'],
        shape=list(label.shape), skeleton_pixels=int(s.sum()), tube_pixels=int(arrays['tube'].sum()),
        boundary_skeleton_pixels=int((s & ~interior(fov)).sum()),
        raw_foreground_outside_fov=int((label.astype(bool) & ~fov).sum()))

def build(root=ROOT, workers=2):
    from retinal_release.verify import verify
    parent = verify(root)
    config = read(root/'configs/m1_pretrial_v1.json')
    if parent['release_id'] != config['parent_release_id']:
        raise ValueError('Parent changed')
    folder = root/DEST
    (folder/'maps').mkdir(parents=True, exist_ok=True)
    if any((folder/'maps').iterdir()):raise ValueError('Map output must be empty')
    with (root/'releases/development_v1/all.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    started = time.perf_counter()
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(build_one, ((str(root), r) for r in rows), chunksize=1):
            records.append(record)
            if len(records) % 25 == 0:
                print(f'maps {len(records)}/{len(rows)} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    radii = []
    for record in records:
        if record['split'] == 'train':
            with np.load(safe(root, record['path']), allow_pickle=False) as a:
                radii.append(a['skeleton_radius'][a['skeleton']])
    values = np.concatenate(radii)
    if len(radii) != 470 or not values.size:
        raise ValueError('Unexpected calibration cohort')
    tau, reference = np.quantile(values, [.25, .5], method='linear').tolist()
    calibration = dict(tau=tau, r_ref=reference, units='native_image_pixels_not_physical_diameter',
        scope='train_only', images=len(radii), skeleton_pixels=int(values.size), quantile_method='linear')
    # Thin maps are derived from sealed skeleton_radius and this scalar, exactly.
    write(folder/'calibration.json', calibration)
    write(folder/'maps.json', dict(parent_release_id=parent['release_id'], records=records,
        stored_arrays={'skeleton':'bool','tube':'bool','skeleton_radius':'float32','tube_radius':'float32'},
        thin_map='skeleton & (skeleton_radius <= calibration.tau)', seconds=time.perf_counter()-started))
    print(dict(status='maps_complete', count=len(records), **calibration), flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers', type=int, default=2);a=p.parse_args()
    if not 1 <= a.workers <= 4: raise ValueError('Use 1-4 workers for bounded RAM')
    build(workers=a.workers)
