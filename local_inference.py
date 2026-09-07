"""Local inference for the restored FIVES development models; does not train."""
import argparse
import sys
sys.dont_write_bytecode = True
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from retinal_release.common import ROOT, read, digest
from retinal_release.freeze import make_fov
from retinal_m0.model import UNet
from retinal_m1.data import SkeletonDataset
from retinal_m1.inference import probability
from retinal_m1.metrics import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('m0', 'm1'), default='m1')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--image', type=Path, help='Centered square FIVES-style fundus RGB image')
    parser.add_argument('--fov', type=Path, help='Optional explicit field-of-view mask for the input image')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 4:
        parser.error('Use batch size 1 to 4 on the 8 GB laptop')
    if args.fov and not args.image:
        parser.error('--fov requires --image')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable')
    torch.set_num_threads(4)
    run = 'm0_seed2026_dev10k_w1' if args.model == 'm0' else 'm1_lambda01_seed2026_dev10k_w1'
    checkpoint = ROOT / 'runs/prefetch_round1' / run / 'last.pt'
    manifest = read(ROOT.parent / 'retinal-release-assets/FILE_MANIFEST.json')
    expected = next(r['sha256'] for r in manifest['files'] if r['path'] == checkpoint.relative_to(ROOT).as_posix())
    actual = digest(checkpoint)
    if actual != expected:
        raise ValueError('Checkpoint SHA-256 mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['step'] != 10000:
        raise ValueError('Expected the completed 10,000-step development checkpoint')
    config = read(ROOT / 'configs/development_v1.json')
    model = UNet(config['widths'], config['group_norm_groups'])
    model.load_state_dict(saved['model'], strict=True)
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('Nonfinite model weights')
    del saved
    model.cuda().eval()
    item = None
    if args.image:
        with Image.open(args.image) as source:
            rgb = np.asarray(source.convert('RGB')).copy()
        if args.fov:
            with Image.open(args.fov) as source:
                field = np.asarray(source.convert('L')) > 0
            if field.shape != rgb.shape[:2] or not field.any():
                raise ValueError('FOV must be nonempty and match the image dimensions')
        else:
            if rgb.shape[0] != rgb.shape[1]:
                raise ValueError('Automatic FOV is only for centered square FIVES-style inputs; provide --fov')
            field = make_fov(rgb)
        stats = read(ROOT / 'releases/development_v1/normalization.json')
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
        image = ((image - torch.tensor(stats['mean'])[:, None, None]) / torch.tensor(stats['std'])[:, None, None]) * torch.from_numpy(field)
        fov = torch.from_numpy(field)
        key = str(args.image.resolve())
    else:
        dataset = SkeletonDataset('validation', full=True, cache_size=1)
        if not 0 <= args.sample_index < len(dataset):
            raise ValueError(f'sample-index must be between 0 and {len(dataset)-1}')
        item = dataset[args.sample_index]
        image, fov, key = item['image'], item['fov'][0], item['key']
        with Image.open(ROOT / dataset.rows[args.sample_index]['image_path']) as source:
            rgb = np.asarray(source.convert('RGB')).copy()
    output = args.output or ROOT / '.runtime/local_inference' / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + args.model)
    output.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    prob, tiles = probability(model, image, fov, batch_size=args.batch_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    values = prob.numpy()
    mask = values >= 0.5
    if not np.isfinite(values).all() or values.shape != rgb.shape[:2]:
        raise ValueError('Invalid prediction')
    np.save(output / 'probability.npy', values, allow_pickle=False)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / 'vessels.png')
    Image.fromarray(rgb).save(output / 'input.png')
    overlay = rgb.copy()
    overlay[mask] = (rgb[mask].astype(np.float32) * .5 + np.array([0, 255, 70]) * .5).astype(np.uint8)
    Image.fromarray(overlay).save(output / 'overlay.png')
    result = dict(status='passed_local_inference', created_at=datetime.now(timezone.utc).isoformat(),
                  model=args.model, checkpoint_sha256=actual, training_step=10000, image=key,
                  scope='single_image_deployment_smoke_not_formal_evaluation',
                  gpu=torch.cuda.get_device_name(0), batch_size=args.batch_size, tiles=tiles,
                  shape=list(values.shape), inference_seconds=elapsed,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), threshold=.5,
                  cross_gpu_equivalence='not_established', output=str(output.resolve()))
    if item is not None:
        result['metrics'] = score(values, item['mask'][0].numpy(), fov.numpy(), item['thin'][0].numpy(), threshold=.5)
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
