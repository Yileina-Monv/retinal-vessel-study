"""One fixed final native-space Dev28 inference with branch-local reconstruction."""
import os, sys, argparse
from common import *


def worker(mode, parent, nonce):
    key, authorize = admit_worker(mode, 'science', 'infer', parent, nonce)
    out = STAGE / 'science' / mode
    os.environ.update(nnUNet_raw=str(BASE / 'raw'), nnUNet_preprocessed=str(BASE / 'preprocessed'),
                      nnUNet_results=str(out / 'inference_constructor'))
    import torch, numpy as np
    from PIL import Image
    from candidate import make_trainer, seed_cpu
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from replication_infer import prevent_native_retry, verify_tile_count
    complete = read(out / 'training_completion.json'); counts = complete['counts']
    if complete['status'] != 'TRAINING_COMPLETE' or counts['attempts'] != 25000 or counts['monitor_complete'] != 5000 or \
       counts['fatal'] or counts['unknown'] or counts['success'] + counts['skip'] != 25000 or counts['phase'] != 'complete':
        raise RuntimeError('Dev closed before complete final')
    final = Path(complete['checkpoint_final'])
    if sha(final) != complete['checkpoint_final_sha256']: raise RuntimeError('wrong final checkpoint')
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(10240 * 1048576 / torch.cuda.get_device_properties(0).total_memory)
    seed_cpu(); q = protocol()
    trainer = make_trainer(mode, torch.device('cpu'), out / 'inference_constructor', BASE)
    checkpoint = torch.load(final, map_location='cpu', weights_only=False)
    trainer.network.load_state_dict(checkpoint['network_weights'], strict=True)
    # CPU-verified final Tensor export; all8 training heads remain in saved state.
    trainer.network.decoder.deep_supervision = False
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=True, device=torch.device('cuda:0'), verbose=False,
        verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(trainer.network.to('cuda:0'), trainer.plans_manager,
        trainer.configuration_manager, [checkpoint['network_weights']], trainer.dataset_json,
        type(trainer).__name__, None)
    rows = read(q['references']['Dev_RGB_manifest'])['images']
    if [r['filename'] for r in rows] != q['Dev_ids']: raise RuntimeError('Dev whitelist/order changed')
    prediction = out / 'dev_predictions'; prediction.mkdir(exist_ok=False)
    inputs = out / 'dev_input'; inputs.mkdir(exist_ok=False)
    export = prediction / 'official_export'; export.mkdir(exist_ok=False)
    budget = {'binding': key, 'status': 'RUNNING', 'forwards': 0,
              'cap': sum(q['expected_tiles_by_image'].values()), 'images': {}}
    active = {'name': None}
    def forward(*_):
        authorize(); name = active['name']; image = budget['images'].get(name)
        if budget['status'] != 'RUNNING' or image is None or image['state'] != 'PENDING' or \
           image['forwards'] >= q['expected_tiles_by_image'][name] or budget['forwards'] >= budget['cap']:
            raise RuntimeError('unapproved extra terminal forward')
        image['forwards'] += 1; budget['forwards'] += 1; atomic(out / 'inference_budget.json', budget)
    def failure(exc):
        budget['status'] = 'FAILED_STOP'; budget['reason'] = repr(exc); atomic(out / 'inference_budget.json', budget)
    prevent_native_retry(predictor, failure, lambda: budget['status'] == 'RUNNING')
    handle = predictor.network.register_forward_pre_hook(forward); records = []
    try:
        for row in rows:
            authorize(); name = row['filename']; active['name'] = name; source = row['inputs']['RGB']
            if sha(source['path']) != source['sha256']: raise RuntimeError('Dev RGB changed')
            rgb = np.asarray(Image.open(source['path']).convert('RGB'))
            if list(rgb.shape[:2]) != row['native_shape']: raise RuntimeError('native RGB size changed')
            files = []
            for channel in range(3):
                p = inputs / (name + f'_{channel:04d}.png'); Image.fromarray(rgb[:, :, channel], mode='L').save(p); files.append(str(p))
            budget['images'][name] = {'state': 'PENDING', 'forwards': 0, 'attempts': 1}; atomic(out / 'inference_budget.json', budget)
            predictor.predict_from_files([files], [str(export / name)], save_probabilities=True,
                overwrite=False, num_processes_preprocessing=1, num_processes_segmentation_export=1)
            with np.load(export / (name + '.npz'), allow_pickle=False) as z: arr = z['probabilities']
            if arr.ndim == 4 and arr.shape[1] == 1: arr = arr[:, 0]
            if arr.shape != (2, *row['native_shape']): raise RuntimeError('inverse export channel/native shape mismatch')
            prob = np.asarray(arr[1], dtype=np.float32)
            if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)): raise RuntimeError('invalid vessel probability')
            verify_tile_count(budget['images'][name]['forwards'], q['expected_tiles_by_image'][name])
            path = prediction / (name + '.npy'); np.save(path, prob, allow_pickle=False)
            budget['images'][name]['state'] = 'COMPLETE'; atomic(out / 'inference_budget.json', budget)
            records.append({'filename': name, 'sha256': sha(path), 'shape': list(prob.shape), 'dtype': str(prob.dtype),
                            'actual_tile_forwards': budget['images'][name]['forwards'], 'peak_reserved_MiB': memory_check()})
        if len(records) != 28 or not 0 < budget['forwards'] <= budget['cap']: raise RuntimeError('incomplete Dev28 budget')
        budget['status'] = 'COMPLETE'; atomic(out / 'inference_budget.json', budget)
        atomic(out / 'prediction_verification.json', {'status': 'PASS', 'images': records, 'threshold': .5,
               'checkpoint_sha256': complete['checkpoint_final_sha256'], 'TTA': False, 'Test_accessed': False}, new=True)
    except BaseException as exc: failure(exc); raise
    finally: handle.remove()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--worker', action='store_true', required=True)
    p.add_argument('--mode', choices=['final'], required=True); p.add_argument('--parent', type=int, required=True); p.add_argument('--nonce', required=True)
    a = p.parse_args(); worker(a.mode, a.parent, a.nonce)
