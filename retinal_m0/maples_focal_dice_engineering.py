"""D-only controller: CONT50 versus SPLIT25+25, never scientific training."""
from __future__ import annotations
import argparse
import ctypes
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from retinal_m0.maples_g1_safety import (
    GateError, atomic_json, canonical_sha, exclusive_lock, file_sha, read_json,
    process_identity, _unique_pairs)
from retinal_m0.maples_focal_dice_engineering_core import Budget, empty_budget

ROOT = Path('${RETINA_PROJECT_ROOT}')
CODE = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1'
DIAG = BASE / 'focal_dice_gpu_diag_v1_20261024'
OUT = BASE / 'focal_dice_engineering_100_v1_20261024'
T0 = ROOT / 'experiments/t0_tw_core_20261016/runs/seed_2026/T0'
BINDING = OUT / 'execution_binding.json'
RELEASE = OUT / 'ENGINEERING_RELEASE.json'
RECEIPT = OUT / 'CPU_PREFLIGHT_REVIEW.json'
LEDGER = OUT / 'engineering_ledger.json'
SESSION = OUT / 'RUN_SESSION.json'
SEGMENT = OUT / 'CURRENT_SEGMENT.json'
GPU_UUID = '${GPU_UUID}'
PHASE = 'D_FOCAL_DICE_ENGINEERING_100_V1'
RELEASE_ID = 'sol_D_engineering100_20261024_01'
START = '<!-- D_FOCAL_DICE_ENG100_RELEASE_CURRENT -->'
END = '<!-- /D_FOCAL_DICE_ENG100_RELEASE_CURRENT -->'
SEGMENTS = {'CONT': ('CONT', 0, 50), 'SPLIT_FIRST': ('SPLIT', 0, 25), 'SPLIT_RESUME': ('SPLIT', 25, 50)}


def write_new(path, obj):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def data_guard(data_root, train_paths):
    """Content reads restricted to frozen Train paths; metadata remains readable."""
    root = Path(data_root).resolve()
    allowed = {Path(p).resolve() for p in train_paths}

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if root in path.parents and path.suffix.lower() in ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.npy', '.npz'] and path not in allowed:
            raise GateError('non-Train dataset content access refused')
    sys.addaudithook(guard)


def resources(idle=False):
    values = [x.strip() for x in subprocess.check_output([
        'nvidia-smi', '--id=0', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip().split(',')]
    if len(values) != 5 or values[0] != '0' or values[1] != GPU_UUID:
        raise GateError('physical GPU0 UUID mismatch')
    pids = [int(x) for x in subprocess.check_output([
        'nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        text=True, timeout=10).splitlines() if x.strip()]
    snap = {'index': 0, 'uuid': values[1], 'total_MiB': int(values[2]),
            'used_MiB': int(values[3]), 'util_percent': int(values[4]),
            'compute_pids': pids, 'GPU1_queried': False}
    if idle and (pids or snap['used_MiB'] >= 512 or snap['util_percent'] >= 10):
        raise GateError('GPU0 busy; no lock stealing, killing or fallback GPU')
    if not idle and any(pid != os.getpid() for pid in pids):
        raise GateError('another GPU0 compute process appeared')
    if shutil.disk_usage(OUT).free < 20 * 1024**3:
        raise GateError('less than20GiB free checkpoint/evidence space')
    return snap


def wait_idle_after_own_worker():
    until = time.monotonic() + 10
    while True:
        snap = resources()
        if not snap['compute_pids'] and snap['used_MiB'] < 512 and snap['util_percent'] < 10:
            return snap
        if time.monotonic() >= until:
            raise GateError('GPU0 did not become idle after own worker')
        time.sleep(.5)


def current_status(text):
    if text.count(START) != 1 or text.count(END) != 1:
        raise GateError('exactly one current D engineering release block required')
    a, b = text.index(START) + len(START), text.index(END)
    if b <= a:
        raise GateError('malformed D engineering status block')
    return json.loads(text[a:b], object_pairs_hook=_unique_pairs)


def update_status(**changes):
    path = CODE / 'STATUS.md'
    original = path.read_text()
    state = current_status(original)
    state.update(changes)
    a, b = original.index(START) + len(START), original.index(END)
    new = original[:a] + '\n' + json.dumps(state, ensure_ascii=False, separators=(',', ':')) + '\n' + original[b:]
    if path.read_text() != original:
        raise GateError('STATUS changed concurrently; no overwrite')
    pending = path.with_name('STATUS.md.D_engineering.pending')
    with pending.open('x', encoding='utf8') as stream:
        stream.write(new)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def authority(*, full_sources=False):
    state = current_status((CODE / 'STATUS.md').read_text())
    if state.get('state') != 'RELEASED' or state.get('phase') != PHASE:
        raise GateError('D engineering100 not independently RELEASED')
    for path in [BINDING, RELEASE, RECEIPT, LEDGER]:
        if not path.is_file() or path.is_symlink():
            raise GateError('missing/symlink D engineering evidence')
    b, release, receipt = read_json(BINDING), read_json(RELEASE), read_json(RECEIPT)
    scope = {'phase': PHASE, 'gamma': 1, 'class_weights': None,
             'global_charged_cap': 100, 'track_caps': {'CONT': 50, 'SPLIT': 50},
             'segments': [['CONT', 0, 50], ['SPLIT', 0, 25], ['SPLIT', 25, 50]],
             'pause_ratio_gt': 2, 'GPU0_uuid': GPU_UUID, 'forward_cap': 100,
             'gradient_cap': 300, 'stage_GPU_wall_seconds': 1800,
             'Dev_Test_inference_or_scientific_training_allowed': False}
    for key, value in scope.items():
        if type(b.get(key)) is not type(value) or b[key] != value:
            raise GateError('D engineering scope drift: ' + key)
    digest = canonical_sha(b)
    expected = {'phase': PHASE, 'authorized': True, 'release_id': RELEASE_ID,
                'global_charged_cap': 100, 'gpu_uuid': GPU_UUID,
                'binding_sha256': digest, 'binding_file_sha256': file_sha(BINDING),
                'cpu_review_sha256': file_sha(RECEIPT), 'ledger_path': str(LEDGER)}
    for key, value in expected.items():
        if type(release.get(key)) is not type(value) or release[key] != value:
            raise GateError('D engineering release mismatch: ' + key)
    if (state.get('release_id') != release['release_id'] or state.get('binding_sha256') != digest
            or state.get('release_file_sha256') != file_sha(RELEASE)):
        raise GateError('STATUS/release execution binding mismatch')
    if (receipt.get('status') != 'CPU_ENGINEERING_TESTS_ACCEPTED_GPU_HELD'
            or receipt.get('binding_sha256') != digest or receipt.get('tests_passed') is not True):
        raise GateError('CPU engineering receipt mismatch')
    for path, sha in receipt['evidence_sha256'].items():
        if file_sha(path) != sha:
            raise GateError('CPU evidence drift: ' + path)
    for path, sha in b['sources_sha256'].items():
        if (full_sources or path in b['runtime_guard_paths']) and file_sha(path) != sha:
            raise GateError('source/data drift: ' + path)
    return b


def worker_authority(track, start, end):
    b = authority()
    session, active = read_json(SESSION), read_json(SEGMENT)
    if session.get('binding_sha256') != canonical_sha(b) or session.get('status') != 'RUNNING':
        raise GateError('no active engineering controller')
    parent = session['controller']
    if parent['pid'] != os.getppid() or process_identity(parent['pid']) != parent:
        raise GateError('worker not direct child of live lock-owning controller')
    if time.time() >= session['deadline_unix']:
        raise GateError('whole-stage30min deadline reached')
    if active.get('binding_sha256') != canonical_sha(b) or active.get('segment') != [track, start, end]:
        raise GateError('worker segment outside active approval')
    return b


def prepare():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from retinal_m0.maples_core import CorePatches
    from torch.utils.data._utils.collate import default_collate
    from retinal_m0.maples_focal_dice_engineering_core import pair_record
    diag = read_json(DIAG / 'completion.json')
    review = read_json(DIAG / 'SOL_GPU_DIAGNOSTIC_REVIEW.json')
    if diag['completed'] != 16 or diag['parameter_updates'] != 0 or review['accepted'] is not True:
        raise GateError('independent D diagnostic admission missing')
    old_binding = read_json(DIAG / 'execution_binding.json')
    accepted = old_binding['source_sha256']
    for path, sha in accepted.items():
        if file_sha(path) != sha:
            raise GateError('D diagnostic accepted source changed: ' + path)
    config = read_json(T0 / 'config.json')
    old_train = read_json(BASE / 'soft_cldice_engineering_100_v1_20261023/execution_binding.json')['sources_sha256']
    ds = CorePatches(config, 'T0')
    train_sources = {}
    for row in ds.native.rows:
        for key in ['image_path', 'label_vessels', 'label_hemorrhages', 'label_brightuncertains', 'label_reduncertains']:
            path = str((Path(config['data_root']) / row[key]).resolve())
            if path not in old_train or file_sha(path) != old_train[path]:
                raise GateError('Train source differs from frozen original source')
            train_sources[path] = old_train[path]
    if len(ds.native.rows) != 110:
        raise GateError('not the original Train110')
    data_guard(config['data_root'], train_sources)
    pairs = [json.loads(x) for x in (T0 / 'pairing.jsonl').read_text().splitlines()][:50]
    actual = []
    for step in range(1, 51):
        batch = default_collate([ds[i] for i in range((step - 1) * 4, step * 4)])
        pair = pair_record(batch, step)
        if pair != pairs[step - 1]:
            raise GateError('first200 native inputs disagree with original T0')
        actual.append(pair)
    files = [T0 / 'config.json', T0 / 'pairing.jsonl', Path(config['source_m0']),
             Path(config['normalization_source']), CODE / 'configs/development_v1.json',
             DIAG / 'SOL_GPU_DIAGNOSTIC_REVIEW.json', DIAG / 'completion.json',
             OUT / 'USER_AUTHORIZATION.json', OUT / 'ENGINEERING_100_PROTOCOL.md']
    files += [CODE / 'retinal_m0' / name for name in [
        'maples_focal_dice.py', 'maples_focal_dice_engineering.py',
        'maples_focal_dice_engineering_core.py', 'maples_soft_cldice_engineering_core.py',
        'maples_soft_cldice.py', 'maples_core.py', 'model.py', 'state.py',
        'maples_g1_safety.py', 'maples_g2_engineering.py']]
    files += [CODE / 'retinal_data' / name for name in ['fives.py', 'maples_native.py', 'maples_fov.py']]
    files += [CODE / 'tests/test_maples_focal_dice_engineering.py']
    formal = read_json(ROOT / 'experiments/t0_tw_core_20261016/execution_binding.json')
    for rel in ['retinal_m0/maples_core.py', 'retinal_m0/model.py', 'retinal_data/fives.py',
                'retinal_data/maples_native.py', 'retinal_data/maples_fov.py', 'configs/development_v1.json']:
        if file_sha(CODE / rel) != formal['sources'][rel]:
            raise GateError('original formal code changed: ' + rel)
    sources = {**train_sources, **{str(path): file_sha(path) for path in files}}
    if file_sha(config['source_m0']) != config['source_m0_sha256']:
        raise GateError('original M0 drift')
    binding = {'phase': PHASE, 'gamma': 1, 'class_weights': None,
               'global_charged_cap': 100, 'track_caps': {'CONT': 50, 'SPLIT': 50},
               'segments': [list(x) for x in SEGMENTS.values()], 'pause_ratio_gt': 2,
               'GPU0_uuid': GPU_UUID, 'memory_allocator_cap_bytes': 8 * 1024**3,
               'memory_fraction_cap': .8, 'forward_cap': 100, 'gradient_cap': 300,
               'audit_every_attempt': True, 'AMP_overflow': 'stop_without_retry',
               'stage_GPU_wall_seconds': 1800, 'sources_sha256': sources,
               'runtime_guard_paths': [str(path) for path in files if str(path) != config['source_m0']],
               'train_content_paths': sorted(train_sources), 'data_root': config['data_root'],
               'M0_path': config['source_m0'], 'M0_sha256': config['source_m0_sha256'],
               'config_path': str(T0 / 'config.json'), 'expected_pairs': actual,
               'torch_version': '2.4.0+cu121',
               'Dev_Test_inference_or_scientific_training_allowed': False}
    write_new(BINDING, binding)
    write_new(OUT / 'cpu_preflight.json', {
        'status': 'PASSED_CPU_ONLY', 'binding_sha256': canonical_sha(binding),
        'original_T0_steps': 50, 'paired_draws': 200, 'Train_images': 110,
        'Train_content_SHA_count': len(train_sources), 'new_M0_forwards': 0,
        'GPU_initialized': False, 'Dev_Test_content_used': False})
    print('PREPARED', canonical_sha(binding), flush=True)


def worker(name):
    track, start, end = SEGMENTS[name]
    b = worker_authority(track, start, end)
    # Linux parent-death signal prevents an orphaned worker holding CUDA.
    parent_pid = os.getppid()
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != parent_pid:
        raise GateError('parent-death guard installation failed')
    resources(idle=True)
    data_guard(b['data_root'], b['train_content_paths'])
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUID
    import torch
    import numpy as np
    from retinal_m0.model import UNet
    from retinal_m0.maples_core import CorePatches
    from retinal_m0.state import state_digest
    from retinal_m0.maples_focal_dice_engineering_core import run_segment
    if torch.cuda.is_initialized() or str(torch.__version__) != b['torch_version']:
        raise GateError('early CUDA initialization or torch version drift')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.device_count() != 1:
        raise GateError('only authorized GPU0 may be visible')
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(min(.8, 8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory), 0)
    config, arch = read_json(b['config_path']), read_json(CODE / 'configs/development_v1.json')
    model = UNet(arch['widths'], arch['group_norm_groups']).cuda()
    if start == 0:
        cp = torch.load(b['M0_path'], map_location='cpu', weights_only=True)
        if cp['step'] != 10000:
            raise GateError('wrong M0 checkpoint step')
        model.load_state_dict(cp['model'], strict=True)
        if state_digest(model.state_dict()) != state_digest(cp['model']):
            raise GateError('loaded M0 tensor mismatch')
        del cp
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'], betas=tuple(config['betas']),
                                 eps=config['eps'], foreach=False)
    scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=config['amp_initial_scale'])
    ds = CorePatches(config, 'T0')
    result = run_segment(model=model, optimizer=optimizer, scaler=scaler, dataset=ds,
                         track=track, start=start, end=end, out=OUT / 'runs' / track,
                         budget=Budget(LEDGER, canonical_sha(b)), binding_sha=canonical_sha(b),
                         expected_pairs=b['expected_pairs'], device='cuda:0', gamma=1)
    write_new(OUT / (name + '_completion.json'), result)


def compare_tracks():
    import torch
    from retinal_m0.state import state_digest
    cont = [json.loads(x) for x in (OUT / 'runs/CONT/train.jsonl').read_text().splitlines()]
    split = [json.loads(x) for x in (OUT / 'runs/SPLIT/train.jsonl').read_text().splitlines()]
    if len(cont) != 50 or len(split) != 50:
        raise GateError('incomplete paired engineering logs')
    fields = ['step', 'pair', 'A0_loss', 'delta_loss', 'loss', 'gamma', 'audit',
              'GT_count', 'FOV_count', 'per_crop_losses', 'loss_scale_before',
              'loss_scale_after', 'preclip_total_L2', 'state_digests']
    for a, b in zip(cont, split):
        for field in fields:
            if a[field] != b[field]:
                raise GateError(f'paired exact mismatch step{a["step"]}: {field}')
    ca = torch.load(OUT / 'runs/CONT/last.pt', map_location='cpu', weights_only=True)
    cb = torch.load(OUT / 'runs/SPLIT/last.pt', map_location='cpu', weights_only=True)
    for key in ['model', 'optimizer', 'scaler', 'rng', 'loader_generator']:
        if state_digest(ca[key]) != state_digest(cb[key]):
            raise GateError('terminal checkpoint mismatch: ' + key)
    restore = read_json(OUT / 'runs/SPLIT/restore_025.json')
    if restore['new_pid'] == restore['checkpoint_pid']:
        raise GateError('resume did not use a new process')
    return cont, split


def controller():
    b = authority(full_sources=True)
    with exclusive_lock(OUT / 'RUN.lock'), exclusive_lock(ROOT / 'experiments/a14_gpu0.lock'):
        if read_json(LEDGER) != empty_budget(canonical_sha(b)):
            raise GateError('initial zero ledger required; no implicit restart')
        budget = Budget(LEDGER, canonical_sha(b))
        started = time.monotonic()
        write_new(SESSION, {'status': 'RUNNING', 'controller': process_identity(),
                           'binding_sha256': canonical_sha(b), 'segments': list(SEGMENTS),
                           'deadline_unix': time.time() + 1800})
        update_status(run_state='RUNNING', active_controller=process_identity(), charged_updates=0)
        try:
            for name, segment in SEGMENTS.items():
                authority(full_sources=True)
                wait_idle_after_own_worker()
                remaining = 1800 - (time.monotonic() - started)
                if remaining <= 0:
                    raise GateError('whole-stage30min budget exhausted')
                atomic_json(SEGMENT, {'binding_sha256': canonical_sha(b), 'segment': list(segment), 'name': name})
                with (OUT / (name + '.log')).open('x') as log:
                    result = subprocess.run([sys.executable, '-B', '-u', '-m',
                                             'retinal_m0.maples_focal_dice_engineering', '--worker', name],
                                            cwd=CODE, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
                if result.returncode != 0:
                    raise GateError('worker stopped: ' + name)
                print('SEGMENT_COMPLETE', name, flush=True)
                update_status(run_state='RUNNING', last_segment_completed=name,
                              charged_updates=budget.snapshot()['charged_events'])
            cont, split = compare_tracks()
            s = budget.snapshot()
            if (s['verified_successes'] != 100 or s['charged_events'] != 100
                    or s['optimizer_call_starts'] != 100 or s['forward_calls'] != 100
                    or s['gradient_calls'] != 300 or s['failed_events'] or s['uncertain_events']
                    or s['failed_compute'] or s['compute_pending']):
                raise GateError('100-update accounting not clean')
            authority(full_sources=True)
            wait_idle_after_own_worker()
            summary = {'status': 'D_ENGINEERING_100_EXACT_RESTORE_PASSED_PENDING_SOL_REVIEW',
                       'binding_sha256': canonical_sha(b), 'charged_updates': 100,
                       'verified_successful_updates': 100, 'forwards': 100, 'gradient_calls': 300,
                       'failed_updates': 0, 'uncertain_updates': 0, 'paired_steps_exact': 50,
                       'new_process_restore_exact': True, 'gamma': 1,
                       'max_delta_to_A0': max(x['audit']['delta_to_A0'] or 0 for x in cont + split),
                       'negative_cosine_updates': sum(x['audit']['cosine_A0_delta'] is not None and x['audit']['cosine_A0_delta'] < 0 for x in cont + split),
                       'GPU_reserved_peak_MiB': max(x['GPU_peak_reserved_MiB'] for x in cont + split),
                       'elapsed_seconds': time.monotonic() - started,
                       'Dev_Test_content_used': False, 'GPU1_queried_or_used': False,
                       'scientific_training_authorized': False}
            write_new(OUT / 'completion.json', summary)
            atomic_json(SESSION, {**read_json(SESSION), 'status': 'COMPLETE_PENDING_SOL_REVIEW'})
            update_status(state='CONSUMED', run_state='COMPLETE_PENDING_SOL_REVIEW',
                          charged_updates=100, successful_updates=100, active_controller=None,
                          next='Sol reviews100 updates; no scientific training authorization')
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        except BaseException as exc:
            for track, value in budget.snapshot()['tracks'].items():
                if value.get('pending') is not None:
                    budget.resolve(track, 'uncertain')
            budget.stop(repr(exc))
            atomic_json(SESSION, {**read_json(SESSION), 'status': 'FAILED_STOP'})
            write_new(OUT / 'failure.json', {'error': repr(exc), 'budget': budget.snapshot(),
                                           'no_auto_restart': True, 'scientific_training_authorized': False})
            update_status(state='HOLD_FAILED', run_state='FAILED_STOP',
                          charged_updates=budget.snapshot()['charged_events'],
                          next='Sol diagnoses retained failure; no automatic retries')
            raise


def main():
    parser = argparse.ArgumentParser(description='D engineering100 only:50 versus25+25; no Dev/Test/science')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--run', action='store_true')
    group.add_argument('--worker', choices=list(SEGMENTS))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker:
        worker(args.worker)
    else:
        controller()


if __name__ == '__main__':
    main()
