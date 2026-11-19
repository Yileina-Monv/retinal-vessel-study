"""Full-budget paired trajectories, only after a separate post-engineering Sol release."""
import os, sys, time, json, argparse, signal, subprocess, shutil
from pathlib import Path
from common import *


def train_worker(mode, parent, nonce):
    key, authorize = admit_worker(mode, 'science', 'train', parent, nonce)
    out = STAGE / 'science' / mode
    os.environ.update(nnUNet_raw=str(BASE / 'raw'), nnUNet_preprocessed=str(BASE / 'preprocessed'),
                      nnUNet_results=str(out / 'nnunet_results'))
    import torch, numpy as np
    from fusion_model import make_trainer, seed_cpu, original_state, legacy_digest
    from seeded_runtime import install_seeded_worker_adapter
    from amp32_runner import bind_train_datasets
    from fast_observer import FastObserver, Journal, AmpLimits, digest, rng_pack
    if torch.cuda.is_initialized(): raise RuntimeError('premature CUDA before science admission')
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
    if torch.cuda.device_count() != 1: raise RuntimeError('one assigned GPU only')
    torch.cuda.set_per_process_memory_fraction(10240 * 1048576 / torch.cuda.get_device_properties(0).total_memory)
    seed_cpu(); torch.cuda.manual_seed_all(2027)
    t = make_trainer(mode, torch.device('cuda:0'), out / 'nnunet_results', BASE)
    backbone = legacy_digest(original_state(t.network)); wanted = read(N0 / 'initialization.json')['model']
    if backbone != wanted: raise RuntimeError('initial original backbone differs from N0')
    seeds = install_seeded_worker_adapter(2027)
    atomic(out / 'initialization.json', {'seed': 2027, 'mode': mode, 'backbone': backbone,
        'module': digest(t.network.decoder.stages[-1].detail.state_dict()), 'model': digest(t.network.state_dict()),
        'optimizer': digest(t.optimizer.state_dict()), 'scaler': digest(t.grad_scaler.state_dict()),
        'RNG': digest(rng_pack()), 'engineering_weights_loaded': False, 'fresh_initialization': True,
        'patch': [1024, 1536], 'batch': 2, 'binding_sha256': key}, new=True)
    q = protocol(); names = set(q['train_ids']); tr, va = bind_train_datasets(t, names)
    atomic(out / 'loader_sets.json', {'Train': sorted(tr.identifiers), 'Train_monitor': sorted(va.identifiers)}, new=True)
    expected = [json.loads(l) for l in (N0 / 'attempts.jsonl').read_text().splitlines()]
    expected_monitor = [json.loads(l)['ids'] for l in (N0 / 'monitor.jsonl').read_text().splitlines()]
    if len(expected) != 25000 or len(expected_monitor) != 5000: raise RuntimeError('baseline tapes incomplete')
    j = Journal(out / 'budget.json', key, 100, 250, 50, AmpLimits(5, 25))
    if j.read()['attempts'] or j.read()['status'] != 'READY': raise RuntimeError('science budget consumed; no automatic resume')
    def event(kind, counts, observer):
        memory_check()
        if kind == 'train' and counts['attempts'] == 1:
            want = [{'pool': 'train', 'worker_count': 1, 'seeds': [2027]},
                    {'pool': 'internal_train_monitor', 'worker_count': 1, 'seeds': [2028]}]
            if seeds != want: raise RuntimeError('actual augmentation seeds drift')
            atomic(out / 'actual_seed_pools.json', seeds, new=True)
        save = (kind == 'train' and (counts['attempts'] in [30, 200] or counts['attempts'] % 250 == 0)) or kind == 'epoch'
        if save:
            observer.save_bundle(out / 'state_last.pt', key)
            if kind == 'train' and counts['attempts'] in [30, 200]:
                shutil.copy2(out / 'state_last.pt', out / ('state_start_%05d.pt' % counts['attempts']))
            if kind == 'epoch' and counts['epoch'] in [20, 40, 60, 80, 100]:
                shutil.copy2(out / 'state_last.pt', out / ('state_epoch_%03d.pt' % counts['epoch']))
        if kind == 'train' and (counts['attempts'] == 1 or counts['attempts'] % 100 == 0):
            print('SCIENCE', mode, counts['attempts'], counts['success'], counts['skip'], round(time.monotonic() - observer.clock, 2), flush=True)
        if shutil.disk_usage(STAGE).free < 50 * 2**30: raise RuntimeError('free disk below50GiB')
    observer = FastObserver(t, j, out, names, authorize, on_event=event)
    actual_train = t.train_step; actual_monitor = t.validation_step
    def paired_step(batch):
        authorize(); index = j.read()['attempts']; want = expected[index]
        got = {'ids': list(map(str, batch.get('keys', []))), 'input_sha256': digest(batch['data']),
               'target_sha256': digest(batch['target'])}
        if any(got[k] != want[k] for k in got) or abs(t.optimizer.param_groups[0]['lr'] - want['lr']) > 1e-12:
            j.stop('N0 input/target/IDs/LR mismatch before forward at' + str(index + 1))
            atomic(out / 'pairing_failure.json', {'attempt': index + 1, 'got': got, 'wanted': want})
            raise RuntimeError('paired input mismatch before model computation')
        return actual_train(batch)
    def paired_monitor(batch):
        index = j.read()['monitor_attempts']; got_ids = list(map(str, batch.get('keys', [])))
        if got_ids != expected_monitor[index]:
            j.stop('historical Train monitor IDs mismatch'); raise RuntimeError('monitor ID sequence drift')
        evidence = {'number': index + 1, 'ids': got_ids, 'input_sha256': digest(batch['data']),
                    'target_sha256': digest(batch['target'])}
        value = actual_monitor(batch); append(out / 'monitor_inputs.jsonl', evidence); return value
    t.train_step = paired_step; t.validation_step = paired_monitor
    def stopped(signum, frame): raise RuntimeError('external signal; preserve full state, no automatic native queue rebuild')
    signal.signal(signal.SIGTERM, stopped)
    try:
        t.run_training()
        counts = j.read(); final = Path(t.output_folder) / 'checkpoint_final.pth'
        if counts['attempts'] != 25000 or counts['monitor_complete'] != 5000 or counts['phase'] != 'complete' or \
           counts['success'] + counts['skip'] != 25000 or counts['fatal'] or counts['unknown'] or \
           counts['status'] != 'READY' or counts['skip'] >= 25 or t.current_epoch != 100 or not final.exists():
            raise RuntimeError('scientific endpoint incomplete')
        observer.save_bundle(out / 'state_last.pt', key)
        atomic(out / 'training_completion.json', {'status': 'TRAINING_COMPLETE', 'mode': mode, 'seed': 2027,
                'counts': counts, 'checkpoint_final': str(final), 'checkpoint_final_sha256': sha(final),
                'elapsed_seconds': time.monotonic() - observer.clock, 'peak_reserved_MiB': memory_check(),
                'all25000_train_inputs_paired_to_N0': True, 'historical5000_monitor_IDs_paired': True,
                'historical_monitor_pixel_hashes_available': False, 'native_queue_resume_supported': False,
                'Dev_Test_forwards_before_final': 0, 'binding_sha256': key}, new=True)
    except BaseException as exc:
        j.stop(repr(exc))
        try: observer.save_bundle(out / 'state_interrupted.pt', key)
        except BaseException as save_exc: atomic(out / 'snapshot_failure.json', {'error': repr(save_exc)})
        atomic(out / 'training_failure.json', {'error': repr(exc), 'budget': j.read(), 'no_automatic_retry': True})
        raise
    finally:
        observer.close()
        for loader in [getattr(t, 'dataloader_train', None), getattr(t, 'dataloader_val', None)]:
            if hasattr(loader, '_finish'):
                try: loader._finish()
                except BaseException: pass


def run_mode(mode):
    validate(mode, 'science', full=True); out = STAGE / 'science' / mode
    with exclusive_lock(out / 'RUN.lock'), exclusive_lock(ROOT / 'experiments' / ('a14_gpu0.lock' if mode == 'local' else 'a14_gpu1.lock')):
        resource = resources(mode)
        if (out / 'controller_session.json').exists(): raise RuntimeError('scientific session consumed; no restart')
        atomic(out / 'controller_session.json', {'identity': process_identity(), 'resource': resource}, new=True)
        update_authority(**{'science_' + mode: {'state': 'RUNNING', 'controller': process_identity()}})
        for role in ['train', 'infer']:
            nonce = os.urandom(24).hex(); env = worker_env(mode, nonce, out)
            script = Path(__file__) if role == 'train' else BRANCH / 'inference.py'
            args = [sys.executable, '-u', '-B', str(script), '--worker', '--mode', mode,
                    '--parent', str(os.getpid()), '--nonce', nonce]
            with (out / (role + '.log')).open('x', encoding='utf-8') as log:
                p = subprocess.Popen(args, env=env, cwd=CODE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                atomic(out / (role + '_session.json'), {'worker': process_identity(p.pid), 'controller': process_identity(),
                        'nonce': nonce, 'mode': mode, 'role': role}, new=True)
                update_authority(**{'science_' + mode: {'state': role.upper(), 'controller': process_identity(), 'worker': process_identity(p.pid)}})
                # No normal-progress wall cutoff. Native numerical/budget rules remain hard stops.
                p.wait()
            atomic(out / (role + '_process_result.json'), {'exit': p.returncode, 'automatic_restart': False}, new=True)
            if p.returncode:
                update_authority(**{'science_' + mode: {'state': 'FAILED_STOP', 'role': role, 'exit': p.returncode}})
                raise RuntimeError('science stopped:' + role)
        atomic(out / 'completion.json', {'status': 'COMPLETE_PENDING_SOL_REVIEW', 'Test_accessed': False}, new=True)
        update_authority(**{'science_' + mode: {'state': 'COMPLETE_PENDING_SOL_REVIEW'}})
    # Only CPU aggregation when both prespecified groups have completed. No next experiment.
    if all((STAGE / 'science' / m / 'completion.json').exists() for m in GPU):
        with exclusive_lock(STAGE / 'ANALYSIS.lock'):
            if not (STAGE / 'analysis/endpoint_100epochs.json').exists():
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
                with (STAGE / 'analysis/analysis.log').open('x', encoding='utf-8') as log:
                    result = subprocess.run([sys.executable, '-B', str(BRANCH / 'analysis.py')], cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode: raise RuntimeError('CPU analysis failed; preserve both trained groups')
            update_authority(state='ALL_COMPLETE_PENDING_SOL_RESEARCH_REVIEW', science_released=False,
                             next='Sol research review of final paired contrasts and protections; no automatic new experiments')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['local', 'multi'], required=True)
    g = p.add_mutually_exclusive_group(required=True); g.add_argument('--worker', action='store_true'); g.add_argument('--run-authorized', action='store_true')
    p.add_argument('--parent', type=int); p.add_argument('--nonce'); a = p.parse_args()
    if a.worker: train_worker(a.mode, a.parent, a.nonce)
    else: run_mode(a.mode)
