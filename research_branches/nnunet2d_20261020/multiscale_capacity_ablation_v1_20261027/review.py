"""Explicit Sol review actions. Never called automatically by either engineering executor."""
import os, sys, math, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from common import *


def verify_binding_all():
    b = binding(); protocol()
    for section in ['all_sources', 'Train_sources', 'reference_sources', 'fixture_sources']:
        for path, want in b[section].items():
            if sha(path) != want: raise RuntimeError('bound evidence drift:' + path)
    cpu = read(STAGE / 'CPU_REVIEW.json'); checks = read(STAGE / 'cpu/checks.json')
    if cpu['status'] != 'CPU_ACCEPTED' or cpu['binding_sha256'] != b['binding_sha256'] or not checks['passed'] or \
       cpu['checks_sha256'] != sha(STAGE / 'cpu/checks.json') or not all(checks['checks'].values()):
        raise RuntimeError('CPU evidence invalid')
    return b


def release(role):
    b = verify_binding_all()
    if role == 'engineering':
        from engineering import Counter
        for mode in GPU:
            key = canonical({'binding': b['binding_sha256'], 'mode': mode, 'phase': role})
            c = Counter(STAGE / role / mode / 'budget.json', key).read()
            if c['updates_reserved'] or c['diagnostics_reserved'] or c['status'] != 'READY': raise RuntimeError('engineering already consumed')
        denials = read(STAGE / 'cpu/authorization_denials.json'); cli = read(STAGE / 'cpu/CLI_denials.json')
        if len(denials['real_denials']) != 6 or len(cli['cases']) != 3 or any(r['returncode'] == 0 for r in cli['cases']):
            raise RuntimeError('actual phase/CLI denials incomplete')
        atomic(STAGE / 'SOL_CPU_RELEASE_REVIEW.json', {'status': 'PASS', 'binding_sha256': b['binding_sha256'],
            'CPU_checks': read(STAGE / 'CPU_REVIEW.json')['checks_count'], 'actual_authorization_denials': 6,
            'actual_CLI_denials': 3, 'no_GPU_before_release': True, 'science_released': False,
            'reviewer': 'Sol independently checked actual CPU evidence; no GPU diagnostic claimed',
            'reviewer_code_sha256': sha(Path(__file__))}, new=True)
    elif role == 'science':
        if read(STAGE / 'SOL_ENGINEERING_REVIEW.json')['status'] != 'PASS': raise RuntimeError('engineering not accepted')
        from fast_observer import Journal, AmpLimits
        for mode in GPU:
            key = canonical({'binding': b['binding_sha256'], 'mode': mode, 'phase': role})
            Journal(STAGE / role / mode / 'budget.json', key, 100, 250, 50, AmpLimits(5, 25)).initialize()
    else: raise RuntimeError('unknown release role')
    r = {'phase': PHASE, 'role': role, 'binding_sha256': b['binding_sha256'],
         'protocol_sha256': sha(STAGE / 'protocol.json'), 'CPU_review_sha256': sha(STAGE / 'CPU_REVIEW.json'),
         'modes': ['local', 'multi'], 'GPU': GPU, 'official_Test_allowed': False}
    if role == 'science': r['engineering_review_sha256'] = sha(STAGE / 'SOL_ENGINEERING_REVIEW.json')
    path = STAGE / (role.upper() + '_RELEASE.json'); atomic(path, r, new=True)
    update_authority(**{role + '_released': True, role + '_release_sha256': sha(path),
                       'state': role.upper() + '_RELEASED', 'next': 'Explicitly start released ' + role + ' controllers; no phase auto-transition'})
    for mode in GPU: validate(mode, role, full=True)
    print(role.upper() + '_INDEPENDENT_RELEASE_VERIFIED', flush=True)


def review_engineering():
    b = verify_binding_all()
    import torch, gc
    from fast_observer import digest, states_finite
    results = {}
    for mode in GPU:
        out = STAGE / 'engineering' / mode; complete = read(out / 'completion.json'); budget = complete['budget']
        want = {'updates_reserved': 32, 'successful_updates': 32, 'diagnostics_reserved': 4,
                'forwards': 36, 'backwards': 32, 'optimizer_calls': 32,
                'failed_updates': 0, 'unknown_updates': 0}
        if any(budget[k] != v for k, v in want.items()) or budget['pending'] or \
           complete['elapsed_seconds'] > 1800 or complete['science_started'] or budget['status'] != 'COMPLETE_PENDING_REVIEW':
            raise RuntimeError('engineering quota/result incomplete:' + mode)
        diag = read(out / 'diagnostics.json')
        if diag['status'] != 'PASS' or diag['updates'] or len(diag['results']) != 2 or \
           not all(r['all8_outputs_exact'] for r in diag['results']): raise RuntimeError('native identity/AMP diagnostics incomplete')
        states = []
        for name in ['cont16.pt', 'split16.pt']:
            state = torch.load(out / name, map_location='cpu', weights_only=False)
            for key, want_digest in state['digests'].items():
                if digest(state[key]) != want_digest: raise RuntimeError('checkpoint self digest failed:' + name + ':' + key)
            if not states_finite(state['model'], state['optimizer']): raise RuntimeError('nonfinite terminal state')
            states.append(state)
        compared = ['model', 'optimizer', 'scaler', 'scheduler', 'rng', 'cursor', 'budget',
                    'partial_train', 'partial_monitor', 'logging', 'best_ema']
        same = {k: digest(states[0][k]) == digest(states[1][k]) for k in compared}
        if not all(same.values()): raise RuntimeError('CONT16 versus split8+8 differs:' + repr(same))
        gradient = [json.loads(l) for l in (out / 'gradient_diagnostics.jsonl').read_text().splitlines()]
        if len(gradient) != 32 or any(r['peak_reserved_MiB'] > 10240 for r in gradient): raise RuntimeError('gradient/resource evidence incomplete')
        if any(not all(math.isfinite(r[k]) for k in ['module_unscaled_L2', 'projection_unscaled_L2', 'internal_unscaled_L2']) for r in gradient):
            raise RuntimeError('nonfinite real unscaled module gradient')
        if any(r['internal_unscaled_L2'] <= 0 for r in gradient if r['fixed_batch'] > 1): raise RuntimeError('no internal gradient after zero-head unlock')
        tapes = []
        baseline = [json.loads(l) for l in (N0 / 'attempts.jsonl').read_text().splitlines()][:16]
        for name in ['cont', 'split']:
            tape = [json.loads(l) for l in (out / name / 'attempts.jsonl').read_text().splitlines()]
            if len(tape) != 16 or any(r['classification'] != 'SUCCESS' for r in tape): raise RuntimeError('engineering success tape incomplete')
            for got, want_row in zip(tape, baseline):
                if any(got[k] != want_row[k] for k in ['ids', 'input_sha256', 'target_sha256', 'lr']): raise RuntimeError('engineering inputs not paired to N0')
            tapes.append(tape)
        if read(out / 'restore_verification.json')['status'] != 'PASS': raise RuntimeError('missing actual new-process restore')
        results[mode] = {'status': 'PASS', 'budget': budget, 'available_state_exact_comparison': same,
                         'same_GPU_fixed_batch_restore_only': True, 'native_queue_resume_claimed': False,
                         'native_FP32_AMP_all8_outputs_initial_identity': True,
                         'max_reserved_MiB': max(diag['reserved_peak_MiB'], max(r['peak_reserved_MiB'] for r in gradient)),
                         'real_unscaled_module_gradients_finite': True, 'all32_Train_batches_paired': True,
                         'elapsed_seconds': complete['elapsed_seconds']}
        del states; gc.collect()
    if read(STAGE / 'engineering/local/cont_initialization.json')['module'] != read(STAGE / 'engineering/multi/cont_initialization.json')['module']:
        raise RuntimeError('two modes did not use identical initial module parameters')
    summary = {'status': 'PASS', 'reviewer': 'Sol independent CPU reconciliation of actual GPU artifacts',
               'binding_sha256': b['binding_sha256'], 'results': results, 'total_charged_updates': 64,
               'total_diagnostic_forwards': 8, 'Test_accessed': False, 'science_started': False,
               'engineering_executor_does_not_release_science': True,
               'reviewer_code_sha256': sha(Path(__file__))}
    atomic(STAGE / 'SOL_ENGINEERING_REVIEW.json', summary, new=True)
    update_authority(state='ENGINEERING_PASS_AWAITING_SEPARATE_SCIENCE_RELEASE', engineering_released=False,
                     next='Sol may separately release approved two scientific trajectories; executor never auto-transitions')
    print('ENGINEERING_INDEPENDENT_REVIEW_PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--release-engineering', action='store_true'); g.add_argument('--review-engineering', action='store_true'); g.add_argument('--release-science', action='store_true')
    a = p.parse_args()
    if a.release_engineering: release('engineering')
    elif a.review_engineering: review_engineering()
    else: release('science')
