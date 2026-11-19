"""One existing STATUS authority, independent phase releases and fail-closed budgets."""
from pathlib import Path
import os, sys, json, hashlib, time, subprocess, shutil

ROOT = Path('${RETINA_PROJECT_ROOT}')
CODE = ROOT / 'code/retinal-vessel-study'
BASE = ROOT / 'experiments/nnunet2d_branch_20261020'
BRANCH = CODE / 'research_branches/nnunet2d_20261020/deep_supervision_ablation_20261028/science100_v1'
PARENT_STAGE = BASE / 'deep_supervision_ablation_20261028'
ENGINEERING_STAGE = PARENT_STAGE / 'engineering_revision_v2'
STAGE = PARENT_STAGE / 'science100_v1'
OLD = CODE / 'research_branches/nnunet2d_20261020'
REPLICATION = OLD / 'replication_seeds_2027_2028_v1_20261026'
N0 = BASE / 'replication_seed2027_2028_100epochs_v1_20261026/runs/seed_2027'
PHASE = 'NNUNET_FINAL_ONLY_SCIENCE100_V1'
GPU = {'final': '${GPU_UUID}'}
MARKERS = ('<!-- NNUNET_FINAL_ONLY_CURRENT -->', '<!-- /NNUNET_FINAL_ONLY_CURRENT -->')
for path in [CODE, OLD / 'no_wall_limit_restart_02', OLD / 'amp_engineering32_20261025',
             OLD / 'amp_recovery_revision_20261021', OLD / 'scientific_resume_v5_20261025', REPLICATION]:
    if str(path) not in sys.path: sys.path.append(str(path))
from retinal_m0.maples_g1_safety import process_identity, exclusive_lock, owner_alive


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value, new=False):
    path = Path(path)
    if new and path.exists(): raise RuntimeError('refuse overwrite: ' + str(path))
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.pending')
    with temp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def authority():
    text = (CODE / 'STATUS.md').read_text()
    a, z = MARKERS
    if text.count(a) != 1 or text.count(z) != 1: raise RuntimeError('unique current authority missing')
    q = json.loads(text.split(a)[1].split(z)[0])
    if q.get('phase') != PHASE: raise RuntimeError('wrong active phase')
    return q


def update_authority(**changes):
    # Only controllers/review write STATUS, with the same existing global document.
    with exclusive_lock(CODE / '.deep_supervision_status_write.lock'):
        path = CODE / 'STATUS.md'; old = path.read_text(); a, z = MARKERS
        lo = old.index(a) + len(a); hi = old.index(z)
        v = json.loads(old[lo:hi]); v.update(changes)
        new = old[:lo] + '\n' + json.dumps(v, ensure_ascii=False) + '\n' + old[hi:]
        temp = path.with_name('STATUS.md.multiscale.' + str(os.getpid()) + '.pending')
        with temp.open('x', encoding='utf-8') as f:
            f.write(new); f.flush(); os.fsync(f.fileno())
        if path.read_text() != old: raise RuntimeError('STATUS concurrently changed; no overwrite')
        os.replace(temp, path)


def protocol():
    q = read(STAGE / 'protocol.json'); a = authority()
    if a.get('protocol_sha256') != sha(STAGE / 'protocol.json') or q['phase'] != PHASE:
        raise RuntimeError('protocol identity drift')
    if q['seed'] != 2027 or q['modes'] != ['final'] or q['official_Test_allowed']:
        raise RuntimeError('research scope drift')
    if not q['science_authorized'] or q['science']['epochs'] != 100 or q['science']['attempt_cap_per_mode'] != 25000 or q['science']['monitor_cap_per_mode'] != 5000 or q['science']['batch'] != 2 or q['science']['patch'] != [1024,1536]: raise RuntimeError('scientific budget drift')
    if q['weights'] != [1,0,0,0,0,0,0,0] or q['science']['AMP_consecutive_stop'] != 5 or q['science']['AMP_total_stop'] != 25 or q['science']['memory_cap_MiB'] != 10240: raise RuntimeError('frozen candidate rules changed')
    if q['science']['replace_skips'] or q['science']['automatic_restart'] or q['evaluation']['threshold'] != .5 or q['evaluation']['endpoint'] != 'unique final100epochs': raise RuntimeError('endpoint/skip rule changed')
    return q


def binding():
    b = read(STAGE / 'source_binding.json')
    if canonical({k: v for k, v in b.items() if k != 'binding_sha256'}) != b['binding_sha256']:
        raise RuntimeError('binding malformed')
    if b['protocol_sha256'] != sha(STAGE / 'protocol.json'): raise RuntimeError('stale protocol binding')
    return b


def validate(mode, phase, full=False):
    if mode not in GPU or phase not in ('science', 'infer'):
        raise RuntimeError('unknown variant/role')
    q = protocol(); b = binding(); a = authority(); cpu = read(STAGE / 'CPU_REVIEW.json')
    if cpu['status'] != 'CPU_ACCEPTED' or cpu['binding_sha256'] != b['binding_sha256']:
        raise RuntimeError('CPU review missing/stale')
    if cpu['checks_sha256'] != sha(STAGE / 'cpu/checks.json') or not read(STAGE / 'cpu/checks.json')['passed']:
        raise RuntimeError('CPU test evidence changed/failed')
    rphase = 'science' if phase == 'infer' else phase
    release_path = STAGE / (rphase.upper() + '_RELEASE.json')
    if not release_path.exists() or not a.get(rphase + '_released'):
        raise RuntimeError(rphase + ' HOLD: independent release missing')
    release = read(release_path)
    want = {'phase': PHASE, 'role': rphase, 'binding_sha256': b['binding_sha256'],
            'protocol_sha256': sha(STAGE / 'protocol.json'),
            'CPU_review_sha256': sha(STAGE / 'CPU_REVIEW.json'),
            'modes': ['final'], 'GPU': GPU, 'official_Test_allowed': False}
    if rphase == 'science':
        review = ENGINEERING_STAGE / 'SOL_ENGINEERING_REVIEW.json'
        if not review.exists() or read(review)['status'] != 'PASS': raise RuntimeError('engineering not independently accepted')
        want['engineering_review_sha256'] = sha(review)
    if release != want or a.get(rphase + '_release_sha256') != sha(release_path):
        raise RuntimeError('release scope/STATUS binding drift')
    for path, h in b['runtime_sources'].items():
        if sha(path) != h: raise RuntimeError('runtime source drift: ' + path)
    if full:
        for section in ['all_sources', 'Train_sources', 'reference_sources', 'fixture_sources']:
            for path, h in b[section].items():
                if sha(path) != h: raise RuntimeError('source/data drift: ' + path)
    return canonical({'binding': b['binding_sha256'], 'mode': mode, 'phase': rphase})


def runtime_authorizer(mode, phase, key, role, parent, nonce):
    session_path = STAGE / phase / mode / (role + '_session.json')
    deadline = time.monotonic() + 5
    while not session_path.exists() and time.monotonic() < deadline:
        if os.getppid() != parent: raise RuntimeError('controller exited before session binding')
        time.sleep(.01)
    session = read(session_path); fixed = sha(session_path)
    def check():
        if sha(session_path) != fixed or process_identity() != session['worker'] or \
           process_identity(parent) != session['controller'] or os.getppid() != parent or \
           nonce != session['nonce'] or nonce != os.environ.get('RETINA_FUSION_NONCE'):
            raise RuntimeError('unbound worker/parent/session')
        if validate(mode, 'infer' if role == 'infer' else phase) != key:
            raise RuntimeError('changed release binding')
    check(); return check


def resources(mode):
    gpu = GPU[mode]
    values = [s.strip() for s in subprocess.check_output(
        ['nvidia-smi', '--id=' + gpu, '--query-gpu=uuid,memory.used,utilization.gpu,memory.total',
         '--format=csv,noheader,nounits'], text=True).strip().split(',')]
    procs = subprocess.check_output(['nvidia-smi', '--id=' + gpu, '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True)
    if values[0] != gpu or int(values[1]) >= 512 or int(values[2]) >= 10 or procs.strip():
        raise RuntimeError('assigned GPU busy; never kill/switch')
    if shutil.disk_usage(STAGE).free < 50 * 2**30: raise RuntimeError('free disk below50GiB')
    return {'GPU': gpu, 'used_MiB': int(values[1]), 'total_MiB': int(values[3])}


def worker_env(mode, nonce, output):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU[mode], PYTHONHASHSEED='2027',
               RETINA_FUSION_MODE=mode, RETINA_FUSION_NONCE=nonce,
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               nnUNet_compile='false', nnUNet_n_proc_DA='1', MPLCONFIGDIR=str(output / 'matplotlib'))
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    return env


def admit_worker(mode, phase, role, parent, nonce):
    if os.getppid() != parent or nonce != os.environ.get('RETINA_FUSION_NONCE') or len(nonce) < 32:
        raise RuntimeError('authenticated direct child required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU[mode] or os.environ.get('PYTHONHASHSEED') != '2027':
        raise RuntimeError('GPU/seed environment mismatch')
    key = validate(mode, 'infer' if role == 'infer' else phase, full=True)
    import ctypes, signal
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != parent:
        raise RuntimeError('parent death guard failed')
    return key, runtime_authorizer(mode, phase, key, role, parent, nonce)


def save_torch(path, payload):
    import torch
    path = Path(path); temp = path.with_name(path.name + '.pending')
    with temp.open('xb') as f: torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def memory_check():
    import torch
    value = torch.cuda.max_memory_reserved(0) / 1048576
    if value > 10240: raise RuntimeError('reserved GPU memory exceeds frozen10GiB limit')
    return value
