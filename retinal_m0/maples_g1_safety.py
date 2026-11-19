"""Fail-closed G1 authorization, immutable binding, and cumulative accounting.

No model, CUDA initialization, release creation, or ledger mutation on import.
"""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

GPU_UUID = '${GPU_UUID}'
CAP = 32
START = '<!-- P1_G1_RELEASE_CURRENT -->'
END = '<!-- /P1_G1_RELEASE_CURRENT -->'
DIGEST_ALGORITHM = 'sha256_canonical_json_utf8_sortkeys_compact_v1'


class GateError(RuntimeError):
    pass


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _unique_pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise GateError('duplicate JSON key: ' + k)
        result[k] = v
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(GateError('nonfinite JSON: '+value)))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    finally:
        if os.path.exists(temp): os.unlink(temp)


@contextlib.contextmanager
def exclusive_lock(path):
    with Path(path).open('a+') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise GateError('busy lock: '+str(path)) from exc
        try: yield stream
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def active_release(status_text):
    if status_text.count(START) != 1 or status_text.count(END) != 1:
        raise GateError('exactly one active G1 release block is required; historical lines are not authority')
    a = status_text.index(START)+len(START); b = status_text.index(END)
    if b <= a: raise GateError('malformed current G1 release block')
    return json.loads(status_text[a:b].strip(), object_pairs_hook=_unique_pairs)


def verify_authority(*, status_text, release_path, binding_path, current_binding, ledger_path,
                     review_receipt_path=None):
    """Caller supplies fixed production paths, not command-line overrides."""
    state = active_release(status_text)
    if state.get('state') != 'RELEASED' or state.get('phase') != 'G1_ONLY':
        raise GateError('current G1 state is not RELEASED')
    release_path, binding_path, ledger_path = map(Path, (release_path, binding_path, ledger_path))
    if release_path.is_symlink() or binding_path.is_symlink() or ledger_path.is_symlink():
        raise GateError('authorization/binding/ledger symlink is not allowed')
    saved = read_json(binding_path)
    if saved != current_binding:
        raise GateError('current implementation/data/environment differs from the reviewed execution binding')
    release = read_json(release_path)
    digest = canonical_sha(current_binding)
    if review_receipt_path is None: raise GateError('reviewed CPU evidence receipt required')
    receipt_path=Path(review_receipt_path)
    if receipt_path.is_symlink():raise GateError('CPU receipt symlink is not allowed')
    receipt=read_json(receipt_path)
    if (receipt.get('status')!='CPU_VALIDATED_GPU_HELD' or receipt.get('binding_sha256')!=digest
        or receipt.get('binding_file_sha256')!=file_sha(binding_path)
        or receipt.get('tests_passed') is not True or receipt.get('preflight_passed') is not True):
        raise GateError('CPU review receipt does not match this tested execution binding')
    evidence=receipt.get('evidence')
    if not isinstance(evidence,dict) or len(evidence)<3:raise GateError('missing CPU evidence hashes')
    for path,expected in evidence.items():
        if file_sha(path)!=expected:raise GateError('reviewed CPU evidence changed: '+path)
    required = {'phase': 'G1_ONLY', 'authorized': True, 'gpu_launch_authorized': True,
                'max_forwards': CAP, 'parameter_updates': 0, 'gpu_uuid': GPU_UUID,
                'digest_algorithm': DIGEST_ALGORITHM, 'binding_sha256': digest,
                'binding_file_sha256': file_sha(binding_path),
                'cpu_review_receipt_sha256':file_sha(receipt_path),
                'ledger_path': str(ledger_path.resolve())}
    for key, value in required.items():
        if type(release.get(key)) is not type(value) or release[key] != value:
            raise GateError('release field mismatch: '+key)
    if not isinstance(release.get('release_id'), str) or not release['release_id'].strip():
        raise GateError('release_id missing')
    for key, value in {'release_id': release['release_id'], 'binding_sha256': digest,
                       'release_file_sha256': file_sha(release_path)}.items():
        if state.get(key) != value: raise GateError('active STATUS binding mismatch: '+key)
    return release


def finite_tree(value, path='metrics'):
    """Recursively reject NaNs, infinities, tensor/object shortcuts, not just scalars."""
    if value is None or type(value) in (str, bool, int): return
    if type(value) is float:
        if not math.isfinite(value): raise GateError('nonfinite at '+path)
        return
    if type(value) is dict:
        for k, child in value.items(): finite_tree(child, path+'.'+str(k))
        return
    if type(value) in (list, tuple):
        for i, child in enumerate(value): finite_tree(child, path+f'[{i}]')
        return
    raise GateError('unsupported diagnostic type at '+path+': '+type(value).__name__)


def safe_evidence(value):
    """Preserve offending values explicitly in FAILURE evidence, never turn them into PASS."""
    if type(value) is float and not math.isfinite(value): return {'NONFINITE': repr(value)}
    if isinstance(value, dict): return {str(k): safe_evidence(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [safe_evidence(v) for v in value]
    if value is None or type(value) in (str, bool, int, float): return value
    return {'UNSUPPORTED': repr(value)}


def process_identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    # Linux /proc start time disambiguates PID reuse, including comm with spaces.
    stat = Path(f'/proc/{pid}/stat').read_text()
    return {'pid': pid, 'start_time': stat[stat.rfind(')')+2:].split()[19]}


def owner_alive(owner):
    try: return process_identity(owner['pid']) == owner
    except FileNotFoundError: return False
    except (PermissionError, KeyError, ValueError): return True  # Fail closed.


def ledger_snapshot(path):
    """Read-only. Never reap a live job or recreate a missing production file."""
    if not Path(path).is_file(): raise GateError('ledger missing; refusing implicit reset')
    state = read_json(path)
    if state.get('schema') == 1:
        if state.get('cap') != CAP or state.get('combinations') != {}:
            raise GateError('legacy ledger is not empty; explicit migration/review required')
        if any(state.get(k) != 0 for k in ('autograd_calls','optimizer_step_attempts')):
            raise GateError('legacy ledger has consumed work')
        return state
    if state.get('schema') != 2 or state.get('cap') != CAP:
        raise GateError('invalid ledger schema/cap')
    entries = state.get('combinations')
    if not isinstance(entries, dict) or len(entries) > CAP:
        raise GateError('invalid cumulative combination count')
    starts = 0
    backwards = 0
    for item in entries.values():
        if item.get('status') not in ('reserved','started','completed','failed'):
            raise GateError('invalid ledger status')
        if type(item.get('forward_starts')) is not int or item['forward_starts'] not in (0,1):
            raise GateError('invalid per-combination forward count')
        starts += item['forward_starts']
        if type(item.get('autograd_calls')) is not int or item['autograd_calls'] not in range(5):
            raise GateError('invalid per-combination autograd count')
        backwards += item['autograd_calls']
        if not isinstance(item.get('owner'), dict): raise GateError('missing ledger owner')
    if state.get('forward_starts') != starts or starts > len(entries):
        raise GateError('forward accounting mismatch')
    if state.get('autograd_calls') != backwards:
        raise GateError('autograd accounting mismatch')
    for key in ('optimizer_step_attempts','blocked_forward_attempts'):
        if type(state.get(key)) is not int or state[key]<0: raise GateError('invalid '+key)
    return state


class Ledger:
    """Production uses ONE fixed path under an exclusive runtime/GPU lease.

    Construction and preflight are read-only. Migration and recovery are explicit.
    """
    def __init__(self, path, binding_id, combo_ids):
        self.path, self.binding_id = Path(path), binding_id
        self.ids = set(combo_ids)
        if len(self.ids) != CAP: raise GateError('exact32 fixed IDs required')
        self.owner = process_identity()

    def initialize_empty_legacy(self, backup_path):
        with exclusive_lock(str(self.path)+'.lock'):
            old = ledger_snapshot(self.path)
            if old['schema'] != 1: raise GateError('ledger already initialized; no implicit restart')
            if Path(backup_path).exists(): raise GateError('legacy migration backup already exists')
            atomic_json(backup_path, old)
            atomic_json(self.path, {'schema':2,'cap':CAP,'binding_id':self.binding_id,
                'combinations':{},'forward_starts':0,'autograd_calls':0,
                'optimizer_step_attempts':0,'blocked_forward_attempts':0})

    def _change(self, action):
        with exclusive_lock(str(self.path)+'.lock'):
            state = ledger_snapshot(self.path)
            if state.get('binding_id') != self.binding_id:
                raise GateError('ledger binding mismatch; cannot reset or change path')
            if not set(state['combinations']).issubset(self.ids): raise GateError('unknown ledger ID')
            action(state)
            atomic_json(self.path,state)
            ledger_snapshot(self.path)

    def reserve(self, combo):
        def action(s):
            if combo not in self.ids or combo in s['combinations'] or len(s['combinations'])>=CAP:
                raise GateError('unknown/duplicate/exhausted combination')
            if any(v['status']=='failed' for v in s['combinations'].values()):
                raise GateError('prior failure; no automatic continuation')
            s['combinations'][combo]={'status':'reserved','owner':self.owner,'forward_starts':0,
                'autograd_calls':0,'reserved_at':time.time(),'failure':None}
        self._change(action)

    def before_forward(self, combo):
        """Called by forward PRE-hook; a second call cannot execute the model body."""
        blocked = [False]
        def action(s):
            item=s['combinations'][combo]
            if item['status']!='reserved' or item['forward_starts']!=0:
                s['blocked_forward_attempts']+=1;blocked[0]=True;return
            if s['forward_starts']>=CAP: raise GateError('32 actual forwards exhausted')
            item['status']='started';item['forward_starts']=1;s['forward_starts']+=1
        self._change(action)
        if blocked[0]: raise GateError('extra model forward blocked BEFORE execution')

    def autograd(self, combo):
        def action(s):
            if s['combinations'][combo]['status']!='started': raise GateError('autograd outside started combo')
            if s['combinations'][combo]['autograd_calls']>=4: raise GateError('unexpected extra autograd call')
            s['autograd_calls']+=1;s['combinations'][combo]['autograd_calls']+=1
        self._change(action)

    def prohibit_optimizer(self, combo):
        def action(s):
            if combo not in s['combinations']: raise GateError('optimizer outside reserved combination')
            s['optimizer_step_attempts']+=1
        self._change(action)
        raise GateError('optimizer.step prohibited in G1')

    def finish(self, combo, *, error=None):
        def action(s):
            item=s['combinations'][combo]
            if item['status'] not in ('reserved','started'): raise GateError('invalid finish transition')
            if error is None and item['forward_starts']!=1: raise GateError('cannot complete without forward')
            item['status']='failed' if error is not None else 'completed'
            item['failure']=error;item['finished_at']=time.time()
        self._change(action)

    def recover_dead_owner(self):
        """Explicit, under runtime lease; never done by reading/preflight."""
        def action(s):
            for item in s['combinations'].values():
                if item['status'] in ('reserved','started'):
                    if owner_alive(item['owner']): raise GateError('owner still alive; cannot reap')
                    item['status']='failed';item['failure']='confirmed_dead_owner_no_budget_reset'
        self._change(action)
