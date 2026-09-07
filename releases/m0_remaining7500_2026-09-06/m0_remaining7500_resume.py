"""Resume the authorized M0 trajectory; no temperature acquisition."""
import sys
import time
import shutil
import subprocess
import traceback
import json
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retinal_release.common import ROOT, read, write, digest
from retinal_release.exchange import pack, ingest

RUN = ROOT / 'runs/prefetch_round1/m0_seed2026_dev10k_w1'
OUT = ROOT / 'releases/m0_remaining7500_2026-09-06'
TICKET = ROOT / '.runtime/coordination/tickets/m0_seed2026_dev10k_w1.json'
assert OUT.is_dir()
started = time.perf_counter()
overall_started_unix = OUT.joinpath('plan.json').stat().st_ctime
results = []
active = OUT / 'active_process.json'

def execute(name, args, timeout):
    begin = time.perf_counter()
    with (OUT / (name + '.log')).open('x', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, *args], cwd=ROOT,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        write(active, dict(phase=name, pid=child.pid, started_unix=time.time()))
        print('Started ' + name, flush=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import psutil
            parent = psutil.Process(child.pid)
            descendants = parent.children(recursive=True)
            for process in descendants:
                process.terminate()
            child.terminate()
            child.wait(timeout=15)
            raise RuntimeError(name + ' exceeded its bounded window; inspect last saved checkpoint')
    if code:
        raise RuntimeError(f'{name} exited {code}; see log')
    write(active, dict(phase='between_phases', pid=None))
    return time.perf_counter() - begin

try:
    progress = read(RUN / 'progress.json')
    if progress['step'] != 5000 or progress['checkpoint_sha256'] != digest(RUN / 'last.pt'):
        raise ValueError('Expected the intact 5000-step checkpoint')
    if (RUN / 'running.lock').exists():
        raise ValueError('Training is already locked')
    write(OUT / 'recovery_plan.json', dict(run_id=RUN.name, start_step=5000, stop_step=10000,
          validation_steps=[5000,7500,10000], threshold=0.5, temperature_recording=False,
          ram_limit_bytes=12000000000, vram_limit_bytes=12000000000,
          ticket=read(TICKET), initial_checkpoint_sha256=progress['checkpoint_sha256'],
          wrapper_sha256=digest(Path(__file__)),
          evaluation_wrapper_sha256=digest(ROOT / '.runtime/m0_remaining7500_eval.py')))
    for stop in (5000, 7500, 10000):
        if stop == 5000:
            train_seconds = read(OUT / 'train_to_5000_resources.json')['seconds']
        else:
            train_seconds = execute(f'train_to_{stop}', ['-m', 'retinal_prefetch.run', str(TICKET),
                                   '--stop-after-step', str(stop), '--max-seconds', '2700'], 2760)
        write(OUT / f'training_{stop}_timing.json', dict(seconds=train_seconds,
              source='resource_guard_elapsed_recovered' if stop==5000 else 'parent_process_elapsed'))
        progress = read(RUN / 'progress.json')
        shutil.copy2(RUN / 'resources.json', OUT / f'train_to_{stop}_resources.json')
        if progress['step'] != stop:
            raise RuntimeError(f'Training paused early at {progress["step"]}; remaining work not silently launched')
        bundle = ROOT / f'.runtime/transfer/m0_seed2026_dev10k_w1_step{stop}.zip'
        if bundle.exists():
            with zipfile.ZipFile(bundle) as z:
                snapshot = json.loads(z.read('snapshot.json'))
        else:
            snapshot = pack(RUN, bundle)
        for attempt in range(3):
            try:
                receipt = ingest(bundle)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(1)
        write(OUT / f'transfer_{stop}.json', dict(snapshot=snapshot, receipt=receipt,
              bundle=str(bundle.relative_to(ROOT)), sha256=digest(bundle)))
        output = OUT / f'evaluation_{stop}.json'
        eval_seconds = execute(f'validation_{stop}', [str(ROOT / '.runtime/m0_remaining7500_eval.py'),
                               str(RUN), str(output), '--threshold', '0.5'], 660)
        results.append(dict(step=stop, train_seconds=train_seconds, evaluation_seconds=eval_seconds,
                       checkpoint_sha256=progress['checkpoint_sha256']))
        write(OUT / 'segments.json', results)
        print(f'Completed and validated step {stop}', flush=True)
    write(OUT / 'window_result.json', dict(status='completed', segments=results,
          seconds=time.time()-overall_started_unix, temperature_recording=False,
          recovered_after_transient_ingest_permission_error=True,
          first_segment_timing_source='resource_guard_elapsed'))
except Exception:
    write(OUT / 'window_result.json', dict(status='failed', segments=results,
          seconds=time.perf_counter()-started, error=traceback.format_exc()))
    raise
finally:
    write(active, dict(phase='stopped', pid=None))
