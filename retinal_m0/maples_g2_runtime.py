"""G2-only execution gate and CPU preflight; never launches from CPU preflight."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
from retinal_m0.maples_g2_engineering import GateError, Budget, lock, atomic_json, canonical_sha, file_sha, read_json, _pairs

ROOT=Path("${RETINA_PROJECT_ROOT}")
CODE=ROOT/"code/retinal-vessel-study"
STAGE=ROOT/"experiments/full_fov_thin_skeleton_bce_20261019"
REVIEW=STAGE/"target_definition_review_v1"
G1=REVIEW/"g1_engineering_v3"
SOL=G1/"sol_repair_20261021"
DOC=ROOT/"docs/full_fov_thin_skeleton_bce_20261019/branch_review_20261021"
EXCEPTION=DOC/"g2_exception_01.json"
PROPOSAL=SOL/"gpu_attempt_01/g2_engineering_proposal.json"
G1_VERIFY=SOL/"cpu_attempt_03/verification.json"
G1_READY=SOL/"ready_binding.json"
G1_RECEIPT=SOL/"cpu_review_receipt.json"
G1_SOURCE=REVIEW/"attempt_02"
G1_TARGETS=STAGE/"targets/train"
G2OUT=REVIEW/"g2_engineering_v3"
STATUS=CODE/"STATUS.md"
G1_RELEASE=G1/"G1_RELEASE.json"
G2_RELEASE=G2OUT/"G2_RELEASE.json"
G2_LEDGER=G2OUT/"g2_budget_ledger.json"
GPU_UUID="${GPU_UUID}"
POLICY_SHA="d083de28136d82b1c0cdd762afd995efff1ef644eeeff304c4ccce492d731aae"
PROPOSAL_SHA="a375865fae90db8900b3e24e43eff71cd42ca8785a322e6c338ddbd5c78db971"
CAPACITY=1024**3
BASE=ROOT/'experiments/t0_tw_core_20261016/runs/seed_2026/T0'
READY=G2OUT/'ready_binding.json'
RECEIPT=G2OUT/'cpu_review_receipt.json'
GPU_LOCK=ROOT/'experiments/a14_gpu0.lock'
RUN_LOCK=G2OUT/'G2_RUNTIME.lock'

def current_g2_state(text):
    a="<!-- P1_G2_RELEASE_CURRENT -->";b="<!-- /P1_G2_RELEASE_CURRENT -->"
    if text.count(a)!=1 or text.count(b)!=1:raise GateError("exactly one current G2 status block required")
    s=json.loads(text.split(a,1)[1].split(b,1)[0].strip(),object_pairs_hook=_pairs)
    if s.get("policy_id")!="P1_B1_B2_G2_ENGINEERING_EXCEPTION_01_20261021" or s.get("policy_sha256")!=POLICY_SHA:
        raise GateError("current G2 policy/status binding mismatch")
    if s.get('phase')!='G2_ONLY' or s.get('user_approved') is not True or s.get('GPU_execution_conditionally_authorized') is not True or type(s.get('charged_update_ceiling')) is not int or s['charged_update_ceiling']!=400 or s.get('scientific_authorized') is not False:
        raise GateError('G2 phase/authorization/budget mismatch')
    return s

def _require_current_hold(text):
    state=current_g2_state(text)
    if state.get("state")!="HOLD_CPU_PREFLIGHT" or state.get("GPU_released") is not False:raise GateError("G2 status is not CPU-only HOLD")
    return state

def observe_resources():
    q=subprocess.run(["nvidia-smi","-i","0","--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10,check=True)
    f=[x.strip() for x in q.stdout.strip().split(",")]
    if len(f)!=5 or f[0]!="0" or f[1]!=GPU_UUID:raise GateError("GPU0 identity mismatch; GPU1 is never queried")
    disk=shutil.disk_usage(G2OUT.parent if G2OUT.parent.exists() else REVIEW)
    return {"gpu0":{"index":0,"uuid":f[1],"total_MiB":int(f[2]),"used_MiB":int(f[3]),"util_percent":int(f[4])},
      "GPU1_queried":False,"free_disk_bytes":disk.free,"observed_at":time.time(),"launch":False}

def build_binding():
    # Hash only: do not deserialize or even open the checkpoint content as a model.
    if file_sha(EXCEPTION)!=POLICY_SHA:raise GateError("approved G2 exception drift")
    if file_sha(PROPOSAL)!=PROPOSAL_SHA:raise GateError("approved proposal drift")
    exception=read_json(EXCEPTION);proposal=read_json(PROPOSAL)
    if exception["authorization_now"]["Luna_CPU_G2_implementation_and_preflight"] is not True:raise GateError("G2 CPU authorization absent")
    if exception["authorization_now"]["G2_GPU_released_now"] is not False:raise GateError("G2 release state in policy unexpected")
    if proposal.get("G1_original_gradient_gate_passed") is not False:raise GateError("proposal historical G1 finding drift")
    from retinal_m0.maples_g1_runtime import build_binding as reviewed_g1_binding
    current_g1,_=reviewed_g1_binding()
    g1=read_json(G1_VERIFY);ready=read_json(G1_READY);receipt=read_json(G1_RECEIPT)
    if current_g1!=ready:raise GateError('reviewed G1 Train/source binding drift')
    if g1.get("status")!="CPU_PREFLIGHT_PASSED_GPU_HELD" or receipt.get("status")!="CPU_VALIDATED_GPU_HELD":raise GateError("G1 CPU evidence not validated")
    if ready.get("phase")!="G1_ONLY" or ready.get("parameter_updates")!=0:raise GateError("G1 readiness binding mismatch")
    # Bind all already reviewed Train/target manifests by their immutable evidence files; no labels are regenerated.
    files=[EXCEPTION,PROPOSAL,G1_VERIFY,G1_READY,G1_RECEIPT,G1/"diagnostic_ledger.json",G1/"G1_RELEASE.json",
      G1_SOURCE/"artifact_sha256.json",G1_SOURCE/"verification.json",
      REVIEW/"attempt_02/verification.json",
      DOC/"gates_amendment_v3.json",DOC/"gates_amendment_v2.json",
      ROOT/"experiments/t0_tw_core_20261016/execution_binding.json",
      ROOT/"experiments/t0_tw_core_20261016/runs/seed_2026/T0/config.json",
      ROOT/"experiments/t0_tw_core_20261016/runs/seed_2026/T0/run_binding.json",
      ROOT/"experiments/t0_tw_core_20261016/runs/seed_2026/T0/pairing.jsonl",
      ROOT/"experiments/t0_tw_core_20261016/runs/seed_2026/T0/step_000200.pt",
      CODE/"retinal_m0/maples_core.py",CODE/"retinal_m0/maples_branch_bce.py",
      CODE/"retinal_m0/maples_g1_runtime.py",CODE/"retinal_m0/maples_g1_safety.py",
      CODE/"retinal_m0/maples_g1_diagnostics.py",CODE/"retinal_m0/maples_g1_entry.py",
      CODE/"retinal_m0/maples_g2_engineering.py",CODE/"retinal_m0/maples_g2_runtime.py",
       CODE/"tests/test_maples_g2_engineering.py",CODE/'retinal_m0/maples_g2_worker.py',
       CODE/'tests/test_maples_g2_repaired.py',CODE/'retinal_m0/state.py',CODE/'retinal_m0/model.py',
       BASE/'train.jsonl']
    for p in files:
        if not p.is_file() or p.is_symlink():raise GateError("missing/symlink binding input: "+str(p))
    # M0 is SHA-only. Do not torch.load/open checkpoint.
    config=read_json(files[13]);run_binding=read_json(files[14])
    if file_sha(config["source_m0"])!=config["source_m0_sha256"] or run_binding["source_m0_sha256"]!=config["source_m0_sha256"]:
        raise GateError("original M0 digest mismatch")
    target_manifest=read_json(G1_SOURCE/"artifact_sha256.json")
    for rel,digest in target_manifest.items():
        p=G1_SOURCE/rel
        if not p.is_file() or file_sha(p)!=digest:raise GateError("bound Train target evidence drift: "+str(p))
    ledger=read_json(G1/"diagnostic_ledger.json");g1_release=read_json(G1/"G1_RELEASE.json")
    if ledger.get("forward_starts")!=32 or len(ledger.get("combinations",{}))!=32 or any(v.get("status")!="completed" for v in ledger["combinations"].values()):raise GateError("G1 production ledger is not the expected exhausted record")
    if g1_release.get("phase")!="G1_ONLY" or g1_release.get("max_forwards")!=32:raise GateError("G1 historical release artifact mismatch")
    source_hashes={**current_g1['source_hashes'],**{str(p):file_sha(p) for p in files}}
    return {"schema":2,"phase":"G2_ONLY","policy_sha256":POLICY_SHA,"proposal_sha256":PROPOSAL_SHA,
       "source_hashes":source_hashes,'Train_source_count':current_g1['Train_source_count'],
       'versions':current_g1['versions'],'production_ledger':str(G2_LEDGER),'gpu_uuid':GPU_UUID,
      "M0_path":config["source_m0"],"M0_sha256":config["source_m0_sha256"],"M0_content_loaded":False,
      "train110_evidence_sha256":file_sha(G1_SOURCE/"verification.json"),
      "train_target_manifest_sha256":file_sha(G1_SOURCE/"artifact_sha256.json"),
      "G1_receipt_sha256":file_sha(G1_RECEIPT),"G1_ledger_sha256":file_sha(G1/"diagnostic_ledger.json"),"G1_release_sha256":file_sha(G1/"G1_RELEASE.json"),"G1_remaining_forwards":0,
      "budget":{"global":400,"tracks":{"R0_LAMBDA0":200,"B1_CONT":50,"B1_SPLIT":50,"B2_CONT":50,"B2_SPLIT":50}},
      "fixed":{"lambda":0.1,"seed":2026,"batch":4,"patch":512,"steps":{"R0":200,"B1_total":100,"B2_total":100}},
      "scientific_training_authorized":False,"G2_RELEASE_issued":False}

def cpu_preflight():
    import torch
    if torch.cuda.is_initialized():raise GateError("CPU preflight cannot follow CUDA initialization")
    outroot=G2OUT/"cpu_attempts";outroot.mkdir(parents=True,exist_ok=True)
    for i in range(1,1000):
        out=outroot/f"attempt_{i:03d}"
        try:out.mkdir();break
        except FileExistsError:continue
    else:raise GateError("CPU attempt namespace exhausted")
    try:
        _require_current_hold(STATUS.read_text(encoding='utf-8'))
        if G2_RELEASE.exists() or G2_LEDGER.exists():raise GateError('CPU preflight cannot replace released/consumed evidence')
        binding=build_binding();resources=observe_resources()
        if resources["free_disk_bytes"]<CAPACITY:raise GateError("less than 1 GiB free for engineering evidence")
        atomic_json(out/"execution_binding.json",binding)
        verification={"status":"CPU_PREFLIGHT_PASSED_GPU_HELD_FOR_SOL_REVIEW","binding_sha256":canonical_sha(binding),
          "binding_file_sha256":file_sha(out/"execution_binding.json"),"resources":resources,
          "CUDA_initialized":torch.cuda.is_initialized(),"retinal_UNet_constructed":False,"checkpoint_deserialized":False,
          "M0_hashed_only":True,"Dev_or_Test_access":False,"production_G1_ledger_touched":False,
          "G2_production_ledger_created":False,"GPU_updates":0,"GPU_RELEASE_issued":False,"science_authorized":False}
        atomic_json(out/"verification.json",verification)
        return out,verification
    except Exception as e:
        atomic_json(out/"failure.json",{"status":"CPU_PREFLIGHT_FAILED","error_type":type(e).__name__,"error":str(e),"GPU_updates":0})
        raise

def verify_release(*,full=False):
    state=current_g2_state(STATUS.read_text(encoding="utf-8"))
    if state.get("state")!="RELEASED" or state.get('GPU_released') is not True or not G2_RELEASE.is_file():
        raise GateError("G2 GPU blocked: Sol-issued evidence-bound G2_RELEASE.json is absent")
    for p in (G2_RELEASE,READY,RECEIPT):
        if p.is_symlink() or not p.is_file():raise GateError('release evidence missing/symlink')
    release=read_json(G2_RELEASE);binding=read_json(READY);receipt=read_json(RECEIPT)
    if release.get('phase')!='G2_ONLY' or release.get('review_owner')!='Sol' or release.get('policy_sha256')!=POLICY_SHA or release.get('charged_update_ceiling')!=400 or release.get('scientific_authorized') is not False:raise GateError('invalid G2 release scope')
    if not release.get('release_id') or state.get('release_id')!=release['release_id'] or state.get('release_file_sha256')!=file_sha(G2_RELEASE):raise GateError('STATUS/release mismatch')
    digest=canonical_sha(binding)
    if state.get('binding_sha256')!=digest or release.get('binding_sha256')!=digest or release.get('binding_file_sha256')!=file_sha(READY) or release.get('cpu_receipt_sha256')!=file_sha(RECEIPT):raise GateError('release/binding/receipt mismatch')
    if receipt.get('status')!='CPU_VALIDATED_GPU_HELD' or receipt.get('binding_sha256')!=digest or not receipt.get('all_passed') or not receipt.get('evidence_hashes'):raise GateError('CPU evidence not accepted')
    for path,sha in receipt['evidence_hashes'].items():
        if not Path(path).resolve().is_relative_to(G2OUT.resolve()) or file_sha(path)!=sha:raise GateError('CPU evidence drift')
    if binding.get('phase')!='G2_ONLY' or binding.get('production_ledger')!=str(G2_LEDGER) or binding.get('gpu_uuid')!=GPU_UUID:raise GateError('execution destination drift')
    for path,sha in binding['source_hashes'].items():
        # Raw data and large checkpoints are rehashed before each process; small
        # code/policy inputs are checked again at every forward and update.
        if full or Path(path).stat().st_size<1048576:
            if file_sha(path)!=sha:raise GateError('bound source drift: '+path)
    if full and build_binding()!=binding:raise GateError('current execution differs from reviewed binding')
    return binding

def run_g2():
    import secrets
    from retinal_m0.maples_g2_worker import SEGMENTS,validate_segment
    binding=verify_release(full=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise GateError('GPU0 UUID mask required before launch')
    with lock(RUN_LOCK):
        verify_release(full=True)
        if G2_LEDGER.exists() or (G2OUT/'matrix_launch.json').exists() or (G2OUT/'runs').exists():raise GateError('prior G2 attempt exists; no automatic restart/reset')
        budget=Budget(G2_LEDGER,canonical_sha(binding));budget.initialize_isolated()
        started=time.perf_counter()
        try:
            for index,segment in enumerate(SEGMENTS):
                verify_release(full=True)
                token=secrets.token_hex(32)
                atomic_json(G2OUT/'matrix_launch.json',{'segment_index':index,'segment':list(segment),'parent_pid':os.getpid(),
                    'token_sha256':__import__('hashlib').sha256(token.encode()).hexdigest(),'binding_sha256':canonical_sha(binding)})
                env=dict(os.environ,RETINA_G2_SEGMENT_TOKEN=token,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                with (G2OUT/f'worker_{index:02d}.log').open('x') as log:
                    subprocess.run([sys.executable,'-B','-m','retinal_m0.maples_g2_runtime','--segment',str(index)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                validate_segment(G2OUT/'runs',index,canonical_sha(binding))
            s=budget.snapshot()
            if s['charged_events']!=400 or s['verified_successes']!=400 or s['failed_events'] or s['uncertain_events'] or s.get('compute_pending') or any(t.get('pending') for t in s['tracks'].values()):raise GateError('final 400-update accounting mismatch')
            atomic_json(G2OUT/'completion.json',{'status':'ENGINEERING_PASSED_FOR_SOL_REVIEW','ledger':s,'elapsed_seconds':time.perf_counter()-started,'scientific_authorized':False})
        except BaseException as e:
            s=budget.snapshot()
            for track,t in s['tracks'].items():
                if t.get('pending'):budget.mark_crash_uncertain(track)
            budget.stop(str(e));atomic_json(G2OUT/'matrix_failure.json',{'error':str(e),'ledger':budget.snapshot(),'elapsed_seconds':time.perf_counter()-started})
            raise

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    m=p.add_mutually_exclusive_group(required=True);m.add_argument("--cpu-preflight",action="store_true");m.add_argument("--run-g2",action="store_true")
    m.add_argument('--segment',type=int,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.cpu_preflight:
        out,v=cpu_preflight();print(json.dumps({"out":str(out),**v},ensure_ascii=False))
    elif a.segment is not None:
        from retinal_m0.maples_g2_worker import gpu_worker
        gpu_worker(a.segment)
    else:run_g2()
if __name__=="__main__":main()

