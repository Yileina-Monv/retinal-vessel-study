"""Source-bound fixed-budget runner. No normal-progress wall-clock deadline."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path("${RETINA_PROJECT_ROOT}")
FREEZE = BASE / "docs/nnunet2d_seed2026_100epochs_20261020/no_wall_limit_restart_02"
PROTO = json.loads((FREEZE / "protocol.json").read_text())
ROOT = Path(PROTO["paths"]["scientific_root"])
BRANCH = Path(PROTO["paths"]["branch_code"])
PYTHON = PROTO["paths"]["python"]
PREFLIGHT = ROOT / "preflight/verification.json"
BINDING = ROOT / "execution_binding.json"
LEDGER = ROOT / "budget_ledger.json"
EVENTS = ROOT / "budget_events.jsonl"
GPU_UUID = PROTO["operational_amendment"]["gpu0_uuid"]
PRIOR_ROOT = Path(PROTO["operational_amendment"]["previous_scientific_root"])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def validate_scientific_invariants(parent, amended):
    for section in ("data", "recipe", "randomness", "monitoring", "evaluation", "statistics"):
        if parent[section] != amended[section]:
            raise RuntimeError("Unauthorized scientific change: " + section)
    for key, value in parent["scope"].items():
        if key != "no_replay_budget" and amended["scope"].get(key) != value:
            raise RuntimeError("Unauthorized scope change: " + key)
    for key in ("python", "data_root", "raw", "preprocessed", "engineering"):
        if parent["paths"][key] != amended["paths"][key]:
            raise RuntimeError("Data/environment path changed: " + key)
    for key, value in parent["inference"].items():
        if key != "no_automatic_oom_fallback" and amended["inference"].get(key) != value:
            raise RuntimeError("Inference rule changed: " + key)
    if amended["budget"]["cumulative_gpu_wall_cap_seconds"] is not None:
        raise RuntimeError("Normal-progress wall-clock cap must be absent")


def parse_gpu_snapshot(rows_text, apps_text, expected_uuid):
    rows = [line.split(",") for line in rows_text.strip().splitlines() if line.strip()]
    selected = [row for row in rows if row[0].strip() == "0"]
    if len(selected) != 1 or len(selected[0]) != 4:
        raise RuntimeError("Physical GPU0 not uniquely identified")
    row = selected[0]
    if row[1].strip() != expected_uuid:
        raise RuntimeError("Physical GPU0 UUID changed")
    memory = int(row[2].strip().split()[0])
    utilization = float(row[3].strip().split()[0])
    selected_apps = []
    for line in apps_text.strip().splitlines():
        if not line.strip() or "No running" in line:
            continue
        fields = line.split(",")
        if len(fields) < 4:
            raise RuntimeError("Unparseable GPU process snapshot")
        if fields[0].strip() == expected_uuid:
            selected_apps.append(line)
    return {"gpu0_uuid": expected_uuid, "memory_mib": memory,
            "utilization_pct": utilization, "compute_apps_on_gpu0": selected_apps,
            "other_gpu_jobs_ignored": True,
            "idle": memory <= 512 and utilization == 0 and not selected_apps}


def gpu_resource_gate():
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader"], text=True)
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader"], text=True)
    result = parse_gpu_snapshot(rows, apps, GPU_UUID)
    if not result["idle"]:
        raise RuntimeError("GPU0 busy; never evict or switch GPU: " + str(result))
    free_gib = int(subprocess.check_output(
        ["df", "-BG", str(BASE)], text=True).splitlines()[1].split()[3].rstrip("G"))
    if free_gib < 50:
        raise RuntimeError("Free disk below50GiB; no deletion authorized")
    result["free_disk_gib"] = free_gib
    return result


def validate_fresh_authorization(auth, preflight_sha, binding_sha, prior_ledger_sha):
    if auth.get("status") != "USER_APPROVED_FRESH_NO_WALL_LIMIT_RESTART":
        raise RuntimeError("Explicit user authorization missing")
    if (auth.get("seed"), auth.get("epochs"), auth.get("attempt_cap"),
        auth.get("wall_clock_hard_cap_seconds")) != (2026, 100, 25000, None):
        raise RuntimeError("Authorization would alter fixed recipe/budget")
    if (auth.get("preflight_sha256") != preflight_sha or
        auth.get("execution_binding_sha256") != binding_sha or
        auth.get("prior_budget_ledger_sha256") != prior_ledger_sha):
        raise RuntimeError("Authorization evidence changed")
    if auth.get("prior_successful_updates") != 916:
        raise RuntimeError("Discarded trajectory is not explicitly accounted")
    if auth.get("automatic_restart") is not False:
        raise RuntimeError("Automatic restart not authorized")


def validate_final_training(training, ledger):
    if (training.get("status") != "TRAINING_COMPLETE" or
        training.get("attempts") != 25000 or training.get("successful_updates") != 25000 or
        training.get("internal_train_monitor_batches") != 5000 or
        ledger.get("optimizer_attempts") != 25000 or ledger.get("successful_updates") != 25000 or
        ledger.get("internal_train_monitor_batches") != 5000):
        raise RuntimeError("Terminal training incomplete; Dev remains sealed")


def dump_atomic(path, doc):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def record_event(doc):
    with EVENTS.open("a") as stream:
        stream.write(json.dumps(doc, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_binding(preflight_sha):
    pre = json.loads(PREFLIGHT.read_text())
    bind = json.loads(BINDING.read_text())
    if pre.get("status") != "PASS" or sha(PREFLIGHT) != preflight_sha:
        raise RuntimeError("Exact passing CPU preflight required")
    if pre.get("execution_binding_sha256") != sha(BINDING):
        raise RuntimeError("Binding changed after preflight")
    if bind.get("protocol_sha256") != sha(FREEZE / "protocol.json"):
        raise RuntimeError("Amended protocol changed")
    for name, expected in bind["implementation_sha256s"].items():
        if sha(BRANCH / name) != expected:
            raise RuntimeError("Implementation drift: " + name)
    parent_path = Path(PROTO["operational_amendment"]["parent_protocol"])
    if sha(parent_path) != PROTO["operational_amendment"]["parent_protocol_sha256"]:
        raise RuntimeError("Original frozen protocol changed")
    validate_scientific_invariants(json.loads(parent_path.read_text()), PROTO)
    return pre, bind


def count_logs():
    events = [json.loads(x) for x in (ROOT / "per_update.jsonl").read_text().splitlines()
              if x.strip()] if (ROOT / "per_update.jsonl").exists() else []
    monitor_path = ROOT / "internal_train_monitor.jsonl"
    monitors = sum(bool(x.strip()) for x in monitor_path.open()) if monitor_path.exists() else 0
    return len(events), sum(x.get("successful_update") is True for x in events), monitors


def run_stage(stage, args):
    verify_binding(args.preflight_sha256)
    resources = gpu_resource_gate()
    auth_path = FREEZE / "authorization.json"
    auth = json.loads(auth_path.read_text())
    prior_sha = sha(PRIOR_ROOT / "budget_ledger.json")
    validate_fresh_authorization(auth, sha(PREFLIGHT), sha(BINDING), prior_sha)
    prior_accounting = json.loads((FREEZE / "prior_trajectory_accounting.json").read_text())
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {
        "status": "PRELAUNCH", "cumulative_gpu_process_seconds": 0.0,
        "prior_scientific_gpu_process_seconds": prior_accounting["gpu_process_seconds"],
        "prior_discarded_optimizer_attempts": prior_accounting["attempts"],
        "prior_discarded_successful_updates": prior_accounting["successful_updates"],
        "wall_clock_hard_cap_seconds": None, "optimizer_attempt_cap": 25000,
        "optimizer_attempts": 0, "successful_updates": 0,
        "internal_train_monitor_batches": 0, "stages": [],
        "automatic_restart": False, "official_test_accessed": False}
    if ledger.get("official_test_accessed") or ledger.get("optimizer_attempts", 0) > 25000:
        raise RuntimeError("Ledger violates fixed scope")
    if stage == "train":
        if not args.start_scientific_training:
            raise RuntimeError("Explicit scientific training flag missing")
        if ledger["stages"] or (ROOT / "initial_state.json").exists() or list(ROOT.rglob("*.pth")):
            raise RuntimeError("Fresh trajectory already attempted; no automatic restart")
        old_state = json.loads((PRIOR_ROOT / "restart_01/job_state.json").read_text())
        if old_state.get("status") != "STOPPED_INCOMPLETE":
            raise RuntimeError("Old job not confirmed stopped")
        model_dir = Path(PROTO["paths"]["nnUNet_results"]) / "Dataset501_MAPLES/nnUNetTrainer_100epochs__nnUNetPlans_8GBtarget__2d"
        if model_dir.exists():
            raise RuntimeError("Prior model directory exists; do not overwrite")
        with (ROOT / "authorization_consumed.json").open("x") as stream:
            json.dump({"utc": utc(), "authorization_sha256": sha(auth_path),
                       "preflight_sha256": args.preflight_sha256}, stream)
        command = [PYTHON, "-B", str(BRANCH / "nnunet2d_scientific_train.py"),
                   "--execute-scientific-run"]
    else:
        if not args.start_final_dev_inference:
            raise RuntimeError("Explicit terminal-only inference flag missing")
        training = json.loads((ROOT / "training_completion.json").read_text())
        validate_final_training(training, ledger)
        with (ROOT / "final_dev_inference_attempted.json").open("x") as stream:
            json.dump({"utc": utc(), "checkpoint_final_sha256": training["checkpoint_final_sha256"]}, stream)
        command = [PYTHON, "-B", str(BRANCH / "nnunet2d_scientific_infer_eval.py"),
                   "--run-final-dev-inference"]
    spent = float(ledger["cumulative_gpu_process_seconds"])
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": "2026", "CUDA_VISIBLE_DEVICES": GPU_UUID,
                "nnUNet_compile": "false", "nnUNet_n_proc_DA": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "SCIENTIFIC_PREFLIGHT_PATH": str(PREFLIGHT),
                "SCIENTIFIC_BINDING_PATH": str(BINDING),
                "nnUNet_raw": PROTO["paths"]["raw"],
                "nnUNet_preprocessed": PROTO["paths"]["preprocessed"],
                "nnUNet_results": PROTO["paths"]["nnUNet_results"]})
    started = time.monotonic()
    env["SCIENTIFIC_PROCESS_START_MONOTONIC"] = str(started)
    log_path = ROOT / (stage + "_supervisor.log")
    stage_doc = {"stage": stage, "started_utc": utc(), "status": "RUNNING",
                 "gpu_uuid": GPU_UUID, "wall_clock_deadline_seconds": None,
                 "command": command, "log": str(log_path), "cumulative_before_seconds": spent}
    ledger["status"] = "RUNNING"
    ledger["stages"].append(stage_doc)
    dump_atomic(LEDGER, ledger)
    record_event({"event": "stage_start", **stage_doc, "resource_check": resources})
    with log_path.open("x") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        stage_doc["pid"] = proc.pid
        dump_atomic(LEDGER, ledger)
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                record_event({"event": "heartbeat", "stage": stage, "utc": utc(),
                              "stage_elapsed_seconds": elapsed,
                              "cumulative_projected_seconds": spent + elapsed,
                              "all_scientific_projected_seconds":
                                  prior_accounting["gpu_process_seconds"] + spent + elapsed,
                              "child_pid": proc.pid, "wall_clock_stop_enabled": False})
                time.sleep(10)
        except BaseException:
            # Only our own child group; never touch another GPU user's processes.
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
            raise
    elapsed = time.monotonic() - started
    stage_doc.update({"finished_utc": utc(), "elapsed_process_seconds": elapsed,
                      "exit_code": proc.returncode,
                      "status": "PASS" if proc.returncode == 0 else "INCOMPLETE"})
    ledger["cumulative_gpu_process_seconds"] = spent + elapsed
    ledger["all_scientific_gpu_process_seconds"] = prior_accounting["gpu_process_seconds"] + spent + elapsed
    attempts, successes, monitors = count_logs()
    ledger.update({"optimizer_attempts": attempts, "successful_updates": successes,
                   "internal_train_monitor_batches": monitors, "updated_utc": utc()})
    if stage == "train":
        complete = False
        if proc.returncode == 0 and (ROOT / "training_completion.json").exists():
            try:
                validate_final_training(json.loads((ROOT / "training_completion.json").read_text()), ledger)
                complete = True
            except RuntimeError:
                pass
        ledger["status"] = "TRAINING_COMPLETE" if complete else "INCOMPLETE"
    else:
        complete = proc.returncode == 0 and (ROOT / "completion_verification.json").exists()
        if complete:
            complete = json.loads((ROOT / "completion_verification.json").read_text()).get("status") == "PASS"
        ledger["status"] = "EVALUATION_COMPLETE" if complete else "INCOMPLETE"
    dump_atomic(LEDGER, ledger)
    record_event({"event": "stage_end", "stage": stage, "utc": utc(),
                  "status": ledger["status"], "exit_code": proc.returncode,
                  "cumulative_gpu_process_seconds": spent + elapsed,
                  "all_scientific_gpu_process_seconds": ledger["all_scientific_gpu_process_seconds"]})
    print(json.dumps(ledger, indent=2))
    return 0 if complete else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "final_dev_inference"), required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--start-scientific-training", action="store_true")
    parser.add_argument("--start-final-dev-inference", action="store_true")
    args = parser.parse_args()
    return run_stage(args.mode, args)


if __name__ == "__main__":
    raise SystemExit(main())
