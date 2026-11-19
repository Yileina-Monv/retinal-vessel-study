"""One explicit user-approved zero-update restart, then fixed final Dev evaluation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

BASE = Path("${RETINA_PROJECT_ROOT}")
BRANCH = BASE / "code/retinal-vessel-study/research_branches/nnunet2d_20261020"
ROOT = BASE / "experiments/nnunet2d_branch_20261020/scientific_seed2026_100epochs_v1"
RESTART = ROOT / "restart_01"
PYTHON = str(BASE / "envs/nnunet2d_v2/bin/python")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved-restart", action="store_true")
    parser.add_argument("--preflight-sha256", required=True)
    args = parser.parse_args()
    if not args.execute_approved_restart:
        raise SystemExit("Explicit approved restart flag required; no automatic launch")
    preflight = RESTART / "preflight/verification.json"
    binding = RESTART / "preflight/execution_binding.json"
    authorization = RESTART / "authorization.json"
    if not all(p.is_file() for p in (preflight, binding, authorization)):
        raise SystemExit("Restart artifacts missing")
    with (RESTART / "job_started.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat()}, stream)
    state_path = RESTART / "job_state.json"
    state = {"status": "RUNNING", "pid": os.getpid(), "stage": "train",
             "scientific_scope": "one fresh seed2026/100epochs; then final-only Dev28",
             "official_test_accessed": False}
    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(tmp, state_path)
    common = [PYTHON, "-B", str(BRANCH / "nnunet2d_scientific_supervisor.py"),
              "--preflight-sha256", args.preflight_sha256,
              "--preflight-file", str(preflight), "--binding-file", str(binding)]
    save()
    try:
        for stage, flag in (("train", "--start-scientific-training"),
                            ("final_dev_inference", "--start-final-dev-inference")):
            state["stage"] = stage
            save()
            command = common + ["--mode", stage, flag]
            if stage == "train":
                command += ["--restart-authorization", str(authorization)]
            with (RESTART / (stage + "_phase_launcher.log")).open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            state[stage + "_exit_code"] = result.returncode
            if result.returncode != 0:
                state["status"] = "STOPPED_INCOMPLETE"
                state["automatic_restart"] = False
                save()
                return result.returncode
        state["status"] = "COMPLETED_PENDING_SOL_REVIEW"
        state["automatic_restart"] = False
        save()
        return 0
    except BaseException as exc:
        state["status"] = "STOPPED_EXCEPTION"
        state["error"] = repr(exc)
        state["traceback"] = traceback.format_exc()
        state["automatic_restart"] = False
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
