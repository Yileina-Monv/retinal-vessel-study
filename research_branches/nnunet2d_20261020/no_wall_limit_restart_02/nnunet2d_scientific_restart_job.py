"""One explicitly approved fresh restart; final-only Dev28; no automatic retries."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

BASE = Path("${RETINA_PROJECT_ROOT}")
FREEZE = BASE / "docs/nnunet2d_seed2026_100epochs_20261020/no_wall_limit_restart_02"
PROTO = json.loads((FREEZE / "protocol.json").read_text())
ROOT = Path(PROTO["paths"]["scientific_root"])
BRANCH = Path(PROTO["paths"]["branch_code"])
PYTHON = PROTO["paths"]["python"]
STATUS = BASE / "code/retinal-vessel-study/STATUS.md"
START_MARKER = "<!-- NNUNET_NO_WALL_RESTART02_CURRENT -->"
END_MARKER = "<!-- /NNUNET_NO_WALL_RESTART02_CURRENT -->"


def write_status(state):
    """Replace only this task's marked block; preserve unrelated P1/newer edits."""
    current = STATUS.read_text()
    if current.count(START_MARKER) != 1 or current.count(END_MARKER) != 1:
        raise RuntimeError("Unique current-task STATUS block missing; no management overwrite")
    start = current.index(START_MARKER)
    end = current.index(END_MARKER, start) + len(END_MARKER)
    summary = (START_MARKER + "\n"
        + "**nnU-Net无墙钟限时重启02：** " + state["status"]
        + "；阶段=" + state["stage"]
        + "；更新UTC=" + state["updated_utc"]
        + "。GPU0固定100epochs/25000次尝试，最终模型才评价Dev28；GPU1留给待另批主线。"
        + "旧916次更新及1441.203855秒累计科学消耗保留、不计入新轨迹成功数；无自动重启，不访问Test，B1/B2未放行。"
        + "证据：" + str(ROOT / "job_state.json") + "、budget_ledger.json、per_update.jsonl。\n"
        + END_MARKER)
    backup_dir = ROOT / "management_backups"
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / ("STATUS_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".md")
    shutil.copy2(STATUS, backup)
    if STATUS.read_text() != current:
        raise RuntimeError("STATUS changed concurrently; preserve and report instead of overwrite")
    tmp = STATUS.with_name("STATUS.nnunet_restart02." + str(os.getpid()) + ".tmp")
    tmp.write_text(current[:start] + summary + current[end:])
    if STATUS.read_text() != current:
        tmp.unlink()
        raise RuntimeError("Concurrent STATUS edit detected; no overwrite")
    os.replace(tmp, STATUS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved-restart", action="store_true")
    parser.add_argument("--preflight-sha256", required=True)
    args = parser.parse_args()
    if not args.execute_approved_restart:
        raise SystemExit("Explicit approved restart flag required")
    with (ROOT / "job_started.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat()}, stream)
    state = {"status": "RUNNING", "pid": os.getpid(), "stage": "train",
             "scientific_scope": "one fresh seed2026/100epochs; final-only Dev28",
             "wall_clock_hard_cap_seconds": None, "automatic_restart": False,
             "official_test_accessed": False, "B1_B2_training_authorized": False}
    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        tmp = ROOT / "job_state.json.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(tmp, ROOT / "job_state.json")
        try:
            write_status(state)
        except Exception as exc:
            with (ROOT / "management_update_warnings.jsonl").open("a") as stream:
                stream.write(json.dumps({"utc": state["updated_utc"], "error": repr(exc)}) + "\n")
    common = [PYTHON, "-B", str(BRANCH / "nnunet2d_scientific_supervisor.py"),
              "--preflight-sha256", args.preflight_sha256]
    save()
    try:
        for stage, flag in (("train", "--start-scientific-training"),
                            ("final_dev_inference", "--start-final-dev-inference")):
            state["stage"] = stage
            save()
            with (ROOT / (stage + "_phase_launcher.log")).open("x") as log:
                result = subprocess.run(common + ["--mode", stage, flag],
                                        stdout=log, stderr=subprocess.STDOUT)
            state[stage + "_exit_code"] = result.returncode
            if result.returncode != 0:
                state["status"] = "STOPPED_INCOMPLETE"
                save()
                return result.returncode
        state["status"] = "COMPLETED_PENDING_SOL_REVIEW"
        save()
        return 0
    except BaseException as exc:
        state["status"] = "STOPPED_EXCEPTION"
        state["error"] = repr(exc)
        state["traceback"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
