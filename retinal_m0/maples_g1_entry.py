"""Fail-closed G1-only entry and CPU preflight for MAPLES B1/B2.

This module never constructs a retinal model or initializes CUDA in preflight.
GPU diagnostics require a separate Sol-issued release and are deliberately not
started by this CPU delivery.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path("${RETINA_PROJECT_ROOT}")
CODE = ROOT / "code/retinal-vessel-study"
STAGE = ROOT / "experiments/full_fov_thin_skeleton_bce_20261019"
REVIEW = STAGE / "target_definition_review_v1"
CPU_EVIDENCE = REVIEW / "sol_readiness_review_20261021/attempt_01"
BATCH_EVIDENCE = REVIEW / "scientific_review_20261021/attempt_01/G1_fixed_batches.json"
SOURCE_EVIDENCE = REVIEW / "attempt_02"
DOC = ROOT / "docs/full_fov_thin_skeleton_bce_20261019/branch_review_20261021"
OUTPUT_ROOT = REVIEW / "g1_engineering_v3"
LEDGER_PATH = OUTPUT_ROOT / "diagnostic_ledger.json"
MAX_COMBINATIONS = 32
EXPECTED_STEPS = (115, 1442, 1660, 1, 4, 3, 66, 400)
EXPECTED_METHODS = ("B1", "B2")
EXPECTED_PRECISIONS = ("FP32", "AMP")


class GateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class BudgetLedger:
    """Atomic cumulative ledger; each fixed combo can be reserved only once."""

    def __init__(self, path: Path, *, recover_stale: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with self._locked():
            if not self.path.exists():
                atomic_json(self.path, self._empty())
            state = self._read_unlocked()
            self._validate_state(state)
            if recover_stale:
                changed = False
                for item in state["combinations"].values():
                    if item["status"] in ("reserved", "started"):
                        item["status"] = "failed"
                        item["failure"] = "recovered_incomplete_after_process_restart"
                        item["finished_at"] = _now()
                        changed = True
                if changed:
                    state["recovery_events"] += 1
                    atomic_json(self.path, state)

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"schema": 1, "cap": MAX_COMBINATIONS, "combinations": {},
                "autograd_calls": 0, "optimizer_step_attempts": 0,
                "recovery_events": 0, "updated_at": _now()}

    def _locked(self):
        class Lock:
            def __init__(inner, outer):
                inner.outer = outer
                inner.stream = None
            def __enter__(inner):
                inner.stream = inner.outer.lock_path.open("a+")
                fcntl.flock(inner.stream.fileno(), fcntl.LOCK_EX)
                return inner
            def __exit__(inner, *_):
                fcntl.flock(inner.stream.fileno(), fcntl.LOCK_UN)
                inner.stream.close()
        return Lock(self)

    def _read_unlocked(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise GateError("budget ledger unreadable; refusing reset") from exc

    @staticmethod
    def _validate_state(state):
        if state.get("schema") != 1 or state.get("cap") != MAX_COMBINATIONS:
            raise GateError("budget ledger schema/cap mismatch; refusing reset")
        if not isinstance(state.get("combinations"), dict):
            raise GateError("budget ledger combinations malformed")
        if len(state["combinations"]) > MAX_COMBINATIONS:
            raise GateError("budget ledger already exceeds fixed cap")

    def _mutate(self, operation):
        with self._locked():
            state = self._read_unlocked()
            self._validate_state(state)
            operation(state)
            state["updated_at"] = _now()
            atomic_json(self.path, state)
            return state

    def reserve(self, combo_id: str, metadata: dict[str, Any]) -> None:
        def apply(state):
            if combo_id in state["combinations"]:
                raise GateError("duplicate combination; no implicit retry or replay")
            if len(state["combinations"]) >= MAX_COMBINATIONS:
                raise GateError("cumulative 32-combination budget exhausted")
            state["combinations"][combo_id] = {
                "metadata": metadata, "status": "reserved", "reserved_at": _now(),
                "started_at": None, "finished_at": None, "failure": None,
            }
        self._mutate(apply)

    def mark_started(self, combo_id: str) -> None:
        def apply(state):
            item = self._require(state, combo_id, "reserved")
            item["status"] = "started"
            item["started_at"] = _now()
        self._mutate(apply)

    def mark_completed(self, combo_id: str, *, autograd_calls: int) -> None:
        def apply(state):
            item = self._require(state, combo_id, "started")
            if type(autograd_calls) is not int or autograd_calls < 0:
                raise GateError("invalid autograd call count")
            item["status"] = "completed"
            item["finished_at"] = _now()
            item["autograd_calls"] = autograd_calls
            state["autograd_calls"] += autograd_calls
        self._mutate(apply)

    def mark_failed(self, combo_id: str, reason: str, *, autograd_calls: int = 0) -> None:
        def apply(state):
            item = state["combinations"].get(combo_id)
            if item is None or item["status"] not in ("reserved", "started"):
                raise GateError("failure transition requires reserved/started combo")
            item["status"] = "failed"
            item["finished_at"] = _now()
            item["failure"] = str(reason)[:500]
            item["autograd_calls"] = int(autograd_calls)
            state["autograd_calls"] += int(autograd_calls)
        self._mutate(apply)

    def record_optimizer_step_attempt(self, combo_id: str) -> None:
        def apply(state):
            if combo_id not in state["combinations"]:
                raise GateError("optimizer step without a reserved G1 combination")
            state["optimizer_step_attempts"] += 1
        self._mutate(apply)

    @staticmethod
    def _require(state, combo_id, status):
        item = state["combinations"].get(combo_id)
        if item is None or item["status"] != status:
            raise GateError(f"invalid transition for {combo_id}; expected {status}")
        return item

    def snapshot(self):
        with self._locked():
            state = self._read_unlocked()
            self._validate_state(state)
            return state


class OptimizerStepProhibited:
    """Install on a CPU/GPU optimizer before G1; every attempted step is logged."""

    def __init__(self, optimizer, ledger: BudgetLedger, combo_id: str):
        self.optimizer = optimizer
        self.ledger = ledger
        self.combo_id = combo_id
        self.original_step = optimizer.step
        self._installed = False

    def __enter__(self):
        if self._installed:
            raise GateError("optimizer guard already installed")
        def blocked(*args, **kwargs):
            self.ledger.record_optimizer_step_attempt(self.combo_id)
            raise GateError("optimizer.step is prohibited in G1")
        self.optimizer.step = blocked
        self._installed = True
        return self

    def __exit__(self, *_):
        if self._installed:
            self.optimizer.step = self.original_step
            self._installed = False


def tensor_state_digest(module) -> str:
    """Digest parameters and buffers without serializing/loading a checkpoint."""
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        h.update(str(value.dtype).encode("ascii"))
        h.update(str(tuple(value.shape)).encode("ascii"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def fixed_combinations(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(batches) != 8:
        raise GateError("fixed G1 manifest must contain exactly 8 batches")
    observed_steps = tuple(item.get("step_original_T0_one_based") for item in batches)
    if observed_steps != EXPECTED_STEPS:
        raise GateError("fixed G1 batch steps/order mismatch")
    combinations = []
    for batch in batches:
        if len(batch.get("draws", [])) != 4:
            raise GateError("fixed G1 batch must have exactly four draws")
        for method in EXPECTED_METHODS:
            for precision in EXPECTED_PRECISIONS:
                combo_id = f"step{batch['step_original_T0_one_based']:04d}_{method}_{precision}"
                combinations.append({"combo_id": combo_id,
                    "step": batch["step_original_T0_one_based"],
                    "method": method, "precision": precision})
    if len({x["combo_id"] for x in combinations}) != MAX_COMBINATIONS:
        raise GateError("fixed combination manifest is not unique 32")
    return combinations


def _verify_sources(source_binding_path: Path) -> tuple[dict[str, str], int]:
    expected = json.loads(source_binding_path.read_text(encoding="utf-8"))
    if not isinstance(expected, dict) or len(expected) < 300:
        raise GateError("Sol Train source binding missing or incomplete")
    for raw_path, expected_hash in expected.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise GateError(f"protected source drift: {raw_path}")
    return expected, len(expected)


def _verify_batches(batches: list[dict[str, Any]], draw_evidence: list[dict[str, Any]],
                    train_names: set[str], source_artifacts: dict[str, str]) -> dict[str, Any]:
    import numpy as np
    from retinal_m0.maples_branch_bce import transform_native_target
    prior = {int(row["draw"]): row for row in draw_evidence}
    if len(prior) != 32 or any(not row.get("target_and_geometry_match") for row in prior.values()):
        raise GateError("prior actual32 draw evidence missing/malformed")
    flattened = [draw for batch in batches for draw in batch.get("draws", [])]
    if len(flattened) != 32:
        raise GateError("G1 manifest must bind exactly 32 draws")
    crop_rows = {}
    with (SOURCE_EVIDENCE / "crops_8000_and_analytic_gradients.csv").open(encoding="utf-8", newline="") as stream:
        import csv
        for row in csv.DictReader(stream):
            crop_rows[(int(row["step"]), int(row["slot"]))] = row
    pairing_path = ROOT / "experiments/t0_tw_core_20261016/runs/seed_2026/T0/pairing.jsonl"
    pairs = [json.loads(line) for line in pairing_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(pairs) != 2000:
        raise GateError("original T0 pairing manifest must contain 2000 steps")
    artifacts = json.loads((SOURCE_EVIDENCE / "artifact_sha256.json").read_text(encoding="utf-8"))
    seen_draws = set()
    checked_targets = 0
    for batch in batches:
        step = int(batch["step_original_T0_one_based"])
        pair = pairs[step - 1]
        for draw in batch["draws"]:
            number, slot = int(draw["draw"]), int(draw["slot"])
            if number in seen_draws:
                raise GateError("duplicate frozen draw")
            seen_draws.add(number)
            if draw["filename"] not in train_names:
                raise GateError(f"draw outside Train whitelist: {draw['filename']}")
            old = prior.get(number)
            if old is None or old["filename"] != draw["filename"]:
                raise GateError("draw/filename differs from independently checked CPU evidence")
            if old["input_sha256"] != draw["input_sha256_original_T0"]:
                raise GateError("RGB tensor fingerprint differs from actual32 evidence")
            if not old["target_and_geometry_match"]:
                raise GateError("target/geometry mismatch in prior CPU evidence")
            geometry = draw["geometry"]
            if (number != (step - 1) * 4 + slot or pair["draws"][slot] != number
                or pair["filenames"][slot] != draw["filename"]
                or pair["geometry"][slot] != geometry):
                raise GateError("draw order/crop/flip/rotation differs from original T0 pairing")
            crop = crop_rows.get((step, slot))
            if crop is None or crop["filename"] != draw["filename"]:
                raise GateError("fixed draw absent from original CPU crop evidence")
            if [int(crop[k]) for k in ("top", "left", "hflip", "vflip", "ccw_rot90_quarters")] != geometry:
                raise GateError("crop geometry differs from original CPU crop evidence")
            if set(draw["target_crop_uint8_C_order_sha256"]) != {"B1", "B2"}:
                raise GateError("missing B1/B2 target crop fingerprints")
            name = draw["filename"]
            rel = f"candidate_masks_NOT_FOR_TRAINING/{name}.npz"
            mask_path = SOURCE_EVIDENCE / rel
            if artifacts.get(rel) != sha256_file(mask_path):
                raise GateError(f"candidate target source hash mismatch: {name}")
            with np.load(mask_path, allow_pickle=False) as masks:
                for label, key, count_key in (("B1", "B1_reference", "N_B1_reference"),
                                               ("B2", "B2_candidate", "N_B2")):
                    target = transform_native_target(masks[key], top=geometry[0], left=geometry[1],
                        size=512, hflip=bool(geometry[2]), vflip=bool(geometry[3]),
                        quarter_turns=geometry[4])
                    digest = hashlib.sha256(target.tobytes()).hexdigest()
                    if digest != draw["target_crop_uint8_C_order_sha256"][label]:
                        raise GateError(f"{label} target crop fingerprint mismatch for {name}/draw{number}")
                    if int(target.sum()) != int(draw[f"N_{label}"]) or int(crop[count_key]) != int(draw[f"N_{label}"]):
                        raise GateError(f"{label} target count mismatch for {name}/draw{number}")
                    checked_targets += 1
            if draw["N_B1"] > draw["N_B2"]:
                raise GateError("B1 is not a subset of B2 in a fixed draw")
    return {"draw_count": len(flattened), "unique_draws": len(seen_draws),
            "target_arrays_recomputed": checked_targets,
            "all_train": True, "all_input_fingerprints_match_prior": True,
            "all_T0_pairing_geometry_match": True, "all_target_crop_fingerprints_recomputed": True}


def _release_check(release_path: Path | None, binding_digest: str, gpu0_uuid: str | None):
    if release_path is None or not release_path.is_file():
        return {"authorized": False, "reason": "G1-only Sol release absent"}
    release = json.loads(release_path.read_text(encoding="utf-8"))
    required = {"phase": "G1_ONLY", "authorized": True, "gpu_launch_authorized": True,
                "execution_binding_sha256": binding_digest, "gpu0_uuid": gpu0_uuid}
    mismatches = [key for key, value in required.items() if release.get(key) != value]
    if mismatches:
        return {"authorized": False, "reason": "release mismatch: " + ",".join(mismatches)}
    return {"authorized": True, "reason": "release fields match; Sol review still required by procedure"}


G1_DIAGNOSTIC_FIELDS = (
    "base_bce", "base_dice", "aux_unweighted", "aux_weighted",
    "grad_l2_base", "grad_l2_aux_weighted", "grad_l2_combined",
    "cosine_base_aux", "clip_norm_combined", "clip_fraction_combined",
    "per_layer_grad_norms", "finite", "loss_scaler",
    "input_fingerprint_sha256", "target_fingerprint_sha256", "precision",
    "zero_lambda_value_exact", "zero_lambda_gradient_exact",
    "empty_aux_value_zero", "empty_aux_gradient_zero", "aux_grad_outside_target_l2",
    "analytic_fp32_max_abs_error", "analytic_fp32_max_rel_error",
    "unscaled_gradients", "gradient_scaling_removed",
)


def _capture_rng(model, torch):
    import random
    import numpy as np
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": None}
    if any(parameter.device.type == "cuda" for parameter in model.parameters()):
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng(state, torch):
    import random
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _rng_digest(state):
    import pickle
    h = hashlib.sha256()
    h.update(pickle.dumps(state["python"], protocol=4))
    h.update(pickle.dumps(state["numpy"], protocol=4))
    h.update(state["torch_cpu"].cpu().numpy().tobytes())
    if state["torch_cuda"] is not None:
        for item in state["torch_cuda"]:
            h.update(item.cpu().numpy().tobytes())
    return h.hexdigest()


def execute_g1_only(*, release_path: Path, execution_binding_sha256: str,
                    gpu0_uuid: str, batches: list[dict[str, Any]], model, optimizer,
                    ledger_path: Path, diagnostic_callback, cpu_test_substitute: bool = False):
    """Run exactly the fixed G1 diagnostic callbacks after an external Sol release.

    Production use requires a CUDA model and a separately acquired GPU0 lock.
    The CPU substitute switch exists only for isolated unit tests; the CLI never
    exposes it. Callback performs one model forward and the required unscaled
    gradient decomposition from that same graph; this controller measures API
    autograd calls, blocks optimizer.step, persists each reservation first and
    verifies complete parameter/buffer identity afterward.
    """
    import math
    import torch

    permit = _release_check(release_path, execution_binding_sha256, gpu0_uuid)
    if not permit["authorized"]:
        raise GateError("G1 release rejected before model execution: " + permit["reason"])
    if not cpu_test_substitute:
        expected_release = (OUTPUT_ROOT / "G1_RELEASE.json").resolve()
        if Path(release_path).resolve() != expected_release:
            raise GateError("production G1 release must use the fixed STATUS-bound release path")
        status_text = (CODE / "STATUS.md").read_text(encoding="utf-8")
        release_sha = sha256_file(Path(release_path))
        required_status_lines = (
            "G1_GPU_RELEASED: true",
            f"G1_GPU_RELEASE_BINDING_SHA256: {execution_binding_sha256}",
            f"G1_GPU_RELEASE_FILE_SHA256: {release_sha}",
        )
        if any(line not in status_text for line in required_status_lines):
            raise GateError("production G1 release is not cross-bound by current STATUS")
    if not gpu0_uuid or not gpu0_uuid.startswith("GPU-"):
        raise GateError("a concrete physical GPU0 UUID is required")
    parameters = list(model.parameters())
    if not parameters:
        raise GateError("G1 diagnostic model has no parameters")
    if not cpu_test_substitute and any(p.device.type != "cuda" for p in parameters):
        raise GateError("production G1 requires an explicitly authorized CUDA model")
    if cpu_test_substitute and any(p.device.type != "cpu" for p in parameters):
        raise GateError("CPU test substitute requires CPU-only model parameters")
    combos = fixed_combinations(batches)
    by_step = {int(row["step_original_T0_one_based"]): row for row in batches}
    ledger = BudgetLedger(ledger_path)
    initial_digest = tensor_state_digest(model)
    rng_initial = _capture_rng(model, torch)
    rng_initial_digest = _rng_digest(rng_initial)
    reports = []
    for combo in combos:
        _restore_rng(rng_initial, torch)
        combo_id = combo["combo_id"]
        ledger.reserve(combo_id, combo)
        ledger.mark_started(combo_id)
        before = tensor_state_digest(model)
        forward_count = [0]
        autograd_count = [0]
        hook = model.register_forward_hook(lambda *_args: forward_count.__setitem__(0, forward_count[0] + 1))
        old_grad, old_backward = torch.autograd.grad, torch.autograd.backward
        old_tensor_backward = torch.Tensor.backward
        torch.autograd.grad = lambda *a, **k: (autograd_count.__setitem__(0, autograd_count[0] + 1) or old_grad(*a, **k))
        torch.autograd.backward = lambda *a, **k: (autograd_count.__setitem__(0, autograd_count[0] + 1) or old_backward(*a, **k))
        torch.Tensor.backward = lambda self, *a, **k: (autograd_count.__setitem__(0, autograd_count[0] + 1) or old_tensor_backward(self, *a, **k))
        expected_input_digest = canonical_sha256([
            {"draw": int(draw["draw"]), "sha256": draw["input_sha256_original_T0"]}
            for draw in by_step[combo["step"]]["draws"]])
        expected_target_digest = canonical_sha256([
            {"draw": int(draw["draw"]), "sha256": draw["target_crop_uint8_C_order_sha256"][combo["method"]]}
            for draw in by_step[combo["step"]]["draws"]])
        try:
            with OptimizerStepProhibited(optimizer, ledger, combo_id):
                metrics = diagnostic_callback(combo, by_step[combo["step"]])
            if forward_count[0] != 1:
                raise GateError(f"{combo_id}: expected exactly one model forward, got {forward_count[0]}")
            if autograd_count[0] < 1:
                raise GateError(f"{combo_id}: no autograd/gradient call recorded")
            if not isinstance(metrics, dict) or any(field not in metrics for field in G1_DIAGNOSTIC_FIELDS):
                raise GateError(f"{combo_id}: required gradient/loss/AMP fields missing")
            if metrics["finite"] is not True:
                raise GateError(f"{combo_id}: nonfinite diagnostic")
            if metrics["input_fingerprint_sha256"] != expected_input_digest:
                raise GateError(f"{combo_id}: actual RGB input fingerprint mismatch")
            if metrics["target_fingerprint_sha256"] != expected_target_digest:
                raise GateError(f"{combo_id}: actual target fingerprint mismatch")
            if metrics["precision"] != combo["precision"]:
                raise GateError(f"{combo_id}: diagnostic precision mismatch")
            for flag in ("zero_lambda_value_exact", "zero_lambda_gradient_exact",
                         "empty_aux_value_zero", "empty_aux_gradient_zero",
                         "unscaled_gradients", "gradient_scaling_removed"):
                if metrics[flag] is not True:
                    raise GateError(f"{combo_id}: required numerical contract failed: {flag}")
            if float(metrics["aux_grad_outside_target_l2"]) > 1e-7:
                raise GateError(f"{combo_id}: auxiliary gradient outside target support")
            if combo["precision"] == "FP32":
                for field, limit in (("analytic_fp32_max_abs_error", 1e-7),
                                     ("analytic_fp32_max_rel_error", 1e-5)):
                    if metrics[field] is None or float(metrics[field]) > limit:
                        raise GateError(f"{combo_id}: analytic FP32 gradient tolerance failed: {field}")
            for field in G1_DIAGNOSTIC_FIELDS:
                value = metrics[field]
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    raise GateError(f"{combo_id}: nonfinite field {field}")
            after = tensor_state_digest(model)
            if before != after or before != initial_digest:
                raise GateError(f"{combo_id}: parameter/buffer changed during zero-update G1")
            state = ledger.snapshot()
            if state["optimizer_step_attempts"] != 0:
                raise GateError(f"{combo_id}: optimizer.step attempt detected")
            ledger.mark_completed(combo_id, autograd_calls=autograd_count[0])
            reports.append({"combo": combo, "metrics": metrics,
                            "forward_calls": forward_count[0],
                            "autograd_calls": autograd_count[0],
                            "parameter_buffer_sha256_before": before,
                            "parameter_buffer_sha256_after": after,
                            "status": "completed"})
        except Exception as exc:
            state = ledger.snapshot()
            current = state["combinations"][combo_id]["status"]
            if current in ("reserved", "started"):
                ledger.mark_failed(combo_id, f"{type(exc).__name__}: {exc}",
                                   autograd_calls=autograd_count[0])
            raise
        finally:
            hook.remove()
            torch.autograd.grad, torch.autograd.backward = old_grad, old_backward
            torch.Tensor.backward = old_tensor_backward
            _restore_rng(rng_initial, torch)
    _restore_rng(rng_initial, torch)
    rng_final_digest = _rng_digest(_capture_rng(model, torch))
    if rng_final_digest != rng_initial_digest:
        raise GateError("G1 diagnostics consumed or changed the caller RNG stream")
    paired = {}
    for row in reports:
        key = (row["combo"]["step"], row["combo"]["method"])
        paired.setdefault(key, {})[row["combo"]["precision"]] = row["metrics"]
    differences = []
    for (step, method), precisions in sorted(paired.items()):
        if set(precisions) != {"FP32", "AMP"}:
            raise GateError("missing FP32/AMP pair for a fixed batch/method")
        fp32, amp = precisions["FP32"], precisions["AMP"]
        fields = ("base_bce", "base_dice", "aux_unweighted", "aux_weighted",
                  "grad_l2_base", "grad_l2_aux_weighted", "grad_l2_combined",
                  "clip_norm_combined", "clip_fraction_combined")
        differences.append({"step": step, "method": method,
            "absolute_differences": {name: abs(float(fp32[name]) - float(amp[name])) for name in fields}})
    hold = any((row["metrics"]["grad_l2_aux_weighted"] > row["metrics"]["grad_l2_base"])
               or (row["metrics"]["grad_l2_base"] == 0 and row["metrics"]["grad_l2_aux_weighted"] > 0)
               for row in reports)
    return {"status": "G1_DIAGNOSTICS_COMPLETED_ZERO_UPDATES",
            "combination_count": len(reports), "optimizer_updates": 0,
            "autograd_calls": sum(row["autograd_calls"] for row in reports),
            "parameter_buffer_sha256_initial": initial_digest,
            "parameter_buffer_sha256_final": tensor_state_digest(model),
            "rng_sha256_initial": rng_initial_digest, "rng_sha256_final": rng_final_digest,
            "FP32_vs_AMP_absolute_differences": differences,
            "weighted_aux_gradient_review_trigger": hold,
            "reports": reports}


def run_cpu_preflight(*, release_path: Path | None = None) -> dict[str, Any]:
    import torch

    if torch.cuda.is_initialized():
        raise GateError("CUDA was initialized before CPU preflight")
    v3_path = DOC / "GATES_AMENDMENT_V3.md"
    v3_json_path = DOC / "gates_amendment_v3.json"
    v3 = json.loads(v3_json_path.read_text(encoding="utf-8"))
    v2_path = DOC / "gates_amendment_v2.json"
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    spec_path = DOC / "review_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if v3.get("status") != "EFFECTIVE_USER_APPROVED_GATE_AMENDMENT_ONLY":
        raise GateError("V3 policy is not effective")
    if v3["authorization_now"]["Luna_G1_CPU_implementation_and_preflight"] is not True:
        raise GateError("G1 CPU implementation not authorized")
    if v3["authorization_now"]["G1_GPU_execution"] or v3["authorization_now"]["G2_GPU_execution"] or v3["authorization_now"]["scientific_training"]:
        raise GateError("authorization unexpectedly permits GPU/scientific execution")
    if v3["medical_review"]["legacy_human_reviewed"] != 0 or v3["medical_review"]["legacy_total_cases"] != 26:
        raise GateError("legacy human review state drift")
    if v3["medical_review"]["legacy_status"] != "UNREVIEWED_0_OF_26_PRESERVE_WITHOUT_SIGNATURES":
        raise GateError("legacy human review must remain explicitly unreviewed")
    if spec["authorization"]["current_gpu_execution_permitted"] or spec["authorization"]["current_scientific_training_permitted"]:
        raise GateError("review spec unexpectedly permits GPU/scientific execution")
    if v3["parent_version"]["sha256"] != sha256_file(v2_path):
        raise GateError("V2 parent digest mismatch")
    if v3["parent_version"]["markdown_sha256"] != sha256_file(DOC / "GATES_AMENDMENT_V2.md"):
        raise GateError("V2 markdown parent digest mismatch")

    expected_sources, count = _verify_sources(CPU_EVIDENCE / "source_binding.json")
    batches = json.loads(BATCH_EVIDENCE.read_text(encoding="utf-8"))["batches"]
    draw_evidence = json.loads((CPU_EVIDENCE / "actual32_draws.json").read_text(encoding="utf-8"))
    with (SOURCE_EVIDENCE / "images_110.csv").open(encoding="utf-8", newline="") as stream:
        import csv
        train_names = {row["filename"] for row in csv.DictReader(stream)}
    batch_check = _verify_batches(batches, draw_evidence, train_names,
                                  {str(k): v for k, v in expected_sources.items()})
    combinations = fixed_combinations(batches)

    import numpy as np
    from retinal_m0.maples_branch_bce import branch_bce_loss, normalized_positive_bce
    # Import is deliberately safe: do not import the old CPU preparation CLI.
    if torch.cuda.is_initialized():
        raise GateError("CUDA initialized by an imported module during preflight")
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "numpy": np.__version__, "cuda_initialized": torch.cuda.is_initialized(),
           "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}

    gpu0_uuid = None
    gpu0_observation = {"status": "unavailable"}
    try:
        probe = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-gpu=uuid,index,memory.used,memory.total",
             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=8)
        fields = [x.strip() for x in probe.stdout.strip().split(",")]
        if len(fields) == 4 and fields[1] == "0":
            gpu0_uuid = fields[0]
            gpu0_observation = {"status": "read_only_observed", "uuid": fields[0], "index": 0,
                                "memory_used_mib": fields[2], "memory_total_mib": fields[3],
                                "ownership_lock_acquired": False}
        else:
            gpu0_observation = {"status": "malformed_or_wrong_gpu_response"}
    except Exception as exc:
        gpu0_observation = {"status": "observation_failed", "type": type(exc).__name__}
    disk = shutil_disk(STAGE)

    t0_dir = ROOT / "experiments/t0_tw_core_20261016/runs/seed_2026/T0"
    t0_binding = json.loads((t0_dir / "run_binding.json").read_text(encoding="utf-8"))
    t0_config = t0_dir / "config.json"
    m0_path = ROOT / "checkpoints/m0_10000_20261008/last.pt"
    if sha256_file(t0_config) != t0_binding["config_sha256"]:
        raise GateError("original T0 config digest mismatch")
    if sha256_file(m0_path) != t0_binding["source_m0_sha256"]:
        raise GateError("M0 source digest mismatch; checkpoint is hash-only and never loaded")
    base_hashes = {
        "entry_source": sha256_file(Path(__file__)),
        "cpu_test_source": sha256_file(CODE / "tests/test_maples_g1_entry.py"),
        "t0_config": sha256_file(t0_config),
        "t0_run_binding": sha256_file(t0_dir / "run_binding.json"),
        "m0_checkpoint_hash_only": t0_binding["source_m0_sha256"],
        "loss_source": sha256_file(CODE / "retinal_m0/maples_branch_bce.py"),
        "base_loss_source": sha256_file(CODE / "retinal_m0/maples_core.py"),
        "v3_markdown": sha256_file(v3_path), "v3_json": sha256_file(v3_json_path),
        "v2_markdown": sha256_file(DOC / "GATES_AMENDMENT_V2.md"),
        "v2_json": sha256_file(v2_path), "review_spec": sha256_file(spec_path),
        "batch_manifest": sha256_file(BATCH_EVIDENCE),
        "train_source_binding": sha256_file(CPU_EVIDENCE / "source_binding.json"),
        "actual32_evidence": sha256_file(CPU_EVIDENCE / "actual32_draws.json"),
        "train_images_manifest": sha256_file(SOURCE_EVIDENCE / "images_110.csv"),
        "T0_pairing": expected_sources[str(ROOT / "experiments/t0_tw_core_20261016/runs/seed_2026/T0/pairing.jsonl")],
    }
    binding = {"phase": "G1_ONLY_CPU_PREFLIGHT", "policy": "V3 plus inherited V2 technical rules",
               "source_hashes": base_hashes, "validated_train_source_hash_count": count,
               "fixed_combination_sha256": canonical_sha256(combinations),
               "fixed_draw_input_sha256": canonical_sha256([
                   {"draw": x["draw"], "filename": x["filename"],
                    "input_sha256": x["input_sha256_original_T0"],
                    "target_sha256": x["target_crop_uint8_C_order_sha256"]}
                   for b in batches for x in b["draws"]]),
                "M0_mode": "hash-only SHA256 verified against T0 binding; never deserialized",
               "environment": env}
    binding_digest = canonical_sha256(binding)
    release = _release_check(release_path, binding_digest, gpu0_uuid)
    ledger = BudgetLedger(LEDGER_PATH)
    ledger_state = ledger.snapshot()
    if len(ledger_state["combinations"]) > MAX_COMBINATIONS:
        raise GateError("persistent G1 budget exceeded")
    if torch.cuda.is_initialized():
        raise GateError("CUDA initialized during preflight")
    return {
        "status": "CPU_PREFLIGHT_PASSED_LAUNCH_BLOCKED" if not release["authorized"] else "CPU_PREFLIGHT_PASSED_RELEASE_PRESENT_SOL_REVIEW_REQUIRED",
        "policy": "V3 effective; V2 technical constraints inherited",
        "authorization": {"g1_cpu_preflight": True, "gpu_launch_authorized": False,
                          "g1_release_present_and_bound": release["authorized"],
                          "g1_release_reason": release["reason"],
                          "g2_authorized": False, "scientific_training_authorized": False},
        "human_review": {"reviewed": 0, "total": 26, "status": "UNREVIEWED_0_OF_26_PRESERVED",
                         "medical_truth": "NOT_ASSESSED_BY_EXPERT", "used_as_gate": False},
        "fixed_diagnostic_budget": {"cap": MAX_COMBINATIONS, "manifest_combinations": len(combinations),
            "reserved": len(ledger_state["combinations"]),
            "started": sum(x["status"] == "started" for x in ledger_state["combinations"].values()),
            "completed": sum(x["status"] == "completed" for x in ledger_state["combinations"].values()),
            "failed": sum(x["status"] == "failed" for x in ledger_state["combinations"].values()),
            "autograd_calls_recorded": ledger_state["autograd_calls"],
            "optimizer_step_attempts": ledger_state["optimizer_step_attempts"],
            "counter_persists_across_attempts": True},
        "input_checks": batch_check,
        "source_hashes": base_hashes,
        "execution_binding": binding,
        "execution_binding_sha256": binding_digest,
        "runtime": env,
        "resource_observation": {"gpu0": gpu0_observation, "disk": disk,
                                 "lock": "not acquired; must be rechecked and mutually acquired before any future authorized launch"},
        "model_checkpoint_load_calls": 0, "retinal_model_construction_calls": 0,
        "retinal_model_forward_calls": 0, "autograd_calls": 0, "optimizer_step_calls": 0,
        "optimizer_updates": 0, "dev_images_or_probabilities_read": 0,
        "official_Test_accessed": False,
        "limitations": ["CPU preflight is not G1 GPU release or numerical network-gradient validation",
            "AMP/GPU memory/time and real parameter gradients remain unmeasured",
            "disk observation is not a future launch clearance; check space and lock again"],
    }


def shutil_disk(path: Path) -> dict[str, Any]:
    import shutil
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free,
            "free_percent": round(100 * usage.free / usage.total, 4) if usage.total else None}


def next_attempt_path() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = OUTPUT_ROOT / f"attempt_{index:02d}"
        if not candidate.exists():
            return candidate
        index += 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-preflight", action="store_true",
                        help="run only the CPU startup preflight; never launches a model")
    parser.add_argument("--release", type=Path, default=None,
                        help="optional Sol-issued G1-only release file; this CLI still does not launch GPU work")
    args = parser.parse_args(argv)
    if not args.cpu_preflight:
        parser.error("explicit --cpu-preflight is required; no implicit launch path exists")
    attempt = next_attempt_path()
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        result = run_cpu_preflight(release_path=args.release)
        result["attempt_dir"] = str(attempt)
        atomic_json(attempt / "preflight.json", result)
        atomic_json(attempt / "execution_binding.json", result["execution_binding"])
        print(json.dumps({"status": result["status"], "attempt_dir": str(attempt),
                          "execution_binding_sha256": result["execution_binding_sha256"],
                          "gpu_launch_authorized": False}, ensure_ascii=False))
        return 0
    except Exception as exc:
        atomic_json(attempt / "failure.json", {"status": "CPU_PREFLIGHT_FAILED",
            "error_type": type(exc).__name__, "error": str(exc),
            "gpu_launch_authorized": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())