"""CPU-only tests for the G1 entry; never construct retinal models or touch CUDA."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from retinal_m0.maples_g1_entry import (
    BudgetLedger, GateError, OptimizerStepProhibited, fixed_combinations,
    run_cpu_preflight, tensor_state_digest, _release_check, execute_g1_only,
    _verify_sources, _verify_batches, CPU_EVIDENCE, BATCH_EVIDENCE, SOURCE_EVIDENCE, canonical_sha256,
)


def manifest():
    steps = (115, 1442, 1660, 1, 4, 3, 66, 400)
    return [{"step_original_T0_one_based": step, "draws": [
        {"draw": (step - 1) * 4 + slot, "input_sha256_original_T0": "a" * 64,
         "target_crop_uint8_C_order_sha256": {"B1": "b" * 64, "B2": "c" * 64}}
        for slot in range(4)]} for step in steps]


class G1EntryTests(unittest.TestCase):
    def test_fixed_manifest_has_exact_32_unique_combinations(self):
        rows = fixed_combinations(manifest())
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({row["combo_id"] for row in rows}), 32)

    def test_fixed_manifest_rejects_wrong_count_order_or_batch_size(self):
        with self.assertRaises(GateError):
            fixed_combinations(manifest()[:-1])
        wrong = manifest()
        wrong[0]["step_original_T0_one_based"] = 999
        with self.assertRaises(GateError):
            fixed_combinations(wrong)
        wrong = manifest()
        wrong[0]["draws"].pop()
        with self.assertRaises(GateError):
            fixed_combinations(wrong)

    def test_ledger_persists_failure_and_rejects_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            first = BudgetLedger(path)
            first.reserve("c1", {"step": 1})
            first.mark_started("c1")
            first.mark_failed("c1", "synthetic failure", autograd_calls=1)
            second = BudgetLedger(path)
            state = second.snapshot()
            self.assertEqual(state["combinations"]["c1"]["status"], "failed")
            self.assertEqual(state["autograd_calls"], 1)
            with self.assertRaisesRegex(GateError, "duplicate"):
                second.reserve("c1", {"step": 1})

    def test_crash_recovery_consumes_reservation_and_does_not_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            first = BudgetLedger(path)
            first.reserve("c1", {"step": 1})
            first.mark_started("c1")
            # Constructing a new ledger models a process restart.
            second = BudgetLedger(path)
            state = second.snapshot()
            self.assertEqual(state["combinations"]["c1"]["status"], "failed")
            self.assertEqual(state["combinations"]["c1"]["failure"],
                             "recovered_incomplete_after_process_restart")
            self.assertEqual(state["recovery_events"], 1)
            with self.assertRaisesRegex(GateError, "duplicate"):
                second.reserve("c1", {"step": 1})

    def test_budget_is_cumulative_and_cap_is_exactly_32(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BudgetLedger(Path(tmp) / "ledger.json")
            for index in range(32):
                ledger.reserve(f"c{index}", {"index": index})
                ledger.mark_started(f"c{index}")
                ledger.mark_completed(f"c{index}", autograd_calls=1)
            with self.assertRaisesRegex(GateError, "exhausted"):
                ledger.reserve("c32", {})
            state = ledger.snapshot()
            self.assertEqual(len(state["combinations"]), 32)
            self.assertEqual(state["autograd_calls"], 32)

    def test_optimizer_step_is_blocked_and_attempt_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BudgetLedger(Path(tmp) / "ledger.json")
            ledger.reserve("c1", {})
            optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=1)
            with OptimizerStepProhibited(optimizer, ledger, "c1"):
                with self.assertRaisesRegex(GateError, "prohibited"):
                    optimizer.step()
            state = ledger.snapshot()
            self.assertEqual(state["optimizer_step_attempts"], 1)

    def test_model_state_digest_detects_parameter_or_buffer_change(self):
        module = torch.nn.BatchNorm1d(2)
        before = tensor_state_digest(module)
        with torch.no_grad():
            module.running_mean.add_(1)
        self.assertNotEqual(before, tensor_state_digest(module))

    def test_release_must_bind_exact_phase_authority_hash_and_gpu0(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "release.json"
            payload = {"phase": "G1_ONLY", "authorized": True,
                       "gpu_launch_authorized": True,
                       "execution_binding_sha256": "abc", "gpu0_uuid": "GPU-0"}
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(_release_check(path, "abc", "GPU-0")["authorized"])
            self.assertFalse(_release_check(path, "drift", "GPU-0")["authorized"])
            payload["phase"] = "G2"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(_release_check(path, "abc", "GPU-0")["authorized"])
            payload["phase"] = "G1_ONLY"
            payload["gpu0_uuid"] = "GPU-WRONG"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(_release_check(path, "abc", "GPU-0")["authorized"])
        self.assertFalse(_release_check(None, "abc", "GPU-0")["authorized"])

    def test_full_cpu_preflight_passes_but_never_authorizes_launch(self):
        result = run_cpu_preflight()
        self.assertEqual(result["status"], "CPU_PREFLIGHT_PASSED_LAUNCH_BLOCKED")
        self.assertFalse(result["authorization"]["gpu_launch_authorized"])
        self.assertFalse(result["authorization"]["g1_release_present_and_bound"])
        self.assertEqual(result["human_review"]["status"], "UNREVIEWED_0_OF_26_PRESERVED")
        self.assertEqual(result["fixed_diagnostic_budget"]["cap"], 32)
        self.assertEqual(result["fixed_diagnostic_budget"]["manifest_combinations"], 32)
        self.assertEqual(result["input_checks"]["draw_count"], 32)
        self.assertTrue(result["input_checks"]["all_input_fingerprints_match_prior"])
        self.assertEqual(result["retinal_model_forward_calls"], 0)
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertFalse(result["runtime"]["cuda_initialized"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_controller_runs_only_32_single_forward_diagnostics_on_cpu_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "release.json"
            release.write_text(json.dumps({"phase": "G1_ONLY", "authorized": True,
                "gpu_launch_authorized": True, "execution_binding_sha256": "abc",
                "gpu0_uuid": "GPU-0"}), encoding="utf-8")
            model = torch.nn.Linear(2, 1, bias=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
            before = tensor_state_digest(model)
            rng_before = torch.get_rng_state().clone()
            calls = {"callback": 0}
            x = torch.tensor([[1.0, 2.0]])
            def callback(combo, batch):
                calls["callback"] += 1
                torch.rand(1)  # prove the controller restores the same RNG start per combo
                logits = model(x)
                base = logits.square().mean()
                aux = logits.abs().mean()
                combined = base + 0.1 * aux
                torch.autograd.grad(base, tuple(model.parameters()), retain_graph=True)
                torch.autograd.grad(0.1 * aux, tuple(model.parameters()), retain_graph=True)
                torch.autograd.grad(combined, tuple(model.parameters()))
                return {"base_bce": float(base.detach()), "base_dice": 0.0,
                    "aux_unweighted": float(aux.detach()), "aux_weighted": float((.1 * aux).detach()),
                    "grad_l2_base": 1.0, "grad_l2_aux_weighted": .1,
                    "grad_l2_combined": 1.1, "cosine_base_aux": 1.0,
                    "clip_norm_combined": 1.1, "clip_fraction_combined": 1.0,
                    "per_layer_grad_norms": {"linear": 1.1}, "finite": True, "loss_scaler": 1.0,
                    "input_fingerprint_sha256": canonical_sha256([
                        {"draw": int(d["draw"]), "sha256": d["input_sha256_original_T0"]}
                        for d in batch["draws"]]),
                    "target_fingerprint_sha256": canonical_sha256([
                        {"draw": int(d["draw"]), "sha256": d["target_crop_uint8_C_order_sha256"][combo["method"]]}
                        for d in batch["draws"]]),
                    "precision": combo["precision"], "zero_lambda_value_exact": True,
                    "zero_lambda_gradient_exact": True, "empty_aux_value_zero": True,
                    "empty_aux_gradient_zero": True, "aux_grad_outside_target_l2": 0.0,
                    "analytic_fp32_max_abs_error": 0.0 if combo["precision"] == "FP32" else None,
                    "analytic_fp32_max_rel_error": 0.0 if combo["precision"] == "FP32" else None,
                    "unscaled_gradients": True, "gradient_scaling_removed": True}
            report = execute_g1_only(release_path=release, execution_binding_sha256="abc",
                gpu0_uuid="GPU-0", batches=manifest(), model=model, optimizer=optimizer,
                ledger_path=root / "ledger.json", diagnostic_callback=callback,
                cpu_test_substitute=True)
            self.assertEqual(report["combination_count"], 32)
            self.assertEqual(report["autograd_calls"], 96)
            self.assertEqual(calls["callback"], 32)
            self.assertEqual(before, tensor_state_digest(model))
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            self.assertEqual(report["rng_sha256_initial"], report["rng_sha256_final"])
            ledger = BudgetLedger(root / "ledger.json").snapshot()
            self.assertEqual(len(ledger["combinations"]), 32)
            self.assertEqual(ledger["autograd_calls"], 96)
            self.assertEqual(ledger["optimizer_step_attempts"], 0)

    def test_production_release_must_be_status_bound_and_cannot_use_arbitrary_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "release.json"
            release.write_text(json.dumps({"phase": "G1_ONLY", "authorized": True,
                "gpu_launch_authorized": True, "execution_binding_sha256": "abc",
                "gpu0_uuid": "GPU-0"}), encoding="utf-8")
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
            called = []
            with self.assertRaisesRegex(GateError, "fixed STATUS-bound release path"):
                execute_g1_only(release_path=release, execution_binding_sha256="abc",
                    gpu0_uuid="GPU-0", batches=manifest(), model=model, optimizer=optimizer,
                    ledger_path=root / "ledger.json",
                    diagnostic_callback=lambda *_: called.append(True),
                    cpu_test_substitute=False)
            self.assertEqual(called, [])
            self.assertFalse((root / "ledger.json").exists())

    def test_controller_consumes_budget_when_numeric_contract_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "release.json"
            release.write_text(json.dumps({"phase": "G1_ONLY", "authorized": True,
                "gpu_launch_authorized": True, "execution_binding_sha256": "abc",
                "gpu0_uuid": "GPU-0"}), encoding="utf-8")
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
            batch = manifest()[0]
            def callback(combo, _batch):
                value = model(torch.ones((1, 2))).sum()
                torch.autograd.grad(value, tuple(model.parameters()))
                return {"base_bce": 1.0, "base_dice": 0.0, "aux_unweighted": .2,
                    "aux_weighted": .02, "grad_l2_base": 1.0,
                    "grad_l2_aux_weighted": .2, "grad_l2_combined": 1.1,
                    "cosine_base_aux": 1.0, "clip_norm_combined": 1.1,
                    "clip_fraction_combined": 1.0, "per_layer_grad_norms": {"linear": 1.1},
                    "finite": True, "loss_scaler": 1024.0,
                    "input_fingerprint_sha256": canonical_sha256([
                        {"draw": int(d["draw"]), "sha256": d["input_sha256_original_T0"]}
                        for d in batch["draws"]]),
                    "target_fingerprint_sha256": canonical_sha256([
                        {"draw": int(d["draw"]), "sha256": d["target_crop_uint8_C_order_sha256"][combo["method"]]}
                        for d in batch["draws"]]),
                    "precision": combo["precision"], "zero_lambda_value_exact": True,
                    "zero_lambda_gradient_exact": False, "empty_aux_value_zero": True,
                    "empty_aux_gradient_zero": True, "aux_grad_outside_target_l2": 0.0,
                    "analytic_fp32_max_abs_error": 0.0, "analytic_fp32_max_rel_error": 0.0,
                    "unscaled_gradients": True, "gradient_scaling_removed": True}
            with self.assertRaisesRegex(GateError, "numerical contract failed"):
                execute_g1_only(release_path=release, execution_binding_sha256="abc",
                    gpu0_uuid="GPU-0", batches=[batch] + manifest()[1:], model=model,
                    optimizer=optimizer, ledger_path=root / "ledger.json",
                    diagnostic_callback=callback, cpu_test_substitute=True)
            state = BudgetLedger(root / "ledger.json").snapshot()
            self.assertEqual(len(state["combinations"]), 1)
            self.assertEqual(next(iter(state["combinations"].values()))["status"], "failed")
            self.assertEqual(state["combinations"][next(iter(state["combinations"]))]["autograd_calls"], 1)

    def test_controller_rejects_wrong_release_before_forward_or_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "release.json"
            release.write_text(json.dumps({"phase": "G2", "authorized": True,
                "gpu_launch_authorized": True, "execution_binding_sha256": "abc",
                "gpu0_uuid": "GPU-0"}), encoding="utf-8")
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
            called = []
            with self.assertRaisesRegex(GateError, "release rejected"):
                execute_g1_only(release_path=release, execution_binding_sha256="abc",
                    gpu0_uuid="GPU-0", batches=manifest(), model=model, optimizer=optimizer,
                    ledger_path=root / "ledger.json",
                    diagnostic_callback=lambda *_: called.append(True),
                    cpu_test_substitute=True)
            self.assertEqual(called, [])
            self.assertFalse((root / "ledger.json").exists())

    def test_source_hash_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "source.bin"
            sample.write_bytes(b"changed")
            manifest_path = root / "binding.json"
            rows = {str(root / f"source-{i}.bin"): "0" * 64 for i in range(300)}
            # The first real path is intentionally bound to the wrong digest.
            rows[str(sample)] = "0" * 64
            manifest_path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(GateError, "source drift"):
                _verify_sources(manifest_path)

    def test_input_target_and_nontrain_changes_fail_closed(self):
        import csv
        batches = json.loads(BATCH_EVIDENCE.read_text(encoding="utf-8"))["batches"]
        actual = json.loads((CPU_EVIDENCE / "actual32_draws.json").read_text(encoding="utf-8"))
        with (SOURCE_EVIDENCE / "images_110.csv").open(encoding="utf-8", newline="") as stream:
            train_names = {row["filename"] for row in csv.DictReader(stream)}
        artifacts = json.loads((SOURCE_EVIDENCE / "artifact_sha256.json").read_text(encoding="utf-8"))
        changed = json.loads(json.dumps(batches))
        changed[0]["draws"][0]["input_sha256_original_T0"] = "f" * 64
        with self.assertRaisesRegex(GateError, "fingerprint"):
            _verify_batches(changed, actual, train_names, artifacts)
        changed = json.loads(json.dumps(batches))
        changed[0]["draws"][0]["filename"] = "NOT_TRAIN"
        with self.assertRaisesRegex(GateError, "outside Train"):
            _verify_batches(changed, actual, train_names, artifacts)
        changed = json.loads(json.dumps(batches))
        changed[0]["draws"][0]["target_crop_uint8_C_order_sha256"]["B1"] = "0" * 64
        with self.assertRaisesRegex(GateError, "fingerprint mismatch"):
            _verify_batches(changed, actual, train_names, artifacts)

    def test_module_import_and_preflight_do_not_initialize_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())
        run_cpu_preflight()
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()