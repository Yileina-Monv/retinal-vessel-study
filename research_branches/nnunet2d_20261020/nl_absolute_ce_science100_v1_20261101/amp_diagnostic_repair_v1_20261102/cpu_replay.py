"""Read-only Train stream reconstruction. No scientific model is initialized or run."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", nnUNet_compile="false", nnUNet_n_proc_DA="1")
import argparse, json, random, time, traceback
from paths import *
from common import BASELINE, DATA_BASE, read, sha, atomic, log, compare_train
import torch
import numpy as np
from fast_observer import digest
from nnunet2d_scientific_constructor import build_fresh_trainer
from seeded_runtime import install_seeded_worker_adapter
from sidecar import install
from selection import choose_groups
from amp32_runner import bind_train_datasets

def main(seed):
    start = time.monotonic()
    out = OUT / ("replay_seed_" + str(seed))
    out.mkdir(exist_ok=False)
    source = GROUP / "runs" / ("seed_" + str(seed))
    baseline = BASELINE / "runs" / ("seed_" + str(seed))
    saved = read(source / "budget.json")
    parent_hash = sha(source / "budget.json")
    stop_at = saved["epoch"] * 250 + saved["iteration"] + 1
    assert saved["status"] == "STOPPED" and saved["unknown"] == 1
    assert stop_at == saved["attempts"] and saved["skip"] == 0
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    def forbidden(*a, **kw):
        raise RuntimeError("CPU replay prohibits CUDA, model initialization, forward and optimizer")
    torch.cuda.init = forbidden; torch.cuda.set_device = forbidden
    torch.cuda._lazy_init = forbidden
    torch.nn.Module.cuda = forbidden
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    nnUNetTrainer.initialize = forbidden; nnUNetTrainer.run_training = forbidden
    nnUNetTrainer.train_step = forbidden; nnUNetTrainer.validation_step = forbidden
    targets = [json.loads(x) for x in (source / "auxiliary_targets.jsonl").read_text().splitlines()]
    old = [json.loads(x) for x in (source / "attempts.jsonl").read_text().splitlines()]
    history = [json.loads(x) for x in (baseline / "attempts.jsonl").read_text().splitlines()]
    monitors = [json.loads(x) for x in (source / "monitor.jsonl").read_text().splitlines()]
    baseline_monitors = [json.loads(x) for x in (baseline / "monitor.jsonl").read_text().splitlines()]
    assert len(old) == stop_at - 1 and len(targets) == stop_at
    assert len(monitors) == saved["monitor_complete"] == saved["epoch"] * 50
    names = set(read(baseline / "protocol.json")["train_ids"])
    first_nonempty = next(x["attempt"] for x in targets if sum(x["group_counts"]))
    os.environ.update(nnUNet_raw=str(DATA_BASE / "raw"), nnUNet_preprocessed=str(DATA_BASE / "preprocessed"),
                      nnUNet_results=str(out / "native_constructor"), MPLCONFIGDIR=str(out / "matplotlib"))
    random.seed(seed); np.random.seed(seed); torch.random.default_generator.manual_seed(seed)
    install()
    t = build_fresh_trainer(read(DATA_BASE / "preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json"),
                            read(DATA_BASE / "preprocessed/Dataset501_MAPLES/dataset.json"), torch.device("cpu"))
    t.batch_size = 2
    pools = install_seeded_worker_adapter(seed)
    train_set, monitor_set = bind_train_datasets(t, names)
    tr = va = None; matched = monitor_count = 0; selected = []
    try:
        tr, va = t.get_dataloaders()
        for number in range(1, stop_at + 9):
            batch = next(tr)
            epoch = (number - 1) // 250
            h = history[number - 1]
            compare_train(batch, h, number, epoch, .01 * (1 - epoch / 100) ** .9, digest)
            assert set(batch["keys"]) <= names
            if number < stop_at:
                previous = old[number - 1]
                for key in ["attempt", "epoch", "iteration", "ids", "input_sha256", "target_sha256", "lr"]:
                    assert previous[key] == h[key], (number, key, "old NL versus N0")
            meta = batch["nl_candidates"]
            assert len(meta) == 2 and [r["image"] for r in meta] == list(batch["keys"])
            state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
            groups = [choose_groups(r["candidates"], seed, number, crop) for crop, r in enumerate(meta)]
            assert digest(state) == digest((random.getstate(), np.random.get_state(), torch.get_rng_state()))
            if number <= stop_at:
                expected = targets[number - 1]
                assert expected["attempt"] == number and expected["ids"] == list(batch["keys"])
                assert expected["selected_groups_sha256"] == digest(groups)
                assert expected["group_counts"] == [len(g) for g in groups]
                assert expected["counts"] == [r["counts"] for r in meta]
            row = {"logical_input": number, "ids": list(batch["keys"]), "input_sha256": h["input_sha256"],
                   "target_sha256": h["target_sha256"], "groups_sha256": digest(groups),
                   "group_counts": [len(g) for g in groups], "existing_NL_verified": number <= stop_at,
                   "N0_input_match": True}
            log(out / "matched_inputs.jsonl", row); matched += 1
            if number in [first_nonempty, stop_at]:
                path = out / ("diagnostic_nonempty.pt" if number == first_nonempty else "diagnostic_interrupted.pt")
                with path.open("xb") as f:
                    torch.save({"seed": seed, "logical_input": number, "data": batch["data"],
                                "target": batch["target"], "groups": groups, "ids": list(batch["keys"]),
                                "provenance": row}, f)
                selected.append({"path": str(path), "sha256": sha(path), **row})
            if number % 250 == 0 and number < stop_at:
                for _ in range(50):
                    monitor = next(va)
                    m = monitors[monitor_count]; bm = baseline_monitors[monitor_count]
                    assert list(monitor["keys"]) == m["ids"] == bm["ids"]
                    assert set(monitor["keys"]) <= names and "nl_candidates" not in monitor
                    log(out / "matched_monitors.jsonl", {"number": monitor_count + 1, "ids": list(monitor["keys"]),
                        "reconstructed_input_sha256": digest(monitor["data"]),
                        "reconstructed_target_sha256": digest(monitor["target"]),
                        "historical_tensor_hash_available": False})
                    monitor_count += 1
            if number % 100 == 0 or number == stop_at + 8:
                value = {"state": "RUNNING", "seed": seed, "matched_train": matched, "target_train": stop_at + 8,
                         "matched_monitor_IDs": monitor_count, "seconds": time.monotonic() - start,
                         "GPU_calls": 0, "model_forwards": 0, "optimizer_updates": 0}
                atomic(out / "progress.json", value)
                print("CPU_REPLAY", seed, matched, stop_at + 8, round(value["seconds"], 2), flush=True)
        assert monitor_count == saved["monitor_complete"] and sha(source / "budget.json") == parent_hash
        assert pools == [{"pool": "train", "worker_count": 1, "seeds": [seed]},
                         {"pool": "internal_train_monitor", "worker_count": 1, "seeds": [seed + 1]}]
        assert not torch.cuda.is_initialized() and t.network is None and t.optimizer is None
        atomic(out / "RESULT.json", {"status": "PASS", "seed": seed, "matched_train": matched,
            "historical_NL_target_matches": stop_at, "matched_monitor_IDs": monitor_count,
            "interrupted_input": stop_at, "next_inputs_verified": 8, "selected_diagnostics": selected,
            "monitor_tensor_equivalence_to_history_proved": False, "live_prefetch_queue_serialized": False,
            "parent_budget_sha256": parent_hash, "seconds": time.monotonic() - start,
            "GPU_calls": 0, "model_forwards": 0, "optimizer_updates": 0}, True)
    finally:
        for loader in [tr, va]:
            if hasattr(loader, "_finish"):
                loader._finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, choices=[2027, 2028], required=True)
    seed = parser.parse_args().seed
    try:
        main(seed)
    except BaseException as exc:
        out = OUT / ("replay_seed_" + str(seed))
        if out.exists():
            atomic(out / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                                         "GPU_calls": 0, "optimizer_updates": 0}, True)
        raise
