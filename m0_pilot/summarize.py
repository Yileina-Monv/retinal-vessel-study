"""Render the completed pilot's local evidence without running a model."""
import json
import os
from pathlib import Path
import statistics

import numpy as np

from retinal_data.fives import trial_dataset
from retinal_data.prepare import ROOT, sha256

RUN = ROOT / "runs/m0_pilot_2026-09-04_01"
OUT = ROOT / "m0_pilot"
FIGURES = RUN / "figures"
FIGURES.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib-m0"))


def read(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def read_lines(name):
    return [json.loads(line) for line in (RUN / name).read_text(encoding="utf-8").splitlines()]


def main():
    train, probes = read_lines("train.jsonl"), read_lines("fixed_probes.jsonl")
    config = read("config.json")
    initial, final = read("validation_initial.json"), read("validation_final.json")
    resume = read("resume_check.json")
    segments = [read("segment_initial.json"), read("segment_resume.json")]
    assert segments[0]["status"] == "paused_at_checkpoint" and segments[1]["status"] == "completed"
    assert resume["status"] == "passed"
    assert [row["step"] for row in train] == list(range(1, config["total_optimizer_steps"] + 1))
    assert len({item["key"] for row in train for item in row["samples"]}) == 32
    stable = [row for row in train if row["step"] > 16]
    resources = {
        "hardware": read("hardware.json"), "microbatch": config["microbatch"], "accumulation": config["accumulation"],
        "optimizer_steps": len(train), "additional_reference_updates": config["resume_comparison_steps"],
        "primary_training_samples": sum(len(row["samples"]) for row in train),
        "training_loop_total_seconds": sum(row["step_wall_seconds"] for row in train),
        "data_wait_total_seconds": sum(row["data_wait_seconds"] for row in train),
        "gpu_regions_total_seconds": sum(row["gpu_region_seconds"] for row in train),
        "step_wall_median_after_16_warmup_seconds": statistics.median(row["step_wall_seconds"] for row in stable),
        "gpu_region_median_after_16_warmup_seconds": statistics.median(row["gpu_region_seconds"] for row in stable),
        "step_wall_p95_after_16_warmup_seconds": float(np.quantile([r["step_wall_seconds"] for r in stable], .95)),
        "measured_process_segments_total_seconds": sum(row["elapsed_seconds"] for row in segments),
        "peak_allocated_gib": max(row["peak_allocated_gib"] for row in segments),
        "peak_reserved_gib": max(row["peak_reserved_gib"] for row in segments),
        "sampled_process_tree_rss_max_gib": max(row["sampled_process_tree_rss_max_gib"] for row in segments),
        "final_eight_images_including_loading_seconds": final["elapsed_seconds_including_loading"],
        "final_tiled_inference_seconds": sum(row["inference_seconds"] for row in final["rows"]),
        "final_tiles": sum(row["tiles"] for row in final["rows"]),
        "timing_notes": ["Step timings include data fetch and GPU synchronization.",
                         "GPU regions are CUDA-event intervals, not an energy or utilization measurement.",
                         "Process timers exclude interpreter/model/data initialization before the timer and inter-process gaps.",
                         "RSS is sampled and sums parent/children, which can double-count shared pages.",
                         "Short-pilot timings do not establish convergence or full-study runtime."]}
    summary = {
        "status": "completed_pilot", "scope": config["scope"], "pixel_domain": config["pixel_domain"],
        "initial_fixed_train_loss": probes[0]["loss"], "final_fixed_train_loss": probes[-1]["loss"],
        "initial_fixed_train_dice": probes[0]["dice"], "final_fixed_train_dice": probes[-1]["dice"],
        "initial_development_macro_dice": initial["macro_dice"], "final_development_macro_dice": final["macro_dice"],
        "final_development_macro_precision": final["macro_precision"], "final_development_macro_recall": final["macro_recall"],
        "gradient_clip_steps": sum(r["gradient_norm_before_clip"] > config["gradient_clip_norm"] for r in train),
        "empty_training_crops": sum(s["empty_crop"] for r in train for s in r["samples"]),
        "resume_check": resume, "successful_steps_contiguous": True,
        "source_test_images_used": 0, "external_images_used": 0,
        "last_checkpoint_sha256": sha256(RUN / "last.pt"), "resources": resources}
    for name, value in (("summary.json", summary), ("resource_profile.json", resources)):
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    steps = [r["step"] for r in train]
    axes[0].plot(steps, [r["loss"] for r in train], alpha=.25, color="#276b9a", label="Training patch loss")
    axes[0].plot(steps[15:], np.convolve([r["loss"] for r in train], np.ones(16)/16, mode="valid"), color="#276b9a", label="16-step average")
    axes[0].plot([r["step"] for r in probes], [r["loss"] for r in probes], "o-", color="#b26b20", label="4 fixed training patches")
    axes[0].set_ylabel("BCE + soft Dice loss")
    axes[1].plot([r["step"] for r in probes], [r["dice"] for r in probes], "o-", label="Fixed training patch Dice")
    axes[1].scatter([0, 256], [initial["macro_dice"], final["macro_dice"]], marker="D", color="#b26b20", label="8 development full images")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Dice at fixed threshold 0.5")
    for ax in axes:
        ax.axvline(128, linestyle=":", color="gray", label="Checkpoint / resume")
        ax.set_xlabel("Successful optimizer updates")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("M0 engineering pilot: source train split only, all pixels, no FOV")
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curves.png")
    plt.close(fig)

    dataset = trial_dataset("overfit")
    before = np.load(RUN / "fixed_before.npz")["probabilities"]
    after = np.load(RUN / "fixed_after.npz")["probabilities"]
    fig, axes = plt.subplots(4, 5, figsize=(15, 12), dpi=140)
    for i in range(4):
        item = dataset[(0, i * dataset.repeats)]
        axes[i, 0].imshow(item["image"].numpy().transpose(1, 2, 0))
        axes[i, 0].set_title(item["key"], fontsize=9)
        for j, (array, title) in enumerate(((item["mask"][0], "Annotation"), (before[i] >= .5, "Before: binary"),
                                           (after[i], "After: probability"), (after[i] >= .5, "After: binary")), 1):
            axes[i, j].imshow(array, cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_title(title, fontsize=9)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.suptitle("Four fixed TRAINING patches: diagnostic examples, not test performance")
    fig.tight_layout()
    fig.savefig(FIGURES / "fixed_training_examples.png")
    plt.close(fig)

    validation = trial_dataset("smoke_val")
    selected = [next(i for i, row in enumerate(validation.rows) if row["disease"] == disease) for disease in "ADGN"]
    fig, axes = plt.subplots(4, 4, figsize=(14, 14), dpi=140)
    for i, index in enumerate(selected):
        item = validation[index]
        stem = item["key"].replace("/", "__").removesuffix(".png")
        probability = np.load(RUN / "predictions" / f"{stem}.npy")
        image = item["image"].numpy().transpose(1, 2, 0)[::4, ::4].copy()
        truth = item["mask"][0].numpy()[::4, ::4] > .5
        pred = probability[::4, ::4] >= .5
        errors = image.copy()
        for mask, color in ((pred & truth, [0, 1, 1]), (pred & ~truth, [1, 0, 1]), (~pred & truth, [1, 1, 0])):
            errors[mask] = .4 * image[mask] + .6 * np.array(color)
        for j, (array, title) in enumerate(((image, item["key"]), (truth, "Annotation"),
                                           (probability[::4, ::4], "Final probability"), (errors, "TP cyan | FP magenta | FN yellow"))):
            axes[i, j].imshow(array, cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_title(title, fontsize=8)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.suptitle("Development examples, one per disease chosen by fixed manifest order; not final test data")
    fig.tight_layout()
    fig.savefig(FIGURES / "development_examples.png")
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
