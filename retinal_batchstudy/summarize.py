"""Paired-seed inference; never treat image x seed rows as independent patients."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import stats
from retinal_data.prepare import ROOT


def exact_signflip_p(differences):
    delta = np.asarray(differences, dtype=float)
    observed = abs(delta.mean())
    signs = np.array(list(itertools.product((-1, 1), repeat=len(delta))))
    permuted = np.abs((signs * delta).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-14))


def holm(pvalues):
    order = np.argsort(pvalues)
    result = np.empty(len(pvalues))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(pvalues)-rank)*pvalues[index]))
        result[index] = running
    return result.tolist()


def mean_interval(values, alpha):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return [None, None]
    half = float(stats.t.ppf(1-alpha/2, len(values)-1)*stats.sem(values))
    return [mean-half, mean+half]


def summarize(root, destination):
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    results = {}
    for path in root.glob("seed_*/*/result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        results[(row["seed"], row["arm"]["id"])] = row
    seeds = config["seeds"]
    baseline = config["baseline"]
    complete = len(results) == len(seeds)*len(config["arms"])
    rows, contrasts = [], []
    for arm in config["arms"]:
        available = [results[(seed, arm["id"])] for seed in seeds if (seed, arm["id"]) in results]
        if not available:
            continue
        dice = [r["final"]["macro_dice"] for r in available]
        rows.append({"arm": arm["id"], "n_seeds": len(available), "mean_dice": float(np.mean(dice)),
                     "sd_across_seeds": float(np.std(dice, ddof=1)) if len(dice)>1 else None,
                     "mean_precision": float(np.mean([r["final"]["macro_precision"] for r in available])),
                     "mean_recall": float(np.mean([r["final"]["macro_recall"] for r in available])),
                     "mean_draws_per_second": float(np.mean([r["draws_per_training_second"] for r in available])),
                     "peak_allocated_gib": max(s["peak_allocated_gib"] for r in available for s in r["segments"]),
                     "seed_dice": dict(zip([str(r["seed"]) for r in available], dice))})
        if arm["id"] == baseline:
            continue
        matched = [seed for seed in seeds if (seed, arm["id"]) in results and (seed, baseline) in results]
        for seed in matched:
            if results[(seed, arm["id"])]["initial_digest"] != results[(seed, baseline)]["initial_digest"]:
                raise ValueError("Unmatched initial weights")
            if results[(seed, arm["id"])]["sample_sequence_digest"] != results[(seed, baseline)]["sample_sequence_digest"]:
                raise ValueError("Unmatched actual input tensors/order")
        delta = [results[(s,arm["id"])]["final"]["macro_dice"]-results[(s,baseline)]["final"]["macro_dice"] for s in matched]
        if not delta:
            continue
        item = {"arm": arm["id"], "baseline": baseline, "matched_seeds": matched,
                "seed_differences": delta, "mean_difference": float(np.mean(delta)),
                "ci95_t": mean_interval(delta, .05), "simultaneous_ci_t": mean_interval(delta, .05/(len(config['arms'])-1))}
        if complete:
            item["exact_signflip_p"] = exact_signflip_p(delta)
        contrasts.append(item)
    if complete:
        corrected = holm([r["exact_signflip_p"] for r in contrasts])
        margin = config["practical_margin_absolute_dice"]
        for item, pvalue in zip(contrasts, corrected):
            item["holm_p"] = pvalue
            item["statistically_detected"] = pvalue < config["alpha_familywise"]
            low, high = item["simultaneous_ci_t"]
            item["practical_conclusion"] = ("harm_exceeds_margin" if high < -margin else
                "benefit_exceeds_margin" if low > margin else
                "within_margin_conditional_on_fixed_development_set" if low > -margin and high < margin else
                "inconclusive")
    report = {"status": "complete_single_gpu" if complete else "partial_no_final_significance_claim",
              "completed_runs": len(results), "expected_runs": len(seeds)*len(config['arms']),
              "hardware": json.loads((root/'hardware.json').read_text()), "arms": rows, "contrasts": contrasts,
              "margin": config["practical_margin_absolute_dice"],
              "limitations": ["32 training images, 8 selected development images", "fixed 2048 draws, not established convergence",
                              "paired-seed inference conditional on fixed development images", "not patient-independent generalization",
                              "single GPU does not establish cross-GPU reproducibility", "timings include input hashing/audit overhead"]}
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ["# Batch 敏感性实测结果", "", f"状态：{report['status']}；已完成 {len(results)}/{report['expected_runs']} 次运行。", "",
             "仅当前 GPU；32 张训练、8 张开发图，每组 2048 个裁块。不是收敛/正式泛化结论。", "",
             "| 组 | 种子数 | Dice 均值 | Precision | Recall | 裁块/秒（含审计） | 峰值已分配 GiB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['n_seeds']} | {r['mean_dice']:.5f} | {r['mean_precision']:.5f} | {r['mean_recall']:.5f} | {r['mean_draws_per_second']:.2f} | {r['peak_allocated_gib']:.2f} |")
    lines += ["", "差值以候选组减参考 m2_a2_e4；百分点为绝对 Dice 差×100。", "",
              "| 组 | 平均差/百分点 | 家族同时区间/百分点 | Holm p | 工程容差结论 |",
              "|---|---:|---|---:|---|"]
    translations = {"harm_exceeds_margin":"损害超过 1 个百分点", "benefit_exceeds_margin":"改善超过 1 个百分点",
                    "within_margin_conditional_on_fixed_development_set":"在本开发集和预算下落在 ±1 个百分点内", "inconclusive":"证据不足"}
    for r in contrasts:
        ci = r['simultaneous_ci_t']
        ci_text = f"[{ci[0]*100:.3f}, {ci[1]*100:.3f}]" if ci[0] is not None else "待重复"
        p = f"{r['holm_p']:.6f}" if 'holm_p' in r else "未裁决"
        lines.append(f"| {r['arm']} | {r['mean_difference']*100:+.3f} | {ci_text} | {p} | {translations.get(r.get('practical_conclusion'), '矩阵未完成')} |")
    lines += ["", "p 值来自 8 个匹配种子的精确符号翻转检验，五比较 Holm 校正；区间为 Bonferroni 校正的配对 t 区间（各 99%），依赖配对差分布假设。",
              "不将不显著解释为无影响；不把本轮容差内结果外推到最终 FOV/骨架损失、完整数据或其他 GPU。", "",
              "耗时包含输入张量摘要计算；显存不包含其他进程及全部驱动开销。GPU 和 batch 速度结论均只覆盖实际执行条件。"]
    (destination/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({"status":report['status'], "completed":len(results), "destination":str(destination)}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-label', required=True)
    args = parser.parse_args()
    if not args.device_label.replace('_','').replace('-','').isalnum():
        raise ValueError('Invalid device label')
    summarize(ROOT/'runs/batch_study_v1'/args.device_label, ROOT/'batch_study/results'/args.device_label)


if __name__ == '__main__':
    main()
