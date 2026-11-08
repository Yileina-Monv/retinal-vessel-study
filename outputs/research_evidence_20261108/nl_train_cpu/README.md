# NL Train-only CPU检查

这是已有实验/诊断的补充归档，不是本次新增实验。源阶段：`experiments/nnunet2d_branch_20261020/deep_supervision_ablation_20261028/science100_v1/analysis/nl_absolute_ce_cpu_v2_alltargets_20261031/results/`。

**历史阶段提醒：CPU报告中的未放行是当时状态；NL后续训练已经完成，终点未达到原预设成功标准，见PR #19及本包终点审查。**

## 文件

- [cpu_implementation_tests.json](cpu_implementation_tests.json)
- [seed_2027_summary.json](seed_2027_summary.json)
- [cpu_selection_tests.json](cpu_selection_tests.json)
- [mean_normalization_cost_check.json](mean_normalization_cost_check.json)
- [seed_2028_summary.json](seed_2028_summary.json)
- [atlas_selection.json](atlas_selection.json)
- [replay_crops.csv](replay_crops.csv)
- [replay_images_110.csv](replay_images_110.csv)
- [replay_targets_351.csv](replay_targets_351.csv)
- [independent_review.json](independent_review.json)
- [cpu_summary.json](cpu_summary.json)
- [atlas_manifest.json](atlas_manifest.json)
- [CPU_一页结论.md](CPU_一页结论.md)

## 阅读与复核边界

数值表保留原分母、NA和全部病例。报告为历史文本，不替代最新结论；AI视觉观察不等于医学人工核验。`source_archive/*.py.txt`是供审查的历史CPU脚本，路径已泛化，不自动运行；依赖A7私有输入的部分不声称公开克隆即可复现。完整原件SHA与公开副本SHA分别见上级source_manifest.json。
