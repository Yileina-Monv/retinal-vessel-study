# Plan B错误分析

这是已有实验/诊断的补充归档，不是本次新增实验。源阶段：`experiments/plan_b_error_analysis_20261019/`。

[77×11完整命中矩阵](all77_conditions.png) · [矩阵数表](all77_conditions.csv)

## 文件

- [execution_binding.json](execution_binding.json)
- [leader_summary.md](leader_summary.md)
- [completion_verification.json](completion_verification.json)
- [validation/verification.json](validation/verification.json)
- [analysis/image_manifest.csv](analysis/image_manifest.csv)
- [analysis/point_catalog.csv](analysis/point_catalog.csv)
- [analysis/per_point_observations.csv](analysis/per_point_observations.csv)
- [analysis/per_condition_image_metrics.csv](analysis/per_condition_image_metrics.csv)
- [analysis/fp_by_region.csv](analysis/fp_by_region.csv)
- [analysis/original_metric_reconciliation.csv](analysis/original_metric_reconciliation.csv)
- [analysis/point_recurrence.csv](analysis/point_recurrence.csv)
- [analysis/formal_paired_point_changes.csv](analysis/formal_paired_point_changes.csv)
- [analysis/class_split_paired_point_changes.csv](analysis/class_split_paired_point_changes.csv)
- [analysis/budget_point_changes.csv](analysis/budget_point_changes.csv)
- [analysis/visual_review_log.csv](analysis/visual_review_log.csv)
- [analysis/summary.json](analysis/summary.json)
- [analysis/case_selection.csv](analysis/case_selection.csv)
- [analysis/formal_paired_metric_changes.csv](analysis/formal_paired_metric_changes.csv)
- [analysis/formal_paired_metric_summary.csv](analysis/formal_paired_metric_summary.csv)
- [analysis/budget_paired_metric_changes.csv](analysis/budget_paired_metric_changes.csv)

## 阅读与复核边界

数值表保留原分母、NA和全部病例。报告为历史文本，不替代最新结论；AI视觉观察不等于医学人工核验。`source_archive/*.py.txt`是供审查的历史CPU脚本，路径已泛化，不自动运行；依赖A7私有输入的部分不声称公开克隆即可复现。完整原件SHA与公开副本SHA分别见上级source_manifest.json。
