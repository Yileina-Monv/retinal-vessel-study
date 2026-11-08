# B1/B2新增误报分析

这是已有实验/诊断的补充归档，不是本次新增实验。源阶段：`experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022/analysis/fp_error_analysis_v1/`。

[28图公开误差图册](public_atlas/README.md) · [下载后浏览HTML](public_atlas/index.html)

## 文件

- [source_archive/fp_analysis.py.txt](source_archive/fp_analysis.py.txt)
- [analysis_spec.json](analysis_spec.json)
- [run_manifest.json](run_manifest.json)
- [membership.csv](membership.csv)
- [per_image_metrics.csv](per_image_metrics.csv)
- [per_image_sets.csv](per_image_sets.csv)
- [regions.csv](regions.csv)
- [distance_density.csv](distance_density.csv)
- [summary.json](summary.json)
- [case_catalog.json](case_catalog.json)
- [figures_manifest.json](figures_manifest.json)
- [input_manifest.json](input_manifest.json)
- [validation.json](validation.json)
- [source_archive/fp_validate.py.txt](source_archive/fp_validate.py.txt)
- [A0_attachment_supplement.csv](A0_attachment_supplement.csv)
- [A0_attachment_supplement.json](A0_attachment_supplement.json)
- [independent_validation.json](independent_validation.json)
- [source_archive/isolated_cases.py.txt](source_archive/isolated_cases.py.txt)
- [isolated_case_selection.json](isolated_case_selection.json)
- [additional_figures_manifest.json](additional_figures_manifest.json)
- [visual_review.json](visual_review.json)
- [visual_review_receipt.json](visual_review_receipt.json)
- [case_review_status.csv](case_review_status.csv)
- [report.md](report.md)
- [native_size_summary.json](native_size_summary.json)
- [final_acceptance.json](final_acceptance.json)
- [artifact_manifest.json](artifact_manifest.json)

## 阅读与复核边界

数值表保留原分母、NA和全部病例。报告为历史文本，不替代最新结论；AI视觉观察不等于医学人工核验。`source_archive/*.py.txt`是供审查的历史CPU脚本，路径已泛化，不自动运行；依赖A7私有输入的部分不声称公开克隆即可复现。完整原件SHA与公开副本SHA分别见上级source_manifest.json。
