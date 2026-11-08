# A0/D/B1/B2概率与阈值诊断

这是已有实验/诊断的补充归档，不是本次新增实验。源阶段：`experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/focal_dice_scientific_2000_v1_20261024/runs/seed_2026/C/analysis/probability_fpfn_diagnosis_v1/`。

## 文件

- [diagnostic_spec.json](diagnostic_spec.json)
- [math_tests.json](math_tests.json)
- [threshold_curves_macro.csv](threshold_curves_macro.csv)
- [per_image_diagnostic.csv](per_image_diagnostic.csv)
- [fixed05_error_decomposition.csv](fixed05_error_decomposition.csv)
- [new_FP_spatial.csv](new_FP_spatial.csv)
- [probability_statistics_per_image.json](probability_statistics_per_image.json)
- [probability_shifts_per_image.json](probability_shifts_per_image.json)
- [matched_operating_points.csv](matched_operating_points.csv)
- [PR_advantage_grid.csv](PR_advantage_grid.csv)
- [input_manifest.json](input_manifest.json)
- [diagnostic_summary.json](diagnostic_summary.json)
- [source_archive/validate_and_plot_probability_diagnosis.py.txt](source_archive/validate_and_plot_probability_diagnosis.py.txt)
- [comparability_review.json](comparability_review.json)
- [D_new_FP_support.json](D_new_FP_support.json)
- [LOIO_threshold_sensitivity.json](LOIO_threshold_sensitivity.json)
- [native_size_descriptive.json](native_size_descriptive.json)
- [FINAL_INPUT_MANIFEST.json](FINAL_INPUT_MANIFEST.json)
- [independent_validation.json](independent_validation.json)
- [operating_advantage_summary.json](operating_advantage_summary.json)
- [report.md](report.md)
- [final_review.json](final_review.json)
- [artifact_manifest.json](artifact_manifest.json)

## 阅读与复核边界

数值表保留原分母、NA和全部病例。报告为历史文本，不替代最新结论；AI视觉观察不等于医学人工核验。`source_archive/*.py.txt`是供审查的历史CPU脚本，路径已泛化，不自动运行；依赖A7私有输入的部分不声称公开克隆即可复现。完整原件SHA与公开副本SHA分别见上级source_manifest.json。
