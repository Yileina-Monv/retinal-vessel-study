# Train110裁块曝光审计

这是已有实验/诊断的补充归档，不是本次新增实验。源阶段：`experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/focal_dice_scientific_2000_v1_20261024/runs/seed_2026/C/analysis/train_crop_exposure_v1/`。

[完整110图册（下载后打开）](atlas.html)

## 文件

- [UFL_diagnostic_reconciliation.json](UFL_diagnostic_reconciliation.json)
- [diagnostic_spec.json](diagnostic_spec.json)
- [images_110.csv](images_110.csv)
- [strata_per_image.csv](strata_per_image.csv)
- [candidate_paths.csv](candidate_paths.csv)
- [crops_8000_reused.csv](crops_8000_reused.csv)
- [summary.json](summary.json)
- [input_manifest.json](input_manifest.json)
- [validation.json](validation.json)
- [independent_validation.json](independent_validation.json)
- [report.md](report.md)
- [completion_review.json](completion_review.json)
- [source_archive/train_crop_exposure_audit.py.txt](source_archive/train_crop_exposure_audit.py.txt)
- [artifact_manifest.json](artifact_manifest.json)
- [source_archive/spatial_stratified_cpu_candidate.py.txt](source_archive/spatial_stratified_cpu_candidate.py.txt)

## 阅读与复核边界

数值表保留原分母、NA和全部病例。报告为历史文本，不替代最新结论；AI视觉观察不等于医学人工核验。`source_archive/*.py.txt`是供审查的历史CPU脚本，路径已泛化，不自动运行；依赖A7私有输入的部分不声称公开克隆即可复现。完整原件SHA与公开副本SHA分别见上级source_manifest.json。
