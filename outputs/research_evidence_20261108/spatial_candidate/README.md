# 空间分层CPU候选检查

这是已有实验/诊断的补充归档，不是本次新增实验。源阶段：`experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/focal_dice_scientific_2000_v1_20261024/runs/seed_2026/C/analysis/train_crop_exposure_v1/spatial_stratified_candidate_v1/`。

其中方法建议为当时研究记录，部分已被后续实验取代；不构成当前待执行任务。

[完整110图册（下载后打开）](atlas.html)

## 文件

- [candidate_spec.json](candidate_spec.json)
- [candidate_trajectory_8000.csv](candidate_trajectory_8000.csv)
- [crop_counts_8000.csv](crop_counts_8000.csv)
- [images_110.csv](images_110.csv)
- [strata_per_image.csv](strata_per_image.csv)
- [candidate_paths.csv](candidate_paths.csv)
- [origin_density_bounds.csv](origin_density_bounds.csv)
- [comparison_summary.json](comparison_summary.json)
- [input_manifest.json](input_manifest.json)
- [validation.json](validation.json)
- [independent_review.json](independent_review.json)
- [report.md](report.md)
- [source_archive/spatial_stratified_cpu_candidate.py.txt](source_archive/spatial_stratified_cpu_candidate.py.txt)
- [completion_review.json](completion_review.json)
- [artifact_manifest.json](artifact_manifest.json)

## 阅读与复核边界

数值表保留原分母、NA和全部病例。报告为历史文本，不替代最新结论；AI视觉观察不等于医学人工核验。`source_archive/*.py.txt`是供审查的历史CPU脚本，路径已泛化，不自动运行；依赖A7私有输入的部分不声称公开克隆即可复现。完整原件SHA与公开副本SHA分别见上级source_manifest.json。
