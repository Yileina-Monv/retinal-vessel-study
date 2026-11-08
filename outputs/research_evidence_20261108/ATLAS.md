# 图册与图表入口

GitHub可直接预览PNG及Markdown；HTML请下载对应目录后打开。

- [B1/B2完整28图、87视图误差图册](b1_b2_fp/public_atlas/README.md)：绿TP、蓝FN、橙FP，无RGB；原生计数，显示缩放不计分。
- [Plan B全部77位置×11条件](plan_b/all77_conditions.png)：每一行是一个位置，不是患者或完整血管。
- [阈值与PR曲线](threshold_diagnosis/threshold_PR_overview.png)；[全部28图阈值曲线](threshold_diagnosis/all28_Dice_thresholds.png)；[同工作点比较](threshold_diagnosis/matched_operating_advantage.png)。最优Dev阈值仅诊断。
- [Train110裁块曝光图册](train_crop_coverage/atlas.html)。
- [Train110空间分层对比图册](spatial_candidate/atlas.html)。

- [T0/TW/N0/NL原6例覆盖诊断图册](nl_terminal_audit/public_atlas/README.md)：共同2027/2028，无RGB。

## A7完整含RGB图册

以下原件未上传公网；不是丢失或删除。授权组员在A7项目根下定位：

|材料|项目相对入口|
|---|---|
|Plan B，77目标位置+28全图|`experiments/plan_b_error_analysis_20261019/report.html`|
|B1/B2，28图误报对照|`experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022/analysis/fp_error_analysis_v1/index.html`|
|D0，77目标位置|`experiments/nnunet2d_branch_20261020/deep_supervision_ablation_20261028/science100_v1/analysis/d0_probability_fp_v1/atlas.html`|
|NL Train目标几何图册|`experiments/nnunet2d_branch_20261020/deep_supervision_ablation_20261028/science100_v1/analysis/nl_absolute_ce_cpu_v2_alltargets_20261031/results/atlas.html`|
|N0/NL评价覆盖图册|`experiments/nnunet2d_branch_20261020/deep_supervision_ablation_20261028/science100_v1/analysis/nl_absolute_ce_cpu_v2_alltargets_20261031/science100_v1/terminal_evaluation_20261103/analysis/metric_review_20261103_atlas_manifest.json`|

可通过已有SSH连接把相应目录转发到本机浏览：在A7的相应图册目录运行 `python -m http.server 8890 --bind 127.0.0.1`，本机另开 `ssh -L 8890:127.0.0.1:8890 <已有A7别名>`，然后打开 `http://127.0.0.1:8890/`。不公开监听，不需要新模型推理；操作前确认端口空闲。

医学人工状态保持Plan B 0/105、分支审核0/26。自动诊断、AI观察与医学确认须分别理解。
