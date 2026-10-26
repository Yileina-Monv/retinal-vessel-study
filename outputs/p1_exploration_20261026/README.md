# P1 原U-Net阶段探索公开包（2026-10-26）

先读[给组长的中文报告](../../docs/research/p1_exploration_summary_20261026.md)。仅固定seed2026、Dev28、2000更新/.5终点；不含独立模型旁支结果。A0复用，B1/B2/C/D/UFL/SPATIAL/SCSE共7条已完成科学轨迹；前期正式三种子数据保留在[原公开包](../maples_training_results_20261019/README.md)。

| 文件 | 内容 |
|---|---|
| training_config_summary.csv | 8组配置摘要，A0标明复用，空间分层标明输入变化 |
| fixed_endpoint_per_image_metrics.csv | 8×28×31=6944条，保留分子/分母、NA、概率来源SHA（不含概率值） |
| absolute_metric_summary.csv | 248个图像宏均值、有效/NA图数、单位、seed_SD为空 |
| paired_image_deltas.csv | 7×28×31=6076个配对差，包含NA |
| paired_comparison_summary.csv | 217项对A0比较，条件图像区间、正/负/持平图数 |
| bootstrap_indices_by_n.json | 原1000次已保存索引，供无NumPy的标准库校验复算 |
| protocol_design_summary.json | 各组原主目标/保护摘要，非替代冻结协议或启动授权 |
| fp_geometry_summary.csv / fp_per_image_diagnostics.json | 新增/消除FP、GT距离、预测邻接、全部28图；不是医学分类 |
| diagnostic_summary.json | 已有阈值/AP/校准、工作点比较和Train覆盖数字 |
| threshold_curve_sample.csv | 原稠密表统一按0.01步长抽样，并保留最佳/匹配点；未重扫或改变评价 |
| matched_operating_points.csv | 原同Recall/同FP诊断，含残差与同Dev选择局限 |
| source_audit.json / export_manifest.json | 私有原件身份与公开文件摘要，二者哈希不可互相冒充 |
| dice_paired_intervals.svg / dice_threshold_diagnostic.svg | 只含汇总数字的矢量图，无原图/GT像素 |

正文百分比与百分点，CSV指标与差值采用比例；最长命中/漏检长度为参考尺度像素单位。候选路径等权宏召回没有单一的像素分子/分母，故对应栏留空，不能伪造计数。hard-clDice采用原硬骨架、FOV、0.5与原空目标规则；不是训练soft-clDice的数值。

已发布数据足够重算本包指标均值和配对区间，但不包含图像或分支级原始资产，不能从CSV独立重建图像预测/医学真实性，也不能重新运行完整训练。Test封存声明来自原运行记录；本次导出自己未读Test内容。

运行：`python scripts/verify_p1_exploration_export.py`。无第三方包、GPU或私有资产需求。校验不新增方法选择，不修改原终点/保护。
