# 公开数值证据

运行 `python -B outputs/nnunet_ablations_20261029/verify.py`，仅需Python标准库。

`fixed_endpoint_evidence.json` 包含四项研究的全部31指标、逐图数值、有效ID、NA与原邻域分子/分母；`bootstrap_indices_by_n.json` 去重保存原索引。`verify.py`复算各组绝对均值、全部图像配对区间与77位置统计，核对跨研究N0以及来源清单。

`budget_summary.json`保留成功/AMP跳步/失败成本；`design_summary.json`说明控制变量及冻结判定；`source_manifest.json`记录A7原始来源摘要，仅公开投影不改历史实验身份。

CSV包含结果表、全部77位置诊断及N0/仅最终输出的逐图像素计数。位置来自公开数据集图像ID，无真实患者身份。保留负结果及NA。

公开包不含原图、GT图、概率数组、权重、凭据、设备编号或私人运行路径。数表复算不是从图像重算指标，也不是完整训练复现。主要判读见[中文报告](../../docs/research/nnunet_ablations_20261029/README.md)。
