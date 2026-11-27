# 正式模型的首折比较、反向迁移与稳健性复核

**主要发现。** 冻结同一组 Refined IDRiD 首折权重和源域阈值后，U-Net 的 1×+1.5× 在 IDRiD 留出图和 MAPLES 上均提高病灶邻域细血管 D1、远区召回和 Dice。正式 nnU-Net 的局部 D1 增量较小，两个队列的远区召回均下降约 2.5 个百分点。单 1.5× 仍是重要对照，例如 U-Net 在 MAPLES 上的 D1 为 62.11%，融合为 46.73%，后者取得四种方案中最高的远区召回和 Dice。

| 模型与评价队列 | 融合减 1× 的 D1，百分点 [95% 区间] | 远区召回变化 | Dice 变化 |
|---|---:|---:|---:|
| U-Net，IDRiD 首折 | +11.21 [2.98, 22.27] | +2.97 | +1.21 |
| nnU-Net，IDRiD 首折 | +1.82 [−0.88, 4.99] | −2.50 | −1.50 |
| U-Net，MAPLES | +14.16 [1.96, 30.07] | +2.52 | +1.55 |
| nnU-Net，MAPLES | +4.85 [0.00, 10.79] | −2.51 | −2.13 |

使用 seed 2027 历史权重，分别在 IDRiD 36 张源图上以每千邻域像素 15 个误报为预算定阈值，其中 35 张有非空邻域。以同折另 18 张及 MAPLES Dev28 评价，四种方案为 1×、1.25×、1.5× 及 1×+1.5× 等权概率融合。主指标为出血外侧 0.45°、宽度代理 ≤0.15°、容差 0.05° 的逐图平均骨架召回。两队列分别有 16 张 / 964 点、15 张 / 172 点提供主指标分母。区间按图配对重抽 20,000 次，条件于权重和本轮源域阈值。MAPLES 曾用于方法开发，IDRiD 官方 Test27 继续封存。

## 结果与解释

- [IDRiD 四种方案主表](fold0/model_comparison.csv) 的比例以 0–1 表示。[MAPLES 主表](transfer/transfer_summary.csv) 中 D1、远区召回和 Dice 以百分数表示，两表的邻域误报均按每千像素计。
- [反向迁移与阈值对照](transfer/RESULTS.md)覆盖图像配对区间与远区召回的两步分解。MAPLES 上两模型换用融合概率图后，在原阈值下远区召回都下降；改用源域标定的融合阈值后，U-Net 补回的幅度更大。
- [严谨性复核](audit/RESULTS.md)覆盖 D0 / D1、按图 / 按点、宽度分层、小分母、阈值条件与模型间增益差。MAPLES 上，U-Net 图均 +14.16 来自 5 张改善、10 张持平，只取至少 5 个目标点的 7 张图时为 +6.54。固定 0.5 时，两模型的 D1 增益排序翻转。
- [重抽标定图及留出图](audit/calibration_resampling.csv)后，IDRiD 首折 U-Net D1 差的 95% 区间为 [2.08, 22.84]。[视角扰动](audit/angle_summary.csv)将名义直径从 50° 改为 47.5°、52.5° 后仍观察到约 10–11 个百分点的收益。逐图事后精确匹配远区假阳性像素数，nnU-Net 的远区召回仍变化 −1.34 个百分点，见[远区敏感性分解](audit/far_summary.csv)。该分析使用留出标签解释机制。

## 复核与本地运行

在本目录运行 `python verify_results.py` 可用标准库从主表复算四组配对均值差，并关联检查敏感性、模型交互与视角回放。每图重采样区间的重算需要逐图预测。

原运行先准备并校验小于 100 MB 的分包、环境与历史权重，Colab 未获目标 A100/L4。最终在 RTX 4070 Laptop GPU 同环境运行两个模型（PyTorch 2.6.0+cu124、nnU-Net 2.6.2），完成单图试跑与全折推理。各模型生成 36 份标定记录、18 份留出记录及 18 份概率缓存；首折归档 167 个文件，11 个分包均已逐文件核对。原尺度逐图 Dice 相对历史值的最大绝对差低于 0.0071 个百分点。

原分析脚本位于 replay/。需要重新推理时，将其目录内容复制到私有实验工作区，保留以下相对路径。输入资料由研究者按数据授权准备。

1. `Colab_训练推理补充_20261126/runner/stage_inputs.py` 接受 `--parent`、`--supplement` 和 `--work .work/colab_stagecheck/work`；assets.json 记录分包哈希和重组规则。MAPLES 的冻结划分放在 runner/maples_split.json。标准 nnU-Net trainer 初始化还需要在历史 plans 指定的 native_numpy_2d 目录下提供已按该 plans 预处理的源域训练样本（原运行恢复了 36 张）；复用原 plans 和权重。
2. 从私有工作区根运行 `python Colab_训练推理补充_20261126/runner/scale_eval.py --work .work/colab_stagecheck/work --out 首折双尺度_本地GPU_20261126/idrid_fold0 --model both --keep-probabilities`。单图检查改用独立输出目录并添加 `--smoke`。
3. 运行 `双尺度反向迁移_20261126/reverse_transfer.py` 和同目录的 summarize_transfer.py。后者还会对照私有历史角度研究的逐图分母。再依次运行 `双尺度严谨性复核_20261126/check.py`、source_bootstrap.py、angle_check.py。GPU 推理和逐图输出均留在私有工作区。

数据来源为 [Refined IDRiD v2](https://doi.org/10.5281/zenodo.18676805)、[MAPLES-DR](https://doi.org/10.1038/s41597-024-03739-6) 与 [FIVES](https://doi.org/10.1038/s41597-022-01564-3)。历史模型及其训练来源见 [PR #22](https://github.com/Yileina-Monv/retinal-vessel-study/pull/22)。
