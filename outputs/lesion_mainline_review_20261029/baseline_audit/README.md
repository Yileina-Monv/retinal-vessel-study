# 公开逐图跨实验审计

本包归档当前对话较早完成的 P1/nnU-Net 公开 CSV 审计。它不包含原图、GT、模型或概率缓存。

## 当前解释

nnU-Net2027/2028 相对各自 A0 的 Dice 提高 1.9779/2.0626 pp，细路径及结构改善，但严格保护未通过。原邻域细骨架 d1 只有 9 张有效图，不能以全 FOV 28 图指标替代。跨 seed 重复使用同图不能增加独立图数。

[历史完整审计说明](initial_baseline_review.txt)包含所有定量比较、尾部代价、训练成本和当时提出的 patch 消融。该 patch 提案在用户明确要求回归出血邻域主线后已暂缓，不是当前首选实验。当前方向与后续讨论见[研究入口](../../../docs/research/lesion_mainline_review_20261029/README.md)。

## 文件与复算

- old/：PR7 固定提交 1f2b10c31dad94e4814597d09cfe139882c38116 的 P1 源文件。
- new/：PR10 固定提交 3cca62335dad337b0a8791b3c8712aca3359fa22 的 nnU-Net 源文件。
- provenance.json：18 个公开输入的固定 URL 与 Git blob SHA。输入按原字节保存，源 README/报告的相对链接仍按原仓库位置解释，以固定 URL 为准。
- analyze.py：公开数表复算与尾部分析，读取本目录 CSV/JSON，写 derived/；无网络、图像、模型、训练或 Test 调用。
- derived/：绝对均值、全部配对汇总、重算逐图差、尾部、2026 历史比较、同图双 seed 平均、验证记录和原运行输出（仅移除两行行末空格，数值不变）。
- verify_local_endpoint.py 与 local_endpoint_summary.json：补充复算决策稿中的 9 图双 seed 局部差值区间，条件于两个固定 seed。

运行环境：Python 3、numpy 2.3.5、pandas 2.2.3。命令为 python outputs/lesion_mainline_review_20261029/baseline_audit/analyze.py，以及 python outputs/lesion_mainline_review_20261029/baseline_audit/verify_local_endpoint.py。

审计覆盖 434 个绝对均值与 310 组既有配对区间；最大误差约 1.42e-14，小于 1e-12。旧包 6944 行含 384 NA，新包 5208 行含 288 NA；A0/2026 的 868 个指标位置一致。源文件哈希、复算和差值一致，不证明像素评价实现或模型医学有效。

## 单位与统计限制

比例指标的派生 pp 列为百分点；连续路径长度保持参考尺度像素。tails_all_metrics.csv 中 worst3/best3 是数值排序名称，FP 等越低越好的指标需反向理解。1 pp、0.5 pp 尾部阈值及去掉最高三图的描述是事后探索，不是采纳门槛；全图原主分析始终保留。

2026 nnU-Net 是中断接续的历史旁证，不加入新增双 seed 汇总。B1 等只有 seed2026，不能与 nnU-Net2027/2028 拼成同 seed 实验。患者对应未知，图像 bootstrap 未覆盖患者聚类、方案筛选和 seed 总体不确定性。

本包不覆盖完整决策稿中正式 T0/TW 的全部复算，也不包含原稿未执行的 D0。源来源与本次实际检查范围分开记录。
