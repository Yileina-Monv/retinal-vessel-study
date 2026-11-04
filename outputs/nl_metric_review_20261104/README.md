# 公开结果与复算范围

固定seed2027/2028的N0/NL原终点指标见`nl_fixed_endpoint.json`，含全部28图和NA、原区间及逐seed方向保护。历史T0/TW保留2026/2027/2028。`historical_*per_image.csv`含280图-模型行，`per_path.csv`1740路径行；common_seeds表仅2027/2028，all种子比较由effects表另列。metric_review表覆盖原生宽度2/3/4/5。空字段表示NA，不补0。

运行 `python -B outputs/nl_metric_review_20261104/verify.py`（Python与NumPy）。检查公开文件SHA、逐图/路径均值、宽度分母、原NL均值、130组历史效应及1000次原固定种子bootstrap区间、FP集中性。NL旧区间为A7已独立复核后的原值导出；本公开脚本不重新生成旧NL区间，不声称原图级复算。

`*_verification.json`为既有A7验收摘要，不是本地重做像素复算。公开数据不含原图、GT像素、坐标、概率、权重或私人路径；数表只能复算聚合和统计，不能从中完整训练或确认医学机制。输入来源SHA见manifest，输出SHA验证不等同独立实验验证。

新路径为事后补充诊断，历史协议与成功标准保留，Test60封存，无新增训练/推理。
