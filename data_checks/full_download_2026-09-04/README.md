# 两套数据的完整下载与文件审计

主报告：[数据集下载与验证报告](../../数据集下载与验证报告_2026-09-04.md)。

- `datasets/HRF/`：官方来源，45 张原图 + 45 张标注 + 45 张 FOV。
- `datasets/Fundus-AVSeg/`：Kaggle 公开镜像版本 1，100 张原图 + 100 张标注 + 3 个附属文件。与 Figshare 原包的逐文件同一性尚未确认。
- `archives/`：本次保存的压缩包；HRF healthy 的 3 包复用相邻最小测试目录。
- `*_download_evidence.json`：下载来源、体积、压缩包指纹。
- `*_format_evidence.json`、`*_manifest.csv`：全量配对、解码、尺寸、标注编码和文件指纹。
- `http_diagnosis_evidence.json`：403 的诊断证据与成功的替代下载路线。

下载和原始文件已经完成核验，不需要重复下载。没有未完成下载文件，没有执行模型训练。所有原始数据和压缩包均被本目录的 `.gitignore` 排除。

需要复核文件时，`verify_hrf_full.py` 和 `verify_fundus_full.py` 可重新运行；脚本不改变原始标注。`inspect_fundus_metadata.py` 使用只读方式检查压缩包里的工作簿，不编辑或重新导出 Excel。
