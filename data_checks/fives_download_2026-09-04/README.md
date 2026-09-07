# FIVES 核心数据集

[完整下载与验证报告](../../FIVES核心数据集下载与验证_2026-09-04.md)。

- `datasets/FIVES/`：800 对 2048×2048 原图与标注、原始质量表；文件内容未修改。
- `archives/fives_kaggle_v6.zip`：完整 Kaggle 镜像归档。
- `archives/quality_primary.xlsx`、`quality_comparison.xlsx`：两份镜像的质量表，仅用于只读对照。
- `fives_manifest.csv`：800 行清单，包含分区、疾病、质量、路径、文件与像素指纹。
- `format_evidence.json`：全量审计，包括 2 张空标注、6 组同图不同标注及对应文件。
- `quality_evidence.json`：800 行质量表对照和分层计数。
- `mirror_comparison_evidence.json`：两份镜像全部 1,600 个 PNG 的名称、大小与 CRC32 比较。

原始数据和归档已被 Git 忽略。下载完成，无需重复下载；没有启动模型训练。

重复组内不能靠任意删除标注解决歧义。正式实验前需明确重复组分配、同图多标注与空标签规则。镜像交叉一致不等于已证明与官方 Figshare 原包同一。
