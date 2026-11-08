# 已有实验详细证据包

[组长阅读入口](../../docs/research/evidence_review_20261108/README.md) · [图册入口](ATLAS.md) · [来源清单](source_manifest.json)

验证命令：

```bash
python -B outputs/research_evidence_20261108/verify.py
```

需要NumPy、pandas、Pillow。只读取公开包，不启动模型、不访问服务器或Test。验证数表算术、已声明科学事实、文件摘要、图像和链接；不重新选择阈值、不重新生成历史置信区间。

所有历史源码仅作审查，放在各阶段source_archive中；`.py.txt`不会被测试发现器误执行。原机路径改为占位符，原件与公开文件摘要分开保存。
