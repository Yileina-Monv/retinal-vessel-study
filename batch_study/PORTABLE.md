# 在 RTX 4070 Laptop 上补齐同一试验

本轮用户已授权跨 GPU 与 batch 对照；当前任务环境仅实际检测到 RTX 5080。此说明不表示 4070 已完成实测。

## 环境和材料

使用本机实际记录的 Python 3.13.11、PyTorch 2.12.1+cu130 及锁定依赖组合。不要直接复制另一台机器的 `.venv`。在本项目目录使用 `environment/setup.ps1` 重建环境，保留原网络代理规则；安装入口仅检查 Python 3.13 大小版本，跨 GPU 比较器还会检查完整版本，不能把其他补丁版默认为完全相同环境。硬件/软件将实际写入输出，不能只复制 5080 的环境核验 JSON 当作笔记本已核验。

迁移包只包含本试验需要的代码、协议、原有 32 张训练与 8 张开发图、相关清单、共同初始权重和 5080 的紧凑结果。图像均来自原 train 分区，不含官方测试图或外部数据。完整实验检查点仍在原机器本地保留。

解压到一个新目录，不覆盖已有研究项目或产物。包内路径与原项目结构一致。包包含 `bundle_manifest.json`，可核对文件 SHA-256。压缩包传输本身不等于训练完成。

## 执行

在解压目录的 PowerShell 中，环境准备完成后执行：

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -p test_batch_study.py -v
& '.\.venv\Scripts\python.exe' -m retinal_batchstudy.run --device-label rtx4070 --mode all --max-seconds 3000
```

3000 秒为 50 分钟运行预算，留出 1 小时窗口的收尾余量；加载初始化、退出写盘可能额外耗时。运行会在完整参数更新边界保存并暂停。之后执行同一命令即可继续；2 小时或更长窗口可设置更大的 `--max-seconds`，它只改变本次运行窗口，不改变各组总样本预算。已完成组自动跳过。

必须让完整矩阵结束后再作最终显著性判断。查看结果：

```powershell
& '.\.venv\Scripts\python.exe' -m retinal_batchstudy.summarize --device-label rtx4070
```

主矩阵暂停或结束、同卡没有其他训练时，可另做容量测试（微批量 1/2/4/8/16；有效 batch 16；不用于精度推断）：

```powershell
& '.\.venv\Scripts\python.exe' -m batch_study.benchmark_native_batch --device-label rtx4070
```

两台完整矩阵和数值文件收齐后：

```powershell
& '.\.venv\Scripts\python.exe' -m retinal_batchstudy.compare_gpu 'runs\batch_study_v1\rtx5080' 'runs\batch_study_v1\rtx4070'
```

另检查 batch 效应是否依赖 GPU：

```powershell
& '.\.venv\Scripts\python.exe' -m batch_study.compare_interactions 'runs\batch_study_v1\rtx5080' 'runs\batch_study_v1\rtx4070' --output-dir 'batch_study\results\cross_gpu'
```

比较器会拒绝不同代码/协议/依赖、不同初始化或输入序列、缺失运行及相同 GPU 型号冒充跨 GPU 的情况。驱动和操作系统差异记录在硬件清单中；即使锁定框架版本，结论也代表两套实际设备环境，不能完全分离驱动和 GPU 架构的影响。

## 返回材料

返回 `batch_study/results/rtx4070/`、`runs/batch_study_v1/rtx4070/` 中的配置/绑定/硬件/数值文件，以及每个种子每个组的 `result.json`、`eval_draw_1024.json`、`eval_draw_2048.json`。除续训排错外，通常不需传输所有 `last.pt`；不要删除原机检查点。

OOM、AMP 跳步、非有限梯度等错误必须原样保留，不静默换微批量、降低裁块或修改学习率。若硬件无法执行某组，应先记录不兼容，再另建清楚标识的修订协议，不将不同配置混入原比较。
