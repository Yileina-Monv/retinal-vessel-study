# M0 工程试跑入口

2026-09-04 的一次 M0 已完成：32 张试训、8 张开发验证、256 次参数更新，独立进程续训核对通过。正式基线仍待配置裁决。

- [中文结果报告](M0试跑报告_2026-09-04.md)
- [运行前协议](试跑协议_v1.md)
- [结果摘要](summary.json)、[资源实测](resource_profile.json)、[输出验收](artifact_check.json)
- [本次原始输出](../runs/m0_pilot_2026-09-04_01/)、[模型和训练配置](../configs/m0_pilot_v1.json)

## 复现入口

在项目根目录的 PowerShell 中执行以下两段。这里使用新的输出目录名，保留已完成的试跑：

```powershell
.\.venv\Scripts\python.exe -m retinal_m0.run --run-dir runs\m0_pilot_repeat_01 --stop-at 128
.\.venv\Scripts\python.exe -m retinal_m0.run --run-dir runs\m0_pilot_repeat_01 --resume
```

第一段保存第 128 步并产生恢复参考，第二段验证恢复后运行至第 256 步。重复运行时应另取新的目录名。恢复要求绑定的代码、配置和数据清单保持一致；改变配置后应新建运行。

训练使用两个 Windows 数据加载进程，需要普通用户进程间通信权限；受限工具沙箱可能拦截，直接在正常的本地终端运行即可。本次工具运行在经自动审批允许的普通用户环境中完成。

```powershell
.\.venv\Scripts\python.exe -m m0_pilot.summarize
```

以上汇总命令仅读取本次固定目录 `runs/m0_pilot_2026-09-04_01`，重建本次摘要与图表，不训练模型，也不会自动汇总其他运行目录。`runs/` 已排除在 Git 跟踪之外，原始权重、预测和图表保留在本地。
