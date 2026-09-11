# 眼底血管分割 · 小组研究

基于 FIVES 的视网膜血管分割与测量可靠性研究。本仓库是小组代码、进度、研究决策和实验结果的共享入口。

**我们想回答：AI 是否把可见血管网络量多或量少，并把这种误差带进疾病组比较？** 当前研究限于公开标签相对的数字测量，尚无患者级独立、外部效果或临床效用验证。

| 从哪里开始 | 内容 |
|---|---|
| [小组进度 STATUS.md](STATUS.md) | 当前结论、已完成工作、下一步和暂停项 |
| [课堂汇报与讲稿（2026-09-12）](outputs/课堂汇报_20260912/README.md) | 糖网病病灶引导方向，12 页课堂汇报与配套讲稿 |
| [任务列表](https://github.com/Yileina-Monv/retinal-vessel-study/issues) | 领取任务、讨论阻塞、验收与关闭 |
| [协作约定](CONTRIBUTING.md) | 分支、PR、结果回传和数据边界 |
| [最新最小测试](outputs/morphology_minimal_20260907/最小测试报告.md) | 118 图测量误差与疾病/质量偏差 |
| [临床问题调研](docs/research/视网膜血管分割_临床痛点与AI研究问题调研_2026-09-07.md) | 主问题、备选路线与技术决策门槛 |
| [公开范围与迁移说明](PUBLICATION.md) | 来源、未公开内容、历史记录的解释 |

## 当前结果

2026-09-07，M0/M1 均为 seed=2026、10,000 步，118 张相同开发图、阈值 0.5：

| 指标 | M0 | M1 |
|---|---:|---:|
| Dice | 0.908480 | 0.910453 |
| 骨架密度平均绝对相对误差 | 7.38% | 6.42% |
| 额外骨架 / 标签骨架，1 像素容差 | 5.27% | 5.97% |

101/118 图测量误差改善，但额外骨架增加，疾病相对正常组的测量偏移基本未消除。**下一步建议：受约束的统一阈值比较；C1 和新损失训练继续暂停。** 这是开发证据，不代表正式方法胜出。

## 项目结构

- `retinal_m0/`、`retinal_m1/`：像素基线与普通 Skeleton Recall 对照。
- `retinal_c1/`：暂停的候选机制代码与合成测试，保留以便复核。
- `retinal_data/`、`retinal_release/`、`retinal_prefetch/`：数据处理、版本与执行控制。
- `configs/`、`releases/`：配置、划分、项目生成的 FOV、已筛选历史结果。
- `outputs/morphology_minimal_20260907/`：最新测量审计，含原始计数和逐图 CSV。
- `environment/`、`tests/`：依赖锁定、环境准备和测试。

## 获取与运行边界

```powershell
git clone https://github.com/Yileina-Monv/retinal-vessel-study.git
Set-Location retinal-vessel-study
```

阅读进度、审查 CSV 和代码不需要 GPU。运行 Python 代码需要自行准备环境；现有锁定环境以 Windows / Python 3.13.11 为基准，见 [环境说明](environment/README.md)。

**Git 克隆不包含原始图像、标签、模型权重、概率缓存或完整冻结运行包。** 下载与数据来源见 [数据说明](docs/DATA.md)。历史封存协议保留原始哈希，不应为了通过检查而修改哈希；真实运行需取得相匹配的本地资产并按原校验入口验证。

已有环境中的基础合成测试可运行：

```powershell
python -m unittest discover -s tests -p test_m0.py
python -m unittest discover -s tests -p test_m1_pretrial.py
```

完整 `retinal_c1.test_core` 还含两项依赖未公开本地计划/真实数据的检查；仅克隆本仓库无法通过这两项，不应把它们当作无需资产的合成测试。

`outputs/morphology_minimal_20260907/run_minimal.py` 是当次限时执行的证据快照，包含已结束的绝对截止时间和禁止覆盖规则，不是通用重跑入口。未来实验使用新任务、新目录和新协议。

公开仓库允许所有人查看；直接写入需要仓库管理员邀请。组员也可通过 fork 和 Pull Request 提交工作。原私有 BME 保留历史备份，后续小组进展以本仓库为准。
