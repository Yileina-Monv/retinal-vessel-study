# 复核与运行边界

## 可以做什么

无需GPU即可阅读源代码、核对协议及公开CSV、查看无RGB图册，并运行本目录verify_delivery.py核对公开副本SHA、Python语法和本地链接。PR #20已有实验结果核验程序；本PR只补代码与方案，不重新计算历史区间。

A7原件是运行来源，公开副本经过路径和设备标识泛化。source_manifest.json逐项保留原件SHA和公开SHA；两者不同不代表修改了旧实验，公开副本不替换冻结配置/摘要。原模块同一身份的算法逻辑未为发布重写。旧权重/真实训练数据/现场授权/安全锁和冻结摘要不公开，也不重新生成。

## 本次实际验证

A7上CUDA_VISIBLE_DEVICES为空、python -B执行test_maples_core.py（4项）和test_maples_class_split.py（7项）均通过，均为合成小张量/几何/损失检查，无模型推理或参数训练。测试使用原服务器源码；公开副本另做摘要/语法/链接检查，不把两者混称全套复现。没有运行全量测试：部分依赖未公开真实资产、Linux多进程或历史授权。

Windows入口当前没有torch，不在本轮安装GPU依赖。environment目录原有Windows锁定文件属于早期环境，不是nnU-Net的Linux环境锁定。A7 retina_a11环境本次只读观察为torch2.4.0、numpy2.2.6、scipy1.15.3、scikit-image0.25.2、Pillow12.3.0、nnunetv2 2.8.1、dynamic-network-architectures0.4.4、batchgenerators0.25.3、batchgeneratorsv2 0.3.5；这是当前安装版本，不单独证明所有历史实验使用完全相同版本。应以对应冻结协议和环境证据为准。

## 运行限制与失败处理

不得直接执行旧controller/train/science/entry脚本。许多脚本会建立运行目录、校验来源、修改旧STATUS或请求CUDA；历史release不是新的授权。公开文件中的占位符不会自动解析，原冻结检查也不会因为公开文件SHA不同而绕过。需要重现实验时，另行取得完整数据/冻结源码与绑定资产，按现行协议授权；本次不提供自动开训入口。

源码保留了历史失败/暂停实现，例如早期nnU-Net恢复方案与后续连续双seed协议。保留这些文件是为审查失败原因，不表示推荐运行。原seed2026中断轨迹不混入2027/2028连续复现均值。

原始RGB、GT、模型、概率数组、环境目录、登录材料及PPT/讲稿未作为本次新增交付内容。数据获取和署名沿用仓库PUBLICATION.md和docs/DATA.md，官方Test60封存。
