> 历史阶段报告归档；其中“下一步”“尚未授权”描述的是当时状态，不构成本轮实验授权。当前状态见本包总索引。

# 以普通 Dice 为首要目标的 Loss 研究评估

日期：2026-10-24；科研规划/实施与审查：Sol。

## 阅读结论与权限

下一轮优先候选：**保留原前景 Dice，以无类别偏置、gamma=1 的 Focal BCE 替换普通 BCE。**备选是 BCE + 轻度偏 FP 的 Tversky，不自动执行备选。当前结论是研究选择，不是方法有效性证明。

本次用户“Implement the plan”批准研究归档、新候选隔离 CPU 实现与 CPU 验证。GPU 零更新诊断、100次工程更新、2000步科学探索均保留单独批准，未自动放行。原 C soft-clDice 正在执行的独立授权不受本提案影响；不改 C 的训练、成功口径或已冻结协议。Test60封存，GPU1不用；原T0/TW负面结论、S3 H1未通过、人工0/26/专家未知均不变。

新研究首要效果量为普通像素 Dice。历史B1/B2主比较和历史骨架主指标不追改。Precision、Recall、FP/FOV、clDice、细分支/粗血管和原病灶邻域结果仍完整报告。

## 第一部分：Loss 入门

TP=找对的血管，FP=把GT背景当血管，FN=漏掉GT血管。普通Dice=2TP/(2TP+FP+FN)。例如GT100像素，TP80/FP15/FN20时Dice82.05%；找回5个TP却新增10个FP后，Recall80%升至85%，Dice降至80.95%。不能只看召回。

### BCE / Cross Entropy

单像素 BCE = -y log(p) -(1-y) log(1-p)。血管低概率、背景高概率都会受罚；对logit的导数为p-y。它提供逐像素纠错，但不直接优化整图Dice。背景很多不代表其真实参数梯度必然占优，必须实际核对。B1/B2保留了A0背景监督，单向的是新增正样本骨架项。

### Dice Loss

项目真实实现：每crop L_D = 1 -(2 sum_F(p*y)+epsilon)/(sum_F(p)+sum_F(y)+epsilon)，epsilon=1e-6，再固定batch平均；不把整个batch展平成一张图。概率和标注始终按原FOV计算，原U仍训练。

Dice对类别不平衡较友好，直接关注重叠，但比值使像素梯度耦合；空/小目标必须检查。它不认识分叉或连接。V-Net原文§3的平方和分母与项目线性和分母不同，不是完全相同的Dice实现。[V-Net](https://arxiv.org/html/1606.04797)

### Focal Loss

p_t=y*p+(1-y)*(1-p)；L_F=-alpha_t*(1-p_t)^gamma*log(p_t)。gamma压低容易样本，alpha_t另管类别权重。取消类别权重且gamma=0等于BCE。gamma=1时p_t=.99像素的Loss乘.01，p_t=.1的错误像素乘.9；既处理困难FP，也处理困难FN，并不专门关注细血管。

原文针对极端前景/背景不平衡的密集检测，§3式(4)(5)、表1(b)：最优配套alpha下gamma0/1/2/5的AP为31.1/33.7/34.0/32.2。没有医学Dice结果。难样本可能包括漏标/边界争议；不是标签噪声稳健Loss。[Focal](https://arxiv.org/html/1708.02002)

### Tversky

统一写TI=(TP_s+e)/(TP_s+a_FP*FP_s+b_FN*FN_s+e)，L_T=1-TI；软TP=sum_F(p*y)，FP=sum_F(p*(1-y))，FN=sum_F((1-p)*y)。a_FP增大偏重FP，b_FN增大偏重FN；均.5并对齐平滑后为Dice。

Salehi2017 §2.2式(2)(3)的alpha是FP系数；Abraham2019式(3)的alpha却是FN系数，禁止只按字母抄参数。Tversky提供取舍旋钮，但不保证普通Dice上升。[Tversky](https://arxiv.org/html/1706.05721)

### Focal Tversky

统一用实际指数q：L_FT=(1-TI)^q。Abraham论文式(4)为q=1/gamma，gamma=4/3；作者losses.py直接pow(1-TI,.75)，数值一致。[原文](https://arxiv.org/html/1810.07842)；[作者实现](https://github.com/nabsabraham/focal-tversky-unet/blob/94fa7f7ce27e42e6a46a8483a72db3db0b635761/losses.py)。

独立数学推导：同crop/类别的梯度为q*(1-TI)^(q-1)*grad(L_T)，并不是逐像素困难选择；逐crop平均时改变crop间权重。q<1时近零误差导数因子变大，需空目标和数值检查，不能把不同论文的gamma含义混在一起。

### Unified Focal：对称和非对称

它是两个“改造后的组成项”的加权组合，不是四个Loss相加。论文§2.8式(16)—(22)以lambda管两项比例、delta管类别/FPFN权重、gamma管聚焦。作者代码TI_c=(TP_c+e)/(TP_c+delta*FN_c+(1-delta)*FP_c+e)。

以p为血管概率、r_c=1-TI_c，作者二分类实现的对称CE项为：

F_sym=mean_F[-delta*y*(1-p)^gamma*log(p) -(1-delta)*(1-y)*p^gamma*log(1-p)]。

非对称CE项仅去掉前景的聚焦乘子；背景仍乘p^gamma。区域项R_sym=(r_0^(1-gamma)+r_1^(1-gamma))/2；R_asym=(r_0+r_1^(1-gamma))/2。代码组合为weight*R+(1-weight)*F。默认weight=.5/delta=.6/gamma=.5，不等于原A0的相同比例和相同前景Dice。

原文部分式子与代码的delta方向、指数或lambda位置有不一致；gamma=.5/weight=.5会掩盖部分差异。实施必须用明确公式和固定快照，不能同时声称遵循彼此不一致的版本。本次固定作者loss_functions.py提交411c9c5ce43b2ef847f0903d1b841512ad8d2eee；CPU语法检查发现asymmetric_focal_loss文档字符串缩进错误。原文件只存档，不导入、不擅自修成所谓原作者结果。r*r^(-gamma)在r=0时也需避免0乘无穷。[论文](https://arxiv.org/html/2102.04525v4)；[固定代码](https://github.com/mlyg/unified-focal-loss/blob/411c9c5ce43b2ef847f0903d1b841512ad8d2eee/loss_functions.py)

## 第二部分：文献实验比较

最贴近本项目的同研究比较为UFL论文表3（DRIVE）：

|Loss|Dice|Precision|Recall|
|---|---:|---:|---:|
|CE|78.9%|87.4%|74.2%|
|Focal|78.1%|72.4%|85.3%|
|Dice|79.9%|81.7%|78.7%|
|Tversky|79.4%|75.0%|84.8%|
|Focal Tversky|79.8%|76.5%|83.9%|
|Combo|79.6%|83.6%|76.3%|
|UFL对称|80.1%|81.6%|79.2%|
|UFL非对称|80.3%|79.3%|81.8%|

表3以95%CI报告，例如Dice79.9±.7%、UFL非对称80.3±.6%；两种UFL之间无显著差异。设置：16Train/4Validation/20Test，缩放512²整图，随机初始化U-Net+instance norm，SGD；验证Loss调LR/早停/选模型。二维UFL还搜索gamma=.1至.9，表中为搜索最好结果。不是我们M0/原生crop/固定2000终点的公平排名。[UFL §3.2、§4.1表3](https://arxiv.org/html/2102.04525v4)

Tversky原文表1（15例MS MRI，3D U-Net、两折、阈值.5）：FP/FN=.5/.5时DSC53.42%，.3/.7时56.42%，.1/.9时46.42%，Recall49.85%→56.85%→65.57%。过度偏召回会损害Dice，不能直接搬FN.7。

Focal Tversky原文表1普通U-Net BUS数据DL/TL/FTL Dice54.7/65.7/66.9%；表2普通U-Net ISIC TL→FTL Dice83.8→82.9%、Precision82.2→79.7%、Recall91.7→92.6%。最佳完整模型还改变注意力、多尺度输入和深监督，不能全归功于Loss。原Focal与Tversky/FTL原文没有视网膜Dice实验；只有上述UFL同框架比较提供相关DRIVE证据。不同数据集、网络、预算及选择协议的数字不能横向排名。

## 第三部分：项目证据与新增只读诊断

|Dev28 seed2026，2000步，阈值.5|A0|B1|B2|
|---|---:|---:|---:|
|Dice|83.506%|83.365%|83.358%|
|Precision|85.272%|84.278%|84.367%|
|全图Recall|82.177%|82.880%|82.777%|
|FP/FOV|2.114%|2.309%|2.289%|
|细路径骨架Recall|74.224%|76.512%|76.326%|
|clDice|87.575%|87.403%|87.402%|

B1/B2净增TP25743/21366，净增FP50599/45045；28/28图Precision下降且FP增加。约93%新增FP所属差集片触及原A0预测，既有分析不支持少数图片/大型孤立岛主导。接触不能自动判定增粗，GT外也不等于已医学证实背景。复用原FP与clDice报告/87图册，不重复人工审核、不改GT或NA。

单向骨架辅助项可能既恢复分支，也放大共享参数产生的GT外响应；这是合理机制，不是已完成的因果证明。原BCE+Dice并非没有FP惩罚。改FP/FN权重只能改变取舍，可能同时压低真实血管。

本目录diagnostic_spec.json、threshold_per_image.csv与diagnostic_summary.json为本次可复算CPU证据。固定7阈值(.3/.4/.45/.5/.55/.6/.7)、5组(A0三seed/B1/B2)、28图，共140张原生概率/980条记录；所有输入SHA前后核对，不加载模型。0.5必须复现原Dice/P/R/FP至1e-12。不是选最佳阈值、AUC或概率校准证明，不改变原0.5口径。

此前同范围预读示例：A0_2026在.45/.5/.55下Dice83.504/83.506/83.472%；B1为83.324/83.365/83.378%；B2为83.326/83.358/83.367%。调整阈值的改善很小且仍低于A0，有限网格不支持靠简单阈值解决差距，也不能证明所有问题来自表征。

A0概率的BCE与Focal gamma1输出logit导数代理另存summary：只是像素L1贡献，未经过网络Jacobian，不是batch归一化后的真实参数梯度剂量。后续选择依据为Train固定M0真实参数梯度，不用Dev标定gamma。

## 第四部分：首选、备选与风险

首选L=mean_b[sum_F((1-p_t)*BCEWithLogits)/sum_F(1)+原Dice_b]，gamma=1，无类别权重。理由：困难FP与FN都纠错，保留重叠目标；只改变像素困难度，gamma0可精确退化A0，比UFL更容易解释。gamma1是温和待验证设置，非论文MAPLES最优，Focal单独在DRIVE失败过，保留Dice不是已经证明能避开失败。

主要风险：容易背景监督减少、标签争议被强调、梯度/工作点变化仍可能降低Dice。Loss标量权重不等于参数梯度；CPU验证不证明医学有效或CUDA AMP安全。

备选BCE+Tversky，TI=(2TP+epsilon)/(2TP+1.2FP+.8FN+epsilon)，即FP.6/FN.4；.5/.5精确退化原Dice。它仅是轻度FP取舍假设，风险是损害血管召回，暂不实现、不调参、不自动训练。单独Tversky去掉BCE、FTL新增指数、UFL新增多项权重/类别归约均不是本轮首选。

## 第五部分：分阶段实施与验收

### 已批准CPU范围

隔离maples_focal_dice.py/CPU审查入口与定向测试，不改旧训练代码。稳定logits BCE；(1-p_t)=sigmoid(-(2y-1)z)避免减法抵消且保留调制梯度。Loss归约FP32、epsilon1e-6、逐crop FOV归一化、固定batch平均；空GT正常背景损失，空FOV报错不重采样。gamma只接受候选1/恒等控制0，不支持扫描。

检查数值极端/空稀疏目标/非二值/域外不变/解析梯度/有限差分/逐crop聚合/原Dice不变/CPU autocast。真实原M0固定8批32个draw，核对原T0配对输入；最多8次CPU前向、32次参数autograd，0更新，失败保留不自动重跑。审查g_A0/g_new/delta=g_new-g_A0的真实L2/方向/逐层贡献、gamma0精确一致、来源/M0/RNG不变。ATen FP32参考后端沿用C已有正确性依据，不更改未来GPU后端。CPU结果与局限见该阶段SOL_CPU_REVIEW.md/json和attempt_01/。

### GPU零更新（待用户单独批准）

最多16前向=8批×FP32/AMP，零更新；每前向最多6个参数autograd，累计96上限，检查gamma0和候选。总梯度相对L2差<=10%、cosine>=.99；Loss相对差<=1%，近零Loss以绝对1e-5、近零梯度仅在两者norm<=1e-12时用绝对L2<=1e-12。delta/A0>2、非有限/AMP异常、来源或资源漂移即停；负cosine记录，不独立等于发散。失败计算占额保留。GPU0共享锁、指定UUID、分配器<=8GiB且<=80%，GPU1不查询不用。不得写了CPU通过就自动GPU执行。

### 100更新工程（再另批）

CONT原M0独立50 vs SPLIT相同原M0独立25+新进程恢复25，总100实际/失败不明占额不退款。100前向/300梯度上限，每次审计A0及delta后新Loss真实backward；每步完整last，25/50不可覆盖。核对输入/参数/AdamW/AMP/RNG/私有loader/采样及预算日志前缀精确一致。无Dev/Test推理，不续接工程权重作科学初始化。

### 科学探索（最后单独授权）

只seed2026原M0新训2000，A0复用；保留原UNet、Train110/Dev28、batch4/crop512、输入顺序/增强/归一化、AdamW LR1e-4/WD1e-4/betas(.9,.999)/eps1e-8、clip1、AMP、无scheduler。每200步固定28图评价；唯一主比较为2000终点/0.5普通Dice，不选中间best，不改旧B1/B2/C结论。2000占额上限，目标2000成功；失败/未知不退款、自动重试或扩额。完整日志/逐次梯度/状态/异常停止支持无人值守。

Dice改善定义：28图宏均配对差>0且预存整图bootstrap95%下限>0。受保护扩seed另要求原Precision/far Recall严格区间下限>=0，FP密度区间上限<=0；全图Recall/粗血管Recall/clDice均值不恶化。细骨架和原邻域完整报告，不能替代Dice成功。Dice提升但保护失败明确标记代价，不改标准。

工程完成以完整公平可复算为准，不以正面效果为条件。简单Focal初估含审计/存盘/评价1—2GPU小时/条，仅粗估，待工程实测更新，不改变2000预算。首轮通过再另批同配置seed2027/2028；先核对同seed原M0、8000draw和全部输入SHA。报告三seed差值/均值/离散程度，不能把84个seed×图像当独立图，也不能把已参与筛选的2026包装成新验证。Test60仍不开放。

## 可复核产物与下一步

本文与sources_manifest.json/sources/保存原论文和固定代码快照；诊断脚本、输入SHA和逐图CSV保留。候选CPU证据在同一target_definition_review_v1下focal_dice_cpu_v1_20261024。最终独立审查以SOL_CPU_REVIEW为准，不能预写通过。

下一步为用户审查CPU交付后决定是否单独批准GPU零更新诊断。暂不让Luna开始GPU，不自动转训练；旧项目进度继续使用服务器唯一STATUS。


## 本次实际CPU交付

# Focal BCE + 原 Dice：Sol CPU 交付审查

**结论：CPU独立验收通过，GPU诊断/更新/科学训练均未放行。**

候选固定 gamma=1、无类别偏置，原前景Dice/FOV/逐crop归一化不变；gamma0仅作原A0精确控制。新增纯Loss和CPU入口隔离，原A0/B1/B2/C源码与冻结包未改。

- 21项定向单元检查、3项CLI拒绝通过。
- 原M0真实CPU前向8次、参数autograd32次，32个原T0输入/几何/指纹完全匹配。
- 8/8批gamma0 Loss及真实参数梯度精确一致；M0与CPU RNG不变，无梯度累积，无optimizer调用或更新。
- 候选/A0参数梯度L2范围 1.003687—1.035031；delta=(新Loss−A0)梯度/A0最大 0.050864。这不是旧正样本辅助梯度，也不是动态缩放策略。
- A0与候选梯度cosine范围 0.999096—0.999865；逐层范数及方向独立重建通过。
- 用时 669.8秒，峰值RSS 6.20GiB，2线程ATen FP32参考后端。没有CUDA AMP实测，不冒称有更新轨迹恢复通过。
- 既有Dev只读诊断140张原生概率、980行；0.5 Dice/P/R/FP原记录复现误差0，174输入来源SHA不变。首次汇总schema错误已保留，修正前后980行CSV字节相同。

真正未解决的条件：CUDA AMP/资源实测、真实短程更新与新进程恢复，以及最终Dice/保护条件。Train-M0八批的初始温和梯度不保证整个2000步稳定，不证明科学收益。

本次无新GPU操作、无Dev推理、无Test内容读取。M0审查仅Train原图/GT，允许读取原划分元数据；后验诊断单独复用既有Dev预测和参考，不能混称所有工作完全没读Dev。

下一步只建议单独批准最多16次固定GPU0零更新诊断（最多96参数梯度调用），不是100工程或2000步授权。保留ratio>2及其他异常停止线，失败不清零。GPU入口尚未实施/运行，不制造release假象。

研究报告：`${RETINA_PROJECT_ROOT}/experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022/analysis/loss_function_review_20261024/LOSS_RESEARCH_REPORT.md`。
复核数据：本目录attempt_01八批原始记录、逐层L2/dot、账本/来源摘要，以及SOL_CPU_REVIEW.json。
