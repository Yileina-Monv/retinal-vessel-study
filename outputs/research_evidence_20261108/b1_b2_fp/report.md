> 历史阶段报告归档；其中“下一步”“尚未授权”描述的是当时状态，不构成本轮实验授权。当前状态见本包总索引。

# P1：A0/B1/B2 新增误报科研审查

执行者/审查者：Sol；只复用 seed=2026、λ=0.03、2000步已有预测，CPU后验描述性分析。

## 结论先行

新增标签相对FP存在可解释的空间模式：以已有A0响应周围的小片扩展/延伸为主，不是少数图片或几个大型孤立岛主导。它们多远离GT，却经常贴着A0原有标签相对FP。GT覆盖局限在部分图中有视觉依据，但真实背景误报与漏标的比例仍不能确定。

这解释了代价的形态，不是免责：固定0.5下，B1/B2的Precision与Dice下降、FP/FOV升高，原保护判定仍不通过。不能据视觉怀疑把FP重新标为TP，也不能宣称医学连接恢复或原TW成功。

## 1. 固定定义与可追溯性

- 三组各28张原生float32概率，共84张；来源来自A0复用绑定、B1/B2固定终点评价及密封Dev参考。RGB沿既有Plan B来源复用。
- P=(prob≥0.5)∩FOV，FP=P∩非GT。整体评价包含U；原区域internal/near/far与FOV内U互斥覆盖FOV。
- B新增=FP_B∩非FP_A0；消除=FP_A0∩非FP_B；保留=两者交集。七个非空成员组合另表，不能把净增当成全部新增。
- GT距离为到GT∩FOV的原生欧氏距离。≤1/>1是本次固定的探索性描述，不是新的医学界线，不改变d1或正式FP；完整连续距离频数保留。另用已有Train固定alpha换算参考尺度。
- 沿既有8连通，不过滤任何小区域。新增掩码的区域是差集碎片，不能当作新生成的独立血管/独立预测对象。
- A0邻接补充同样仅描述：记录距A0 FP≤1，以及新增片是否以8邻接触及A0预测；0.45分带沿原Plan B，只用于描述A0置信度，不生成新阈值预测。

## 2. 固定终点效果及原保护条件

|指标（图像宏平均）|A0|B1|B2|
|---|---:|---:|---:|
|候选路径骨架d1 Recall|74.22%|76.51%|76.33%|
|Dice|83.51%|83.36%|83.36%|
|Precision|85.27%|84.28%|84.37%|
|FP/FOV|2.11%|2.31%|2.29%|
|全图Recall|82.18%|82.88%|82.78%|
|远处Recall|82.36%|83.05%|82.95%|
|粗血管d0 Recall|98.52%|98.60%|98.60%|

唯一主比较仍为B2−A0：+2.1015个百分点，整图配对bootstrap95%区间[+1.7519,+2.4582]个百分点；B1−A0为+2.2876个百分点。B2−B1为−0.1861个百分点，不能把B1辅助结果替换主比较。两组全部28张均增加FP且降低Precision。仍是单seed、反复使用Dev后的条件探索，未覆盖seed不确定性或独立验证。

## 3. FP分解与集中程度

|项目（像素池化计数）|B1|B2|
|---|---:|---:|
|新增FP|57856|52495|
|消除FP|7257|7450|
|净增FP|50599|45045|
|新增距GT≤1比例|20.34%|19.61%|
|新增距GT原生中位数|7.21px|8.06px|
|新增距GT参考尺度中位数|5.64|6.33|
|新增面积前三图贡献|16.53%|17.54%|
|新增差集8连通片数|27437|25322|
|新增片面积中位数 / 90分位数|1 / 4|1 / 4|
|最大新增片面积|80|75|
|单像素片占新增面积|34.41%|34.77%|

A0/B1/B2共有FP为578166像素；B1/B2共有新增为49134像素，占B1新增84.92%、B2新增93.60%。B1独有新增8722、B2独有新增3361。三组FP总量分别586940/637539/631985。

新增FP在所有28图出现；B1范围1372–3676、B2范围1260–3527像素。最大图只贡献约6%–7%，前三图约17%，不支持少数图片主导。面积中位数1、90分位数4并不证明是随机噪点：差集会将原有连续预测的边缘切成许多小片。

## 4. 最主要空间模式：贴着原有响应扩展

|新增FP的条件描述|B1|B2|
|---|---:|---:|
|距A0原有FP≤1的像素比例|67.00%|68.23%|
|属于触及A0原有FP的新增片的像素比例|87.29%|87.95%|
|属于触及A0原预测的新增片的像素比例|92.71%|93.13%|
|属于不触及A0原预测的新增片的像素比例|7.29%|6.87%|
|A0分数处于[0.45,0.5)的比例|42.55%|44.89%|

这不是说约93%的每个像素都直接紧邻A0，而是它们所属新增8连通片接触A0。距A0 FP≤1约67%–68%与距GT≤1约20%的反差，说明多数代价不是单纯GT边界的一像素扩张，而是也扩展了A0已有GT外响应。

新增位置的A0概率中位数为0.4385/0.4426，B1/B2为0.5505/0.5485；超过一半在A0时低于0.45。注意：在“新增FP”条件集合里观察到分数上升是选择条件本身，不能据此证明全FOV统一概率抬升、校准失效或某个Loss的因果机制。空间依附与跨方法重叠才是此处额外的描述性证据。

## 5. 区域和尺寸：不能混淆面积贡献与风险密度

|区域|B1新增像素|B2新增像素|背景分母|B1新增/背景|B2新增/背景|
|---|---:|---:|---:|---:|---:|
|internal|551|645|142119|0.39%|0.45%|
|near|1412|1401|578986|0.24%|0.24%|
|far|55872|50422|24584844|0.23%|0.21%|
|U_in_FOV|21|27|13409|0.16%|0.20%|

far贡献约96%的新增面积，但它本身约占97.1%的可评价背景面积，不能据绝对贡献说误报对远处有额外偏好。按背景面积归一化，internal密度反而更高，但该区面积小、只有18图有非零分母，不能推断病灶因果或再次自动改为区域加权。逐图表保留零分母NA：internal/near有效18图，far28图，U13图。图像宏平均与上述像素池化量分开保存。

原生尺寸与FOV面积差异通过逐图原始面积、FP/FOV、组件面积/FOV、alpha换算距离/面积和原生尺寸描述表保留，不选择新的宽度或距离阈值。

## 6. 全28图逐图变化（原生像素）

|图片|B1新增|B1消除|B2新增|B2消除|
|---|---:|---:|---:|---:|
|20051116_44615_0400_PP|1929|415|1677|420|
|20051116_44785_0400_PP|2395|347|2051|408|
|20051117_37078_0400_PP|2075|117|1704|180|
|20051117_37130_0400_PP|1426|81|1260|121|
|20051202_51655_0400_PP|2934|329|2506|390|
|20051202_54530_0400_PP|1664|52|1449|76|
|20051205_31934_0400_PP|2127|214|1718|264|
|20051205_35224_0400_PP|3676|672|3527|568|
|20051205_35276_0400_PP|1930|45|1789|64|
|20051205_35305_0400_PP|1542|202|1533|191|
|20051208_41707_0400_PP|2306|436|1924|542|
|20060407_41937_0200_PP|1881|80|1661|117|
|20060411_57853_0200_PP|1518|134|1416|131|
|20060411_61214_0200_PP|1631|144|1506|135|
|20060411_61478_0200_PP|1452|78|1369|91|
|20060412_51656_0200_PP|1618|184|1562|186|
|20060412_58287_0200_PP|1477|142|1386|159|
|20060412_58889_0200_PP|1372|131|1388|111|
|20060412_60913_0200_PP|1577|84|1384|67|
|20060412_61299_0200_PP|1838|73|1700|75|
|20060522_45248_0100_PP|1999|330|1734|345|
|20060522_45583_0100_PP|2951|902|2934|793|
|20060523_43038_0100_PP|2002|453|1858|443|
|20060523_49859_0100_PP|2914|282|2747|244|
|20060530_36895_0100_PP|2265|114|1938|138|
|20060530_53062_0100_PP|2144|347|1879|348|
|20060530_54390_0100_PP|2320|554|2269|519|
|20060530_55390_0100_PP|2893|315|2626|324|

## 7. 图册和真实审核范围

- 全部28图总览及固定局部图册（历史A7原件：`index.html`；本包未收录）：28张全图总览、57张不同原生裁块，共85张。每图三条固定选择规则有84条记录；相同bbox去重，不是漏图。
- 另为少数不接触A0的失败模式，按面积选每组前两例，合并相同bbox后增加2张不利图。总计87张，补充规则和全部排名见isolated_case_selection.json。
- Sol实际查看10张8面板原生图，涉及9张Dev；不是全部87张或全部像素的医学审核。原人工记录0/26和专家未核验状态保持。原case_catalog保留初始状态，实际AI覆盖见case_review_status.csv与visual_review_receipt.json。
- 本轮没有医学确认的“明确背景误报”案例。可以确认的是GT相对FP；RGB中背景/斑块样证据、可能漏标/边界证据及未知必须分开。不能把疑似漏标用来重标GT、扣除误报或提高指标。

|实际查看图片/裁块|AI观察，不是医学确认|证据图|
|---|---|---|
|20051116_44615_0400_PP [1024, 512, 1536, 1024]|possible_annotation_or_boundary：RGB contains numerous thin, branching dark lines. GT includes only part of this network. Existing and added FP include thin extensions and rims around labeled or unlabeled linear structures; much of the conspicuous FP is retained from A0, not newly introduced.|原生图（历史A7原件：`atlas/20051116_44615_0400_PP_far_gt_new_tile.png`；本包未收录）|
|20051205_35224_0400_PP [512, 0, 1024, 512]|possible_annotation_or_boundary：Several branching dark linear trajectories visible in RGB are absent from GT. All three models already predict many of these trajectories, with small additional strips or tips in B1/B2. The large red/white network must not be described as entirely new FP.|原生图（历史A7原件：`atlas/20051205_35224_0400_PP_largest_new_component.png`；本包未收录）|
|20051202_54530_0400_PP [512, 0, 1024, 512]|possible_annotation_or_boundary：Many small added segments lie along labeled vessel edges and short tips. Thin RGB branches continue through places not fully represented by the white GT mask. No gross whole-image GT/prediction displacement is visually apparent in this sheet.|原生图（历史A7原件：`atlas/20051202_54530_0400_PP_largest_new_component.png`；本包未收录）|
|20060522_45583_0100_PP [1024, 512, 1536, 1024]|background_like_visual_evidence_and_unknown：The crop contains bright textured patches and short irregular dark marks as well as plausible thin linear structures. Some label-relative FP follows irregular blotch-like features, while other FP follows linear RGB features. A0 already contains substantial FP here; B1/B2 add smaller boundary/tip increments.|原生图（历史A7原件：`atlas/20060522_45583_0100_PP_far_gt_new_tile.png`；本包未收录）|
|20060412_58889_0200_PP [512, 512, 1024, 960]|unknown：Small extra rims and fragments occur around the GT vascular network and near faint RGB lines. Some isolated predicted marks have weak visible support at this displayed resolution. No clear dominant cause can be assigned.|原生图（历史A7原件：`atlas/20060412_58889_0200_PP_far_gt_new_tile.png`；本包未收录）|
|20051117_37130_0400_PP [512, 512, 1024, 960]|unknown_with_possible_boundary_error：Added pixels are sparse relative to the shared prediction. Several follow existing vessel rims or short extensions; other faint red fragments are not confidently resolved in RGB. GT/predictions remain spatially aligned overall.|原生图（历史A7原件：`atlas/20051117_37130_0400_PP_far_gt_new_tile.png`；本包未收录）|
|20051202_51655_0400_PP [1024, 0, 1536, 512]|unknown_with_possible_annotation_or_boundary：B1/B2 changes appear mostly as small increments on shared rims and short branches, not a new large isolated region. Faint RGB detail makes some GT-free short predictions ambiguous.|原生图（历史A7原件：`atlas/20051202_51655_0400_PP_far_gt_new_tile.png`；本包未收录）|
|20051205_31934_0400_PP [1536, 512, 2048, 1024]|possible_boundary_error_and_unknown：FP and added changes are prominent along the boundaries between crowded larger vessel-like structures. GT and predictions disagree locally about coverage near the bright crowded center. The two methods remain visually very similar.|原生图（历史A7原件：`atlas/20051205_31934_0400_PP_B1_B2_disagreement_tile.png`；本包未收录）|
|20051208_41707_0400_PP [1024, 1024, 1536, 1536]|unknown：The selected isolated component is a small short fragment in a GT-free portion of the crop, with faint background linear detail. Its separation from A0 is a computed fact, but its biological identity is not visually secure.|原生图（历史A7原件：`atlas/20051208_41707_0400_PP_isolated_new_B1.png`；本包未收录）|
|20051116_44615_0400_PP [1536, 1024, 2048, 1536]|unknown：A small isolated added prediction appears near a weak narrow dark feature toward the outer FOV. The raw RGB is low contrast locally and does not establish whether this is an omitted vessel or background texture.|原生图（历史A7原件：`atlas/20051116_44615_0400_PP_isolated_new_B1.png`；本包未收录）|

典型模式包括：20051205_35224中若干可见细线网络在GT中缺失（仅怀疑）；20051202_54530更偏GT边界/末端增量；20060522_45583同时有不规则斑块样和细线样响应；20051205_31934的拥挤较粗根部边界仍模糊；两例最大孤立新增片仍无法仅凭RGB判定真伪。它们不是随机抽样，不能据10张裁块估算全Dev真实背景误报比例。

## 8. B1与B2，以及下一步最小假设

B2总FP比B1少5554像素，同时候选路径召回低0.1861个百分点；25/28图的主指标低于B1。两组主要新增区域高度重叠，未发现B2特有的巨大新误报岛。它更像响应程度的不同，而非已经证实的全新错误类别；同λ不等于同归一化分母或同梯度剂量，不能把差异纯归因于“完整分支拓扑”。

**最值得下一步验证的单一假设：当前细骨架辅助项是否主要产生一般正类驱动力带来的覆盖—误报权衡，而非细分支位置特异的净收益。**

最小的后续科学对照应优先考虑此前已提出但未授权的普通血管正样本BCE辅助项，而不是继续加步数或扫描λ。若未来获批，必须预先按Train目标和梯度设计可比剂量、固定终点/阈值/保护，且不能把Dev观察到的FP位置制成训练标签。本轮仅提出机制假设，不选择新系数、不实施Loss、不发布训练交接。

在此之前，对本轮已选的“GT外可见细线、斑块样和孤立片”安排少量专家复核可以降低解释风险，但不虚构专家结论、不改原评价，也不以专家可得性无限阻塞工程。仍缺普通正样本对照、三seed不确定性和独立验证；本轮不自动扩大任何一项。

## 9. 验收和局限

- 14项合成边界检查通过；84张预测来源/尺寸/float32/有限性/范围核验，原始逐图Dice/Precision/Recall/FP精确复算。
- 七类集合、新增−消除、四区域、全部连通片和连续距离频数可对账；独立CPU程序从原始概率重建全部28张保存掩码并核验。
- 全部输入摘要前后不变；87张PNG可解码，图册和报告链接核验；没有模型、权重加载、训练、GPU调用或Test访问。
- 原validation.json包含运行中execution.log的早期摘要，该条并非稳定证据；最终artifact_manifest.json记录完成后的文件摘要并明确取代此暂态条目。原日志不删改。
- 工程分析完成不等于方法通过保护。全部不利结果、NA、原正式T0/TW负面结论、S3 H1状态和Test封存继续保留。

## 可复核数据

[逐图正式指标](per_image_metrics.csv)；[逐图FP集合](per_image_sets.csv)；[七类成员关系](membership.csv)；全部连通区域（历史A7原件：`components.csv`；本包未收录）；[区域/背景分母](regions.csv)；距离频数（历史A7原件：`distance_histogram.csv`；本包未收录）；[距离分层密度](distance_density.csv)；[A0邻接补充](A0_attachment_supplement.csv)；[真实AI观察](visual_review_receipt.json)；[独立核验](independent_validation.json)；[分析定义](analysis_spec.json)。

服务器权威结果位于当前科学实验 analysis/fp_error_analysis_v1/。下载副本只供查看，不是第二套项目进度。
