from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
s=json.loads((P/'summary.json').read_text(encoding='utf-8'));v=json.loads((P/'verification.json').read_text(encoding='utf-8'))
rows=list(csv.DictReader((P/'per_image.csv').open(encoding='utf-8-sig')))
assert len(rows)==472 and len({r['key'] for r in rows})==118
for arm,metrics in s['arms'].items():
    rr=[r for r in rows if r['arm']==arm]
    assert len(rr)==118
    for k,m in metrics.items():assert abs(sum(float(r[k]) for r in rr)/118-m['mean'])<1e-12
tags=['M0_fixed','M1_fixed','M1_match_M0','M0_match_M1'];labels=['M0 fixed','M1 fixed','M1 matched to M0','M0 matched to M1']
fig,ax=plt.subplots(1,3,figsize=(12,4))
for a,k,title in zip(ax,['fp_fov','thin_recall','mare'],['False positive / FOV (%)','Thin centerline recall (%)','Density MARE (%)']):
    a.bar(range(4),[s['arms'][t][k]['mean']*100 for t in tags],color=['#3572a5','#e28f28','#5b9f70','#8677b5']);a.set_title(title);a.set_xticks(range(4),labels,rotation=30,ha='right');a.grid(axis='y',alpha=.2)
fig.suptitle('Exploratory two-fold development comparison; actual FP levels shown')
fig.tight_layout();fig.savefig(P/'comparison.png',dpi=150);plt.close(fig)
def val(t,k):return f"{100*s['arms'][t][k]['mean']:.4f}"
lines=['# 同等误报水平下 M0/M1 小测试','',
'## 结论与下一步','',
'M1 的优势在本次像素误报匹配后仍保留：主比较 FP/FOV 为 M0 0.6068%、M1 0.6064%，细血管召回提升 1.5324 个百分点，密度 MARE 降低 0.7494 个百分点；反向比较方向一致。因此现有优势不能由这两组统一阈值调整完全解释，值得保留 M1 作为后续机制对照。','',
'结构代价仍在：相同主比较中，无标注支持骨架/参考骨架从 5.2696% 增至 5.6861%。像素误报面积接近，不等于错误骨架量相同；需要进一步结合原图检查新增结构属于变粗、背景误认还是标注疑点，再决定训练改动。不能直接把残余差异归因于某一种机制或宣称已证明损失函数优越。','',
'暂不据此新训练或改网络。下一步优先小规模错误类型复核；正式方法结论仍需重复训练及独立评价。','',
'## 测试方法','',
'复用 118 张开发图的概率；按疾病分层，固定哈希分成两折各 59 张。在另一折上选择统一阈值，回到本折评价并交换。主比较以 M0@0.5 的平均 FP/FOV 为目标调整 M1；反向对照以 M1@0.5 为目标调整 M0。阈值网格 0.1–0.9，间隔 0.005，仅按误报距离选择，不按召回或密度挑结果。','',
'FP/FOV 为多画前景像素占有效眼底区域的比例；细血管使用原有训练集半径阈值 4 像素，召回采用 1 像素容差。各指标先逐图计算再等权平均。密度为骨架像素密度，不是物理血管长度或毛细血管密度。','',
'## 阈值','',
'| 评价折 | M1 匹配 M0 的阈值 | M0 匹配 M1 的阈值 |','|---|---:|---:|']
for f,r in s['thresholds'].items():lines.append(f"| {f} | {r['M0']['threshold']:.3f} | {r['M1']['threshold']:.3f} |")
lines+=['','## 结果','',
'所有数值以百分数表示。无支持骨架比例的分母是参考骨架数，并非预测骨架数。','',
'| 方案 | FP/FOV % | 细血管召回 % | 密度 MARE % | 无支持骨架/参考骨架 % | 精确率 % |','|---|---:|---:|---:|---:|---:|']
for tag in tags:lines.append('| '+tag+' | '+' | '.join(val(tag,k) for k in ('fp_fov','thin_recall','mare','extra','precision'))+' |')
lines+=['','## 配对差与不确定性','',
'均为 M1 减 M0，单位为百分点。主比较为 M1_match_M0 − M0_fixed；反向比较为 M1_fixed − M0_match_M1。区间是选定阈值条件下的图像配对 bootstrap，未包括阈值选择不确定性及未知患者聚类，也未校正多重比较。','',
'| 比较 | 指标 | 差值（百分点） | 描述性 95% 区间 |','|---|---|---:|---|']
for name,rr in s['comparisons'].items():
    for k,r in rr.items():lines.append(f"| {name} | {k} | {r['mean']*100:.4f} | {r['ci95'][0]*100:.4f} 至 {r['ci95'][1]*100:.4f} |")
lines+=['','![比较图](comparison.png)','','## 核验与范围','',
f"完成 {v['pairs']} 对、{v['curve_checkpoints']} 个阈值统计恢复点及 {v['evaluation_checkpoints']} 个评价恢复点。核对原始来源哈希；0.5 阈值下的既有指标完全复现；排序统计的误报及召回与直接掩膜评价一致。此次运行 {v['seconds']:.1f} 秒，未训练、未推理、未读取正式测试集。另从导出 CSV 重算了各组均值，全部一致。",'',
'匹配阈值在另一折选择，评价折上误报不保证严格相等，应结合实际差距解释。该数据此前已经用于探索；这不是独立最终验证、患者级外推、临床价值证明或新方法创新认证。主目标是判断固定阈值优势是否在控制像素误报后保留；它不回答血管拓扑是否正确。']
(P/'测试报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(P/'independent_crosscheck.json').write_text(json.dumps(dict(status='PASS',rows=472,images=118,arm_means_recomputed=True),indent=2),encoding='utf-8')
print(json.dumps(s,ensure_ascii=False,indent=2))
