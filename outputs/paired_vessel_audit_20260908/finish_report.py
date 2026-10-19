from pathlib import Path
import csv,json
P=Path(__file__).resolve().parent
s=json.loads((P/'summary.json').read_text(encoding='utf-8'));v=json.loads((P/'verification.json').read_text(encoding='utf-8'))
rows=list(csv.DictReader((P/'paired_transitions.csv').open(encoding='utf-8-sig')))
old=list(csv.DictReader((P.parent/'morphology_minimal_20260907/per_image_measurements.csv').open(encoding='utf-8-sig')))
old={(r['key'],r['model']):r for r in old}
assert len(rows)==118 and len(list((P/'checkpoints').glob('*.json')))==118
for r in rows:
    for d in (0,1,2):
        delta=float(old[r['key'],'M0'][f'missing_gt_skeleton_fraction_d{d}'])-float(old[r['key'],'M1'][f'missing_gt_skeleton_fraction_d{d}'])
        assert abs(float(r[f'recovered_d{d}'])-float(r[f'regressed_d{d}'])-delta)<1e-12
for k,val in s['metrics'].items():assert abs(sum(float(r[k]) for r in rows)/118-val['mean'])<1e-12
m=s['metrics']
def pct(k):return f"{100*m[k]['mean']:.3f}%"
def interval(k):return ' 至 '.join(f'{100*x:.3f}%' for x in m[k]['ci95'])
extra=m['delta_extra_d1']['mean'];supported=m['delta_supported_d1']['mean']
lines=['# M0/M1 配对血管错误分析','',
'结论：M1 同时找回部分参考血管、增加无参考支持结构。它有真实的参考覆盖改善，但密度误差下降不能全部当作结构准确性改善。当前证据不足以决定加入新损失或网络模块。','',
'## 范围与训练差异','',
'118 张既有开发图、固定阈值 0.5、相同参考标注和 FOV；未训练、未推理、未读取正式测试集。M0 为 BCE + soft Dice；M1 保留该项，加 0.1 × 软 Skeleton Recall。额外监督作用于人工中心线膨胀并限制在参考前景内的 tube，原有损失仍惩罚假阳性。不同目标的总训练 loss 不能直接比较高低来判断优劣。','',
'## 主结果：相同参考中心线的位置转移','',
'主描述采用 1 像素容差；先按图归一化，再对 118 图等权平均。分母为每张图的参考中心线像素数。','',
'| 项目 | 结果 | 描述性 95% 图像 bootstrap 区间 |','|---|---:|---:|',
f"| M0 漏画、M1 找回 | {pct('recovered_d1')} | {interval('recovered_d1')} |",
f"| M0 覆盖、M1 新增漏画 | {pct('regressed_d1')} | {interval('regressed_d1')} |",
f"| 两者仍漏画 | {pct('both_missed_d1')} | {interval('both_missed_d1')} |",
f"| 新增无参考支持预测骨架 / 参考骨架 | {pct('delta_extra_d1')} | {interval('delta_extra_d1')} |",
f"| 参考掩膜支持范围内预测骨架净增 / 参考骨架 | {pct('delta_supported_d1')} | {interval('delta_supported_d1')} |",'',
f"密度 MARE：{pct('M0_mare')} → {pct('M1_mare')}。{s['improved']}/118 图改善，其中 {s['improved_with_more_extra']} 图同时出现无参考支持骨架增加。",'',
f'严格计数分解：预测骨架总量净增中，约 {100*extra/(extra+supported):.1f}% 来自无参考支持部分（宏平均净增量之比）。这不是误差改善归因百分比；支持范围内的骨架也可能受到变粗、位移、分叉与骨架化的影响。','',
'## 粗细分层','',
'使用既有训练集校准：参考中心线局部半径 ≤ 4 原图像素为细血管，其余为较粗血管。不是物理直径或毛细血管分类。各列分母为对应图内粗细层的参考中心线像素数，等权图像平均。','',
'| 层 | M0 漏画 | M1 漏画 | 找回 | 新增漏画 |','|---|---:|---:|---:|---:|']
for label,r in s['width'].items():lines.append('| '+label+' | '+' | '.join(f'{r[k]*100:.3f}%' for k in ('M0_missed','M1_missed','recovered','regressed'))+' |')
lines+=['','## 残余漏画的概率诊断','','以下严格使用 0 像素容差下两模型共同漏掉的参考中心线位置。分母仍为全部参考中心线，因此可加至共同漏画率；不是阈值优劣实验。','',
'| 模型 | 概率 < 0.1 | 0.1 ≤ 概率 < 0.3 | 0.3 ≤ 概率 < 0.5 |','|---|---:|---:|---:|']
for model in ('M0','M1'):lines.append('| '+model+' | '+' | '.join(pct(f'{model}_bothmiss_prob_{k}_per_gt') for k in ('low','mid','near'))+' |')
lines+=['','## 疾病及模糊标签分层','','这些是观察性分层，疾病与图像质量混杂，不能解释为疾病导致模型失效。','',
'| 分层 | n | 找回 | 新增漏画 | 无支持骨架净增 |','|---|---:|---:|---:|---:|']
for label,r in s['subgroups'].items():lines.append(f"| {label} | {r['n']} | "+' | '.join(f"{r[k]['mean']*100:.3f}%" for k in ('recovered_d1','regressed_d1','delta_extra_d1'))+' |')
lines+=['','## 案例与下一步','','![极端案例](cases.png)','','案例按最大找回、最大无支持骨架增加、最大密度恶化自动选择；裁剪选择局部变化最多的窗口。为示意性极端案例，不是代表性抽样。绿色是 1 像素容差下找回的参考中心线；紫色是 0 像素下新增漏画；橙色是新增参考掩膜外像素。两种容差明确分开，橙色不等于主统计的骨架假阳性。','',
'下一步先比较同等误报水平下的中心线覆盖和密度误差，判断 M1 的优势是否只是输出概率变化。若仍有稳定优势，再优先针对细血管错误开展有控制的训练改动；目前不据此启动新损失或网络优化。低概率漏画不能单凭概率认定“特征没有学到”，也不能保证降低阈值会有益。','',
'## 核验与断点','',
f"完成 {v['pairs']} 对；472 个输入文件哈希核验，原有骨架及额外骨架统计复现；配对找回减新增漏画与前次漏画率差逐图一致。完整运行 {v['elapsed_seconds']:.1f} 秒。每图检查点已刷盘保存，可恢复；报告不会同步发布到 GitHub。",'',
'局限：单次训练种子、已探索开发集、无患者身份独立性确认；图像 bootstrap 区间仅描述性，未校正多重比较；未验证拓扑、血管身份、临床测量或病理意义。']
(P/'分析报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(P/'independent_crosscheck.json').write_text(json.dumps(dict(status='PASS',pairs=118,checkpoint_count=118,paired_transition_identity=True,summary_recomputed=True),indent=2),encoding='utf-8')
print(json.dumps(dict(metrics={k:m[k] for k in ('recovered_d1','regressed_d1','both_missed_d1','delta_extra_d1','delta_supported_d1')},width=s['width'],extra_share=extra/(extra+supported),improved=s['improved'],improved_with_more_extra=s['improved_with_more_extra'],seconds=v['elapsed_seconds']),ensure_ascii=False,indent=2))
