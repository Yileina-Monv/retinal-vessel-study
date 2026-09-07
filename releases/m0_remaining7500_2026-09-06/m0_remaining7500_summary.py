"""Audit the completed run and collect development-only learning curves."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import math
import shutil
import zipfile
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from retinal_release.common import ROOT, read, write, digest
from retinal_release.exchange import authority
from retinal_prefetch.release import verify
from retinal_m0.state import state_digest
from retinal_m1.data import SkeletonDataset, fingerprint
from retinal_data.fives import EpochDrawSampler

torch.set_num_threads(2)
OUT = ROOT / 'releases/m0_remaining7500_2026-09-06'
RUN = ROOT / 'runs/prefetch_round1/m0_seed2026_dev10k_w1'
OLD = ROOT / 'releases/m0_first2500_2026-09-05'
window = read(OUT / 'window_result.json')
assert window['status'] == 'completed'
assert not (RUN / 'running.lock').exists()
release = verify()
progress = read(RUN / 'progress.json')
assert progress['step'] == 10000 and progress['status'] == 'completed'
assert progress['checkpoint_sha256'] == digest(RUN / 'last.pt')
saved = torch.load(RUN / 'last.pt', map_location='cpu', weights_only=True)
assert saved['step'] == 10000 and saved['consumed'] == 40000
assert len(saved['history']) == 10000 and len(saved['signatures']) == 40000
assert all(bool(torch.isfinite(t).all()) for t in saved['model'].values())
optimizer_steps = sorted({int(state['step']) for state in saved['optimizer']['state'].values()})
assert optimizer_steps == [10000]
with zipfile.ZipFile(ROOT / '.runtime/transfer/m0_seed2026_dev10k_w1_step2500.zip') as z:
    with z.open('last.pt') as f:
        previous = torch.load(f, map_location='cpu', weights_only=True)
assert state_digest(saved['history'][:2500]) == state_digest(previous['history'])
assert saved['signatures'][:10000] == previous['signatures']
assert saved['initial_digest'] == previous['initial_digest']
del previous
config = read(ROOT / 'configs/development_v1.json')
for i, row in enumerate(saved['history']):
    assert row['step'] == i + 1
    expected = config['minimum_learning_rate'] + .5 * (config['learning_rate'] - config['minimum_learning_rate']) * (1 + math.cos(math.pi*i/10000))
    assert abs(row['lr'] - expected) < 1e-12
ds = SkeletonDataset('train', seed=2026, cache_size=1)
plan = []
epoch = 0
while len(plan) < 40000:
    plan.extend(EpochDrawSampler(ds, epoch=epoch, shuffle=True))
    epoch += 1
for sig, (epoch, draw) in zip(saved['signatures'], plan):
    assert (sig['epoch'], sig['draw']) == (epoch, draw)
    assert sig['key'] == ds.rows[draw // ds.repeats]['key']
boundary_indices = [9999,10000,19999,20000,29999,30000,39999]
for i in boundary_indices:
    assert fingerprint(ds[plan[i]]) == saved['signatures'][i]
write(OUT / 'checkpoint_audit_10000.json', dict(status='passed', step=10000,
      consumed_crops=40000, optimizer_steps=optimizer_steps, model_finite=True,
      history_prefix_2500_unchanged=True, sample_prefix_10000_unchanged=True,
      schedule_10000_all_updates_verified=True, sample_order_40000_verified=True,
      independently_recomputed_fingerprint_indices=boundary_indices,
      all_fingerprints_recomputed=False, release=release,
      checkpoint_sha256=progress['checkpoint_sha256']))
evaluations = [read(OLD / 'evaluation_2500.json')] + [read(OUT / f'evaluation_{s}.json') for s in (5000,7500,10000)]
base_keys = [(r['key'], r['disease']) for r in evaluations[0]['rows']]
for e in evaluations:
    assert e['threshold'] == .5 and [(r['key'], r['disease']) for r in e['rows']] == base_keys
    assert len(e['rows']) == 118
metrics = ['dice','precision','recall','cldice','average_precision','thin_recall_d0','thin_recall_d1','thin_recall_d2']
curve = [dict(step=e['step'], **{k:e['summary']['all'][k]['mean'] for k in metrics}) for e in evaluations]
comparisons = []
for earlier,later in zip(evaluations,evaluations[1:]):
    comparison = dict(from_step=earlier['step'],to_step=later['step'],metrics={})
    for k in metrics:
        delta = np.array([y[k]-x[k] for x,y in zip(earlier['rows'],later['rows'])])
        comparison['metrics'][k] = dict(mean_change_pp=float(delta.mean()*100),
              improved_images=int((delta>0).sum()),decreased_images=int((delta<0).sum()),
              unchanged_images=int((delta==0).sum()))
    comparisons.append(comparison)
write(OUT / 'paired_descriptive_changes.json',dict(scope='descriptive_only_no_significance_test',comparisons=comparisons))
with (OUT / 'learning_curve.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(curve[0])); writer.writeheader(); writer.writerows(curve)
with (OUT / 'training_history.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(saved['history'][0])); writer.writeheader(); writer.writerows(saved['history'])
resources = [read(OUT / f'train_to_{s}_resources.json') for s in (5000,7500,10000)]
resources += [read(OUT / f'evaluation_{s}_resources.json') for s in (5000,7500,10000)]
peaks = {k:max(r[k] for r in resources) for k in ('peak_rss_bytes','peak_private_bytes','peak_cuda_reserved_bytes','peak_cuda_allocated_bytes','peak_device_vram_used_bytes')}
assert all(v < 12000000000 for v in peaks.values())
summary = dict(status='completed_10000_development_only', curve=curve, window=window,
          resource_peaks=peaks, temperature_recording=False, threshold_selected=False,
          groups={str(e['step']):e['summary']['disease'] for e in evaluations},
          checkpoint_sha256=progress['checkpoint_sha256'],
          best_observed_dice_step=max(curve,key=lambda r:r['dice'])['step'],
          train_loss_last50=float(np.mean([r['loss'] for r in saved['history'][-50:]])))
write(OUT / 'summary.json', summary)
steps = [r['step'] for r in curve]
fig, axes = plt.subplots(1, 2, figsize=(11,4), layout='constrained')
for k,label in [('dice','Dice'),('cldice','clDice'),('precision','Precision'),('recall','Recall')]:
    axes[0].plot(steps,[r[k] for r in curve],marker='o',label=label)
for k,label in [('thin_recall_d0','Thin recall, tolerance 0 px'),('thin_recall_d1','Thin recall, tolerance 1 px (primary)'),('thin_recall_d2','Thin recall, tolerance 2 px')]:
    axes[1].plot(steps,[r[k] for r in curve],marker='o',label=label)
for ax in axes:
    ax.set_xlabel('Optimizer updates'); ax.set_ylabel('Per-image macro mean')
    ax.set_xticks(steps); ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.suptitle('M0 development validation: same 118 images, fixed threshold 0.5')
fig.savefig(OUT / 'validation_curve.png',dpi=160); plt.close(fig)
loss = np.array([r['loss'] for r in saved['history']])
fig, ax = plt.subplots(figsize=(10,3.5),layout='constrained')
ax.plot(np.arange(1,10001),loss,color='#b7c8cf',linewidth=.4,label='Per update')
ax.plot(np.arange(50,10001),np.convolve(loss,np.ones(50)/50,mode='valid'),color='#087f8c',label='50-update moving mean')
for s in (2500,5000,7500): ax.axvline(s,color='#999999',linestyle=':',linewidth=.8)
ax.set(xlabel='Optimizer updates',ylabel='Training loss',title='M0: original 10,000-update cosine schedule')
ax.legend(); fig.savefig(OUT / 'training_loss.png',dpi=150); plt.close(fig)
shutil.copy2(RUN / 'ticket.json',OUT / 'ticket.json')
shutil.copy2(RUN / 'progress.json',OUT / 'progress.json')
shutil.copy2(OLD / 'evaluation_2500.json',OUT / 'evaluation_2500.json')
for name in ('m0_remaining7500_window.py','m0_remaining7500_resume.py','m0_remaining7500_eval.py','m0_remaining7500_summary.py'):
    shutil.copy2(ROOT / '.runtime' / name, OUT / name)
labels = {'dice':'Dice','precision':'Precision','recall':'Recall','cldice':'clDice','average_precision':'AP','thin_recall_d1':'细血管召回（1 px）'}
lines = ['# M0 续训至 10,000 步：开发验证结果','',
    '2026-09-06。按授权从 2,500 步继续到 10,000 步，共新增 7,500 次优化器更新。沿用原票据、种子、数据、有效 batch 4 和 10,000 步余弦学习率轨迹；本次没有采集或记录温度。', '',
    '在 5,000 / 7,500 / 10,000 步暂停训练并验证相同的 118 张开发图像，固定阈值 0.5。各检查点均独立打包并由单一进度登记表接受。未使用官方测试集，也未启动 M1。','',
    '| 指标 | 2,500 | 5,000 | 7,500 | 10,000 | 相对 2,500 的变化（百分点） |',
    '|---|---:|---:|---:|---:|---:|']
for k,label in labels.items():
    lines.append('| '+label+' | '+' | '.join(f'{r[k]:.4f}' for r in curve)+f' | {(curve[-1][k]-curve[0][k])*100:+.2f} |')
lines += ['', '| 疾病组 | 图像数 | Dice：2,500 → 10,000 | 细血管召回：2,500 → 10,000 |', '|---|---:|---:|---:|']
for d in ('A','D','G','N'):
    a=evaluations[0]['summary']['disease'][d]; b=evaluations[-1]['summary']['disease'][d]
    lines.append(f'| {d} | {b["images"]} | {a["dice"]["mean"]:.4f} → {b["dice"]["mean"]:.4f} | {a["thin_recall_d1"]["mean"]:.4f} → {b["thin_recall_d1"]["mean"]:.4f} |')
train_seconds=sum(r['train_seconds'] for r in window['segments'])
eval_seconds=sum(r['evaluation_seconds'] for r in window['segments'])
lines += ['',f'本次训练 {train_seconds/60:.2f} 分钟；三次验证合计 {eval_seconds/60:.2f} 分钟；窗口总耗时 {window["seconds"]/60:.2f} 分钟。训练与验证合计约 {(train_seconds+eval_seconds)/3600:.3f} 个 RTX 5080 占机小时。',
    '',f'资源采样峰值：程序常驻 RAM {peaks["peak_rss_bytes"]/1e9:.2f} GB，私有提交量 {peaks["peak_private_bytes"]/1e9:.2f} GB；CUDA 保留显存 {peaks["peak_cuda_reserved_bytes"]/1e9:.2f} GB，整卡显存 {peaks["peak_device_vram_used_bytes"]/1e9:.2f} GB（包含其他应用）。所有记录均低于 12 GB 限制；采样峰值不是连续监测的绝对峰值。',
    '', '完整性核验通过：优化器实际更新 10,000 次，消费 40,000 个裁块；原 2,500 步历史和前 10,000 个裁块签名未改动；全部采样次序与学习率记录匹配计划；恢复边界的 7 个样本指纹独立重算一致。没有重新计算全部 40,000 个输入指纹，也没有重跑一条完整连续轨迹来证明本次逐位等价。',
    '', '5,000 步进度登记曾遇到一次 Windows 文件夹移动权限错误；独立重试后成功接收原检查点，没有重跑或丢弃训练更新。中断记录与恢复计划均保留。第一段训练耗时从资源监测器的持续时间恢复，其余两段按父进程计时；窗口总耗时包含登记修复等待。',
    '', '结果属于单种子的开发学习曲线，不是最终测试成绩或统计显著性证据；不能据此认定某个步数最优。M0/M1 的正式训练预算仍需在开发比较后决定。',
    '', f'最终检查点 SHA-256：`{progress["checkpoint_sha256"]}`。', '',
    '![验证曲线](validation_curve.png)', '', '![训练损失](training_loss.png)', '']
(OUT / 'README.md').write_text('\n'.join(lines),encoding='utf-8')
write(OUT / 'analysis_manifest.json', dict(checkpoint_sha256=progress['checkpoint_sha256'],
      files={p.relative_to(OUT).as_posix():digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='analysis_manifest.json'}))
bundle=ROOT / '.runtime/transfer/m0_remaining7500_analysis.zip'
with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():
        if p.is_file(): z.write(p,p.relative_to(OUT).as_posix())
with authority() as registry:
    record=registry['jobs'][RUN.name]
    assert record['accepted_step']==10000 and record['status']=='completed'
    record.setdefault('analysis_history',[]).append(record['accepted_analysis'])
    record['accepted_analysis']=dict(step=10000,checkpoint_sha256=progress['checkpoint_sha256'],
         bundle=str(bundle.relative_to(ROOT)).replace('\\','/'),sha256=digest(bundle))
print(summary,flush=True)
