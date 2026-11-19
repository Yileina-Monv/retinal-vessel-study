"""All frozen31 metrics and both prespecified Multi contrasts; no threshold selection."""
import os, json, csv
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from common import *
import numpy as np
from retinal_m0.maples_branch_science_analysis import evaluate_image, metric_values
from retinal_m0.maples_stratified_science_analysis import hard_cldice, paired_bootstrap, bootstrap_indices, dice_readout


def analyze(fixture=False):
    q = protocol(); base = read(N0 / 'analysis/endpoint_100epochs.json'); ids = base['fixed_ids_by_metric']
    refs = read(q['references']['science_reference_manifest'])['images']
    rows = []
    for ref in refs:
        name = ref['filename']
        if sha(ref['reference_npz']) != ref['reference_npz_sha256']: raise RuntimeError('reference drift')
        with np.load(ref['reference_npz'], allow_pickle=False) as z: arrays = {k: z[k].copy() for k in z.files}
        values = {}; full = {}; hashes = {}
        for method, mode in [('N0', None), ('N-Local', 'local'), ('N-Multi', 'multi')]:
            path = N0 / 'dev_predictions' / (name + '.npy') if mode is None or fixture else STAGE / 'science' / mode / 'dev_predictions' / (name + '.npy')
            hashes[method] = sha(path); p = np.load(path, allow_pickle=False)
            if p.dtype != np.float32 or list(p.shape) != ref['native_shape'] or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise RuntimeError('invalid probability:' + method + ':' + name)
            record = evaluate_image(p, arrays, 'B2', threshold=.5); v = metric_values(record)
            h = hard_cldice((p >= .5) & arrays['FOV'].astype(bool), arrays['GT'], arrays['FOV'])
            v['hard_cldice'] = h['cldice']; record.update(hard_cldice=h['cldice'], topology_precision=h['topology_precision'], topology_sensitivity=h['topology_sensitivity'])
            values[method] = {k: v.get(k) for k in ids}; full[method] = record
        rows.append({'filename': name, 'methods': values, 'full_records': full, 'probability_sha256': hashes})
    by = {r['filename']: r for r in rows}; indices = {n: bootstrap_indices(n) for n in {len(v) for v in ids.values()}}
    means = {m: {} for m in ['N0', 'N-Local', 'N-Multi']}; stats = {}
    for metric, names in ids.items():
        for m in means:
            v = [by[n]['methods'][m][metric] for n in names]
            if any(x is None or not np.isfinite(x) for x in v): raise RuntimeError('NA on fixed effective GT cohort')
            means[m][metric] = float(np.mean(v))
        if abs(means['N0'][metric] - base['absolute_means']['NNUNET'][metric]) > 1e-12: raise RuntimeError('N0 reconciliation failed')
        stats[metric] = {}
        for new, old in [('N-Multi', 'N0'), ('N-Multi', 'N-Local'), ('N-Local', 'N0')]:
            d = paired_bootstrap([by[n]['methods'][new][metric] for n in names], [by[n]['methods'][old][metric] for n in names], indices=indices[len(names)])
            stats[metric][new + '-' + old] = {k: v for k, v in d.items() if k not in ['indices', 'estimates']}
    judgments = {}
    for contrast in ['N-Multi-N0', 'N-Multi-N-Local', 'N-Local-N0']:
        judgments[contrast] = dice_readout({k: {'SPATIAL-A0': s[contrast]} for k, s in stats.items()})
    success = all(stats['dice'][c]['lower'] > 0 for c in ['N-Multi-N0', 'N-Multi-N-Local']) and all(
        stats[k][c]['lower'] >= 0 for c in ['N-Multi-N0', 'N-Multi-N-Local'] for k in ['precision', 'far_recall']) and all(
        stats['fp_density'][c]['upper'] <= 0 for c in ['N-Multi-N0', 'N-Multi-N-Local']) and all(
        stats[k][c]['estimate'] >= 0 for c in ['N-Multi-N0', 'N-Multi-N-Local'] for k in ['recall', 'coarse_vessel_d0', 'hard_cldice'])
    result = {'status': 'CPU_FIXTURE_ONLY' if fixture else 'COMPLETE_PENDING_SOL_REVIEW', 'seed': 2027,
              'threshold': .5, 'endpoint': 'final100epochs', 'absolute_means': means, 'metrics': stats,
              'fixed_ids_by_metric': ids, 'per_image': rows, 'readout': judgments, 'protected_multiscale_signal': bool(success),
              'bootstrap_indices_by_n': {str(n): v.tolist() for n, v in indices.items()},
              'repeated_Dev_not_independent_validation': True, 'Test_accessed': False}
    out = STAGE / ('cpu/analysis_fixture' if fixture else 'analysis'); out.mkdir(exist_ok=True)
    atomic(out / 'endpoint_100epochs.json', result, new=True)
    with (out / 'metrics_per_image.csv').open('x', encoding='utf-8-sig', newline='') as f:
        columns = ['filename'] + [m + '_' + k for m in means for k in ids]; w = csv.DictWriter(f, columns); w.writeheader()
        for row in rows: w.writerow({'filename': row['filename'], **{m + '_' + k: row['methods'][m][k] for m in means for k in ids}})
    lines = ['# nnU-Net多尺度等参数对照' + ('：CPU替身，非新训练成绩' if fixture else '：待Sol科研复盘'),
             '', '固定seed2027、100epochs、阈值0.5；重复Dev28，不是独立测试。', '', '|指标|N0|N-Local|N-Multi|', '|---|---:|---:|---:|']
    for k in ['dice', 'precision', 'recall', 'fp_density', 'hard_cldice', 'candidate_path_d1_macro', 'far_recall', 'coarse_vessel_d0']:
        lines.append('|' + k + '|' + '|'.join(f'{means[m][k]:.8f}' for m in means) + '|')
    for c in judgments:
        d = stats['dice'][c]; lines += ['', f'{c}: Dice差{100*d["estimate"]:+.4f}pp，95%区间[{100*d["lower"]:+.4f},{100*d["upper"]:+.4f}]pp。', json.dumps(judgments[c], ensure_ascii=False)]
    lines += ['', '受保护多尺度信号：' + str(bool(success)), '全部逐图分母、NA及不利案例保留。工程完成不等于科研成功；不自动扩种子/模块/阈值/预算。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if fixture and any(v['estimate'] != 0 for metric in stats.values() for v in metric.values()): raise RuntimeError('nonzero CPU fixture contrast')
    return result


if __name__ == '__main__': analyze()
