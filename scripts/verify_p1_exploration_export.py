#!/usr/bin/env python3
"""Recompute the public P1 export with stdlib only; no images/models/GPU required."""
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def verify(root):
    def read(name):
        with (root / name).open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))

    rows = read('fixed_endpoint_per_image_metrics.csv')
    means = read('absolute_metric_summary.csv')
    pairs = read('paired_image_deltas.csv')
    stats = read('paired_comparison_summary.csv')
    configs = read('training_config_summary.csv')
    indices = json.loads((root / 'bootstrap_indices_by_n.json').read_text(encoding='utf-8'))
    assert len(rows) == 6944 and len(means) == 248
    assert len(pairs) == 6076 and len(stats) == 217 and len(configs) == 8
    arms = {'A0', 'B1', 'B2', 'C', 'D', 'UFL', 'SPATIAL', 'SCSE'}
    assert {r['method'] for r in rows} == arms
    assert all(r['seed'] == '2026' and r['endpoint'] == '2000' and float(r['threshold']) == .5 for r in rows + pairs)
    assert len({r['metric'] for r in rows}) == 31
    assert all(r['successful_updates'] == '2000' and r['official_Test_in_this_export'] == 'False' for r in configs)
    assert [r['method'] for r in configs if r['baseline_reused'] == 'True'] == ['A0']
    assert {r['method'] for r in configs if r['additional_parameters'] != '0'} == {'SCSE'}
    assert all(len(v) == 1000 and all(len(x) == int(n) and all(0 <= i < int(n) for i in x) for x in v) for n, v in indices.items())

    values = {}
    grouped = defaultdict(list)
    na = Counter()
    image_sets = defaultdict(set)
    for r in rows:
        k = (r['method'], r['image'], r['metric'])
        assert k not in values
        image_sets[r['method']].add(r['image'])
        if r['value'] == '':
            assert r['na_reason']
            if r['denominator'] != '':
                assert float(r['denominator']) == 0
            values[k] = None
            na[r['method'], r['metric']] += 1
        else:
            v = float(r['value'])
            assert math.isfinite(v) and r['na_reason'] == ''
            if r['denominator'] != '':
                assert float(r['denominator']) > 0
                assert math.isclose(v, float(r['numerator']) / float(r['denominator']), abs_tol=1e-12)
            values[k] = v
            grouped[r['method'], r['metric']].append(v)
        assert len(r['probability_sha256']) == 64, (r['method'], r['image'])
    reference_ids = image_sets['A0']
    assert len(reference_ids) == 28 and all(v == reference_ids for v in image_sets.values())
    for r in means:
        k = (r['method'], r['metric'])
        assert math.isclose(statistics.fmean(grouped[k]), float(r['mean']), abs_tol=1e-12)
        assert len(grouped[k]) == int(r['n_images']) and na[k] == int(r['na_images'])
        assert r['seed_SD'] == ''
    for arm in arms:
        assert na[arm, 'near_thin_recall_d1'] == 19
        assert len(grouped[arm, 'near_thin_recall_d1']) == 9

    delta_groups = defaultdict(list)
    for r in pairs:
        arm = r['contrast'].removesuffix('-A0')
        x = values[arm, r['image'], r['metric']]
        y = values['A0', r['image'], r['metric']]
        if x is None or y is None:
            assert r['delta'] == '' and r['na_reason']
        else:
            d = float(r['delta'])
            assert math.isclose(d, x-y, abs_tol=1e-12)
            delta_groups[r['contrast'], r['metric']].append(d)
    max_error = 0.
    for r in stats:
        v = delta_groups[r['contrast'], r['metric']]
        assert len(v) == int(r['n_images']) and len(v) + int(r['na_images']) == 28
        bs = sorted(sum(v[i] for i in ix)/len(v) for ix in indices[str(len(v))])
        def quantile(p):
            x = (len(bs)-1)*p
            a = math.floor(x)
            return bs[a] + (bs[min(a+1,len(bs)-1)]-bs[a])*(x-a)
        expected = [statistics.fmean(v), quantile(.025), quantile(.975)]
        for a,b in zip(expected,[float(r[k]) for k in ['estimate','lower','upper']]):
            max_error = max(max_error,abs(a-b))
            assert math.isclose(a,b,abs_tol=1e-12), (r['contrast'],r['metric'],a,b)
        assert int(r['increased']) == sum(x>0 for x in v)
        assert int(r['decreased']) == sum(x<0 for x in v)
        assert int(r['unchanged']) == sum(x==0 for x in v)

    for r in read('fp_geometry_summary.csv'):
        ints = {k:int(r[k]) for k in ['new_FP_pixels','removed_FP_pixels','net_FP_pixels','within_1_native_pixel_A0','only_adjacent_A0_TP','only_adjacent_A0_FP','adjacent_both','adjacent_neither']}
        assert ints['new_FP_pixels'] - ints['removed_FP_pixels'] == ints['net_FP_pixels']
        assert sum(ints[k] for k in ['only_adjacent_A0_TP','only_adjacent_A0_FP','adjacent_both','adjacent_neither']) == ints['new_FP_pixels']
        assert sum(ints[k] for k in ['only_adjacent_A0_TP','only_adjacent_A0_FP','adjacent_both']) == ints['within_1_native_pixel_A0']
    diag = json.loads((root/'diagnostic_summary.json').read_text(encoding='utf-8'))
    for arm, d in diag['probability']['methods'].items():
        for m,v in d['fixed_05'].items():
            assert abs(v-statistics.fmean(grouped[arm,m])) < 1e-12
        b = d['best_on_same_Dev']
        assert abs(b['dice']-d['fixed_05']['dice']-b['gain_over_05']) < 1e-12
        assert b['grid_optimistic_not_validation'] is True
    curve = read('threshold_curve_sample.csv')
    assert len(curve) == 364
    for r in curve:
        assert r['method'] in {'A0','B1','B2','D'}
        assert .1 <= float(r['threshold']) <= .9
        if float(r['threshold']) == .5:
            assert abs(float(r['dice'])-statistics.fmean(grouped[r['method'],'dice'])) < 1e-12
    assert len(read('matched_operating_points.csv')) == 8
    audit = json.loads((root/'source_audit.json').read_text(encoding='utf-8'))
    assert audit['official_Test_access_in_this_export'] is False and audit['new_training_or_inference'] is False
    assert len(audit['sources']) == 35
    manifest_path = root/'export_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        repo = root.parent.parent
        for item in manifest['files']:
            p = repo/item['path']
            assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], item['path']
    return dict(status='PASS',methods=8,metrics=31,per_image_rows=len(rows),paired_rows=len(pairs),comparisons=len(stats),NA_rows=sum(na.values()),bootstrap_replicates=1000,max_statistic_error=max_error,source_files=len(audit['sources']),new_inference=False,Test_accessed=False)


if __name__ == '__main__':
    directory = Path(sys.argv[1]) if len(sys.argv)>1 else Path('outputs/p1_exploration_20261026')
    print(json.dumps(verify(directory),ensure_ascii=False))
