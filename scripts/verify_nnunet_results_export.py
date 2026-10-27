#!/usr/bin/env python3
"""Verify the public numerical export using Python's standard library only.

This checks statistics, not image-derived metrics or full training reproduction.
Run from any directory; --root optionally selects a repository checkout.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    fraction = index - lower
    return ordered[lower] + fraction * (ordered[math.ceil(index)] - ordered[lower])


def verify(root):
    root = Path(root).resolve()
    package = root / 'outputs/nnunet_replication_20261027'

    def read(name):
        with (package / name).open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))

    manifest = json.loads((package / 'export_manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], relative
        assert path.stat().st_size == expected['bytes'], relative
    rows = read('fixed_endpoint_per_image_metrics.csv')
    means = read('absolute_metric_summary.csv')
    pairs = read('paired_image_deltas.csv')
    comparisons = read('paired_comparison_summary.csv')
    configs = read('training_config_summary.csv')
    losses = read('training_loss_steps.csv')
    epochs = read('training_loss_epochs.csv')
    seed_summaries = read('seed_replication_summary.csv')
    indices = json.loads((package / 'bootstrap_indices_by_n.json').read_text(encoding='utf-8'))
    assert len(rows) == 5208 and len(means) == 186 and len(pairs) == 2460
    assert len(comparisons) == 93 and len(configs) == 6 and len(epochs) == 300
    assert len(losses) == 75000 and len(seed_summaries) == 31
    assert set(indices) == {'9', '18', '28'}
    for n, samples in indices.items():
        assert len(samples) == 1000
        assert all(len(x) == int(n) and all(isinstance(i, int) and 0 <= i < int(n) for i in x) for x in samples)

    def cohort(seed):
        return 'historical_interrupted' if seed == '2026' else 'prospective_replication'

    data, grouped, image_sets = {}, defaultdict(list), defaultdict(set)
    na_count = 0
    for row in rows:
        seed, method, metric = row['seed'], row['method'], row['metric']
        assert seed in {'2026', '2027', '2028'} and method in {'A0', 'NNUNET'}
        assert row['cohort'] == cohort(seed) and float(row['threshold']) == 0.5
        assert row['endpoint'] == ('2000' if method == 'A0' else '100epochs')
        key = seed, method, row['image'], metric
        assert key not in data
        image_sets[seed, method].add(row['image'])
        if row['value'] == '':
            assert row['na_reason']
            value = None
            na_count += 1
        else:
            assert row['na_reason'] == ''
            value = float(row['value'])
            assert math.isfinite(value)
            grouped[seed, method, metric].append(value)
            if row['numerator'] != '' and row['denominator'] != '':
                assert float(row['denominator']) > 0
                assert abs(value - float(row['numerator']) / float(row['denominator'])) < 1e-12
        data[key] = value
    assert na_count == 288 and all(len(x) == 28 for x in image_sets.values())
    assert len({tuple(sorted(x)) for x in image_sets.values()}) == 1
    max_error = 0.0
    for row in means:
        values = grouped[row['seed'], row['method'], row['metric']]
        assert int(row['n_valid']) == len(values) and int(row['n_na']) == 28-len(values)
        max_error = max(max_error, abs(statistics.mean(values) - float(row['mean'])))
    delta_groups = defaultdict(list)
    for row in pairs:
        seed, metric, image = row['seed'], row['metric'], row['image']
        assert row['cohort'] == cohort(seed)
        left, right = data[seed, 'NNUNET', image, metric], data[seed, 'A0', image, metric]
        assert left is not None and right is not None
        delta = left-right
        max_error = max(max_error, abs(delta-float(row['delta'])))
        delta_groups[seed, metric].append((int(row['pair_index']), image, delta))
    estimates = {}
    for row in comparisons:
        seed, metric = row['seed'], row['metric']
        ordered = sorted(delta_groups[seed, metric])
        assert [x[0] for x in ordered] == list(range(len(ordered)))
        values = [x[2] for x in ordered]
        n = len(values)
        assert int(row['n_images']) == n and row['generator'] == 'PCG64'
        assert row['bootstrap_seed'] == '20261016' and row['replicates'] == '1000'
        samples = [sum(values[i] for i in sample)/n for sample in indices[str(n)]]
        actual = [statistics.mean(values), quantile(samples, .025), quantile(samples, .975)]
        wanted = [float(row[k]) for k in ['estimate', 'lower', 'upper']]
        max_error = max(max_error, *(abs(x-y) for x, y in zip(actual, wanted)))
        estimates[seed, metric] = actual[0]
    for row in seed_summaries:
        metric = row['metric']
        values = [estimates[str(seed), metric] for seed in [2027, 2028]]
        assert row['n_seeds'] == '2' and row['seeds'] == '2027;2028'
        max_error = max(max_error, abs(statistics.mean(values)-float(row['mean_delta'])),
                        abs(statistics.stdev(values)-float(row['sample_sd_ddof1'])))
    assert max_error < 1e-12, max_error

    loss_groups = defaultdict(list)
    success_counts = Counter()
    for row in losses:
        seed = row['seed']
        assert row['cohort'] == cohort(seed)
        attempt = int(row['attempt'])
        assert int(row['epoch']) == (attempt-1)//250+1 and int(row['iteration']) == (attempt-1)%250
        assert row['classification'] in {'SUCCESS', 'AMP_SKIP'}
        assert math.isfinite(float(row['loss'])) and math.isfinite(float(row['lr']))
        assert abs(float(row['lr']) - .01*(1-(int(row['epoch'])-1)/100)**.9) < 1e-12
        if row['classification'] == 'SUCCESS':
            success_counts[seed] += 1
        assert int(row['cumulative_successful_updates']) == success_counts[seed]
        loss_groups[seed].append(row)
    assert dict(success_counts) == {'2026': 24994, '2027': 24993, '2028': 24994}
    for seed, items in loss_groups.items():
        assert [int(row['attempt']) for row in items] == list(range(1, 25001))
    for row in epochs:
        items = loss_groups[row['seed']][(int(row['epoch'])-1)*250:int(row['epoch'])*250]
        assert len(items) == 250 and int(row['attempts']) == 250
        successes = [x for x in items if x['classification'] == 'SUCCESS']
        assert len(successes) == int(row['successful_updates'])
        assert 250-len(successes) == int(row['amp_skips'])
        assert abs(statistics.mean(float(x['loss']) for x in items)-float(row['train_loss_mean_all_attempts'])) < 1e-12
        assert abs(statistics.mean(float(x['loss']) for x in successes)-float(row['train_loss_mean_successful_updates'])) < 1e-12
        assert int(row['train_monitor_batches']) == 50 and math.isfinite(float(row['train_only_monitor_loss_mean']))
    for row in configs:
        assert row['official_test_accessed'] == 'False' and float(row['threshold']) == .5
        if row['method'] == 'NNUNET':
            seed = row['seed']
            assert int(row['successful_updates']) == success_counts[seed]
            assert int(row['training_attempts']) == 25000 and int(row['train_monitor_batches']) == 5000
            assert int(row['amp_skips']) == 25000-success_counts[seed]
        else:
            assert row['successful_updates'] == '2000' and row['baseline_reused'] == 'True'
    # Re-evaluate the guards instead of treating an automatic label as scientific evidence.
    cmp = {(r['seed'], r['metric']): r for r in comparisons}
    guards = {}
    for seed in ['2026', '2027', '2028']:
        directions = all(estimates[seed, m] >= 0 for m in ['precision', 'recall', 'far_recall', 'coarse_vessel_d0', 'hard_cldice']) and estimates[seed, 'fp_density'] <= 0
        strict = float(cmp[seed, 'precision']['lower']) >= 0 and float(cmp[seed, 'far_recall']['lower']) >= 0 and float(cmp[seed, 'fp_density']['upper']) <= 0
        guards[seed] = {'mean_guards_nonadverse': directions, 'strict_intervals_pass': strict}
    assert guards['2027']['mean_guards_nonadverse'] is False
    assert guards['2028']['mean_guards_nonadverse'] is True
    assert all(not x['strict_intervals_pass'] for x in guards.values())
    return {'status': 'PASS', 'statistics_max_absolute_error': max_error, 'per_image_rows': len(rows),
            'NA_rows': na_count, 'paired_rows': len(pairs), 'comparisons': len(comparisons),
            'loss_rows': len(losses), 'epoch_rows': len(epochs), 'manifest_files_verified': len(manifest['files']),
            'guards_recomputed': guards, 'verification_scope': 'Public numerical and file-integrity checks; no image-metric, model or clinical validation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))
