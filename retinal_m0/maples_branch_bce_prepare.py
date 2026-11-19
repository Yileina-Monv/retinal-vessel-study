"""CPU-only preparation of the approved review framework; no model/GPU entry."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from retinal_m0.maples_branch_bce import human_review_gate, transform_native_target

ROOT = Path('${RETINA_PROJECT_ROOT}')
CODE = ROOT / 'code/retinal-vessel-study'
REVIEW = ROOT / 'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1'
SOURCE = REVIEW / 'attempt_02'
DOC = ROOT / 'docs/full_fov_thin_skeleton_bce_20261019/branch_review_20261021'
BASE = ROOT / 'experiments/t0_tw_core_20261016/runs/seed_2026/T0'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != (REVIEW / 'scientific_review_20261021').resolve() or not out.name.startswith('attempt_'):
        raise ValueError('Output must be a new attempt under the existing review stage')
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    def bind(path, expected=None):
        path = Path(path)
        value = sha(path)
        if expected is not None and value != expected:
            raise RuntimeError('source drift: ' + str(path))
        sources[str(path)] = value
        return value
    try:
        torch.set_num_threads(2)
        protected = json.loads((SOURCE / 'protected_sources_sha256.json').read_text())
        critical = [ROOT / 'docs/full_fov_thin_skeleton_bce_20261019' / n
                    for n in ('protocol.json', 'train_manifest.json', 'freeze_receipt.json')]
        critical += [CODE / 'retinal_m0/maples_core.py', BASE / 'pairing.jsonl']
        for path in critical:
            bind(path, protected[str(path)])
        artifacts = json.loads((SOURCE / 'artifact_sha256.json').read_text())
        bind(SOURCE / 'artifact_sha256.json')
        for name in ('candidate_definition_DRAFT.json', 'images_110.csv', 'branches_all_Train.csv',
                     'crops_8000_and_analytic_gradients.csv', 'human_review_BLANK.csv', 'atlas_cases.csv', 'verification.json'):
            bind(SOURCE / name, artifacts[name])
        for path in (DOC / 'review_spec.json', DOC / 'report.md',
                     CODE / 'retinal_m0/maples_branch_bce.py', Path(__file__), CODE / 'tests/test_maples_branch_bce.py'):
            bind(path)
        spec = json.loads((DOC / 'review_spec.json').read_text())
        assert not spec['authorization']['current_gpu_execution_permitted']
        assert not spec['authorization']['current_scientific_training_permitted']
        assert spec['loss']['lambda'] == .1
        assert spec['engineering']['G2']['max_successful_updates_total'] == 400
        assert sum(spec['engineering']['G2'][k] for k in ('lambda0_replay', 'B1_continuous50',
                   'B1_independent25plus25', 'B2_continuous50', 'B2_independent25plus25')) == 400
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        result = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v', 'tests.test_maples_branch_bce'],
                                cwd=CODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out / 'unittest.log').write_text(result.stdout, encoding='utf-8')
        if result.returncode:
            raise RuntimeError('CPU regression failed; see unittest.log')

        images = {row['filename']: row for row in read_csv(SOURCE / 'images_110.csv')}
        crops = read_csv(SOURCE / 'crops_8000_and_analytic_gradients.csv')
        assert len(images) == 110 and len(crops) == 8000
        by_step = {}
        for row in crops:
            by_step.setdefault(int(row['step']), []).append(row)
        pairs = [json.loads(line) for line in (BASE / 'pairing.jsonl').read_text().splitlines() if line.strip()]
        assert len(pairs) == 2000
        steps = spec['engineering']['G1']['fixed_batch_steps_one_based']
        assert len(steps) == len(set(steps)) == 8
        selected = []
        masks = {}
        for step in steps:
            rs = sorted(by_step[step], key=lambda row: int(row['slot']))
            assert len(rs) == 4
            pair = pairs[step - 1]
            assert pair['step'] == step
            draws = []
            for row in rs:
                slot, draw = int(row['slot']), int(row['draw'])
                name = row['filename']
                assert draw == (step - 1) * 4 + slot == pair['draws'][slot]
                assert pair['filenames'][slot] == name
                geom = [int(row[k]) for k in ('top', 'left', 'hflip', 'vflip', 'ccw_rot90_quarters')]
                assert geom == pair['geometry'][slot]
                if name not in masks:
                    key = f'candidate_masks_NOT_FOR_TRAINING/{name}.npz'
                    bind(SOURCE / key, artifacts[key])
                    with np.load(SOURCE / key, allow_pickle=False) as z:
                        masks[name] = (z['B1_reference'].copy(), z['B2_candidate'].copy())
                    assert not np.any(masks[name][0].astype(bool) & ~masks[name][1].astype(bool))
                target_hashes = {}
                for arr, method, countkey in zip(masks[name], ('B1', 'B2'), ('N_B1_reference', 'N_B2')):
                    cropped = transform_native_target(arr, top=geom[0], left=geom[1], size=512,
                                      hflip=bool(geom[2]), vflip=bool(geom[3]), quarter_turns=geom[4])
                    assert int(cropped.sum()) == int(row[countkey])
                    target_hashes[method] = hashlib.sha256(cropped.tobytes()).hexdigest()
                draws.append({'draw': draw, 'slot': slot, 'filename': name,
                              'native_shape': [int(images[name]['height']), int(images[name]['width'])],
                              'geometry': geom, 'input_sha256_original_T0': pair['input_sha256'][slot],
                              'N_B1': int(row['N_B1_reference']), 'N_B2': int(row['N_B2']),
                              'target_crop_uint8_C_order_sha256': target_hashes})
            selected.append({'step_original_T0_one_based': step, 'draws': draws})
        flat = [d for batch in selected for d in batch['draws']]
        assert len(set(tuple(d['native_shape']) for d in flat)) == 3
        assert sorted(d['N_B2'] for d in flat if not d['N_B1'] and d['N_B2']) == [1, 4, 14]
        assert any(d['N_B2'] == 0 for d in flat)
        write_json(out / 'G1_fixed_batches.json', {'status': 'CPU_TARGET_BINDING_ONLY_NOT_GPU_RELEASE',
                    'selection': 'GT-only predefined sparse / size / empty / densest; not selected by model results',
                    'batches': selected})

        branch_rows = read_csv(SOURCE / 'branches_all_Train.csv')
        candidates = [row for row in branch_rows if row['candidate_B2'] == 'True']
        signed = read_csv(SOURCE / 'human_review_BLANK.csv')
        human = human_review_gate(signed, [row['case_id'] for row in read_csv(SOURCE / 'atlas_cases.csv')])
        assert human['expected_cases'] == 26
        write_json(out / 'human_gate_status.json', human)
        # Recheck only the explicitly read/bound sources, not all343 historic files.
        for path, value in sources.items():
            if sha(path) != value:
                raise RuntimeError('concurrent source change: ' + path)
        assert not torch.cuda.is_initialized()
        verification = {
            'status': 'CPU_PREPARATION_PASSED_HUMAN_AND_GPU_GATES_PENDING',
            'tests': {'passed': True, 'log': 'unittest.log'},
            'protected_critical_old_files_rehashed': len(critical),
            'source_file_count_rehashed': len(sources),
            'old343_protection': 'historical evidence only; not rerun or newly claimed',
            'selected_batches': len(selected), 'selected_draws': len(flat),
            'three_native_sizes': True, 'sparse_B2_only_counts': [1, 4, 14],
            'original_T0_metadata_and_target_geometry': 'exact match on selected32 draws; RGB inputs not reconstructed this run',
            'candidate_paths': len(candidates),
            'human_signed_rows': human['signed_rows'], 'human_gate_passed': False,
            'real_parameter_gradients': 'NOT_MEASURED', 'GPU_AMP_and_resume': 'NOT_RUN',
            'cuda_initialized': False, 'retinal_model_forward_calls': 0, 'optimizer_updates': 0,
            'new_target_masks_generated': 0, 'Dev_images_or_probabilities_read': 0,
            'official_Test_accessed': False, 'gpu_release': False,
            'scientific_training_authorized': False,
            'remaining': ['genuine human26 and target semantic acceptance', 'Sol explicit engineering gate release',
                          'Luna G1/G2 implementation and GPU validation', 'new evaluation reference/decision freeze before scientific run']}
        write_json(out / 'source_binding.json', sources)
        write_json(out / 'verification.json', verification)
        print(json.dumps(verification, ensure_ascii=False))
    except Exception as exc:
        write_json(out / 'failure.json', {'status': 'CPU_PREPARATION_FAILED', 'type': type(exc).__name__,
                                         'error': str(exc), 'gpu_release': False})
        raise


if __name__ == '__main__':
    main()
