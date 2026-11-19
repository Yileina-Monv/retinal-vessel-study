"""CPU fixtures only; production U-Net/GPU accounting is never touched."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from retinal_m0.maples_ufl_engineering_core import Budget, run_segment, pair_record
from retinal_m0.maples_g1_safety import GateError, atomic_json, canonical_sha, file_sha
import retinal_m0.maples_ufl_engineering as entry

torch.set_num_threads(2)


class TinyDataset:
    def __len__(self): return 200

    def __getitem__(self, index):
        g = np.random.default_rng(500 + index)
        x = g.normal(size=(3, 12, 12)).astype(np.float32)
        y = (g.random((1, 12, 12)) > .65).astype(np.float32)
        if index % 7 == 0:
            y[:] = 0
        return {'image': torch.tensor(x), 'mask': torch.tensor(y),
                'fov': torch.ones(1, 12, 12), 'draw': index,
                'filename': f'fixture_{index}', 'geometry': torch.tensor([index, 0, 0, 0, 0]),
                'input_sha256': f'fixture_hash_{index}'}


class EngineeringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/dev/shm')
        self.root = Path(self.tmp.name)

    def tearDown(self): self.tmp.cleanup()

    def budget(self, name='budget'):
        b = Budget(self.root / (name + '.json'), 'fixture')
        b.initialize_fixture()
        return b

    def ready(self, b, track='CONT', step=1):
        b.before_forward(track, step)
        for _ in range(3): b.before_gradient(track, step)
        b.reserve(track, step, 1)

    def model(self):
        torch.manual_seed(2026)
        m = torch.nn.Conv2d(3, 1, 1)
        return m, torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4, foreach=False), torch.amp.GradScaler('cpu', enabled=False)

    def settings(self):
        ds = TinyDataset()
        pairs = [pair_record(default_collate([ds[i] for i in range(s * 4, s * 4 + 4)]), s + 1) for s in range(50)]
        return ds, pairs

    def segment(self, b, track='CONT', start=0, end=50, out='CONT', **extra):
        ds, pairs = self.settings()
        m, o, s = self.model()
        return run_segment(model=m, optimizer=o, scaler=s, dataset=ds, track=track,
                           start=start, end=end, out=self.root / out, budget=b,
                           binding_sha='fixture', expected_pairs=pairs, device='cpu', **extra)

    def test_same_production_loop_exact50_vs25_restore25(self):
        ds, pairs = self.settings()
        b = self.budget()
        common = dict(dataset=ds, budget=b, binding_sha='fixture', expected_pairs=pairs, device='cpu')
        m, o, s = self.model()
        random.seed(2026); np.random.seed(2026)
        run_segment(model=m, optimizer=o, scaler=s, track='CONT', start=0, end=50, out=self.root / 'CONT', **common)
        m, o, s = self.model()
        random.seed(2026); np.random.seed(2026)
        run_segment(model=m, optimizer=o, scaler=s, track='SPLIT', start=0, end=25, out=self.root / 'SPLIT', **common)
        m, o, s = self.model()
        random.seed(19); np.random.seed(33); torch.manual_seed(12)
        run_segment(model=m, optimizer=o, scaler=s, track='SPLIT', start=25, end=50, out=self.root / 'SPLIT', **common)
        cont = [json.loads(x) for x in (self.root / 'CONT/train.jsonl').read_text().splitlines()]
        split = [json.loads(x) for x in (self.root / 'SPLIT/train.jsonl').read_text().splitlines()]
        for a, c in zip(cont, split):
            for key in ['pair', 'A0_loss', 'delta_loss', 'loss', 'audit', 'per_crop_losses', 'state_digests', 'preclip_total_L2']:
                self.assertEqual(a[key], c[key])
        z = b.snapshot()
        self.assertEqual((z['verified_successes'], z['charged_events'], z['forward_calls'], z['gradient_calls']), (100, 100, 100, 300))
        self.assertTrue(torch.load(self.root / 'CONT/last.pt', weights_only=True)['NOT_FOR_SCIENTIFIC_INITIALIZATION'])

    def test_no_ledger_reset(self):
        b = self.budget()
        with self.assertRaises(GateError): b.initialize_fixture()

    def test_failures_and_unknown_are_charged(self):
        for outcome in ['failed', 'uncertain']:
            b = self.budget(outcome); self.ready(b); b.before_optimizer('CONT', 1); b.resolve('CONT', outcome)
            self.assertEqual(b.snapshot()['charged_events'], 1)
            self.assertEqual(b.snapshot()['verified_successes'], 0)
            with self.assertRaises(GateError): b.before_forward('CONT', 1)

    def test_optimizer_requires_reservation(self):
        b = self.budget()
        with self.assertRaises(GateError): b.before_optimizer('CONT', 1)

    def test_duplicate_optimizer_denied(self):
        b = self.budget(); self.ready(b); b.before_optimizer('CONT', 1)
        with self.assertRaises(GateError): b.before_optimizer('CONT', 1)

    def test_forward_and_gradient_hard_limits(self):
        b = self.budget(); b.before_forward('CONT', 1)
        with self.assertRaises(GateError): b.before_forward('CONT', 1)
        for _ in range(3): b.before_gradient('CONT', 1)
        with self.assertRaises(GateError): b.before_gradient('CONT', 1)

    def test_fifty_per_track_one_hundred_total(self):
        b = self.budget()
        for track in ['CONT', 'SPLIT']:
            for step in range(1, 51):
                self.ready(b, track, step); b.before_optimizer(track, step); b.resolve(track, 'success')
        self.assertEqual(b.snapshot()['charged_events'], 100)
        for track in ['CONT', 'SPLIT']:
            with self.assertRaises(GateError): b.before_forward(track, 51)

    def test_amp_skip_keeps_compute_and_stops(self):
        b = self.budget(); self.ready(b); b.resolve('CONT', 'amp_overflow', amp_retry=True)
        z = b.snapshot()
        self.assertEqual((z['forward_calls'], z['failed_compute'], z['charged_events']), (1, 1, 0))
        with self.assertRaises(GateError): b.before_forward('CONT', 1)

    def test_stop_is_idempotent(self):
        b = self.budget(); b.before_forward('CONT', 1); b.stop('first'); b.stop('second')
        self.assertEqual(b.snapshot()['failed_compute'], 1)
        self.assertEqual(b.snapshot()['first_stop_reason'], 'first')

    def test_invalid_counter_rejected(self):
        b = self.budget(); z = b.snapshot(); z['forward_calls'] = -1; atomic_json(b.path, z)
        with self.assertRaises(GateError): b.snapshot()

    def test_input_mismatch_before_forward(self):
        ds, pairs = self.settings(); pairs[0]['input_sha256'][0] = 'wrong'; b = self.budget(); m, o, s = self.model()
        with self.assertRaises(GateError):
            run_segment(model=m, optimizer=o, scaler=s, dataset=ds, track='CONT', start=0, end=50,
                        out=self.root / 'bad', budget=b, binding_sha='fixture', expected_pairs=pairs, device='cpu')
        self.assertEqual(b.snapshot()['forward_calls'], 0)

    def test_science_segment_denied(self):
        with self.assertRaises(GateError): self.segment(self.budget(), end=2000)

    def test_gamma_scan_denied(self):
        for gamma in [0, 1, 2, True]:
            with self.assertRaises(GateError): self.segment(self.budget(str(gamma)), gamma=gamma)

    def test_ratio_pause_before_optimizer_update(self):
        b = self.budget()
        with mock.patch('retinal_m0.maples_ufl_engineering_core.audit', return_value={'pause_ratio_gt2': True}):
            with self.assertRaises(GateError): self.segment(b)
        self.assertEqual(b.snapshot()['forward_calls'], 1)
        self.assertEqual(b.snapshot()['charged_events'], 0)
        self.assertEqual(b.snapshot()['status'], 'FAILED_STOP')

    def test_resume_log_tamper_denied(self):
        b = self.budget(); self.segment(b, track='SPLIT', end=25, out='SPLIT')
        with (self.root / 'SPLIT/train.jsonl').open('a') as f: f.write('\n')
        with self.assertRaises(GateError): self.segment(b, track='SPLIT', start=25, end=50, out='SPLIT')
        self.assertEqual(b.snapshot()['verified_successes'], 25)

    def test_cli_disallows_steps_seed_gpu_gamma_and_noargs(self):
        for args in [[], ['--run', '--steps', '2000'], ['--run', '--device', 'cuda:1'],
                     ['--run', '--seed', '2027'], ['--run', '--gamma', '2']]:
            with mock.patch('sys.argv', ['eng', *args]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                entry.main()
            self.assertEqual(error.exception.code, 2)

    def test_worker_authority_before_resource_or_CUDA(self):
        with mock.patch.object(entry, 'worker_authority', side_effect=GateError('HOLD')), mock.patch.object(entry, 'resources') as resources:
            with self.assertRaises(GateError): entry.worker('CONT')
            resources.assert_not_called()

    def test_data_guard_allows_Train_but_refuses_other_image_content(self):
        train = self.root / 'train.png'; dev = self.root / 'dev.png'
        train.write_bytes(b'fixture'); dev.write_bytes(b'fixture')
        entry.data_guard(self.root, [train])
        self.assertEqual(train.read_bytes(), b'fixture')
        with self.assertRaises(GateError): dev.read_bytes()


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/dev/shm'); self.root = Path(self.tmp.name)
        names = ['CODE', 'OUT', 'BINDING', 'RELEASE', 'RECEIPT', 'LEDGER', 'SESSION', 'SEGMENT']
        values = [self.root, self.root, self.root / 'binding.json', self.root / 'release.json',
                  self.root / 'receipt.json', self.root / 'ledger.json', self.root / 'session.json', self.root / 'segment.json']
        self.patches = [mock.patch.object(entry, n, v) for n, v in zip(names, values)]
        for patch in self.patches: patch.start()
        self.source = self.root / 'source.txt'; self.source.write_text('fixture only')
        self.evidence = self.root / 'evidence.txt'; self.evidence.write_text('CPU fixture only')
        self.b = {'phase': entry.PHASE, 'gamma': .5, 'class_weights': {'background': .4, 'foreground': .6}, 'global_charged_cap': 100,
                  'track_caps': {'CONT': 50, 'SPLIT': 50}, 'segments': [list(x) for x in entry.SEGMENTS.values()],
                  'pause_ratio_gt': 2, 'GPU0_uuid': entry.GPU_UUID, 'forward_cap': 100, 'gradient_cap': 300,
                  'stage_GPU_wall_seconds': 1800, 'Dev_Test_inference_or_scientific_training_allowed': False,
                  'sources_sha256': {str(self.source): file_sha(self.source)}, 'runtime_guard_paths': [str(self.source)]}
        self.make_release()

    def tearDown(self):
        for patch in reversed(self.patches): patch.stop()
        self.tmp.cleanup()

    def make_release(self):
        atomic_json(entry.BINDING, self.b)
        digest = canonical_sha(self.b)
        receipt = {'status': 'CPU_ENGINEERING_TESTS_ACCEPTED_GPU_HELD', 'binding_sha256': digest,
                   'tests_passed': True, 'evidence_sha256': {str(self.evidence): file_sha(self.evidence)}}
        atomic_json(entry.RECEIPT, receipt)
        release = {'phase': entry.PHASE, 'authorized': True, 'release_id': entry.RELEASE_ID,
                   'global_charged_cap': 100, 'gpu_uuid': entry.GPU_UUID,
                   'binding_sha256': digest, 'binding_file_sha256': file_sha(entry.BINDING),
                   'cpu_review_sha256': file_sha(entry.RECEIPT), 'ledger_path': str(entry.LEDGER)}
        atomic_json(entry.RELEASE, release)
        atomic_json(entry.LEDGER, {})
        state = {'state': 'RELEASED', 'phase': entry.PHASE, 'release_id': entry.RELEASE_ID,
                 'binding_sha256': digest, 'release_file_sha256': file_sha(entry.RELEASE)}
        (entry.CODE / 'STATUS.md').write_text(entry.START + '\n' + json.dumps(state) + '\n' + entry.END)

    def test_consistent_release_accepted(self): self.assertEqual(entry.authority(), self.b)

    def test_missing_release_denied(self):
        entry.RELEASE.unlink()
        with self.assertRaises(GateError): entry.authority()

    def test_HOLD_denied(self):
        f = entry.CODE / 'STATUS.md'; f.write_text(f.read_text().replace('RELEASED', 'HOLD'))
        with self.assertRaises(GateError): entry.authority()

    def test_wrong_phase_denied_even_if_rebound(self):
        self.b['phase'] = 'C_SCIENCE'; self.make_release()
        with self.assertRaises(GateError): entry.authority()

    def test_wrong_gamma_denied_even_if_rebound(self):
        self.b['gamma'] = 2; self.make_release()
        with self.assertRaises(GateError): entry.authority()

    def test_expanded_budget_denied_even_if_rebound(self):
        self.b['global_charged_cap'] = 2000; self.make_release()
        with self.assertRaises(GateError): entry.authority()

    def test_source_drift_denied(self):
        self.source.write_text('changed')
        with self.assertRaises(GateError): entry.authority()

    def test_CPU_evidence_drift_denied(self):
        self.evidence.write_text('changed')
        with self.assertRaises(GateError): entry.authority()

    def test_release_drift_denied(self):
        r = json.loads(entry.RELEASE.read_text()); r['authorized'] = False; atomic_json(entry.RELEASE, r)
        with self.assertRaises(GateError): entry.authority()

    def test_duplicate_status_block_denied(self):
        f = entry.CODE / 'STATUS.md'; f.write_text(f.read_text() * 2)
        with self.assertRaises(GateError): entry.authority()

    def test_no_active_parent_denied(self):
        with self.assertRaises(FileNotFoundError): entry.worker_authority('CONT', 0, 50)

    def test_wrong_parent_identity_denied(self):
        atomic_json(entry.SESSION, {'status': 'RUNNING', 'binding_sha256': canonical_sha(self.b),
                                   'controller': {'pid': -123, 'start_time': '0'}, 'deadline_unix': 1e20})
        atomic_json(entry.SEGMENT, {'binding_sha256': canonical_sha(self.b), 'segment': ['CONT', 0, 50]})
        with self.assertRaises(GateError): entry.worker_authority('CONT', 0, 50)


if __name__ == '__main__':
    unittest.main(verbosity=2)
