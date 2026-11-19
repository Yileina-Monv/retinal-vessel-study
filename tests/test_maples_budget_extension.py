import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from retinal_m0 import maples_budget_extension as run
from retinal_m0 import maples_budget_extension_analysis as analysis

class BudgetExtensionGateTests(unittest.TestCase):
    def test_parent_sources_and_hash_rejection(self):
        for method in ('T0','TW'):
            parent=run.parent_info(method)
            self.assertEqual(parent['sha256'],run.protocol()['resume']['sources'][method]['checkpoint_sha256'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source';p.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                run.require_sha(p,'0'*64,'synthetic source rejection')

    def test_continuation_cursor_boundaries(self):
        self.assertTrue(run.validate_cursor(2000,8000))
        self.assertTrue(run.validate_cursor(4000,16000))
        for step,draw in ((1999,7996),(4001,16004),(3000,12004)):
            with self.assertRaises(ValueError):run.validate_cursor(step,draw)

    def test_only_frozen_scope_and_endpoint(self):
        self.assertTrue(run.validate_scope(2026,'T0',4000))
        self.assertTrue(run.validate_scope(2026,'TW',4000))
        for seed,method,target in ((2027,'T0',4000),(2026,'other',4000),(2026,'T0',4200)):
            with self.assertRaises(ValueError):run.validate_scope(seed,method,target)
        self.assertFalse(run.save_endpoint_probabilities(3800))
        self.assertTrue(run.save_endpoint_probabilities(4000))

    def test_pair_log_schema_and_draws(self):
        row={'step':2001,'draws':[8000,8001,8002,8003],'filenames':['a']*4,'geometry':[[0,0,0,0,0]]*4,'input_sha256':['a'*64]*4}
        self.assertTrue(run.validate_pair_row(row))
        row['draws']=[8001,8002,8003,8004]
        with self.assertRaises(ValueError):run.validate_pair_row(row)
        row.pop('input_sha256')
        with self.assertRaises(ValueError):run.validate_pair_row(row)

    def test_frozen_evaluation_statistics_schema(self):
        p=run.protocol()
        self.assertEqual(p['evaluation']['new_fixed_threshold_steps'],list(range(2200,4001,200)))
        self.assertEqual(p['evaluation']['primary']['valid_n'],9)
        self.assertEqual(p['evaluation']['primary']['target_points'],77)
        self.assertEqual(p['statistics']['bootstrap']['resamples'],1000)
        self.assertEqual(p['statistics']['bootstrap']['seed'],20261016)
        self.assertEqual(analysis.CORE_METRICS,('near_thin_recall_d1','dice','precision','fp_density','far_recall','recall','near_pixel_recall'))
        self.assertEqual(analysis.DIAMETER_METRICS,('near_thin_recall_d1_diameter_le2','near_thin_recall_d1_diameter_le4','near_thin_recall_d1_diameter_le5'))

if __name__=='__main__':unittest.main()
