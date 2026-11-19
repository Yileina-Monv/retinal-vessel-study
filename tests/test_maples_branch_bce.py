"""CPU synthetic tensors only: no model, image dataset, checkpoint, or CUDA."""
import copy
import random
import unittest
import numpy as np
import torch
from retinal_m0.maples_core import weighted_masked_loss
from retinal_m0.maples_branch_bce import (
    branch_bce_loss, normalized_positive_bce, transform_native_target, human_review_gate)


class BranchBCETests(unittest.TestCase):
    def setUp(self):
        self.z = torch.linspace(-4, 4, 4 * 4 * 4).reshape(4, 1, 4, 4).requires_grad_(True)
        self.y = torch.ones_like(self.z)
        self.f = torch.ones_like(self.z)
        self.s = torch.zeros_like(self.z)
        self.s[0, 0, 0, 0] = 1
        self.s[1, 0, :2, :2] = 1
        self.s[2, 0, :3, :] = 1

    def test_zero_coefficient_value_and_gradient_bitwise(self):
        a = weighted_masked_loss(self.z, self.y, self.f, self.f)
        b = branch_bce_loss(self.z, self.y, self.f, self.s, 0)
        self.assertTrue(torch.equal(a, b))
        ga, = torch.autograd.grad(a, self.z, retain_graph=True)
        gb, = torch.autograd.grad(b, self.z)
        self.assertTrue(torch.equal(ga, gb))

    def test_original_base_is_unchanged(self):
        base = weighted_masked_loss(self.z, self.y, self.f, self.f)
        aux = normalized_positive_bce(self.z, self.y, self.f, self.s)
        self.assertTrue(torch.equal(branch_bce_loss(self.z, self.y, self.f, self.s), base + .1 * aux))

    def test_per_crop_mean_not_pooled(self):
        values = torch.nn.functional.softplus(-self.z)
        manual = (values[0, 0, 0, 0] + values[1, 0, :2, :2].mean()
                  + values[2, 0, :3, :].mean()) / 4
        got = normalized_positive_bce(self.z, self.y, self.f, self.s)
        self.assertTrue(torch.allclose(got, manual, atol=1e-7))
        self.assertFalse(torch.isclose(got, (values * self.s).sum() / self.s.sum()))

    def test_analytic_gradient_and_l1_bound(self):
        value = .1 * normalized_positive_bce(self.z, self.y, self.f, self.s)
        observed, = torch.autograd.grad(value, self.z)
        counts = self.s.sum((1, 2, 3)).clamp_min(1)[:, None, None, None]
        expected = .1 * self.s * (self.z.detach().sigmoid() - 1) / (4 * counts)
        self.assertTrue(torch.allclose(observed, expected, rtol=1e-5, atol=1e-7))
        self.assertLessEqual(observed.abs().sum().item(), .1 + 1e-7)

    def test_empty_target_zero_value_and_gradient(self):
        s = torch.zeros_like(self.s)
        value = normalized_positive_bce(self.z, self.y, self.f, s)
        grad, = torch.autograd.grad(value, self.z)
        self.assertEqual(value.item(), 0)
        self.assertEqual(torch.count_nonzero(grad).item(), 0)

    def test_target_outside_gt_rejected(self):
        y = self.y.clone(); y[0, 0, 0, 0] = 0
        with self.assertRaises(ValueError): branch_bce_loss(self.z, y, self.f, self.s)

    def test_target_outside_fov_rejected(self):
        f = self.f.clone(); f[0, 0, 0, 0] = 0
        with self.assertRaises(ValueError): branch_bce_loss(self.z, self.y, f, self.s)

    def test_empty_fov_rejected_no_resampling(self):
        with self.assertRaisesRegex(ValueError, 'empty_fov'):
            branch_bce_loss(self.z, self.y, torch.zeros_like(self.f), torch.zeros_like(self.s))

    def test_nonbinary_target_rejected(self):
        s = self.s.clone(); s[0, 0, 0, 0] = .5
        with self.assertRaises(ValueError): branch_bce_loss(self.z, self.y, self.f, s)

    def test_nonfinite_logit_rejected(self):
        z = self.z.detach().clone(); z[0, 0, 0, 0] = float('inf')
        with self.assertRaises(ValueError): branch_bce_loss(z, self.y, self.f, self.s)

    def test_lambda_validation(self):
        for lam in (-1, float('nan'), float('inf'), True):
            with self.subTest(lam=lam), self.assertRaises(ValueError):
                branch_bce_loss(self.z, self.y, self.f, self.s, lam)

    def test_single_channel_and_shape_required(self):
        with self.assertRaises(ValueError):
            branch_bce_loss(self.z.expand(-1, 2, -1, -1), self.y, self.f, self.s)

    def test_extreme_confident_misses_finite(self):
        z = torch.full_like(self.z, -1000, requires_grad=True)
        loss = branch_bce_loss(z, self.y, self.f, self.s)
        grad, = torch.autograd.grad(loss, z)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(grad).all().item())

    def test_low_probability_bce_vs_paper_soft_recall(self):
        z = torch.tensor([-4.59511985], requires_grad=True)
        gb, = torch.autograd.grad(torch.nn.functional.softplus(-z).mean(), z, retain_graph=True)
        gr, = torch.autograd.grad(-(z.sigmoid().sum() + 1e-5) / (1 + 1e-5), z)
        self.assertAlmostEqual((gb / gr).item(), 100.001, places=2)

    def test_paper_empty_constant_has_zero_gradient(self):
        s = torch.zeros_like(self.z)
        loss = -((self.z.sigmoid() * s).sum() + 1e-5) / (s.sum() + 1e-5)
        grad, = torch.autograd.grad(loss, self.z)
        self.assertEqual(loss.item(), -1)
        self.assertEqual(torch.count_nonzero(grad).item(), 0)

    def test_fp32_reduction_from_half_inputs(self):
        z = self.z.detach().half().requires_grad_(True)
        value = normalized_positive_bce(z, self.y, self.f, self.s)
        self.assertEqual(value.dtype, torch.float32)
        grad, = torch.autograd.grad(value, z)
        self.assertTrue(torch.isfinite(grad).all().item())

    def test_all_geometry_matches_bound_crop_order(self):
        from retinal_m0.maples_core import paired_weighted_crop
        native = np.zeros((8, 9), dtype=np.uint8); native[2:6, 3:7] = 1; native[2, 3] = 0
        image = np.zeros((8, 9, 3), dtype=np.uint8)
        for hf in (False, True):
            for vf in (False, True):
                for q in range(4):
                    got = transform_native_target(native, top=1, left=2, size=6,
                                                  hflip=hf, vflip=vf, quarter_turns=q)
                    ref = paired_weighted_crop(image, native, np.ones_like(native), native.astype(np.float32),
                                               top=1, left=2, size=6, hflip=hf, vflip=vf, quarter_turns=q)
                    self.assertTrue(np.array_equal(got, ref['mask']))
                    self.assertTrue(got.flags.c_contiguous)

    def test_invalid_geometry_rejected(self):
        for patch in ({'top': -1}, {'quarter_turns': 4}, {'hflip': 1}, {'size': 100}):
            kwargs = dict(top=0, left=0, size=4, hflip=False, vflip=False, quarter_turns=0)
            kwargs.update(patch)
            with self.assertRaises(ValueError): transform_native_target(np.ones((8, 8)), **kwargs)

    def test_primitives_do_not_consume_rng(self):
        py, npstate, ts = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        branch_bce_loss(self.z, self.y, self.f, self.s)
        transform_native_target(np.ones((8, 8)), top=0, left=0, size=4,
                                hflip=True, vflip=False, quarter_turns=1)
        self.assertEqual(py, random.getstate())
        now = np.random.get_state()
        self.assertEqual(npstate[0], now[0]); self.assertTrue(np.array_equal(npstate[1], now[1]))
        self.assertEqual(npstate[2:], now[2:]); self.assertTrue(torch.equal(ts, torch.get_rng_state()))

    def test_unsigned_review_never_releases_gpu(self):
        out = human_review_gate([{'case_id': 'CASE_01'}], ['CASE_01'])
        self.assertEqual(out['signed_rows'], 0)
        self.assertFalse(out['structurally_complete_and_accepted'])
        self.assertFalse(out['gpu_release'])

    def test_even_complete_fixture_is_not_gpu_permission(self):
        fields = ('reviewer_name', 'reviewed_at', 'GT_path_matches_intended_branch', 'RGB_visibility',
                  'visually_fine_branch', 'junction_or_crossing_uncertainty', 'GT_gap_or_artifact_uncertainty',
                  'width_scale_definition_acceptable', 'B1_B2_target_alignment')
        row = {k: 'SYNTHETIC_TEST_ONLY' for k in fields}
        row.update(case_id='CASE_01', accept_pause_uncertain='accept')
        original = copy.deepcopy(row)
        out = human_review_gate([row], ['CASE_01'])
        self.assertTrue(out['structurally_complete_and_accepted'])
        self.assertFalse(out['gpu_release']); self.assertFalse(out['human_authenticity_verified'])
        self.assertEqual(row, original)

    def test_duplicate_missing_or_paused_review_blocks(self):
        for rows in ([{'case_id': 'CASE_01'}] * 2,
                     [{'case_id': 'CASE_02', 'accept_pause_uncertain': 'pause'}], []):
            self.assertFalse(human_review_gate(rows, ['CASE_01'])['structurally_complete_and_accepted'])


if __name__ == '__main__':
    unittest.main()
