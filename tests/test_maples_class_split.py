import unittest
import numpy as np
import torch
from retinal_m0.maples_class_split import (
    build_class_split_regions, class_split_weights,
    paired_class_split_crop, class_split_loss_with_contributions,
)
from retinal_m0.maples_core import weighted_masked_loss


class ClassSplitPrimitiveTests(unittest.TestCase):
    def test_mapping_keeps_explicit_A_when_background_weight_is_one(self):
        fov = np.ones((4, 6), dtype=bool); fov[0, 0] = False
        vessel = np.zeros((4, 6), dtype=bool); vessel[1, 1] = True; vessel[2, 2] = True
        area = np.zeros_like(fov); area[1, 1:3] = True; area[2, 2] = True
        w = class_split_weights(fov, vessel, area)
        self.assertEqual(w[1, 1], 2); self.assertEqual(w[1, 2], 1)
        self.assertEqual(w[2, 2], 2); self.assertEqual(w[0, 0], 0)
        self.assertTrue(np.all(w[fov & ~area] == 1))
        self.assertTrue(area[1, 2] and w[1, 2] == 1)

    def test_one_one_and_two_two_match_frozen_loss_and_logits_gradients(self):
        torch.manual_seed(19)
        z0 = torch.randn(2, 1, 5, 7, dtype=torch.float32)
        y = (torch.rand_like(z0) > .72).float()
        f = (torch.rand_like(z0) > .16).float()
        a = (torch.rand_like(z0) > .48).float() * f
        for vw, bw in ((1., 1.), (2., 2.)):
            weight = f * (1 + a * ((vw - 1) * y + (bw - 1) * (1 - y)))
            z_old = z0.clone().requires_grad_(True)
            old = weighted_masked_loss(z_old, y, f, weight)
            old_grad, = torch.autograd.grad(old, z_old)
            z_new = z0.clone().requires_grad_(True)
            new, _, _, _ = class_split_loss_with_contributions(
                z_new, y, f, a, vessel_weight=vw, background_weight=bw)
            new_grad, = torch.autograd.grad(new, z_new)
            torch.testing.assert_close(new, old, rtol=0, atol=1e-7)
            torch.testing.assert_close(new_grad, old_grad, rtol=0, atol=1e-7)

    def test_two_one_uses_protocol_denominator_and_unweighted_dice(self):
        z = torch.tensor([[[[0., 1., -1., .5]]]], requires_grad=True)
        y = torch.tensor([[[[1., 0., 1., 0.]]]])
        f = torch.tensor([[[[1., 1., 1., 0.]]]])
        a = torch.tensor([[[[1., 1., 0., 0.]]]])
        total, bce, dice, parts = class_split_loss_with_contributions(z, y, f, a)
        raw = torch.nn.functional.binary_cross_entropy_with_logits(z.float(), y, reduction='none')
        expected_d = 4.  # nF=3 and n(A_V1)=1; A_V0 has weight 1.
        expected_bce = (raw * f * parts['weights']).sum() / expected_d
        expected_dice = 1 - (2 * (z.sigmoid() * y * f).sum() + 1e-6) / (
            (z.sigmoid() * f).sum() + (y * f).sum() + 1e-6)
        torch.testing.assert_close(parts['D'], torch.tensor([expected_d]))
        torch.testing.assert_close(bce, expected_bce.reshape(1))
        torch.testing.assert_close(dice, expected_dice.reshape(1))
        torch.testing.assert_close(total, (expected_bce + expected_dice).reshape(1).mean())
        self.assertEqual(parts['nA_V0'].item(), 1)
        self.assertEqual(parts['groups']['A_V0']['n'].item(), 1)
        self.assertGreater(parts['groups']['A_V0']['C'].item(), 0)

    def test_crop_transforms_A_with_rgb_vessel_fov_and_weight(self):
        image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        fov = np.ones((16, 16), dtype=bool); vessel = np.zeros_like(fov)
        vessel[4:8, 4:8] = True; area = np.zeros_like(fov); area[3:9, 3:9] = True
        got = paired_class_split_crop(image, vessel, fov, area, top=0, left=0, size=16,
            hflip=True, vflip=False, quarter_turns=1)
        self.assertTrue(np.array_equal(got['weight'], class_split_weights(
            got['fov'], got['mask'], got['area'])))
        self.assertTrue(np.array_equal(got['area'], np.rot90(area[:, ::-1], 1)))

    def test_empty_fov_fails_without_resampling(self):
        with self.assertRaisesRegex(ValueError, 'empty_fov'):
            class_split_loss_with_contributions(torch.zeros(1, 1, 2, 2),
                torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))

    def test_full_native_regions_keep_uncertain_at_unit_weight(self):
        fov = np.ones((9, 9), dtype=bool); h = np.zeros_like(fov); h[4, 4] = True
        u = np.zeros_like(fov); u[4, 5] = True
        regions = build_class_split_regions(fov, h, u, image_width=9)
        self.assertTrue(regions['area'][4, 4]); self.assertFalse(regions['area'][4, 5])
        vessel = np.zeros_like(fov); vessel[4, 4] = True
        weights = class_split_weights(fov, vessel, regions['area'])
        self.assertEqual(weights[4, 4], 2); self.assertEqual(weights[4, 5], 1)


if __name__ == '__main__':
    unittest.main()

class ClassSplitDatasetTests(unittest.TestCase):
    def test_real_seed_draws_match_old_t0_inputs(self):
        from retinal_m0.maples_class_split import ClassSplitPatches
        from retinal_m0.maples_core import CorePatches
        from retinal_m0.maples_core_experiment import config
        c = config(2026, 'T0', False)
        old = CorePatches(c, 'T0')
        new = ClassSplitPatches(c)
        for draw in (0, 1, 3, 4, 7, 7199, 7200, 7999):
            a, b = old[draw], new[draw]
            self.assertEqual(a['input_sha256'], b['input_sha256'])
            self.assertEqual(a['filename'], b['filename'])
            self.assertTrue(torch.equal(a['geometry'], b['geometry']))
            self.assertTrue(torch.equal(a['image'], b['image']))
            self.assertTrue(torch.equal(a['mask'], b['mask']))
            self.assertTrue(torch.equal(a['fov'], b['fov']))
            self.assertTrue(torch.equal(b['area'] * b['mask'], (b['weight'] == 2).float()))
