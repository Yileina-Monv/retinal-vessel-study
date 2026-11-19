import unittest

import numpy as np
import torch

from retinal_m0.maples_core import (
    build_training_regions,
    paired_weighted_crop,
    weighted_masked_loss,
)
from retinal_release.data import masked_loss


class CoreRegionTests(unittest.TestCase):
    def test_regions_partition_omega_and_keep_uncertain_weight_one(self):
        fov = np.ones((31, 41), dtype=bool)
        fov[0, :] = False
        h = np.zeros_like(fov)
        h[15, 20] = True
        u = np.zeros_like(fov)
        u[15, 20] = True
        result = build_training_regions(fov, h, u, image_width=41)
        omega = fov & ~u
        membership = result["rin"].astype(np.uint8) + result["rnear"].astype(np.uint8) + result["rfar"].astype(np.uint8)
        self.assertTrue(np.array_equal(membership, omega.astype(np.uint8)))
        self.assertTrue(np.array_equal(result["weights"][u], np.ones(int(u.sum()), np.float32)))
        self.assertTrue(np.all(result["weights"][~fov] == 0))
        self.assertTrue(np.all(result["weights"][result["rin"] | result["rnear"]] == 2))

    def test_region_dilation_is_native_full_image_before_crop(self):
        fov = np.ones((61, 71), dtype=bool)
        h = np.zeros_like(fov)
        h[30, 23] = True
        u = np.zeros_like(fov)
        full = build_training_regions(fov, h, u, image_width=71)
        crop = (slice(20, 50), slice(24, 54))
        rebuilt = build_training_regions(fov[crop], h[crop], u[crop], image_width=30)
        self.assertFalse(np.array_equal(full["weights"][crop], rebuilt["weights"]))
        image = np.zeros((61, 71, 3), dtype=np.uint8)
        item = paired_weighted_crop(image, np.zeros_like(fov), fov, full["weights"],
                                    top=20, left=24, size=30, hflip=True,
                                    vflip=False, quarter_turns=1)
        expected = np.rot90(full["weights"][crop][:, ::-1], 1)
        self.assertTrue(np.array_equal(item["weight"], expected))

    def test_weighted_loss_matches_hand_computed_bce_and_unweighted_dice(self):
        logits = torch.tensor([[[[0.0, 1.0], [-1.0, 0.5]]]], requires_grad=True)
        target = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
        fov = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])
        weights = torch.tensor([[[[2.0, 1.0], [1.0, 0.0]]]])
        got = weighted_masked_loss(logits, target, fov, weights)
        z = logits.detach().float().requires_grad_(True)
        bce_map = torch.nn.functional.binary_cross_entropy_with_logits(z, target, reduction="none")
        bce = (bce_map * fov * weights).sum() / (fov * weights).sum()
        p = z.sigmoid()
        dice = 1 - (2 * (p * target * fov).sum() + 1e-6) / ((p * fov).sum() + (target * fov).sum() + 1e-6)
        expected = bce + dice
        self.assertTrue(torch.allclose(got, expected, atol=0, rtol=0))
        grad_got, = torch.autograd.grad(got, logits)
        grad_expected, = torch.autograd.grad(expected, z)
        self.assertTrue(torch.allclose(grad_got, grad_expected, atol=0, rtol=0))

    def test_all_one_weights_equal_frozen_masked_loss_value_and_gradient(self):
        torch.manual_seed(7)
        a = torch.randn(2, 1, 5, 7, requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        target = torch.randint(0, 2, (2, 1, 5, 7)).float()
        fov = torch.randint(0, 2, (2, 1, 5, 7)).float()
        fov[:, :, 0, 0] = 1
        weights = fov.clone()
        lhs = weighted_masked_loss(a, target, fov, weights)
        rhs = masked_loss(b, target, fov, epsilon=1e-6)
        ga, = torch.autograd.grad(lhs, a)
        gb, = torch.autograd.grad(rhs, b)
        self.assertTrue(torch.equal(lhs, rhs))
        self.assertTrue(torch.equal(ga, gb))


if __name__ == "__main__":
    unittest.main()
