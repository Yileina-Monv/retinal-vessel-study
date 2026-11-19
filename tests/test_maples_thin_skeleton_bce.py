import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retinal_m0.maples_thin_skeleton_bce_cpu_gate import s3_target, synthetic_checks, thin_bce_loss
import numpy as np
import torch


class ThinSkeletonCpuGateTests(unittest.TestCase):
    def test_frozen_synthetic_gate(self):
        self.assertTrue(synthetic_checks()['passed'])

    def test_target_empty_and_single_line(self):
        v = np.zeros((9, 9), dtype=bool)
        f = np.ones_like(v)
        g, sk, radius, s = s3_target(v, f)
        self.assertEqual(int(s.sum()), 0)
        v[4, 2:7] = True
        g, sk, radius, s = s3_target(v, f)
        self.assertTrue(np.all(s <= g))
        self.assertTrue(s.any())

    def test_auxiliary_gradient_is_zero_for_empty_target(self):
        z = torch.zeros((4, 1, 3, 3), requires_grad=True)
        y = torch.zeros_like(z)
        f = torch.ones_like(z)
        s = torch.zeros_like(z)
        loss = thin_bce_loss(z, y, f, s, .1)
        grad, = torch.autograd.grad(loss, z)
        # The total includes the base loss; compare against the exact T0 gradient.
        z0 = torch.zeros((4, 1, 3, 3), requires_grad=True)
        base = thin_bce_loss(z0, y, f, s, 0)
        basegrad, = torch.autograd.grad(base, z0)
        self.assertTrue(torch.equal(grad, basegrad))


if __name__ == '__main__':
    unittest.main()
