import math
import unittest

import torch
import torch.nn.functional as F

from retinal_m0.maples_core import weighted_masked_loss
from retinal_m0.maples_focal_dice import focal_dice_loss, focal_dice_terms


class FocalDiceTests(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(20261024)
        self.z = torch.randn(3, 1, 8, 8, generator=gen, requires_grad=True)
        self.y = (torch.rand(3, 1, 8, 8, generator=gen) > .7).float()
        self.f = torch.ones_like(self.y)
        self.f[1, :, :4] = 0

    def grad(self, loss, z=None):
        return torch.autograd.grad(loss, self.z if z is None else z, retain_graph=True)[0]

    def test_gamma_zero_exact_value_and_logit_gradient(self):
        a = weighted_masked_loss(self.z, self.y, self.f, self.f)
        b = focal_dice_loss(self.z, self.y, self.f, gamma=0)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(self.grad(a), self.grad(b)))

    def test_gamma_zero_exact_real_parameter_gradient_small_network(self):
        torch.manual_seed(1)
        net = torch.nn.Conv2d(3, 1, 1)
        z = net(torch.randn(3, 3, 8, 8))
        a = weighted_masked_loss(z, self.y, self.f, self.f)
        b = focal_dice_loss(z, self.y, self.f, gamma=0)
        ga = torch.autograd.grad(a, tuple(net.parameters()), retain_graph=True)
        gb = torch.autograd.grad(b, tuple(net.parameters()))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(ga, gb)))

    def test_independent_probability_formula(self):
        t = focal_dice_terms(self.z, self.y, self.f)
        p = self.z.sigmoid()
        pt = torch.where(self.y.bool(), p, 1-p)
        expected = ((1-pt)*F.binary_cross_entropy_with_logits(
            self.z, self.y, reduction='none')*self.f).sum((1, 2, 3))/self.f.sum((1, 2, 3))
        torch.testing.assert_close(t['per_crop_focal'], expected, atol=1e-7, rtol=1e-6)

    def test_modulation_is_not_detached(self):
        p = self.z.sigmoid()
        q = torch.where(self.y.bool(), 1-p, p)
        ce = F.binary_cross_entropy_with_logits(self.z, self.y, reduction='none')
        detached = ((q.detach()*ce*self.f).sum((1,2,3))/self.f.sum((1,2,3))).mean()
        actual = focal_dice_terms(self.z, self.y, self.f)['per_crop_focal'].mean()
        self.assertGreater(float((self.grad(actual)-self.grad(detached)).abs().max()), 1e-4)

    def test_analytic_focal_gradient(self):
        t = focal_dice_terms(self.z, self.y, self.f)
        q = torch.sigmoid(-(2*self.y-1)*self.z)
        pt = 1-q
        ce = F.binary_cross_entropy_with_logits(self.z, self.y, reduction='none')
        expected = (self.z.sigmoid()-self.y)*(q+pt*ce)*self.f
        expected /= self.f.sum((1,2,3))[:,None,None,None]*len(self.y)
        torch.testing.assert_close(self.grad(t['per_crop_focal'].mean()), expected, atol=1e-8, rtol=1e-5)

    def test_original_dice_arithmetic_unchanged(self):
        t = focal_dice_terms(self.z, self.y, self.f)
        torch.testing.assert_close((t['per_crop_bce']+t['per_crop_dice']).mean(),
                                  weighted_masked_loss(self.z, self.y, self.f, self.f), rtol=0, atol=0)

    def test_per_crop_mean_and_gradient(self):
        batch = focal_dice_loss(self.z, self.y, self.f)
        singles = torch.stack([focal_dice_loss(self.z[i:i+1], self.y[i:i+1], self.f[i:i+1])
                               for i in range(3)]).mean()
        torch.testing.assert_close(batch, singles, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(self.grad(batch), self.grad(singles), atol=1e-8, rtol=1e-5)

    def test_outside_fov_zero_gradient_and_invariance(self):
        a = focal_dice_loss(self.z, self.y, self.f)
        z2 = torch.where(self.f.bool(), self.z, self.z+30)
        b = focal_dice_loss(z2, self.y, self.f)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.all(self.grad(a)[self.f==0] == 0))

    def test_empty_gt_still_penalizes_false_positive(self):
        y = torch.zeros_like(self.y)
        t = focal_dice_terms(self.z, y, self.f)
        self.assertTrue(torch.all(t['per_crop_focal'] > 0))
        self.assertTrue(torch.all(self.grad(t['loss'])[self.f.bool()] > 0))

    def test_single_positive_gt(self):
        y = torch.zeros_like(self.y); y[:, :, 5, 5] = 1
        a = focal_dice_loss(self.z, y, self.f)
        self.assertTrue(torch.isfinite(a) and torch.isfinite(self.grad(a)).all())

    def test_all_foreground(self):
        a = focal_dice_loss(self.z, torch.ones_like(self.y), self.f)
        self.assertTrue(torch.all(self.grad(a)[self.f.bool()] < 0))

    def test_extreme_logits_finite(self):
        z = torch.tensor([-10000., -100., 0., 100., 10000.]).reshape(1,1,1,5).requires_grad_()
        for y in [torch.zeros_like(z), torch.ones_like(z)]:
            a = focal_dice_loss(z, y, torch.ones_like(z))
            self.assertTrue(torch.isfinite(a) and torch.isfinite(self.grad(a,z)).all())

    def test_correct_predictions_approach_zero(self):
        z = (2*self.y-1)*100
        self.assertLess(float(focal_dice_loss(z, self.y, self.f)), 1e-6)

    def test_wrong_predictions_large_loss(self):
        z = -(2*self.y-1)*100
        self.assertGreater(float(focal_dice_loss(z, self.y, self.f)), 99)

    def test_empty_fov_rejected_even_if_other_crop_valid(self):
        f = self.f.clone(); f[0] = 0
        with self.assertRaisesRegex(ValueError,'empty_fov'): focal_dice_loss(self.z,self.y,f)

    def test_binary_masks_required(self):
        for y,f in [(self.y+.1,self.f),(self.y,self.f+.1)]:
            with self.assertRaises(ValueError): focal_dice_loss(self.z,y,f)

    def test_shape_and_channel_rejection(self):
        for z,y,f in [(self.z[:,0],self.y[:,0],self.f[:,0]),
                      (self.z.expand(-1,2,-1,-1),self.y.expand(-1,2,-1,-1),self.f.expand(-1,2,-1,-1)),
                      (self.z,self.y[:1],self.f)]:
            with self.assertRaises(ValueError): focal_dice_loss(z,y,f)

    def test_nonfinite_rejected(self):
        for val in [float('nan'),float('inf')]:
            z=self.z.detach().clone(); z[0,0,0,0]=val
            with self.assertRaises(ValueError): focal_dice_loss(z,self.y,self.f)

    def test_no_gamma_scan_or_boolean(self):
        for gamma in [-1,.5,2,True,float('nan'),'1']:
            with self.assertRaises(ValueError): focal_dice_loss(self.z,self.y,self.f,gamma=gamma)

    def test_fp32_loss_under_cpu_autocast(self):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            a=focal_dice_loss(self.z,self.y,self.f)
        self.assertEqual(a.dtype,torch.float32)
        self.assertTrue(torch.equal(a,focal_dice_loss(self.z,self.y,self.f)))

    def test_finite_difference_total_loss(self):
        z=torch.tensor([-.7,.2,1.1,-1.3]).reshape(1,1,2,2).requires_grad_()
        y=torch.tensor([0.,1.,0.,1.]).reshape_as(z); f=torch.ones_like(z)
        gradient=self.grad(focal_dice_loss(z,y,f),z).reshape(-1)
        # FP32 public primitive: h chosen above reduction-roundoff scale.
        h=1e-3
        for i in range(4):
            left=z.detach().clone();right=z.detach().clone()
            left.reshape(-1)[i]-=h;right.reshape(-1)[i]+=h
            fd=(float(focal_dice_loss(right,y,f))-float(focal_dice_loss(left,y,f)))/(2*h)
            self.assertAlmostEqual(fd,float(gradient[i]),delta=1.5e-4)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
