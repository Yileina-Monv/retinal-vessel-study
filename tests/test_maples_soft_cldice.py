"""CPU-only candidate mathematical tests; no optimizer.step or GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import unittest
from unittest import mock
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from retinal_m0.maples_core import weighted_masked_loss
from retinal_m0.maples_soft_cldice import (candidate_loss, soft_cldice_terms,
    soft_erode_zero, soft_skeleton, calibrate_eta)

torch.set_num_threads(2)


class SoftClDiceTests(unittest.TestCase):
    def case(self, batch=1):
        g=torch.zeros(batch,1,32,32);g[:,:,16,4:28]=1
        f=torch.ones_like(g)
        p=g*.9+(1-g)*.05
        return torch.logit(p).requires_grad_(),g,f

    def test_zero_exact_value_and_logit_gradient(self):
        z,g,f=self.case();a=weighted_masked_loss(z,g,f,f)
        with mock.patch('retinal_m0.maples_soft_cldice.soft_cldice_terms',side_effect=AssertionError('must bypass')):
            b=candidate_loss(z,g,f,k=8,eta=0)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(torch.autograd.grad(a,z)[0],torch.autograd.grad(b,z)[0]))

    def test_zero_real_small_model_parameters(self):
        from retinal_m0.model import UNet
        torch.manual_seed(3);m=UNet((8,16),groups=8)
        x=torch.randn(1,3,32,32);z=m(x);_,g,f=self.case()
        a=weighted_masked_loss(z,g,f,f);b=candidate_loss(z,g,f,k=8,eta=0)
        ga=torch.autograd.grad(a,tuple(m.parameters()),retain_graph=True)
        gb=torch.autograd.grad(b,tuple(m.parameters()))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(ga,gb)))
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_empty_gt_exact_zero_gradient(self):
        z,g,f=self.case();g.zero_();r=soft_cldice_terms(z,g,f,k=8)
        self.assertEqual(r['loss'].item(),0.)
        self.assertEqual(torch.count_nonzero(torch.autograd.grad(r['loss'],z)[0]).item(),0)

    def test_fixed_batch_mean_including_empty(self):
        z,g,f=self.case(2);g[1]=0
        a=soft_cldice_terms(z,g,f,k=8)['loss']
        b=soft_cldice_terms(z[:1],g[:1],f[:1],k=8)['loss']
        self.assertTrue(torch.equal(a,b/2))

    def test_fov_empty_error(self):
        z,g,f=self.case();f.zero_()
        with self.assertRaisesRegex(ValueError,'empty_fov'):candidate_loss(z,g,f,k=8,eta=0)

    def test_fov_outside_logits_no_aux_gradient(self):
        z,g,f=self.case();f[:,:,:,:3]=0
        q=torch.autograd.grad(soft_cldice_terms(z,g,f,k=8)['loss'],z)[0]
        self.assertEqual(torch.count_nonzero(q[f==0]).item(),0)

    def test_nonfinite_rejected(self):
        z,g,f=self.case();z=z.detach();z[0,0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):soft_cldice_terms(z,g,f,k=8)

    def test_nonbinary_rejected(self):
        z,g,f=self.case();g[0,0,0,0]=.5
        with self.assertRaisesRegex(ValueError,'binary'):soft_cldice_terms(z,g,f,k=8)

    def test_single_channel_only(self):
        z,g,f=self.case()
        with self.assertRaises(ValueError):soft_cldice_terms(z.repeat(1,2,1,1),g.repeat(1,2,1,1),f.repeat(1,2,1,1),k=8)

    def test_negative_eta_and_wrong_k(self):
        z,g,f=self.case()
        for kw in [dict(k=8,eta=-1),dict(k=0,eta=.1),dict(k=True,eta=.1),dict(k=8,eta=float('inf'))]:
            with self.assertRaises(ValueError):candidate_loss(z,g,f,**kw)

    def test_perfect_hard_probabilities(self):
        _,g,f=self.case();z=torch.where(g.bool(),torch.tensor(100.),torch.tensor(-100.)).requires_grad_()
        r=soft_cldice_terms(z,g,f,k=8)
        self.assertLess(r['loss'].item(),1e-6)
        self.assertTrue(torch.isfinite(torch.autograd.grad(r['loss'],z)[0]).all())

    def test_missing_segment_penalty_and_direction(self):
        z,g,f=self.case();a=soft_cldice_terms(z,g,f,k=8)['loss'].item()
        zz=z.detach().clone();zz[:,:,16,12:19]=-5;zz.requires_grad_()
        b=soft_cldice_terms(zz,g,f,k=8)['loss'];q=torch.autograd.grad(b,zz)[0]
        self.assertGreater(b.item(),a)
        self.assertLess(q[:,:,16,12:19].sum().item(),0)

    def test_false_axis_penalty_and_direction(self):
        z,g,f=self.case();a=soft_cldice_terms(z,g,f,k=8)['loss'].item()
        zz=z.detach().clone();zz[:,:,6,5:25]=3;zz.requires_grad_()
        b=soft_cldice_terms(zz,g,f,k=8)['loss'];q=torch.autograd.grad(b,zz)[0]
        self.assertGreater(b.item(),a)
        self.assertGreater(q[:,:,6,5:25].sum().item(),0)

    def test_single_point_sparse_finite(self):
        z,g,f=self.case();g.zero_();g[:,:,16,16]=1
        a=soft_cldice_terms(z,g,f,k=8)['loss'];q=torch.autograd.grad(a,z)[0]
        self.assertTrue(torch.isfinite(a) and torch.isfinite(q).all())

    def test_constant_and_extreme_logits_finite(self):
        for value in [-100.,0.,100.]:
            z,g,f=self.case();z=torch.full_like(z,value,requires_grad=True)
            a=soft_cldice_terms(z,g,f,k=8)['loss'];q=torch.autograd.grad(a,z)[0]
            self.assertTrue(torch.isfinite(a) and torch.isfinite(q).all())

    def test_explicit_zero_erosion_matches_scipy(self):
        x=np.ones((9,11),bool);x[4,5]=0
        ref=binary_erosion(x,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]),border_value=0)
        out=soft_erode_zero(torch.tensor(x)[None,None].float())[0,0].numpy()
        self.assertTrue(np.array_equal(ref,out))

    def test_k_is_effective(self):
        x=torch.zeros(1,1,21,21);x[:,:,3:18,3:18]=1
        a=soft_skeleton(x,1);b=soft_skeleton(x,10)
        self.assertFalse(torch.equal(a,b));self.assertGreater(b.sum().item(),a.sum().item())

    def test_rotations_flip_equivariant(self):
        z,g,f=self.case();a=soft_cldice_terms(z,g,f,k=8)['loss']
        for axis in [(2,),(3,)]:
            b=soft_cldice_terms(z.flip(axis),g.flip(axis),f.flip(axis),k=8)['loss']
            self.assertTrue(torch.allclose(a,b,atol=1e-7,rtol=1e-6))
        b=soft_cldice_terms(z.rot90(1,[2,3]),g.rot90(1,[2,3]),f.rot90(1,[2,3]),k=8)['loss']
        self.assertTrue(torch.allclose(a,b,atol=1e-7,rtol=1e-6))

    def test_fp32_under_cpu_autocast(self):
        z,g,f=self.case()
        with torch.autocast('cpu',dtype=torch.bfloat16):r=soft_cldice_terms(z,g,f,k=8)
        self.assertEqual(r['loss'].dtype,torch.float32)
        self.assertTrue(torch.equal(r['loss'],soft_cldice_terms(z,g,f,k=8)['loss']))

    def test_batch_composition_does_not_reweight_valid_crop(self):
        z,g,f=self.case(3);g[1:]=0
        a=torch.autograd.grad(soft_cldice_terms(z,g,f,k=8)['loss'],z)[0]
        zz=z[:1].detach().requires_grad_();b=torch.autograd.grad(soft_cldice_terms(zz,g[:1],f[:1],k=8)['loss'],zz)[0]
        self.assertTrue(torch.allclose(a[:1]*3,b,atol=1e-7,rtol=1e-5))

    def test_eta_train_only_policy(self):
        self.assertEqual(calibrate_eta([1,2,3,4,5,6,7,8]),.1/8)
        self.assertEqual(calibrate_eta([.001]*8),1.)
        for x in [[1]*7,[0]*8,[float('nan')]*8]:
            with self.assertRaises(ValueError):calibrate_eta(x)

    def test_finite_difference_away_from_pool_ties(self):
        torch.manual_seed(77);z,g,f=self.case();z=(z.detach()+.2*torch.rand_like(z)).requires_grad_()
        q=torch.autograd.grad(soft_cldice_terms(z,g,f,k=8)['loss'],z)[0]
        pos=(0,0,16,14);delta=.002
        with torch.no_grad():
            zp=z.clone();zm=z.clone();zp[pos]+=delta;zm[pos]-=delta
            fd=(soft_cldice_terms(zp,g,f,k=8)['loss']-soft_cldice_terms(zm,g,f,k=8)['loss'])/(2*delta)
        self.assertAlmostEqual(q[pos].item(),fd.item(),delta=5e-5)

if __name__=='__main__':unittest.main(verbosity=2)
