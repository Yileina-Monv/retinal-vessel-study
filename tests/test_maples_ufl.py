"""Numerical, class-direction and reduction tests; no real retinal model/GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import unittest
import numpy as np
import torch
from retinal_m0.maples_ufl import ufl_terms, ufl_loss, PARAMETERS

torch.set_num_threads(2)


def numpy_oracle(z,y,f):
    p=1/(1+np.exp(-z));q=np.clip(np.concatenate([1-p,p],1),1e-7,1-1e-7)
    t=np.concatenate([1-y,y],1);tp=(f*t*q).sum((2,3))
    fn=(f*t*(1-q)).sum((2,3));fp=(f*(1-t)*q).sum((2,3))
    ti=(tp+1e-7)/(tp+.6*fn+.4*fp+1e-7)
    afl=(-( .6*y[:,0]*np.log(q[:,1]) + .4*(1-y[:,0])*(1-q[:,0])**.5*np.log(q[:,0]))*f[:,0]).sum((1,2))/f.sum((1,2,3))
    aft=((1-ti[:,0])+(1-ti[:,1])**.5)/2
    return afl,aft,(.5*afl+.5*aft).mean()


class UFLTests(unittest.TestCase):
    def inputs(self,dtype=torch.float64):
        g=torch.Generator().manual_seed(245)
        z=torch.randn(3,1,7,9,generator=g,dtype=dtype).requires_grad_()
        y=(torch.rand(3,1,7,9,generator=g)>.7).to(dtype)
        f=torch.ones_like(y);f[1,:,:2,:]=0;f[2,:,:,:4]=0
        return z,y,f

    def test_fixed_author_parameters(self):
        self.assertEqual(PARAMETERS,{'lambda_code':.5,'delta':.6,'gamma':.5,'epsilon':1e-7})

    def test_independent_numpy_formula_and_both_components(self):
        z,y,f=self.inputs();a=ufl_terms(z,y,f);fl,ft,total=numpy_oracle(z.detach().numpy(),y.numpy(),f.numpy())
        np.testing.assert_allclose(a['per_crop_afl'].detach(),fl,rtol=0,atol=1e-10)
        np.testing.assert_allclose(a['per_crop_aft'].detach(),ft,rtol=0,atol=1e-10)
        self.assertAlmostEqual(float(a['loss']),float(total),places=12)

    def test_autograd_finite_difference(self):
        z,y,f=self.inputs();self.assertTrue(torch.autograd.gradcheck(ufl_loss,(z,y,f),eps=1e-6,atol=1e-6,rtol=1e-4))

    def test_sigmoid_head_equivalent_to_two_class_softmax(self):
        z,y,f=self.inputs();q=torch.softmax(torch.cat([torch.zeros_like(z),z],1),1)
        self.assertTrue(torch.allclose(q[:,1:2],z.sigmoid(),atol=1e-15,rtol=1e-14))
        self.assertTrue(torch.allclose(q[:,0:1],1-z.sigmoid(),atol=1e-15,rtol=1e-14))

    def test_TP_FP_FN_direction_foreground_and_background(self):
        p=torch.tensor([.8,.1,.7,.2],dtype=torch.float64).reshape(1,1,2,2)
        y=torch.tensor([1,1,0,0],dtype=torch.float64).reshape_as(p);f=torch.ones_like(y)
        t=ufl_terms(torch.logit(p),y,f)
        self.assertAlmostEqual(float(t['TP'][0,1]),.9)
        self.assertAlmostEqual(float(t['FN'][0,1]),1.1)
        self.assertAlmostEqual(float(t['FP'][0,1]),.9)
        self.assertAlmostEqual(float(t['FN'][0,0]),.9)
        self.assertAlmostEqual(float(t['FP'][0,0]),1.1)
        self.assertAlmostEqual(float(t['TI'][0,1]),(.9+1e-7)/(.9+.6*1.1+.4*.9+1e-7))

    def test_fov_outside_values_and_gradients_irrelevant(self):
        z,y,f=self.inputs();v=ufl_loss(z,y,f);g=torch.autograd.grad(v,z)[0]
        self.assertTrue(torch.all(g[f==0]==0))
        zz=z.detach().clone();yy=y.clone();zz[f==0]=1000;yy[f==0]=1-yy[f==0]
        self.assertEqual(float(v),float(ufl_loss(zz,yy,f)))

    def test_fov_not_full_crop_denominator(self):
        z,y,f=self.inputs();a=ufl_loss(z,y,f)
        b=[]
        for i in range(len(z)):
            mask=f[i].bool();b.append(ufl_loss(z[i][mask].reshape(1,1,1,-1),y[i][mask].reshape(1,1,1,-1),torch.ones(1,1,1,int(mask.sum()),dtype=z.dtype)))
        self.assertTrue(torch.allclose(a,torch.stack(b).mean(),atol=1e-14,rtol=1e-14))

    def test_batch_mean_not_pooled_TI(self):
        z,y,f=self.inputs();a=ufl_loss(z,y,f)
        b=torch.stack([ufl_loss(z[i:i+1],y[i:i+1],f[i:i+1]) for i in range(3)]).mean()
        self.assertEqual(float(a),float(b))
        ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z)[0]
        self.assertTrue(torch.allclose(ga,gb,atol=1e-15,rtol=1e-14))

    def test_empty_foreground_retained_and_background_penalized(self):
        z=torch.zeros(2,1,5,7,requires_grad=True);y=torch.zeros_like(z);f=torch.ones_like(z)
        a=ufl_terms(z,y,f);g=torch.autograd.grad(a['loss'],z)[0]
        self.assertTrue(torch.all(a['per_crop_afl']>0));self.assertTrue(torch.all(g>0))

    def test_empty_background_finite_and_correct_gradient(self):
        z=torch.zeros(2,1,5,7,requires_grad=True);y=torch.ones_like(z);f=torch.ones_like(z)
        a=ufl_terms(z,y,f);g=torch.autograd.grad(a['loss'],z)[0]
        self.assertTrue(torch.isfinite(g).all());self.assertTrue(torch.all(g<0))

    def test_extreme_logits_finite_for_both_empty_classes(self):
        for val in [-1000.,1000.]:
            for label in [0.,1.]:
                z=torch.full((2,1,16,17),val,requires_grad=True);y=torch.full_like(z,label);f=torch.ones_like(z)
                a=ufl_loss(z,y,f);g=torch.autograd.grad(a,z)[0]
                self.assertTrue(torch.isfinite(a));self.assertTrue(torch.isfinite(g).all())

    def test_float32_fp64_loss_and_gradient(self):
        z,y,f=self.inputs();a=ufl_loss(z,y,f);g=torch.autograd.grad(a,z)[0]
        z32=z.detach().float().requires_grad_();b=ufl_loss(z32,y.float(),f.float());g32=torch.autograd.grad(b,z32)[0]
        torch.testing.assert_close(a.float(),b,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(g.float(),g32,atol=1e-6,rtol=1e-4)

    def test_half_inputs_reduce_in_FP32(self):
        z,y,f=self.inputs(torch.float32);z=z.detach().half().requires_grad_()
        a=ufl_terms(z,y.half(),f.half());self.assertEqual(a['loss'].dtype,torch.float32)
        self.assertTrue(torch.isfinite(torch.autograd.grad(a['loss'],z)[0]).all())

    def test_empty_fov_rejected_including_one_bad_crop(self):
        z,y,f=self.inputs();f[1]=0
        with self.assertRaisesRegex(ValueError,'empty_fov'):ufl_loss(z,y,f)

    def test_shapes_binary_and_nonfinite_rejected(self):
        z,y,f=self.inputs()
        for zz,yy,ff in [(z.repeat(1,2,1,1),y,f),(z,y[:1],f),(z,y+.5,f),(z,y,f+.1),(z*float('nan'),y,f)]:
            with self.assertRaises(ValueError):ufl_loss(zz,yy,ff)

    def test_component_gradients_sum_without_detachment(self):
        z,y,f=self.inputs();a=ufl_terms(z,y,f)
        ga=torch.autograd.grad(.5*a['per_crop_afl'].mean(),z,retain_graph=True)[0]
        gt=torch.autograd.grad(.5*a['per_crop_aft'].mean(),z,retain_graph=True)[0]
        g=torch.autograd.grad(a['loss'],z)[0]
        self.assertTrue(torch.allclose(ga+gt,g,atol=1e-15,rtol=1e-13))

if __name__=='__main__':unittest.main(verbosity=2)
