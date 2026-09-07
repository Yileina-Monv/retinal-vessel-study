import unittest
import numpy as np
import torch
from retinal_m1.maps import targets,interior
from retinal_m1.loss import m1_loss
from retinal_m1.data import transform
from retinal_m1.metrics import score,aggregate
from retinal_m1.inference import probability

class PretrialTests(unittest.TestCase):
    def test_ticket_binds_method_and_addon_and_rejects_reassignment(self):
        from unittest.mock import patch
        import uuid
        from retinal_release.common import ROOT,machine
        from retinal_release.exchange import check_ticket
        from retinal_m1 import jobs
        state=ROOT/'.runtime/m1_test_fixtures'/uuid.uuid4().hex
        with patch.object(jobs,'verify',return_value={'parent_release_id':'parent','addon_id':'addon'}):
            ticket=jobs.issue('m1',machine(),2026,10,'fixture-gpu','M1',.1,state=state)
            check_ticket(ticket)
            with self.assertRaises(ValueError):jobs.issue('m1',machine(),2026,10,'fixture-gpu','M1',.1,state=state)
            changed=dict(ticket,addon_id='different')
            with self.assertRaises(ValueError):check_ticket(changed)
            with self.assertRaises(ValueError):jobs.issue('bad',machine(),2026,10,'fixture-gpu','M0',1.,state=state)

    def test_maps_subset_radii_empty_and_full(self):
        for y in (np.zeros((17,17),bool),np.ones((17,17),bool),np.eye(17,dtype=bool)):
            f=np.ones_like(y);f[:2]=0;a=targets(y,f)
            self.assertFalse(np.any(a['tube'] & ~(y&f)))
            self.assertFalse(np.any(a['skeleton'] & ~a['tube']))
            self.assertTrue(np.all(a['tube_radius'][a['tube']]>0))
            self.assertTrue(np.isfinite(a['skeleton_radius']).all())
        self.assertFalse(targets(np.ones((5,5)),np.zeros((5,5)))['skeleton'].any())

    def test_fov_boundary_kept_primary_and_interior_sensitivity(self):
        y=np.zeros((32,32),bool);y[16,2:29]=1
        f=np.zeros_like(y);f[4:28,4:28]=1;a=targets(y,f)
        self.assertTrue(a['skeleton'][16,4]);self.assertFalse(interior(f)[16,4])
        self.assertTrue(interior(f)[16,6]);self.assertFalse(a['skeleton'][16,3])

    def test_geometry_all_flip_rotation_combinations(self):
        a=np.arange(8*8).reshape(8,8)
        for hf in (0,1):
            for vf in (0,1):
                for k in range(4):
                    expected=a[1:5,2:6]
                    if hf:expected=np.flip(expected,axis=1)
                    if vf:expected=np.flip(expected,axis=0)
                    expected=np.rot90(expected,k)
                    np.testing.assert_array_equal(transform(a,[1,2,4,4,hf,vf,k]),expected)

    def test_loss_accumulation_unequal_valid_denominators(self):
        torch.manual_seed(7);x=torch.randn(4,1,8,8,requires_grad=True)
        y=torch.ones_like(x);f=torch.ones_like(x);f[0]=0
        tube=torch.zeros_like(x);tube[1:3,:,2:6,3:5]=1
        loss,log=m1_loss(x,y,f,tube,coefficient=1.)
        loss.backward();expected=x.grad.clone();x.grad=None
        for start in (0,1,3):
            end={0:1,1:3,3:4}[start]
            part,_=m1_loss(x[start:end],y[start:end],f[start:end],tube[start:end],coefficient=1.,total_images=4,total_valid=2)
            part.backward()
        torch.testing.assert_close(x.grad,expected,rtol=1e-5,atol=1e-7)
        self.assertEqual(log['empty_skeleton_crops'],2);self.assertEqual(float(x.grad[0].abs().sum()),0.)
        z=torch.zeros_like(x)
        loss,log=m1_loss(x,y,f,z,coefficient=1.);self.assertTrue(torch.isfinite(loss));self.assertEqual(float(log['skeleton']),0.)
        with self.assertRaises(ValueError):m1_loss(x,y,f,tube,coefficient=1.,total_images=4)

    def line(self):
        y=np.zeros((40,40),bool);y[20,5:35]=1
        return y,np.ones_like(y)

    def test_gap_and_slide_boundary_gap(self):
        y,f=self.line();perfect=score(y.astype(float),y,f,y)
        p=y.copy();p[20,18:23]=0;r=score(p.astype(float),y,f,y)
        self.assertLess(r['thin_recall_d1'],perfect['thin_recall_d1']);self.assertLess(r['cldice'],1.)
        self.assertLess(r['recall'],1.)

    def test_shift_tolerance(self):
        y,f=self.line();p=np.roll(y,1,axis=0);r=score(p.astype(float),y,f,y)
        self.assertEqual(r['thin_recall_d0'],0.);self.assertEqual(r['thin_recall_d1'],1.)
        p=np.roll(y,2,axis=0);r=score(p.astype(float),y,f,y)
        self.assertEqual(r['thin_recall_d1'],0.);self.assertEqual(r['thin_recall_d2'],1.)

    def test_bridge_and_thickening_do_not_fake_precision(self):
        y,f=self.line();y[12,5:35]=1;p=y.copy();p[12:21,20]=1
        r=score(p.astype(float),y,f,y);self.assertLess(r['precision'],1.);self.assertEqual(r['thin_recall_d1'],1.)
        y,f=self.line();p=y|np.roll(y,1,0)|np.roll(y,-1,0)
        r=score(p.astype(float),y,f,y);self.assertLess(r['precision'],1.);self.assertEqual(r['thin_recall_d0'],1.)

    def test_empty_and_short_branch_and_group_na(self):
        z=np.zeros((32,32),bool);f=np.ones_like(z);r=score(z.astype(float),z,f,z)
        self.assertIsNone(r['dice']);self.assertIsNone(r['cldice']);self.assertIsNone(r['thin_recall_d1'])
        y=z.copy();y[15,15]=1;a=targets(y,f);self.assertEqual(int(a['skeleton'].sum()),1)
        r=score(z.astype(float),y,f,a['skeleton']);self.assertEqual(r['thin_recall_d1'],0.);self.assertEqual(r['dice'],0.)
        self.assertIsNone(r['precision']);self.assertEqual(r['cldice'],0.);r['disease']='A';summary=aggregate([r])
        self.assertEqual(summary['disease']['D']['images'],0);self.assertEqual(summary['all']['precision']['undefined'],1)

    def test_tiling_small_and_nonmultiple_batch_geometry(self):
        class Pointwise(torch.nn.Module):
            def forward(self,x):return x[:,:1]*2-1
        torch.manual_seed(9)
        for h,w in ((9,13),(45,53)):
            x=torch.rand(3,h,w);f=torch.ones(h,w);f[:2]=0
            outputs=[]
            for batch in (1,4):
                p,n=probability(Pointwise(),x,f,patch=16,stride=8,batch_size=batch,device='cpu',amp=False)
                torch.testing.assert_close(p,(x[0]*2-1).sigmoid()*f,rtol=2e-6,atol=1e-7);outputs.append(p)
            torch.testing.assert_close(*outputs,rtol=1e-6,atol=1e-7)

if __name__=='__main__':unittest.main()
