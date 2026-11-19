import unittest
import importlib.util
import numpy as np
import torch

class T0Contract(unittest.TestCase):
    def test_training_module_present(self):
        self.assertIsNotNone(importlib.util.find_spec("retinal_m0.maples_t0"), "New batch/resume entry is missing")

    def test_architecture_and_log_recovery(self):
        from retinal_m0.maples_t0 import validate_model_settings, align_log
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json
        with self.assertRaises(ValueError):validate_model_settings({'widths':[32,64,128,256,512],'group_norm_groups':4})
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'train.jsonl'
            p.write_text(''.join(json.dumps({'step':i})+'\n' for i in (1,2,3)))
            align_log(p,2)
            self.assertEqual(len(p.read_text().splitlines()),2)
            self.assertEqual(len(list(Path(tmp).glob('*.recovery*'))),1)
            p.write_text(json.dumps({'step':2})+'\n')
            with self.assertRaises(ValueError):align_log(p,1)

    def test_geometry_and_loss(self):
        if importlib.util.find_spec("retinal_m0.maples_t0") is None:
            self.skipTest("entry missing; first contract must fail")
        from retinal_m0.maples_t0 import draw_geometry
        from retinal_release.data import masked_loss
        a=draw_geometry(2026,3,900,900)
        self.assertEqual(a,draw_geometry(2026,3,900,900))
        self.assertNotEqual(a,draw_geometry(2026,4,900,900))
        self.assertTrue(0<=a[0]<=388 and 0<=a[1]<=388 and a[4] in range(4))
        z=torch.tensor([[[[0.,1.],[2.,-1.]]],[[[3.,-2.],[1.,0.]]]],requires_grad=True)
        y=torch.tensor([[[[0.,1.],[1.,0.]]],[[[1.,0.],[0.,1.]]]])
        f=torch.tensor([[[[1.,1.],[0.,0.]]],[[[1.,1.],[1.,1.]]]])
        expected=[]
        for i in range(2):
            p=z[i].sigmoid();b=torch.nn.functional.binary_cross_entropy_with_logits(z[i],y[i],reduction="none")
            expected.append((b*f[i]).sum()/f[i].sum()+1-(2*(p*y[i]*f[i]).sum()+1e-6)/((p*f[i]).sum()+(y[i]*f[i]).sum()+1e-6))
        self.assertTrue(torch.allclose(masked_loss(z,y,f),torch.stack(expected).mean()))
        masked_loss(z,y,f).backward();self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(z.grad[0,0,1,0].item(),0.)

if __name__=="__main__":unittest.main()
