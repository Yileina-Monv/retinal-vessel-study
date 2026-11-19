import unittest
import numpy as np, torch
from retinal_m0.maples_core import weighted_masked_loss
from retinal_m0.maples_weight_ablation import align_contribution_log,build_ablation_weights,reweight_base_sample,weighted_loss_with_contributions
class WeightAblationTests(unittest.TestCase):
 def test_weights_uncertainty_empty_and_allowed_alphas(self):
  f=np.ones((31,41),bool);f[0]=0;h=np.zeros_like(f);h[15,20]=1;u=np.zeros_like(f);u[14,20]=1
  for a in (1.,1.5,2.,3.):
   r=build_ablation_weights(f,h,u,image_width=41,alpha=a);self.assertTrue(np.all(r["weights"][r["area"]]==a));self.assertTrue(np.all(r["weights"][u]==1));self.assertTrue(np.all(r["weights"][~f]==0))
  with self.assertRaises(ValueError):build_ablation_weights(np.zeros_like(f),h,u,image_width=41,alpha=2)
  with self.assertRaises(ValueError):build_ablation_weights(f,h,u,image_width=41,alpha=2.5)
 def test_reweight_preserves_inputs_geometry_and_float(self):
  f=torch.tensor([[[[1.,1.],[1.,0.]]]]);w=torch.tensor([[[[2.,1.],[2.,0.]]]]);s={"fov":f,"weight":w,"input_sha256":"same","geometry":torch.tensor([1,2,0,1,3])}
  o=reweight_base_sample(s,1.5);self.assertEqual(o["input_sha256"],"same");self.assertTrue(torch.equal(o["geometry"],s["geometry"]));self.assertTrue(torch.equal(o["weight"],torch.tensor([[[[1.5,1.],[1.5,0.]]]])))
 def test_loss_gradient_and_contribution_reconcile(self):
  torch.manual_seed(22);z=torch.randn(2,1,9,11);y=torch.randint(0,2,(2,1,9,11)).float();f=torch.randint(0,2,(2,1,9,11)).float();f[:,:,0,0]=1
  area=torch.zeros_like(f);area[:,:,2:6,4:8]=f[:,:,2:6,4:8]
  for a in (1.,1.5,2.,3.):
   w=f.clone()
   if a!=1:w[area.bool()]=a
   x=z.clone().requires_grad_();q=z.clone().requires_grad_();got,bce,dice,p=weighted_loss_with_contributions(x,y,f,w,a);expected=weighted_masked_loss(q,y,f,w)
   self.assertTrue(torch.equal(got,expected));g1,=torch.autograd.grad(got,x);g2,=torch.autograd.grad(expected,q);self.assertTrue(torch.equal(g1,g2))
   n=sum(p["groups"][g]["n"] for g in ("A_V1","A_V0","O_V1","O_V0"));self.assertTrue(torch.equal(n,p["nF"]));self.assertTrue(torch.allclose(sum(p["groups"][g]["C"] for g in ("A_V1","A_V0","O_V1","O_V0")),bce,rtol=1e-5,atol=1e-7))
  a1,*_=weighted_loss_with_contributions(z,y,f,f,1.);l0=weighted_masked_loss(z,y,f,f);self.assertTrue(torch.equal(a1,l0))
 def test_log_resume_uses_step_draw_and_preserves_tail(self):
  import json,tempfile
  from pathlib import Path
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"bce.jsonl";p.write_text("\n".join(json.dumps({"step":s,"batch_slot":j,"global_draw":(s-1)*4+j}) for s in range(1,4) for j in range(4))+"\n")
   r=align_contribution_log(p,2,8);self.assertEqual(r["kept_rows"],8);self.assertEqual(len(p.read_text().splitlines()),8);self.assertTrue(Path(r["backup"]).is_file())
   with self.assertRaises(RuntimeError):align_contribution_log(p,2,7)
if __name__=="__main__":unittest.main()
