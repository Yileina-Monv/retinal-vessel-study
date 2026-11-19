import unittest,importlib.util
class ProtocolTests(unittest.TestCase):
 def test_protocol_helpers_exist(self):
  self.assertIsNotNone(importlib.util.find_spec('retinal_m0.maples_protocol'),'Explicit diameter/radius and selection helpers missing')
 def test_radius_U_and_native_diameter_proxy(self):
  import numpy as np
  from retinal_m0.maples_protocol import proposed_references
  f=np.ones((21,101),bool);v=np.zeros_like(f);v[10,5:90]=1;h=np.zeros_like(f);h[5,20]=1;u=np.zeros_like(f);u[5,22]=1
  b,m=proposed_references(v,f,h,u,np.zeros_like(f),diameter_max=3)
  self.assertEqual(m['radius'],2);self.assertTrue(b['thin'][10,50]);self.assertFalse(b['near'][5,22]);self.assertTrue(b['near'][5,21]);self.assertFalse(b['near'][6,22])
  v[9:12,5:90]=1;b,m=proposed_references(v,f,h,u,u,diameter_max=3);self.assertFalse(b['thin'][10,50])
  b,m=proposed_references(v,f,h,u,u,diameter_max=4);self.assertTrue(b['thin'][10,50])
 def test_selection_gates_ties_and_none(self):
  from retinal_m0.maples_protocol import select_checkpoint
  z=dict(step=0,precision=.9,fp_density=.01,far_recall=.7,near_thin_recall_d1=.3,primary_n=1,reference_id='ref',primary_ids=['a'])
  a=dict(z,step=200,near_thin_recall_d1=.5);b=dict(a,step=400);bad=dict(a,step=600,precision=.899,near_thin_recall_d1=.9)
  self.assertEqual(select_checkpoint([bad,b,a],z)['step'],200);self.assertIsNone(select_checkpoint([bad],z))
  with self.assertRaises(ValueError):select_checkpoint([dict(a,primary_ids=['b'])],z)
 def test_matched_density_reports_no_feasible(self):
  from retinal_m0.maples_protocol import matched_threshold_choice
  rows=[dict(threshold=.4,fp_density=.02),dict(threshold=.5,fp_density=.01),dict(threshold=.6,fp_density=.009)]
  self.assertEqual(matched_threshold_choice(rows,.01)['threshold'],.5);self.assertIsNone(matched_threshold_choice(rows,.001))
if __name__=='__main__':unittest.main()

