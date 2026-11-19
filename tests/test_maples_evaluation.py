import unittest,importlib.util
import numpy as np
class EvaluationTests(unittest.TestCase):
 def test_evaluator_exists(self):
  self.assertIsNotNone(importlib.util.find_spec("retinal_m0.maples_evaluate"),"Native-resolution evaluator missing")
 def test_hand_computable_confusion_and_NA(self):
  if importlib.util.find_spec("retinal_m0.maples_evaluate") is None:self.skipTest("module absent")
  from retinal_m0.maples_evaluate import evaluate_arrays,aggregate
  f=np.ones((4,5),bool);y=np.zeros_like(f);y[1,1:4]=1
  h=np.zeros_like(f);h[0,0]=1;u=np.zeros_like(f);u[1,3]=1
  thin=y.copy();p=np.zeros(f.shape);p[1,1]=.9;p[0,2]=.9;p[1,3]=.9
  near=np.zeros_like(f);near[1,1:4]=1;far=f&~near&~h&~u;near&=~u;internal=h&~u
  z=evaluate_arrays(p,y,f,threshold=.5,u=u,regions={'internal':internal,'near':near,'far':far},thin=thin,matching_support='omega')
  self.assertEqual(z['counts']['tp'],2);self.assertEqual(z['counts']['fp'],1)
  self.assertAlmostEqual(z['dice']['value'],2/3);self.assertAlmostEqual(z['precision']['value'],2/3);self.assertAlmostEqual(z['recall']['value'],2/3)
  self.assertAlmostEqual(z['fp_density']['value'],1/20)
  self.assertEqual(z['near_pixel_recall']['value'],.5);self.assertEqual(z['near_thin_recall_d1']['value'],1.)
  self.assertIsNone(z['far_recall']['value']);self.assertEqual(aggregate([z])['far_recall']['n'],0)
  self.assertEqual(z['near_thin_recall_d1']['denominator'],2)
 def test_U_predictions_cannot_match_and_shape_checks(self):
  if importlib.util.find_spec("retinal_m0.maples_evaluate") is None:self.skipTest("module absent")
  from retinal_m0.maples_evaluate import evaluate_arrays
  f=np.ones((3,4),bool);y=np.zeros_like(f);y[1,1]=1;u=np.zeros_like(f);u[1,2]=1;p=np.zeros(f.shape);p[1,2]=1
  regions={'internal':np.zeros_like(f),'near':y.copy(),'far':f&~u&~y}
  z=evaluate_arrays(p,y,f,threshold=.5,u=u,regions=regions,thin=y,matching_support='omega')
  self.assertEqual(z['near_thin_recall_d1']['value'],0.)
  z=evaluate_arrays(p*0,y,f,threshold=.5,u=u,regions=regions,thin=y,matching_support='omega')
  self.assertEqual(z['near_thin_recall_d1']['value'],0.)
  with self.assertRaises(ValueError):evaluate_arrays(p[:2],y,f,threshold=.5)
  with self.assertRaises(ValueError):evaluate_arrays(p+2,y,f,threshold=.5)
class EdgeCases(unittest.TestCase):
 def test_fake_dev_membership_rejected_before_reading_predictions(self):
  from tempfile import TemporaryDirectory
  from pathlib import Path
  import json
  from retinal_m0.maples_evaluate import evaluate_manifest
  with TemporaryDirectory() as tmp:
   p=Path(tmp)/'m.json';p.write_text(json.dumps({'scope':'MAPLES_dev28','images':[{'filename':'fake_'+str(i),'split':'dev','source_split':'train'} for i in range(28)]}))
   with self.assertRaises(ValueError):evaluate_manifest(p,Path(tmp)/'probs',Path(tmp)/'out',threshold=.5)
 def test_euclidean_d1_excludes_diagonal(self):
  from retinal_m0.maples_evaluate import evaluate_arrays
  f=np.ones((3,3),bool);v=np.zeros_like(f);v[1,1]=1;p=np.zeros(f.shape);p[0,0]=1
  r={'internal':np.zeros_like(f),'near':v,'far':f&~v}
  z=evaluate_arrays(p,v,f,threshold=.5,u=np.zeros_like(f),regions=r,thin=v,matching_support='omega')
  self.assertEqual(z['near_thin_recall_d1']['value'],0.)
 def test_empty_reference_and_empty_FOV(self):
  from retinal_m0.maples_evaluate import evaluate_arrays
  a=np.zeros((3,3));f=np.ones((3,3),bool)
  z=evaluate_arrays(a,a,f,threshold=.5)
  self.assertIsNone(z['dice']['value']);self.assertIsNone(z['precision']['value']);self.assertIsNone(z['recall']['value']);self.assertEqual(z['fp_density']['value'],0.)
  z=evaluate_arrays(a,a,a,threshold=.5);self.assertIsNone(z['fp_density']['value'])
 def test_partition_and_empty_hemorrhage(self):
  from retinal_m0.maples_evaluate import make_regions,evaluate_arrays
  f=np.ones((4,4),bool);zero=np.zeros_like(f);r=make_regions(f,zero,zero,radius=1,kernel='euclidean_disk')
  self.assertTrue(np.array_equal(r['far'],f));self.assertFalse(r['near'].any())
  y=zero.copy();y[1,1]=1
  z=evaluate_arrays(y,y,f,threshold=.5,u=zero,regions=r,thin=y,matching_support='omega')
  self.assertIsNone(z['near_thin_recall_d1']['value']);self.assertEqual(z['far_recall']['value'],1.)
  r['near'][1,1]=1
  with self.assertRaises(ValueError):evaluate_arrays(y,y,f,threshold=.5,u=zero,regions=r,thin=y,matching_support='omega')
if __name__=='__main__':unittest.main()

