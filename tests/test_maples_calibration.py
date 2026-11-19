import unittest,importlib.util
class CalibrationTests(unittest.TestCase):
 def test_calibration_driver_exists(self):
  self.assertIsNotNone(importlib.util.find_spec('retinal_m0.maples_calibrate'),'Bound Z0 and calibration driver missing')
 def test_flatten_keeps_valid_counts_and_reference_identity(self):
  import numpy as np
  from retinal_m0.maples_calibrate import flatten
  from retinal_m0.maples_evaluate import evaluate_arrays,aggregate
  f=np.ones((4,4),bool);v=np.zeros_like(f);v[1,1]=1
  result=evaluate_arrays(v.astype(float),v,f,threshold=.5,u=np.zeros_like(f),regions=dict(internal=np.zeros_like(f),near=v,far=f&~v),thin=v,matching_support='omega')
  result['filename']='fixture';report=dict(images=[result],aggregate=aggregate([result]))
  flat=flatten(report,200,'bound_ref')
  self.assertEqual(flat['primary_ids'],['fixture']);self.assertEqual(flat['primary_n'],1);self.assertEqual(flat['valid_n']['far_recall'],0);self.assertIsNone(flat['far_recall']);self.assertEqual(flat['near_thin_recall_d1'],1.)
 def test_supplementary_grid_matches_hand_count(self):
  import numpy as np,json
  from pathlib import Path
  from tempfile import TemporaryDirectory
  from retinal_m0.maples_calibrate import supplementary
  from retinal_m0.maples_evaluate import evaluate_arrays,aggregate
  with TemporaryDirectory() as tmp:
   out=Path(tmp);(out/'probabilities').mkdir();f=np.ones((2,2),bool);v=np.zeros_like(f);v[0,0]=1;p=np.array([[.9,.5],[0.,1.]],np.float32)
   np.save(out/'v.npy',v);np.save(out/'f.npy',f);np.save(out/'probabilities/a.npy',p)
   np.savez(out/'r.npz',U=np.zeros_like(f),internal=np.zeros_like(f),near=f,far=np.zeros_like(f),thin=v,thin_2=v,thin_4=v,thin_5=v)
   doc=dict(images=[dict(filename='a',vessel_path=str(out/'v.npy'),fov_path=str(out/'f.npy'),region_bundle=str(out/'r.npz'))])
   primary=dict(aggregate=aggregate([evaluate_arrays(p,v,f,threshold=.5)]));supplementary(out,doc,primary,0)
   result=json.loads((out/'matched_FP.json').read_text());self.assertEqual(result['actual_FP_density'],.5);self.assertLessEqual(result['actual_FP_density'],result['target_FP_density'])
 def test_macro_grid_preserves_exact_equal_boundary(self):
  import numpy as np
  from retinal_m0.maples_calibrate import macro_density_grid
  values=np.array([0.007908962383510998, 0.014476422002473231, 0.021766750753707008, 0.018422625891009427, 0.009604623511882366, 0.015013278144526965, 0.009764102663255676, 0.026344805258271025, 0.013212590209280941, 0.015299720265166586, 0.01871970926945485, 0.0112825966296799, 0.007665754447070531, 0.011501826833866127, 0.009848219906274044, 0.0068255669435988535, 0.011686127992110737, 0.007373370710392024, 0.013452956561556552, 0.011583497712678893, 0.0060523918086606435, 0.01241042406177301, 0.005424580434781157, 0.00735442871478771, 0.0036645403202483803, 0.007599442645135581, 0.013761117875187856, 0.007734545236461048],np.float64)
  matrix=np.tile(values[:,None],(1,1001))
  target=float(np.mean(values.tolist()))
  actual=macro_density_grid(matrix)
  self.assertEqual(actual[500],target)
  self.assertTrue(np.all(actual<=target))
if __name__=='__main__':unittest.main()


