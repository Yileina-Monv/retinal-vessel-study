import unittest
import numpy as np
from retinal_m0.maples_reliability import choose_threshold_from_counts, metric_from_counts, paired_bootstrap, target_hit_distance
class ReliabilityHelpersTests(unittest.TestCase):
    def test_loo_threshold_uses_training_images_only(self):
        counts=np.array([[4,2,0],[6,2,0],[100,99,90]],dtype=np.int64); fov=np.array([10,10,100]); z0=np.array([.2,.2,.9]); grid=np.array([0.,.5,1.])
        a=choose_threshold_from_counts(counts,fov,z0,[0,1],grid); c=counts.copy(); c[2]=[0,0,0]; z=z0.copy(); z[2]=0
        b=choose_threshold_from_counts(c,fov,z,[0,1],grid)
        self.assertEqual(a,b); self.assertEqual(a['threshold'],.5); self.assertAlmostEqual(a['target_fp_density'],.2)
    def test_threshold_equal_to_target_and_no_feasible(self):
        grid=np.array([0.,.5]); a=choose_threshold_from_counts(np.array([[3,2],[3,2]]),np.array([10,10]),np.array([.2,.2]),[0,1],grid)
        self.assertEqual(a['threshold'],.5); self.assertAlmostEqual(a['actual_fp_density'],.2)
        b=choose_threshold_from_counts(np.array([[5,5]]),np.array([10]),np.array([.1]),[0],grid)
        self.assertIsNone(b['threshold']); self.assertIsNone(b['actual_fp_density'])
    def test_zero_denominator_remains_na(self):
        self.assertEqual(metric_from_counts(0,0),{'value':None,'numerator':0,'denominator':0,'NA_reason':'zero_denominator'})
    def test_paired_bootstrap_reproducible_with_correct_n(self):
        a=paired_bootstrap([.2,-.1,.3],reps=1000,seed=20261016); b=paired_bootstrap([.2,-.1,.3],reps=1000,seed=20261016)
        self.assertEqual({k:v for k,v in a.items() if k!='draw_means'},{k:v for k,v in b.items() if k!='draw_means'})
        self.assertTrue(np.array_equal(a['draw_means'],b['draw_means'])); self.assertEqual(a['n_images'],3); self.assertEqual(a['reps'],1000)
    def test_native_coordinate_and_d1_respect_omega_boundary(self):
        p=np.zeros((5,5),np.float32); f=np.ones((5,5),bool); u=np.zeros((5,5),bool); p[2,3]=.9
        a=target_hit_distance(p,(2,2),f,u,.5)
        self.assertEqual((a['target_row'],a['target_col'],a['nearest_row'],a['nearest_col'],a['distance'],a['hit_d1']),(2,2,2,3,1.,True))
        u[2,3]=True; b=target_hit_distance(p,(2,2),f,u,.5); self.assertIsNone(b['nearest_row']); self.assertFalse(b['hit_d1'])
if __name__=='__main__': unittest.main()
