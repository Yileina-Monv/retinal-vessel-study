import time
import unittest
from unittest.mock import patch
from retinal_prefetch.loader import OrderedBatches

class PrefetchTests(unittest.TestCase):
    def test_order_bound_and_resume_cursor(self):
        def fake(ds,indices):
            time.sleep(.015 if indices==0 else .001)
            return indices
        with patch('retinal_prefetch.loader.SkeletonDataset',return_value=None),patch('retinal_prefetch.loader.batch',side_effect=fake):
            with OrderedBatches(range(8),workers=2,depth=3) as batches:
                self.assertEqual(len(batches.pending),3)
                results=[]
                for item in batches:
                    results.append(item);self.assertLessEqual(len(batches.pending),3)
            self.assertEqual(results,list(range(8)))
            with OrderedBatches(range(3,8),workers=2,depth=3) as batches:self.assertEqual(list(batches),list(range(3,8)))
    def test_exception_is_propagated_and_shutdown(self):
        def fake(ds,index):
            if index==1:raise ValueError('fixture corrupt source')
            return index
        with patch('retinal_prefetch.loader.SkeletonDataset',return_value=None),patch('retinal_prefetch.loader.batch',side_effect=fake):
            b=OrderedBatches(range(5),workers=2,depth=2)
            with self.assertRaisesRegex(ValueError,'corrupt source'):
                with b:list(b)
            self.assertTrue(b.closed)
    def test_profile_limit(self):
        for workers,depth in ((5,8),(4,9),(4,2)):
            with self.assertRaises(ValueError):OrderedBatches([],workers=workers,depth=depth)

if __name__=='__main__':unittest.main()
