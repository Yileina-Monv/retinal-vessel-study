import os,sys,unittest,types,tempfile,json,random
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['nnUNet_n_proc_DA']='1'
from pathlib import Path
import replication_gate as g
from seeded_runtime import install_seeded_worker_adapter
from unittest.mock import patch

class FakeAug:
    def __init__(self,*args,**kwargs):self.kwargs=kwargs;self.finished=False
    def _finish(self):self.finished=True
def fixture(seed):
    trainer=types.SimpleNamespace(NonDetMultiThreadedAugmenter=FakeAug);producer=types.SimpleNamespace(producer=lambda *args:random.random());used=install_seeded_worker_adapter(seed,trainer,producer);return trainer,producer,used

class ReplicationTests(unittest.TestCase):
    def test_seed_contract_both_new_seeds(self):
        for s in [2027,2028]:
            t,p,used=fixture(s);a=t.NonDetMultiThreadedAugmenter(num_processes=1);b=t.NonDetMultiThreadedAugmenter(num_processes=1)
            self.assertEqual([a.kwargs['seeds'],b.kwargs['seeds']],[[s],[s+1]]);self.assertEqual([x['worker_count'] for x in used],[1,1]);self.assertIsInstance(a,FakeAug);self.assertIsInstance(a,t.NonDetMultiThreadedAugmenter)
    def test_legacy2026_worker_seeding_semantics_preserved(self):
        t,p,used=fixture(2026);t.NonDetMultiThreadedAugmenter();t.NonDetMultiThreadedAugmenter();self.assertEqual([x['seeds'] for x in used],[[2026],[2027]])
    def test_producer_python_random_reproducible(self):
        for seed in [2027,2028,2029]:
            t,p,_=fixture(2027);a=p.producer(None,None,None,0,seed,None);b=p.producer(None,None,None,0,seed,None);self.assertEqual(a,b);self.assertEqual(a,random.Random(seed).random())
    def test_extra_pool_and_extra_worker_refused(self):
        t,p,_=fixture(2027)
        with self.assertRaises(RuntimeError):t.NonDetMultiThreadedAugmenter(num_processes=2)
        t.NonDetMultiThreadedAugmenter();t.NonDetMultiThreadedAugmenter()
        with self.assertRaises(RuntimeError):t.NonDetMultiThreadedAugmenter()
    def test_bad_type_failure_reproduced_and_fixed(self):
        with self.assertRaises(TypeError):isinstance(object(),(lambda:None,FakeAug))
        t,p,_=fixture(2027);a=t.NonDetMultiThreadedAugmenter();self.assertTrue(isinstance(a,(t.NonDetMultiThreadedAugmenter,FakeAug)))
    def test_actual_native_on_train_end_saves_then_cleans_both_pools(self):
        import torch
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        t,p,used=fixture(2027);tr=t.NonDetMultiThreadedAugmenter();va=t.NonDetMultiThreadedAugmenter();saved=[];printed=[]
        with tempfile.TemporaryDirectory() as d:
            trainer=types.SimpleNamespace(current_epoch=100,local_rank=0,output_folder=d,dataloader_train=tr,dataloader_val=va,device=torch.device('cpu'),print_to_log_file=lambda *x:printed.append(x))
            trainer.save_checkpoint=lambda path:saved.append({'path':path,'serialized_epoch':trainer.current_epoch+1})
            stdout=sys.stdout
            with patch.dict(nnUNetTrainer.on_train_end.__globals__,{'NonDetMultiThreadedAugmenter':t.NonDetMultiThreadedAugmenter}):nnUNetTrainer.on_train_end(trainer)
            self.assertIs(sys.stdout,stdout);self.assertEqual(trainer.current_epoch,100);self.assertEqual(saved[0]['serialized_epoch'],100);self.assertTrue(saved[0]['path'].endswith('checkpoint_final.pth'));self.assertTrue(tr.finished and va.finished);self.assertEqual(printed[-1],('Training done.',))
    def test_seed_context_and_baseline_paths_are_seed_specific(self):
        paths=[]
        for seed in [2027,2028]:
            g.set_context(seed);q=g.protocol();self.assertEqual(q['worker_seeds'],{'main':seed,'train':seed,'internal_monitor':seed+1,'workers_each':1});paths.append(q['references']['A0_reuse_binding'])
        self.assertNotEqual(*paths)
        for seed in [2026,2029]:
            with self.assertRaises(RuntimeError):g.set_context(seed)
        g.set_context(2027)
    def test_real_native_geometry_and_new_budget_scope(self):
        g.set_context(2027);q=g.protocol();self.assertEqual(sum(q['inference']['expected_tiles_by_image'].values()),67);self.assertEqual(q['scope']['planned_optimizer_attempts'],25000);self.assertEqual(g.group_protocol()['total_train_attempt_cap'],50000);self.assertFalse(q['budget']['GPU1_allowed'])

if __name__=='__main__':unittest.main(verbosity=2)
