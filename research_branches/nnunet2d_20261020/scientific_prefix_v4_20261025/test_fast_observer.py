import os,sys,unittest,tempfile,subprocess,json
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
B=Path('${RETINA_PROJECT_ROOT}/code/retinal-vessel-study/research_branches/nnunet2d_20261020');sys.path.extend([str(B/'amp_engineering32_20261025'),str(B/'amp_recovery_revision_20261021')])
import torch
torch.set_num_threads(1)
import test_amp_observer_v2 as legacy
from fast_observer import FastObserver,Journal,AmpLimits,digest,states_finite,restore_rng
legacy.Observer=FastObserver

def resume_partial(folder):
    folder=Path(folder);saved=torch.load(folder/'partial.pt',weights_only=False,map_location='cpu');t=legacy.Tiny()
    t.network.load_state_dict(saved['model']);t.optimizer.load_state_dict(saved['optimizer']);t.grad_scaler.load_state_dict(saved['scaler']);t.lr_scheduler.load_state_dict(saved['scheduler']);t.current_epoch=saved['native_current_epoch'];restore_rng(saved['rng'])
    j=Journal(folder/'budget.json','fixture',2,3,2,AmpLimits(5,25));assert j.read()==saved['budget'];o=FastObserver(t,j,folder,['train_fixture']);o.partial_train=saved['partial_train'];o.partial_monitor=saved['partial_monitor']
    # This deliberately resumes within an epoch, using a fixed input rather than native worker queues.
    while j.read()['iteration']<3:t.train_step(next(t.dataloader_train))
    t.on_train_epoch_end(o.partial_train.copy());t.on_validation_epoch_start()
    while j.read()['monitor_iteration']<2:t.validation_step(next(t.dataloader_val))
    t.on_validation_epoch_end(o.partial_monitor.copy());t.on_epoch_end();legacy.nnUNetTrainer.run_training(t)
    o.save_bundle(folder/'resumed.pt','fixture');o.close()

class FastSpecificTests(unittest.TestCase):
    def test_mid_epoch_partial_outputs_log_prefixes_resume_new_process_exact(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'cont';a.mkdir();t,j,o=legacy.setup(a);legacy.nnUNetTrainer.run_training(t);want=o.save_bundle(a/'full.pt','fixture');o.close()
            b=root/'split';b.mkdir();t,j,o=legacy.setup(b);t.on_epoch_start();t.on_train_epoch_start();t.train_step(next(t.dataloader_train));o.save_bundle(b/'partial.pt','fixture');o.close()
            p=subprocess.run([sys.executable,'-B',__file__,'resume',str(b)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90);self.assertEqual(p.returncode,0,p.stdout)
            got=torch.load(b/'resumed.pt',weights_only=False)['digests'];self.assertEqual({k:v for k,v in want.items() if k!='log_prefixes'},{k:v for k,v in got.items() if k!='log_prefixes'})
    def test_vector_finite_detects_nonfinite_optimizer_and_keeps_rng(self):
        before=digest(legacy.rng_pack());self.assertTrue(states_finite({'x':torch.ones(2),'y':.01}));self.assertFalse(states_finite({'state':torch.tensor([float('inf')])}));self.assertFalse(states_finite({'lr':float('nan')}));self.assertEqual(before,digest(legacy.rng_pack()))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='resume':resume_partial(sys.argv[2])
    else:
        suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(legacy.IntegrationTests),unittest.defaultTestLoader.loadTestsFromTestCase(FastSpecificTests)])
        result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful() else 1)
