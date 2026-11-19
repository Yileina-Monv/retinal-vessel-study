import os,sys,unittest,tempfile,json,subprocess,copy
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from test_fast_observer import legacy,FastObserver,Journal,AmpLimits,digest,restore_rng
from resume_support import migrate_budget,rebuild_logger,continue_mid_monitor_then_official
from fast_observer import atomic
import torch,numpy as np

def resume(folder):
    folder=Path(folder);s=torch.load(folder/'partial.pt',weights_only=False,map_location='cpu');t=legacy.Tiny()
    t.network.load_state_dict(s['model']);t.optimizer.load_state_dict(s['optimizer']);t.grad_scaler.load_state_dict(s['scaler']);t.lr_scheduler.load_state_dict(s['scheduler']);t.current_epoch=s['native_current_epoch'];restore_rng(s['rng'])
    j=Journal(folder/'budget.json','fixture',2,3,2,AmpLimits(5,25));o=FastObserver(t,j,folder,['train_fixture']);o.partial_train=s['partial_train'];o.partial_monitor=s['partial_monitor'];t.run_training=lambda:legacy.nnUNetTrainer.run_training(t)
    continue_mid_monitor_then_official(t,o,j);o.save_bundle(folder/'full.pt','fixture');o.close()

class ResumeTests(unittest.TestCase):
    def test_committed_mid_monitor_resume_new_process_matches_continuous(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'continuous';a.mkdir();t,j,o=legacy.setup(a);legacy.nnUNetTrainer.run_training(t);want=o.save_bundle(a/'full.pt','fixture');o.close()
            b=root/'split';b.mkdir();t,j,o=legacy.setup(b);t.on_epoch_start();t.on_train_epoch_start()
            for _ in range(3):t.train_step(next(t.dataloader_train))
            t.on_train_epoch_end(o.partial_train.copy());t.on_validation_epoch_start()
            with torch.no_grad():t.validation_step(next(t.dataloader_val))
            o.save_bundle(b/'partial.pt','fixture');j.stop('simulated controller false-stop');old=j.read();atomic(j.path,migrate_budget(old,'fixture'));o.close()
            p=subprocess.run([sys.executable,'-B',__file__,'resume',str(b)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90);self.assertEqual(p.returncode,0,p.stdout)
            actual=torch.load(b/'full.pt',weights_only=False)['digests'];self.assertEqual({k:v for k,v in want.items() if k!='log_prefixes'},{k:v for k,v in actual.items() if k!='log_prefixes'})
    def test_migration_retains_all_spending_caps_and_cursor(self):
        old={'pending':None,'fatal':0,'unknown':0,'skip':0,'attempts':1000,'success':1000,'phase':'monitor','iteration':250,'monitor_iteration':2,'schedule':[100,250,50],'binding':'old','status':'STOPPED','reason':'watchdog','monitor_complete':152,'limits':{'consecutive':5,'total':25}}
        new=migrate_budget(old,'new');self.assertEqual({k:v for k,v in old.items() if k not in ['binding','status','reason']},{k:v for k,v in new.items() if k not in ['binding','status']});self.assertEqual(old['status'],'STOPPED')
        for k,v in [('pending',{'reserved':1}),('unknown',1),('iteration',249),('phase','train')]:
            bad=copy.deepcopy(old);bad[k]=v
            with self.assertRaises(RuntimeError):migrate_budget(bad,'new')
    def test_logger_uses_exact_recorded_outputs_not_rounded_console(self):
        logging={k:[1,2,3] for k in ['train_losses','val_losses','lrs','epoch_start_timestamps','epoch_end_timestamps','mean_fg_dice','ema_fg_dice','dice_per_class_or_region']}
        native={'current_epoch':3,'logging':logging};bundle={'cursor':{'epoch':3,'monitor_iteration':2},'budget':{'schedule':[100,250,50]},'partial_train':[{'loss':np.asarray(.12345678,dtype=np.float32)} for _ in range(250)],'partial_monitor':[{},{}],'optimizer':{'param_groups':[{'lr':.00972959}]}}
        got,meta=rebuild_logger(native,bundle,'2026-10-25 21:03:43.331182: Epoch 3\n');self.assertEqual(len(got['train_losses']),4);self.assertEqual(len(got['val_losses']),3);self.assertAlmostEqual(float(got['train_losses'][-1]),.12345678,places=7);self.assertEqual(logging['train_losses'],[1,2,3]);self.assertFalse(meta['native_queue_resume_supported'])
    def test_watchdog_cannot_auto_terminate_child(self):
        source=(Path(__file__).parent/'science100_controller.py').read_text();self.assertNotIn('st_mtime',source);self.assertNotIn('now-last',source);self.assertIn('PROGRESS_WARNING_ONLY',source)
        from progress_watchdog import ProgressWatchdog
        w=ProgressWatchdog(1800);self.assertFalse(w.observe((1,),10)['stop']);self.assertFalse(w.observe((2,),2000)['stop']);self.assertTrue(w.observe((2,),4000)['stop'])

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='resume':resume(sys.argv[2])
    else:unittest.main(verbosity=2)
