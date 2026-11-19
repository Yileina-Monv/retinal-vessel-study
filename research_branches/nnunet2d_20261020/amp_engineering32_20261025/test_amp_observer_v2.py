import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,itertools,json,random,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from amp_observer_v2 import Journal,Observer,digest,rng_pack,restore_rng,save_full_state
from revised_policy import AmpLimits,RevisionStop,StepEvidence
torch.set_num_threads(1)

class Tiny:
    def __init__(self):
        random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
        self.network=torch.nn.Sequential(torch.nn.Conv2d(3,2,3,padding=1),torch.nn.Dropout(.2))
        self.optimizer=torch.optim.SGD(self.network.parameters(),lr=.01,momentum=.99,nesterov=True,weight_decay=3e-5)
        self.grad_scaler=torch.amp.GradScaler('cpu',init_scale=1024)
        self.lr_scheduler=PolyLRScheduler(self.optimizer,.01,100)
        self.loss=torch.nn.CrossEntropyLoss();self.device=torch.device('cpu');self.current_epoch=0
        self.num_epochs=2;self.num_iterations_per_epoch=3;self.num_val_iterations_per_epoch=2;self.losses=[]
        batch={'data':torch.arange(3*8*8,dtype=torch.float32).reshape(1,3,8,8)/100,'target':torch.arange(64).reshape(1,8,8)%2,'keys':['train_fixture']}
        self.dataloader_train=itertools.cycle([batch]);self.dataloader_val=itertools.cycle([batch])
    def train_step(self,batch):
        result=nnUNetTrainer.train_step(self,batch);self.losses.append(float(result['loss']));return result
    def validation_step(self,batch):
        z=self.network(batch['data']);return {'loss':self.loss(z,batch['target']).detach().numpy()}
    def on_train_start(self):pass
    def on_train_end(self):pass
    def on_epoch_start(self):pass
    def on_train_epoch_start(self):self.network.train();self.lr_scheduler.step(self.current_epoch)
    def on_train_epoch_end(self,outputs):pass
    def on_validation_epoch_start(self):self.network.eval()
    def on_validation_epoch_end(self,outputs):pass
    def on_epoch_end(self):self.current_epoch+=1

def setup(path,force=None):
    t=Tiny();j=Journal(path/'budget.json','fixture',2,3,2,AmpLimits(5,25));j.initialize();o=Observer(t,j,path,['train_fixture'],force_overflow=force);return t,j,o

def run_epoch(t):
    t.on_epoch_start();t.on_train_epoch_start();tr=[t.train_step(next(t.dataloader_train)) for _ in range(3)];t.on_train_epoch_end(tr)
    t.on_validation_epoch_start()
    with torch.no_grad():va=[t.validation_step(next(t.dataloader_val)) for _ in range(2)]
    t.on_validation_epoch_end(va);t.on_epoch_end()

def resume_worker(folder):
    folder=Path(folder);saved=torch.load(folder/'state.pt',weights_only=False,map_location='cpu');t=Tiny()
    t.network.load_state_dict(saved['model']);t.optimizer.load_state_dict(saved['optimizer']);t.grad_scaler.load_state_dict(saved['scaler']);t.lr_scheduler.load_state_dict(saved['scheduler']);t.current_epoch=saved['native_current_epoch'];restore_rng(saved['rng'])
    j=Journal(folder/'budget.json','fixture',2,3,2,AmpLimits(5,25));assert j.read()==saved['budget'];o=Observer(t,j,folder,['train_fixture'])
    nnUNetTrainer.run_training(t);save_full_state(folder/'resumed.pt',t,j);o.close()

class IntegrationTests(unittest.TestCase):
    def test_actual_official_run_training_and_step_observer_is_value_and_rng_neutral(self):
        baseline=Tiny();nnUNetTrainer.run_training(baseline);want={k:digest(v) for k,v in [('model',baseline.network.state_dict()),('opt',baseline.optimizer.state_dict()),('scaler',baseline.grad_scaler.state_dict()),('scheduler',baseline.lr_scheduler.state_dict()),('rng',rng_pack())]}
        with tempfile.TemporaryDirectory() as d:
            t,j,o=setup(Path(d));nnUNetTrainer.run_training(t)
            got={k:digest(v) for k,v in [('model',t.network.state_dict()),('opt',t.optimizer.state_dict()),('scaler',t.grad_scaler.state_dict()),('scheduler',t.lr_scheduler.state_dict()),('rng',rng_pack())]}
            self.assertEqual(want,got);self.assertEqual(baseline.losses,t.losses);s=j.read();self.assertEqual((s['attempts'],s['success'],s['monitor_complete'],s['train_forwards'],s['backwards'],s['optimizer_calls']),(6,6,4,6,6,6));self.assertEqual(s['phase'],'complete');o.close()
    def test_fixed_input_resume_in_new_process_exact_all_available_states(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'cont';a.mkdir();t,j,o=setup(a);nnUNetTrainer.run_training(t);want=save_full_state(a/'state.pt',t,j);o.close()
            b=root/'split';b.mkdir();t,j,o=setup(b);run_epoch(t);save_full_state(b/'state.pt',t,j);o.close()
            p=subprocess.run([sys.executable,'-B',__file__,'resume',str(b)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(p.returncode,0,p.stdout);actual=torch.load(b/'resumed.pt',weights_only=False)['digests'];self.assertEqual(want,actual)
            self.assertFalse(torch.load(b/'resumed.pt',weights_only=False)['native_queue_resume_supported'])
    def test_handled_cpu_overflow_skips_unchanged_states_and_next_update_recovers(self):
        with tempfile.TemporaryDirectory() as d:
            t,j,o=setup(Path(d),force=2);nnUNetTrainer.run_training(t);r=[json.loads(x) for x in (Path(d)/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual([v['classification'] for v in r],['SUCCESS','AMP_SKIP','SUCCESS','SUCCESS','SUCCESS','SUCCESS']);self.assertFalse(r[1]['model_changed']);self.assertFalse(r[1]['optimizer_changed']);self.assertEqual(r[1]['optimizer_calls'],0);self.assertLess(r[1]['scale_after'],r[1]['scale_before']);self.assertEqual(j.read()['success'],5);o.close()
    def test_crashed_reserved_attempt_charged_unknown_no_reset_no_forward_retry(self):
        with tempfile.TemporaryDirectory() as d:
            j=Journal(Path(d)/'budget.json','x',1,1,1,AmpLimits(5,25));j.initialize();j.start('train',0);j.compute('forward');j.stop('killed');s=j.read();self.assertEqual((s['attempts'],s['unknown']),(1,1))
            with self.assertRaises(RevisionStop):j.start('train',0)
            with self.assertRaises(RevisionStop):j.initialize()
    def test_extra_forward_backward_optimizer_and_monitor_optimizer_rejected(self):
        for duplicate in ['forward','backward','optimizer']:
            with tempfile.TemporaryDirectory() as d:
                j=Journal(Path(d)/'b.json','x',1,1,1,AmpLimits(5,25));j.initialize();j.start('train',0)
                for k in ['forward','backward','optimizer']:
                    j.compute(k)
                    if k==duplicate:break
                with self.assertRaises(RevisionStop):j.compute(duplicate)
        with tempfile.TemporaryDirectory() as d:
            j=Journal(Path(d)/'b.json','x',1,1,1,AmpLimits(5,25));j.initialize();j.start('train',0);j.compute('forward');j.compute('backward');j.compute('optimizer');j.resolve_train(StepEvidence(1,1,1,1,1,True,True),True);j.finish_train();j.start('monitor',0);j.compute('forward')
            with self.assertRaises(RevisionStop):j.compute('optimizer')
    def test_nontrain_id_and_wrong_epoch_and_early_finalization_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            t,j,o=setup(Path(d));batch=dict(next(t.dataloader_train),keys=['Dev_or_Test'])
            with self.assertRaises(RevisionStop):t.train_step(batch)
            self.assertEqual(j.read()['attempts'],0)
            with self.assertRaises(RevisionStop):j.finish_train()
            with self.assertRaises(RevisionStop):j.finish_epoch(1)
            with self.assertRaises(RevisionStop):j.start('train',1)
            o.close()
    def test_skip_pause_boundaries_5_and25_are_persistent(self):
        for limit,consecutive in [(5,True),(25,False)]:
            with tempfile.TemporaryDirectory() as d:
                j=Journal(Path(d)/'b.json','x',1,60,1,AmpLimits(5,25));j.initialize()
                for i in range(limit):
                    if not consecutive and i:
                        j.start('train',0);j.compute('forward');j.compute('backward');j.compute('optimizer');j.resolve_train(StepEvidence(1,1,1,1,1,True,True),True)
                    j.start('train',0);j.compute('forward');j.compute('backward');j.resolve_train(StepEvidence(1,float('inf'),2,1,0,True,True),False)
                self.assertEqual(j.read()['status'],'STOPPED');self.assertEqual(j.read()['skip'],limit)
                with self.assertRaises(RevisionStop):j.start('train',0)
    def test_budget_binding_and_loss_fatal_stops_not_forged_success(self):
        with tempfile.TemporaryDirectory() as d:
            t,j,o=setup(Path(d));t.loss=lambda z,y:z.sum()*torch.tensor(float('nan'));t.on_train_epoch_start()
            with self.assertRaises(RevisionStop):t.train_step(next(t.dataloader_train))
            self.assertEqual(j.read()['success'],0);self.assertEqual(j.read()['status'],'STOPPED')
            other=Journal(j.path,'wrong',2,3,2,AmpLimits(5,25))
            with self.assertRaises(RevisionStop):other.read()
            o.close()

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='resume':resume_worker(sys.argv[2])
    else:unittest.main()
