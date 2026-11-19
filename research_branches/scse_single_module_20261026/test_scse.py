import os,sys,json,tempfile,unittest,subprocess,copy
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from scse_gate import CODE,STAGE as S,protocol
import torch,numpy as np,random
from scse_model import SCSE,SCSEUNet,load_m0
from retinal_m0.model import UNet
from retinal_m0.maples_core import weighted_masked_loss
from scse_core import Budget,run_updates,state_digest,pair_record
from retinal_m0.maples_g1_safety import GateError
torch.set_num_threads(1)

class Fixture:
    trajectory_sha256='cpu-fixture'
    def __len__(self):return 16
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(1000+i);x=torch.randn(3,32,32,generator=g);y=(torch.rand(1,32,32,generator=g)>.8).float();f=torch.ones_like(y);f[:,:,:3]=0
        return {'image':x,'mask':y,'fov':f,'draw':i,'filename':'train_fixture','geometry':torch.tensor([0,0,0,0,0]),'input_sha256':state_digest([x,y,f])}
def fixture_build():
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);m=SCSEUNet((8,16,32));o=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.0001,foreach=False);s=torch.amp.GradScaler('cuda',enabled=False);ds=Fixture();pairs=[pair_record(torch.utils.data.default_collate([ds[j] for j in range(i*4,i*4+4)]),i+1) for i in range(4)];return m,o,s,ds,pairs
def fixture_run(root,start,end,initialize):
    m,o,s,ds,pairs=fixture_build();j=Budget(root/'budget.json','fixture',global_cap=4,track_caps={'SCSE':4},forward_cap=4,gradient_cap=4)
    if initialize:j.initialize()
    return run_updates(model=m,optimizer=o,scaler=s,dataset=ds,track='SCSE',start=start,end=end,out=root/'run',budget=j,binding_sha='fixture',expected_pairs=pairs,device='cpu')

class ModuleTests(unittest.TestCase):
    def test_initial_identity_and_input_gradient_exact(self):
        m=SCSE();x=torch.randn(2,32,16,16,requires_grad=True);self.assertTrue(torch.equal(m(x),x));self.assertTrue(torch.equal(torch.autograd.grad(m(x).sum(),x)[0],torch.ones_like(x)))
    def test_formula_and_autograd_agree_with_paper_sum(self):
        m=SCSE().double()
        with torch.no_grad():m.fc2.weight.normal_(0,.02);m.spatial.weight.normal_(0,.02)
        x=torch.randn(1,32,8,8,dtype=torch.double,requires_grad=True);c=torch.sigmoid(m.fc2(torch.relu(m.fc1(x.mean((2,3))))))[:,:,None,None];s=torch.sigmoid(m.spatial(x));a=m(x);b=c*x+s*x
        self.assertTrue(torch.allclose(a,b,atol=1e-14,rtol=1e-14));ga=torch.autograd.grad(a.square().mean(),(x,*m.parameters()),retain_graph=True);gb=torch.autograd.grad(b.square().mean(),(x,*m.parameters()));self.assertTrue(all(torch.allclose(p,q,atol=1e-13,rtol=1e-12) for p,q in zip(ga,gb)))
    def test_module_init_does_not_change_main_rng(self):
        before=torch.get_rng_state().clone();m=SCSE();self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(sum(p.numel() for p in m.parameters()),1105);self.assertFalse(any(isinstance(p,torch.nn.modules.batchnorm._BatchNorm) for p in m.modules()))
    def test_whole_network_common_keys_rng_and_outputs_exact(self):
        torch.manual_seed(2026);a=UNet((8,16,32));want=torch.get_rng_state().clone();torch.manual_seed(2026);b=SCSEUNet((8,16,32));self.assertTrue(torch.equal(want,torch.get_rng_state()));load_m0(b,a.state_dict());x=torch.randn(1,3,32,32);self.assertTrue(torch.equal(a(x),b(x)));b.bypass=True;self.assertTrue(torch.equal(a(x),b(x)))
    def test_true_parameter_gradients_and_expected_first_W1_zero(self):
        m,o,s,ds,pairs=fixture_build();x=ds[0]['image'][None];y=ds[0]['mask'][None];f=ds[0]['fov'][None]
        for i in range(3):
            o.zero_grad();loss=weighted_masked_loss(m(x),y,f,f);loss.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()));self.assertGreater(float(m.scse.fc2.weight.grad.norm()),0)
            if i==0:self.assertEqual(float(m.scse.fc1.weight.grad.norm()),0)
            if i==2:self.assertGreater(float(m.scse.fc1.weight.grad.norm()),0)
            o.step()
    def test_empty_GT_partial_FOV_and_empty_FOV_error(self):
        m=SCSEUNet((8,16,32));x=torch.randn(2,3,32,32);y=torch.zeros(2,1,32,32);f=torch.ones_like(y);f[:,:,:,:8]=0;loss=weighted_masked_loss(m(x),y,f,f);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
        with self.assertRaises(ValueError):weighted_masked_loss(m(x),y,torch.zeros_like(f),torch.zeros_like(f))
    def test_real_M0_all_shared_parameters_exact(self):
        q=protocol();saved=torch.load(q['baseline_config']['source_m0'],map_location='cpu',weights_only=True);m=SCSEUNet();load_m0(m,saved['model']);self.assertTrue(all(torch.equal(saved['model'][k],m.state_dict()[k]) for k in saved['model']));self.assertEqual(sum(p.numel() for p in m.scse.parameters()),1105)
    def test_full_fixed_input_resume_in_new_process(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'continuous';a.mkdir();fixture_run(a,0,4,True);want=torch.load(a/'run/last.pt',weights_only=True)['digests'];b=Path(d)/'split';b.mkdir();fixture_run(b,0,2,True)
            p=subprocess.run([sys.executable,'-B',__file__,'resume',str(b)],cwd=CODE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120);self.assertEqual(p.returncode,0,p.stdout);got=torch.load(b/'run/last.pt',weights_only=True)['digests'];self.assertEqual(want,got)
    def test_budget_duplicate_extra_call_and_rebind_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'b.json';j=Budget(p,'x',global_cap=1,track_caps={'SCSE':1},forward_cap=1,gradient_cap=1);j.initialize();j.before_forward('SCSE',1)
            with self.assertRaises(GateError):j.before_forward('SCSE',1)
            j.before_gradient('SCSE',1)
            with self.assertRaises(GateError):j.before_gradient('SCSE',1)
            j.reserve('SCSE',1);j.optimizer_started('SCSE',1);j.resolve('SCSE','success')
            with self.assertRaises(GateError):j.before_forward('SCSE',2)
            with self.assertRaises(GateError):j.initialize()
            with self.assertRaises(GateError):Budget(p,'wrong',global_cap=1,track_caps={'SCSE':1},forward_cap=1,gradient_cap=1).snapshot()

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='resume':fixture_run(Path(sys.argv[2]),2,4,False)
    else:unittest.main(verbosity=2)
