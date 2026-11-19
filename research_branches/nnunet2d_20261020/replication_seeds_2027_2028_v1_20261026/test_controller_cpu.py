"""Exercise actual serial controller paths with CPU-only fake workers and temporary artifacts."""
import os,sys,json,tempfile,unittest,types,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from unittest.mock import patch
import replication_controller as ctl
import replication_gate as real
from retinal_m0.maples_g1_safety import atomic_json,read_json

class ControllerTests(unittest.TestCase):
    def environment(self,path):
        root=Path(path);group=root/'experiments/group';group.mkdir(parents=True);fake=types.SimpleNamespace(GROUP=group,ROOT=root,CODE=real.CODE,BRANCH=real.BRANCH,GPU=real.GPU,GROUP_PROTOCOL_SHA='fixture',group_protocol=lambda:{},sha=real.sha,update_group=lambda **kw:None,update_status=lambda **kw:None,validate=lambda **kw:'key')
        def context(seed):fake.SEED=seed;fake.STAGE=group/'runs'/f'seed_{seed}';fake.STAGE.mkdir(parents=True,exist_ok=True)
        fake.set_context=context;atomic_json(group/'CPU_REVIEW.json',{'status':'CPU_ACCEPTED'});atomic_json(group/'source_binding.json',{'binding_sha256':'fixture'})
        atomic_json(group/'SERIAL_RELEASE.json',{'seeds':[2027,2028],'group_protocol_sha256':'fixture','group_binding_sha256':'fixture','CPU_review_sha256':real.sha(group/'CPU_REVIEW.json'),'GPU':real.GPU,'total_attempt_cap':50000,'order':[2027,2028],'no_outcome_selection':True,'no_automatic_retry':True,'Test_allowed':False});return fake
    def test_actual_serial_main_keeps2028_after_negative2027(self):
        with tempfile.TemporaryDirectory() as d:
            fake=self.environment(d);calls=[]
            def release(seed):
                if seed==2028:self.assertTrue((fake.GROUP/'runs/seed_2027/completion.json').exists())
                fake.set_context(seed);calls.append(('release',seed));return 'key'
            def child(role,key):
                calls.append((role,fake.SEED));atomic_json(fake.STAGE/'training_completion.json',{'attempts':25000,'internal_train_monitor_batches':5000});atomic_json(fake.STAGE/'analysis/endpoint_100epochs.json',{'readout':{'classification':'NO_DICE_IMPROVEMENT_SIGNAL'},'absolute_means':{'NNUNET':{'dice':.80},'A0':{'dice':.83}}});return {'exit_code':0}
            with patch.object(ctl,'g',fake),patch.object(ctl,'release_seed',side_effect=release),patch.object(ctl,'wait_resources'),patch.object(ctl,'run_child',side_effect=child),patch.object(ctl.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='CPU fixture')),patch.object(ctl,'record_header'),patch.object(ctl,'aggregate',return_value={}):ctl.main()
            self.assertEqual(calls,[('release',2027),('train',2027),('infer',2027),('release',2028),('train',2028),('infer',2028)])
    def test_actual_serial_main_technical_failure_stops_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            fake=self.environment(d);released=[]
            def release(seed):fake.set_context(seed);released.append(seed);return 'key'
            with patch.object(ctl,'g',fake),patch.object(ctl,'release_seed',side_effect=release),patch.object(ctl,'wait_resources'),patch.object(ctl,'run_child',side_effect=RuntimeError('injected CPU failure')):
                with self.assertRaises(RuntimeError):ctl.main()
            self.assertEqual(released,[2027]);self.assertTrue((fake.GROUP/'failure.json').exists());self.assertFalse((fake.GROUP/'runs/seed_2028').exists())
    def test_busy_GPU_waits_without_switch_or_eviction(self):
        with tempfile.TemporaryDirectory() as d:
            fake=self.environment(d);fake.resources=unittest.mock.Mock(side_effect=[RuntimeError('GPU0 busy; never switch'),{'GPU':real.GPU}])
            with patch.object(ctl,'g',fake),patch.object(ctl.time,'sleep') as sleep:self.assertEqual(ctl.wait_resources(2027),{'GPU':real.GPU});sleep.assert_called_once_with(30)
            self.assertEqual(read_json(fake.GROUP/'job_state.json')['state'],'WAITING_GPU0')
    def test_actual_aggregate_on_A0_fixtures_preserves_zero_deltas_and_context(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);fake=types.SimpleNamespace(GROUP=path,ROOT=real.ROOT,group_protocol=real.group_protocol,sha=real.sha)
            for seed in [2027,2028]:
                p=path/f'runs/seed_{seed}/analysis';p.mkdir(parents=True);shutil.copy2(real.GROUP/f'runs/seed_{seed}/cpu_reference/full_endpoint_fixture/endpoint_100epochs.json',p/'endpoint_100epochs.json');atomic_json(p.parent/'train_process_result.json',{'elapsed_process_seconds':0,'CPU_FIXTURE_ONLY':True})
            with patch.object(ctl,'g',fake):out=ctl.aggregate()
            self.assertTrue(all(v['mean_delta']==v['sample_SD_ddof1_n2']==0 for v in out['metrics'].values()));self.assertFalse(out['Dice_replication_signal_both_seeds']);self.assertEqual(set(out['historical_context']),{'nnUNet2026','SCSE2026'})

if __name__=='__main__':unittest.main(verbosity=2)
