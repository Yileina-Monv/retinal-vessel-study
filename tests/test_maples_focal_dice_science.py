import tempfile
import unittest
import numpy as np
import torch
torch.set_num_threads(2)

from retinal_m0.maples_focal_dice_science_core import Budget, verify_pair_rows, validate_scope
from retinal_m0.maples_focal_dice_science_analysis import hard_cldice, paired_bootstrap


class BudgetTests(unittest.TestCase):
    def ledger(self):
        td=tempfile.TemporaryDirectory(dir='/dev/shm'); self.addCleanup(td.cleanup)
        b=Budget(td.name+'/budget.json',binding='fixture',global_cap=2,track_caps={'D':2},forward_cap=2,gradient_cap=6)
        b.initialize(); return b

    def attempt(self,b,step,outcome):
        b.before_forward('D',step); b.before_gradient('D',step,3); b.reserve('D',step)
        b.optimizer_started('D',step); b.resolve('D',outcome)

    def test_failed_optimizer_update_is_charged_and_stops_without_refund(self):
        b=self.ledger(); self.attempt(b,1,'failed'); s=b.snapshot()
        self.assertEqual(s['charged_events'],1); self.assertEqual(s['failed_events'],1)
        self.assertEqual(s['verified_successes'],0)
        with self.assertRaises(RuntimeError): b.before_forward('D',2)

    def test_unknown_optimizer_update_is_charged_and_stops_without_refund(self):
        b=self.ledger(); self.attempt(b,1,'uncertain'); s=b.snapshot()
        self.assertEqual(s['charged_events'],1); self.assertEqual(s['uncertain_events'],1)
        with self.assertRaises(RuntimeError): b.before_forward('D',2)

    def test_optimizer_cannot_be_reserved_without_three_audited_gradients(self):
        b=self.ledger(); b.before_forward('D',1); b.before_gradient('D',1,2)
        with self.assertRaises(RuntimeError): b.reserve('D',1)


class BindingTests(unittest.TestCase):
    def test_pair_rows_require_exact_identity_and_order(self):
        expected=[{'draw':0,'filename':'a.png','geometry':[1,2]}]
        self.assertTrue(verify_pair_rows(expected,expected))
        with self.assertRaises(RuntimeError): verify_pair_rows([{'draw':0,'filename':'b.png','geometry':[1,2]}],expected)

    def test_scope_rejects_any_protocol_widening(self):
        scope={'method':'D','seed':2026,'updates':2000,'gamma':1}
        self.assertTrue(validate_scope(scope))
        with self.assertRaises(RuntimeError): validate_scope({**scope,'seed':2027})


class HardClDiceTests(unittest.TestCase):
    def test_empty_case_rules_are_fixed(self):
        z=np.zeros((7,7),dtype=bool)
        self.assertEqual(hard_cldice(z,z,np.ones_like(z))['cldice'],1.0)
        self.assertIsNone(hard_cldice(z,z,np.ones_like(z))['topology_precision'])
        line=z.copy(); line[3,1:6]=True
        self.assertEqual(hard_cldice(line,z,np.ones_like(z))['cldice'],0.0)

    def test_bootstrap_stops_on_prediction_na_in_fixed_image_set(self):
        with self.assertRaises(ValueError): paired_bootstrap([0.5, None], [0.4, 0.3])

    def test_fov_mask_is_applied_before_skeletonization_and_keeps_u(self):
        gt=np.zeros((9,9),dtype=bool); gt[4,1:8]=True
        pred=gt.copy(); pred[4,5:8]=False
        fov=np.ones_like(gt); fov[:,7:]=False
        got=hard_cldice(pred,gt,fov)
        self.assertAlmostEqual(got['topology_precision'],1.0)
        self.assertLess(got['topology_sensitivity'],1.0)
        uncertain=np.zeros_like(gt); uncertain[4,1:4]=True
        self.assertEqual(hard_cldice(pred,gt,fov,uncertain=uncertain)['topology_sensitivity'],got['topology_sensitivity'])


if __name__ == '__main__': unittest.main()

class TinyPairs:
    def __len__(self): return 8
    def __getitem__(self, i):
        import torch
        x=torch.zeros(3,9,9); x[0]=float(i%2); x[1,2:7,4]=1
        y=torch.zeros(1,9,9); y[0,4,2:7]=1
        f=torch.ones(1,9,9)
        return {'image':x,'mask':y,'fov':f,'draw':i,'filename':f'image_{i//4}.png',
                'geometry':torch.tensor([i,1,2]),'input_sha256':f'input-{i}'}


def tiny_expected(start,end):
    return [{'step':j//4+1,'draws':list(range(j,j+4)),
             'filenames':[f'image_{j//4}.png']*4,
             'geometry':[[i,1,2] for i in range(j,j+4)],
             'input_sha256':[f'input-{i}' for i in range(j,j+4)]}
            for j in range(start*4,end*4,4)]


class SharedLoopTests(unittest.TestCase):
    def _run(self, out, split):
        import torch
        from retinal_m0.maples_focal_dice_science_core import run_updates
        td=tempfile.TemporaryDirectory(dir='/dev/shm'); self.addCleanup(td.cleanup)
        ledger=Budget(td.name+'/ledger.json',binding='loop',global_cap=2,track_caps={'D':2},forward_cap=2,gradient_cap=6); ledger.initialize()
        import random
        torch.manual_seed(41); np.random.seed(41); random.seed(41)
        model=torch.nn.Conv2d(3,1,1)
        opt=torch.optim.AdamW(model.parameters(),lr=0.001,foreach=False)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        if split:
            run_updates(model=model,optimizer=opt,scaler=scaler,dataset=TinyPairs(),track='D',start=0,end=1,out=out,budget=ledger,binding_sha='loop',expected_pairs=tiny_expected(0,1),device='cpu',gamma=1)
            return model,opt,scaler,ledger,td
        run_updates(model=model,optimizer=opt,scaler=scaler,dataset=TinyPairs(),track='D',start=0,end=2,out=out,budget=ledger,binding_sha='loop',expected_pairs=tiny_expected(0,2),device='cpu',gamma=1)
        return model,opt,scaler,ledger,td

    def test_continuous_and_checkpoint_resume_share_same_update_loop(self):
        import torch
        from pathlib import Path
        from retinal_m0.maples_focal_dice_science_core import run_updates
        with tempfile.TemporaryDirectory(dir='/dev/shm') as root:
            aout=root+'/continuous'; bout=root+'/split'
            am,ao,asc,ab,_=self._run(aout,False)
            bm,bo,bsc,bb,btd=self._run(bout,True)
            prefix=(Path(bout)/'train.jsonl').read_bytes()
            import os, subprocess, sys
            child = '''from pathlib import Path\nimport torch\nfrom tests.test_maples_focal_dice_science import TinyPairs,tiny_expected\nfrom retinal_m0.maples_focal_dice_science_core import Budget,run_updates\nout=Path(__import__('sys').argv[1]); ledger=Budget(__import__('sys').argv[2],binding='loop',global_cap=2,track_caps={'D':2},forward_cap=2,gradient_cap=6)\ntorch.manual_seed(41); import numpy as np, random; np.random.seed(41); random.seed(41); model=torch.nn.Conv2d(3,1,1); opt=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False); scaler=torch.amp.GradScaler('cuda',enabled=False)\nrun_updates(model=model,optimizer=opt,scaler=scaler,dataset=TinyPairs(),track='D',start=1,end=2,out=out,budget=ledger,binding_sha='loop',expected_pairs=tiny_expected(0,2),device='cpu',gamma=1)\n'''
            cp=subprocess.run([sys.executable,'-c',child,bout,btd.name+'/ledger.json'],cwd='${RETINA_PROJECT_ROOT}/code/retinal-vessel-study',env=dict(os.environ,PYTHONPATH='${RETINA_PROJECT_ROOT}/code/retinal-vessel-study'),text=True,capture_output=True)
            self.assertEqual(cp.returncode,0,cp.stdout+'\n'+cp.stderr)
            self.assertEqual(ab.snapshot()['verified_successes'],2)
            self.assertEqual(bb.snapshot()['verified_successes'],2)
            self.assertEqual((Path(bout)/'train.jsonl').read_bytes().startswith(prefix),True)
            from retinal_m0.maples_g1_safety import read_json
            import json
            ar=[json.loads(x) for x in (Path(aout)/'train.jsonl').read_text().splitlines()]
            br=[json.loads(x) for x in (Path(bout)/'train.jsonl').read_text().splitlines()]
            for x,y in zip(ar,br):
                self.assertEqual({k:v for k,v in x.items() if k not in ('elapsed_seconds','pid')}, {k:v for k,v in y.items() if k not in ('elapsed_seconds','pid')})
            ca=torch.load(Path(aout)/'last.pt',map_location='cpu',weights_only=True)
            cb=torch.load(Path(bout)/'last.pt',map_location='cpu',weights_only=True)
            self.assertEqual(ca['digests'],cb['digests'])
            self.assertEqual(ca['step'],cb['step'])
class OptimizerStateTests(unittest.TestCase):
    def test_nonfinite_optimizer_post_hook_is_failed_not_success(self):
        import torch
        from retinal_m0.maples_focal_dice_science_core import run_updates
        with tempfile.TemporaryDirectory(dir='/dev/shm') as root:
            ledger=Budget(root+'/budget.json',binding='nan',global_cap=1,track_caps={'D':1},forward_cap=1,gradient_cap=3); ledger.initialize()
            torch.manual_seed(8); model=torch.nn.Conv2d(3,1,1); opt=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False)
            opt.register_step_post_hook(lambda optimizer,args,kwargs: optimizer.param_groups[0]['params'][0].data.fill_(float('nan')))
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            with self.assertRaises(RuntimeError):
                run_updates(model=model,optimizer=opt,scaler=scaler,dataset=TinyPairs(),track='D',start=0,end=1,out=root+'/run',budget=ledger,binding_sha='nan',expected_pairs=tiny_expected(0,1),device='cpu',gamma=1)
            snap=ledger.snapshot()
            self.assertEqual(snap['status'],'FAILED_STOP')
            self.assertEqual(snap['failed_events'],1)
            self.assertEqual(snap['verified_successes'],0)


class SessionTests(unittest.TestCase):
    def test_real_cli_worker_must_be_direct_child(self):
        from retinal_m0.maples_focal_dice_science import cpu_session_probe
        self.assertTrue(cpu_session_probe()['passed'])

class FinalWiringTests(unittest.TestCase):
    def test_probability_cache_requires_exact_ids_dtype_shape_and_range(self):
        from retinal_m0.maples_focal_dice_science import validate_probability_records
        from pathlib import Path
        import json
        with tempfile.TemporaryDirectory(dir='/dev/shm') as root:
            d=Path(root); refs=[{'filename':f'im{i}','native_shape':[2,3]} for i in range(28)]
            from retinal_m0.maples_focal_dice_science import sha
            rows=[]
            for r in refs:
                f=d/(r['filename']+'.npy'); np.save(f,np.zeros((2,3),np.float32)); rows.append({'filename':r['filename'],'probability_sha256':sha(f)})
            self.assertTrue(validate_probability_records(rows,refs,d))
            with self.assertRaises(RuntimeError):validate_probability_records(rows[:-1],refs,d)
            with self.assertRaises(RuntimeError):validate_probability_records(rows[::-1],refs,d)
            f=d/'im0.npy';np.save(f,np.zeros((2,3),np.float64));rows[0]['probability_sha256']=sha(f)
            with self.assertRaises(RuntimeError):validate_probability_records(rows,refs,d)

    def test_checkpoint_cannot_self_certify_state_different_from_committed_log(self):
        import torch
        from pathlib import Path
        from retinal_m0.maples_focal_dice_science_core import verify_checkpoint,atomic_checkpoint,state_digest
        with tempfile.TemporaryDirectory(dir='/dev/shm') as root:
            out=Path(root)/'run'; helper=SharedLoopTests(); helper._run(str(out),False)
            saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
            verify_checkpoint(out/'last.pt',expected_step=2,binding_sha='loop',run_dir=out)
            first=next(iter(saved['model']));saved['model'][first].add_(1);saved['digests']['model']=state_digest(saved['model'])
            atomic_checkpoint(out/'last.pt',saved,replace=True)
            with self.assertRaises(RuntimeError):verify_checkpoint(out/'last.pt',expected_step=2,binding_sha='loop',run_dir=out)
            helper.doCleanups()

    def test_stop_does_not_charge_same_failed_compute_twice(self):
        helper=BudgetTests();b=helper.ledger();b.before_forward('D',1);b.stop('first');b.stop('controller')
        self.assertEqual(b.snapshot()['failed_compute'],1);helper.doCleanups()

    def test_fixed_gt_validity_includes_secondary_metrics_and_neighborhood_omega(self):
        from retinal_m0.maples_focal_dice_science_analysis import fixed_gt_metric_validity
        a={'GT':np.ones((3,3),bool),'FOV':np.ones((3,3),bool),'Omega':np.ones((3,3),bool),
           'near':np.zeros((3,3),bool),'thin':np.zeros((3,3),bool),'far':np.ones((3,3),bool),
           'full_S3':np.eye(3,dtype=bool),'coarse_vessel_reference':np.eye(3,dtype=bool)}
        r={'B2_valid_paths':2}
        v=fixed_gt_metric_validity(a,r)
        self.assertTrue(v['full_S3_d1']);self.assertTrue(v['arc_recall_d1_macro']);self.assertFalse(v['near_thin_recall_d1'])
        a['near'][0,0]=True;a['thin'][0,0]=True;a['Omega'][0,0]=False
        self.assertFalse(fixed_gt_metric_validity(a,r)['near_thin_recall_d1'])


class EvaluationHookTests(unittest.TestCase):
    def test_fixed_evaluation_callback_does_not_consume_training_forward_budget(self):
        import torch
        from pathlib import Path
        from retinal_m0.maples_focal_dice_science_core import run_updates
        with tempfile.TemporaryDirectory(dir='/dev/shm') as root:
            b=Budget(root+'/b.json','evalhook',global_cap=2,track_caps={'D':2},forward_cap=2,gradient_cap=6);b.initialize()
            torch.manual_seed(5);m=torch.nn.Conv2d(3,1,1);o=torch.optim.AdamW(m.parameters(),lr=.001,foreach=False);s=torch.amp.GradScaler('cuda',enabled=False)
            called=[]
            def evaluation(step,payload):
                with torch.inference_mode():m(torch.zeros(1,3,9,9))
                called.append(step)
            run_updates(model=m,optimizer=o,scaler=s,dataset=TinyPairs(),track='D',start=0,end=2,out=Path(root)/'run',budget=b,binding_sha='evalhook',expected_pairs=tiny_expected(0,2),device='cpu',on_update=evaluation)
            self.assertEqual(called,[1,2]);self.assertEqual(b.snapshot()['forward_calls'],2)


class DiceReadoutTests(unittest.TestCase):
    def analysis(self):
        result={m:{'D-A0':{'estimate':0.,'lower':0.,'upper':0.}} for m in ['precision','far_recall','fp_density','recall','coarse_vessel_d0','hard_cldice']}
        result['dice']={'D-A0':{'estimate':.002,'lower':.001,'upper':.003}}
        result['candidate_path_d1_macro']={'D-A0':{'estimate':-.02,'lower':-.03,'upper':-.01}}
        return result

    def test_ordinary_dice_is_only_primary_not_skeleton_recall(self):
        from retinal_m0.maples_focal_dice_science_analysis import dice_readout
        r=dice_readout(self.analysis());self.assertTrue(r['eligible_to_request_additional_seeds'])
        self.assertEqual(list(r['primary_mean_differences']),['dice'])
        self.assertFalse(r['independent_validation']);self.assertFalse(r['automatic_seed_expansion'])

    def test_positive_dice_mean_with_zero_crossing_is_insufficient(self):
        from retinal_m0.maples_focal_dice_science_analysis import dice_readout
        a=self.analysis();a['dice']['D-A0']['lower']=-.001
        self.assertEqual(dice_readout(a)['classification'],'DICE_EVIDENCE_INSUFFICIENT')

    def test_dice_signal_with_precision_cost_not_overall_success(self):
        from retinal_m0.maples_focal_dice_science_analysis import dice_readout
        a=self.analysis();a['precision']['D-A0']={'estimate':-.001,'lower':-.002,'upper':0.}
        r=dice_readout(a);self.assertTrue(r['dice_positive_95pct'])
        self.assertEqual(r['classification'],'DICE_SIGNAL_WITH_PROTECTION_COST')
        self.assertFalse(r['eligible_to_request_additional_seeds'])

    def test_guard_interval_crossing_zero_blocks_seed_expansion(self):
        from retinal_m0.maples_focal_dice_science_analysis import dice_readout
        a=self.analysis();a['fp_density']['D-A0']={'estimate':-.001,'lower':-.002,'upper':.0001}
        self.assertFalse(dice_readout(a)['eligible_to_request_additional_seeds'])

    def test_zero_dice_mean_not_positive_signal(self):
        from retinal_m0.maples_focal_dice_science_analysis import dice_readout
        a=self.analysis();a['dice']['D-A0']={'estimate':0.,'lower':0.,'upper':0.}
        self.assertEqual(dice_readout(a)['classification'],'NO_DICE_IMPROVEMENT_SIGNAL')

    def test_undefined_primary_is_not_silently_dropped(self):
        from retinal_m0.maples_focal_dice_science_analysis import dice_readout
        a=self.analysis();a['dice']['D-A0']['estimate']=None
        with self.assertRaises(ValueError):dice_readout(a)


class EvaluationBudgetTests(unittest.TestCase):
    def fixture(self):
        return ({'slots':{'D:200':{'images':{'im':{'status':'INCOMPLETE','attempts':1,'forward_calls':0,'tiles':0}}}}},
                {'runtime':{'evaluation_image_profile':{'im':{'forward_calls':1,'tiles':4}},'evaluation_forward_cap':1,'evaluation_tile_cap':4}})

    def test_inference_counted_before_forward_and_extra_denied(self):
        from retinal_m0.maples_focal_dice_science import reserve_evaluation_forward
        b,q=self.fixture();reserve_evaluation_forward(b,'D:200','im',4,q)
        with self.assertRaises(RuntimeError):reserve_evaluation_forward(b,'D:200','im',4,q)
        self.assertEqual(b['slots']['D:200']['images']['im']['forward_calls'],1)

    def test_completed_image_cannot_be_retried(self):
        from retinal_m0.maples_focal_dice_science import reserve_evaluation_forward
        b,q=self.fixture();b['slots']['D:200']['images']['im']['status']='COMPLETE'
        with self.assertRaises(RuntimeError):reserve_evaluation_forward(b,'D:200','im',4,q)

    def test_unplanned_slot_refused(self):
        from retinal_m0.maples_focal_dice_science import reserve_evaluation_forward
        b,q=self.fixture()
        with self.assertRaises(RuntimeError):reserve_evaluation_forward(b,'D:100','im',4,q)

    def test_gamma_bool_zero_or_two_is_not_candidate(self):
        for gamma in [True,0,2]:
            with self.assertRaises(RuntimeError):validate_scope({'method':'D','seed':2026,'updates':2000,'gamma':gamma})


class LiveSessionTests(unittest.TestCase):
    def test_real_parent_identity_is_required_for_each_runtime_permission(self):
        import os
        from retinal_m0.maples_focal_dice_science import verify_live_session,process_token
        session={'controller_pid':os.getppid(),'worker_pid':os.getpid(),'controller_token':process_token(os.getppid()),'worker_token':process_token(os.getpid()),'nonce':'x'*48}
        self.assertTrue(verify_live_session(session,'x'*48))
        session['controller_token']='dead_or_replaced'
        with self.assertRaises(RuntimeError):verify_live_session(session,'x'*48)
