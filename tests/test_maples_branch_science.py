"""CPU-only scientific preparation checks; never build or forward the retinal model."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from retinal_m0.maples_branch_science_reference import build_image_reference, valid_runs
from retinal_m0.maples_branch_science_analysis import path_d1, bootstrap_indices
from retinal_m0.maples_branch_science_worker import BudgetLedger, exact_equal, atomic_checkpoint
from retinal_m0.maples_branch_science import validate_science_release

class ReferenceTests(unittest.TestCase):
    def test_U_splits_path_and_counts_only_Omega_coordinates(self):
        path=np.array([[1,1],[1,2],[1,3],[1,4],[1,5]],dtype=np.int32)
        valid=np.ones((4,7),bool); valid[1,3]=False
        runs=valid_runs(path,valid)
        self.assertEqual([x.tolist() for x in runs],[[[1,1],[1,2]],[[1,4],[1,5]]])
    def test_shared_anchor_is_counted_per_path_but_once_in_union(self):
        gt=np.zeros((5,5),bool); gt[2,1:4]=True; fov=np.ones_like(gt); u=np.zeros_like(gt)
        p1=np.array([[2,1],[2,2]],np.int32); p2=np.array([[2,2],[2,3]],np.int32)
        paths=[dict(coords=p1,width_native=np.ones(2)*.1,arc_native=np.array([0.,1.]),body=np.array([0,1])),
               dict(coords=p2,width_native=np.ones(2)*.1,arc_native=np.array([0.,1.]),body=np.array([0,1]))]
        ref=build_image_reference(gt,fov,u,paths,alpha=20.0)
        self.assertEqual(ref['paths'][0]['denominator'],2)
        self.assertEqual(ref['paths'][1]['denominator'],2)
        self.assertEqual(ref['B2_unique_points'],3)
    def test_closed_short_or_wide_paths_are_not_candidates(self):
        gt=np.ones((6,6),bool); fov=gt.copy(); u=np.zeros_like(gt)
        paths=[dict(coords=np.array([[1,1],[1,2],[1,1]]),width_native=np.ones(3),arc_native=np.array([0,1,2]),body=np.ones(3)),
               dict(coords=np.array([[2,1],[2,2]]),width_native=np.ones(2)*20,arc_native=np.array([0,1]),body=np.ones(2))]
        ref=build_image_reference(gt,fov,u,paths,alpha=1.0)
        self.assertTrue(all(not x['candidate_B2'] for x in ref['paths']))

class AnalysisTests(unittest.TestCase):
    def test_d1_radius_is_axial_one_not_diagonal(self):
        ref=np.zeros((4,4),bool); ref[1,1]=True
        pred=np.zeros_like(ref); pred[2,2]=True
        self.assertEqual(path_d1([np.array([[1,1]])],pred,ref),0.0)
        pred[1,2]=True
        self.assertEqual(path_d1([np.array([[1,1]])],pred,ref),1.0)
    def test_bootstrap_same_n_shared_and_reproducible(self):
        a=bootstrap_indices(20261016,7,1000); b=bootstrap_indices(20261016,7,1000)
        self.assertTrue(np.array_equal(a,b)); self.assertEqual(a.shape,(1000,7))

class WorkerTests(unittest.TestCase):
    def test_budget_refuses_over_cap_without_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=BudgetLedger(Path(d)/'budget.json','fixture');ledger.initialize_isolated()
            ledger.before_forward('B1',1,1);ledger.before_gradient('B1',1,1);ledger.reserve('B1',1,1)
            ledger.resolve('B1','failed');before=ledger.snapshot()
            with self.assertRaises(RuntimeError):ledger.before_forward('B2',1,1)
            self.assertEqual(ledger.snapshot(),before)
    def test_nested_state_comparison_and_atomic_checkpoint(self):
        a={'w':torch.tensor([1.,2.]),'r':[1,{'x':'s'}]}; b={'w':torch.tensor([1.,2.]),'r':[1,{'x':'s'}]}
        self.assertTrue(exact_equal(a,b)); b['w'][0]=9; self.assertFalse(exact_equal(a,b))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.pt'; atomic_checkpoint(p,a); loaded=torch.load(p,map_location='cpu',weights_only=True)
            self.assertTrue(exact_equal(a,loaded)); self.assertFalse(list(Path(d).glob('*.pending')))
    def test_45_update_loop_exact_across_process_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cont=root/'cont'; split=root/'split'
            def run(out,start,end):
                subprocess.run([sys.executable,'-B','-m','retinal_m0.maples_branch_science_worker',
                    '--fixture-output',str(out),'--start',str(start),'--end',str(end),'--seed','2026'],
                    check=True,capture_output=True,text=True)
            run(cont,0,45); run(split,0,25); run(split,25,45)
            a=torch.load(cont/'step_000045.pt',map_location='cpu',weights_only=False)
            b=torch.load(split/'step_000045.pt',map_location='cpu',weights_only=False)
            for k in ('model','optimizer','scaler','rng','step','next_global_draw','budget','initial'):
                self.assertTrue(exact_equal(a[k],b[k]),k)
            self.assertEqual((cont/'train.jsonl').read_text(),(split/'train.jsonl').read_text())

class GateTests(unittest.TestCase):
    def test_scientific_release_missing_or_wrong_phase_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'release.json'
            with self.assertRaises(RuntimeError): validate_science_release(p,'0'*64)
            p.write_text('{"state":"RELEASED","phase":"G2_ONLY"}')
            with self.assertRaises(RuntimeError): validate_science_release(p,'0'*64)

class SolRegressionTests(unittest.TestCase):
    def arrays(self):
        f=np.ones((8,8),np.uint8); g=np.zeros_like(f); g[2,1:7]=1
        c=np.array([[2,x] for x in range(1,7)],np.int32)
        return dict(GT=g,FOV=f,Omega=f.copy(),U=np.zeros_like(f),far=f.copy(),
          coarse_vessel_reference=g.copy(),coords=c,offsets=np.array([0,6]),
          candidate_B2=np.array([1]),B1_selected_points=np.array([1,1,0,0,0,0]),
          arc_native=np.arange(6,dtype=np.float32),width_native=np.ones(6),body=np.ones(6))
    def test_R5_left_minus_right_not_reversed(self):
        from retinal_m0.maples_branch_science_analysis import paired_analysis
        rows=[dict(filename=str(i),fixed_primary_valid=True,A0={'recall':.4},B1={'recall':.5},B2={'recall':.6}) for i in range(3)]
        d=paired_analysis(rows)['comparisons']
        self.assertAlmostEqual(d['B2-A0']['recall']['estimate'],.2)
        self.assertAlmostEqual(d['B1-A0']['recall']['estimate'],.1)
        self.assertAlmostEqual(d['B2-B1']['recall']['estimate'],.1)
    def test_R6_full_FOV_includes_U(self):
        from retinal_m0.maples_branch_science_analysis import evaluate_image
        a=self.arrays(); a['U']=a['GT'].copy(); a['Omega']=a['FOV'] & ~a['U']
        d=evaluate_image(a['GT'].astype(np.float32),a,'B2')
        self.assertEqual(d['dice']['value'],1.)
    def test_R6_coarse_d0_cannot_take_axial_neighbor(self):
        from retinal_m0.maples_branch_science_analysis import evaluate_image
        a=self.arrays(); p=np.zeros_like(a['GT'],np.float32);p[1,1:7]=1
        self.assertEqual(evaluate_image(p,a,'B2')['coarse_vessel_d0']['value'],0.)
    def test_R6_same_reference_for_A0_B1_B2(self):
        from retinal_m0.maples_branch_science_analysis import evaluate_image
        a=self.arrays(); p=np.zeros_like(a['GT'],np.float32);p[2,1:3]=1
        results=[evaluate_image(p,a,m)['candidate_path_d1_macro'] for m in ['A0','B1','B2']]
        self.assertEqual(results,[results[0]]*3)
    def test_R6_U_intervals_keep_one_original_path(self):
        from retinal_m0.maples_branch_science_analysis import evaluate_image
        a=self.arrays();a['Omega'][2,3]=0
        p=np.zeros_like(a['GT'],np.float32);p[2,1]=1
        # d1 hits first two nodes; U removes node3, remaining original-path denominator5.
        self.assertAlmostEqual(evaluate_image(p,a,'B2')['candidate_path_d1_macro'],2/5)
    def test_R1_real_shared_training_loop_exists(self):
        from retinal_m0 import maples_branch_science_worker as w
        self.assertTrue(callable(getattr(w,'run_segment',None)))
        self.assertTrue(callable(getattr(w,'run_scientific_training',None)))
    def test_R7_matching_fields_do_not_authorize_an_unbound_release(self):
        from retinal_m0.maples_branch_science import validate_science_release, PROTOCOL_SHA
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fake.json'
            p.write_text(json.dumps(dict(state='RELEASED',phase='SCIENTIFIC_B1_B2_2000',protocol_sha256=PROTOCOL_SHA,
             methods=['B1','B2'],seed=2026,**{'lambda':.03},updates_per_arm=2000,official_Test_allowed=False,gpu1_allowed=False)))
            with self.assertRaises(RuntimeError):validate_science_release(p,PROTOCOL_SHA)

class ScienceBoundaryTests(unittest.TestCase):
    def budget(self,root):
        b=BudgetLedger(Path(root)/'budget.json','fixture');b.initialize_isolated();return b
    def compute(self,b,track='B1',step=1,attempt=1):
        b.before_forward(track,step,attempt);b.before_gradient(track,step,attempt)
    def test_pending_update_blocks_other_track(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.budget(d);self.compute(b);b.reserve('B1',1,1)
            with self.assertRaises(RuntimeError):b.before_forward('B2',1,1)
            other=BudgetLedger(Path(d)/'budget.json','fixture')
            self.assertIsNotNone(other.snapshot()['tracks']['B1']['pending'])
            b.mark_crash_uncertain('B1');s=b.snapshot()
            self.assertEqual(s['charged_events'],1);self.assertEqual(s['verified_successes'],0)
            with self.assertRaises(RuntimeError):b.before_forward('B1',1,1)
    def test_AMP_ninth_attempt_stops_without_update(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.budget(d)
            for attempt in range(1,10):
                self.compute(b,attempt=attempt);b.reserve('B1',1,attempt);b.resolve('B1','amp_overflow',amp_retry=True)
            s=b.snapshot();self.assertEqual(s['verified_successes'],0);self.assertEqual(s['charged_events'],0)
            self.assertEqual(s['forward_calls'],9);self.assertNotEqual(s['status'],'READY')
            with self.assertRaises(RuntimeError):b.before_forward('B1',1,10)
    def test_wrong_retry_draw_cannot_skip(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.budget(d);self.compute(b);b.reserve('B1',1,1);b.resolve('B1','amp_overflow',amp_retry=True)
            with self.assertRaises(RuntimeError):b.before_forward('B1',2,1)
            with self.assertRaises(RuntimeError):b.before_forward('B1',1,1)
            b.before_forward('B1',1,2)
    def test_no_implicit_missing_ledger_reset_or_binding_change(self):
        with tempfile.TemporaryDirectory() as d:
            b=BudgetLedger(Path(d)/'budget.json','fixture')
            with self.assertRaises(RuntimeError):b.snapshot()
            b.initialize_isolated()
            with self.assertRaises(RuntimeError):b.initialize_isolated()
            with self.assertRaises(RuntimeError):BudgetLedger(Path(d)/'budget.json','wrong').snapshot()
    def test_gradient_budget_no_extra_audit_calls(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.budget(d);b.before_forward('B1',1,1)
            for i in range(3):b.before_gradient('B1',1,1)
            with self.assertRaises(RuntimeError):b.before_gradient('B1',1,1)
    def test_real_optimizer_failure_is_charged_not_refunded(self):
        from retinal_m0.maples_g2_engineering import optimizer_update
        with tempfile.TemporaryDirectory() as d:
            b=self.budget(d);self.compute(b);m=torch.nn.Linear(1,1);opt=torch.optim.AdamW(m.parameters());scaler=torch.amp.GradScaler('cpu',enabled=False)
            for p in m.parameters():p.grad=torch.ones_like(p)
            def corrupt(*args):
                with torch.no_grad():next(m.parameters()).fill_(float('nan'))
            handle=opt.register_step_post_hook(corrupt)
            with self.assertRaises(RuntimeError):optimizer_update(b,'B1',1,1,opt,scaler,amp_enabled=False)
            handle.remove();s=b.snapshot();self.assertEqual(s['charged_events'],1);self.assertEqual(s['failed_events'],1);self.assertEqual(s['verified_successes'],0)
    def test_science_call_without_session_refuses_before_CUDA(self):
        from retinal_m0.maples_branch_science_worker import run_scientific_training
        with self.assertRaises(RuntimeError):run_scientific_training('B1',Path('/tmp/not_science'))
        self.assertFalse(torch.cuda.is_initialized())
    def test_audit_uses_unscaled_real_gradients_no_side_effect(self):
        from retinal_m0.maples_branch_science_worker import gradient_audit
        m=torch.nn.Linear(1,1,bias=False);m.weight.data.fill_(1.)
        z=m(torch.ones(1,1));base=z.square().sum();aux=.03*z.sum();scaler=torch.amp.GradScaler('cpu',init_scale=1024)
        before=m.weight.detach().clone();rng=torch.get_rng_state().clone()
        a=gradient_audit(m,z,base,aux,scaler=scaler,audit_step=1)
        self.assertAlmostEqual(a['base_grad_l2'],2.);self.assertAlmostEqual(a['weighted_aux_grad_l2'],.03,places=6)
        self.assertTrue(torch.equal(before,m.weight));self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertIsNone(m.weight.grad)
    def test_zero_base_positive_aux_pause(self):
        from retinal_m0.maples_branch_science_worker import gradient_audit
        m=torch.nn.Linear(1,1,bias=False);z=m(torch.ones(1,1));scaler=torch.amp.GradScaler('cpu',init_scale=1024)
        a=gradient_audit(m,z,z.sum()*0,.03*z.sum(),scaler=scaler,audit_step=1)
        self.assertTrue(a['pause_ratio_gt_2']);self.assertTrue(a['zero_base_positive_aux'])
    def test_actual_shared_loop_B2_cross_process_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def run(name,start,end):
                subprocess.run([sys.executable,'-B','-m','retinal_m0.maples_branch_science_worker','--fixture-output',str(root/name),'--start',str(start),'--end',str(end),'--method','B2'],check=True,capture_output=True,text=True)
            run('cont',0,45);run('split',0,25);run('split',25,45)
            a=torch.load(root/'cont/last.pt',map_location='cpu',weights_only=True);b=torch.load(root/'split/last.pt',map_location='cpu',weights_only=True)
            for k in ['model','optimizer','scaler','rng','loader_generator_state','step','next_global_draw','initial','budget']:self.assertTrue(exact_equal(a[k],b[k]),k)
            self.assertEqual((root/'cont/train.jsonl').read_text(),(root/'split/train.jsonl').read_text())
            self.assertEqual(a['budget']['verified_successes'],45)
            self.assertEqual(a['budget']['forward_calls'],45);self.assertEqual(a['budget']['gradient_calls'],55)
            self.assertEqual(a['scaler']['scale'],1024.)
            for state in a['optimizer']['state'].values():self.assertEqual(float(state['step']),45.)
    def test_pair_drift_prevents_any_real_forward_or_update(self):
        from retinal_m0 import maples_branch_science_worker as w
        with tempfile.TemporaryDirectory() as d:
            model,opt,scaler=w._fixture_components(2026);data=w.FixtureDataset();pairs=w.fixture_pairs(data,1);pairs[0]['input_sha256'][0]='wrong'
            b=self.budget(d)
            with self.assertRaises(RuntimeError):w.run_segment(model=model,optimizer=opt,scaler=scaler,dataset=data,track='B1',start=0,end=1,out=Path(d)/'run',budget=b,binding_sha='fixture',expected_pairs=pairs,device='cpu',authorize=lambda:None)
            self.assertEqual(b.snapshot().get('forward_calls',0),0);self.assertEqual(b.snapshot()['verified_successes'],0)
    def test_aux_empty_retains_batch4_and_lambda_once(self):
        from retinal_m0.maples_branch_bce import normalized_positive_bce
        z=torch.zeros(4,1,2,2,requires_grad=True);v=torch.ones_like(z);s=torch.zeros_like(z);s[0,0,0,0]=1
        aux=.03*normalized_positive_bce(z,v,v,s);aux.backward()
        self.assertAlmostEqual(float(aux),.03*np.log(2)/4)
        self.assertAlmostEqual(float(z.grad[0,0,0,0]),-.03/8)
        self.assertEqual(int((z.grad!=0).sum()),1)
    def test_fixed_valid_metric_NA_not_silently_deleted(self):
        from retinal_m0.maples_branch_science_analysis import paired_analysis
        rows=[dict(filename='x',fixed_primary_valid=True,A0={'recall':.4},B1={'recall':None},B2={'recall':.6})]
        with self.assertRaises(ValueError):paired_analysis(rows)
    def test_absent_method_metric_not_silently_intersected_away(self):
        from retinal_m0.maples_branch_science_analysis import paired_analysis
        rows=[dict(filename='x',fixed_primary_valid=True,A0={'recall':.4},B1={},B2={'recall':.6})]
        with self.assertRaises(ValueError):paired_analysis(rows)
    def test_metric_specific_n_and_same_n_bootstrap(self):
        from retinal_m0.maples_branch_science_analysis import paired_analysis
        rows=[dict(filename=str(i),fixed_primary_valid=i<2,fixed_valid_metrics={'near':i<1},
          **{m:{'recall':.2+i*.1+j*.02,'near':.2+j*.03 if i<1 else None} for j,m in enumerate(['A0','B1','B2'])}) for i in range(3)]
        r=paired_analysis(rows);self.assertEqual(r['comparisons']['B2-A0']['recall']['n_images'],3);self.assertEqual(r['comparisons']['B2-A0']['near']['n_images'],1)
        self.assertEqual(set(r['bootstrap_indices_by_n']),{'1','3'})

class SolAdditionalAcceptanceTests(unittest.TestCase):
    def test_config_checks_reject_budget_or_loss_dose_drift(self):
        from retinal_m0 import maples_branch_science as e
        cfg=json.loads((e.A0/'config.json').read_text());cfg.update(method='B1',**{'lambda':.03},protocol_sha256=e.PROTOCOL_SHA,safe_stop_seconds=None)
        e.validate_config(cfg,'B1')
        cfg['learning_rate']=1e-2
        with self.assertRaises(RuntimeError):e.validate_config(cfg,'B1')
        cfg['learning_rate']=1e-4;cfg['lambda']=.3
        with self.assertRaises(RuntimeError):e.validate_config(cfg,'B1')
    def test_environment_inspection_never_initializes_CUDA(self):
        from retinal_m0.maples_branch_science import environment_versions
        v=environment_versions();self.assertIn('torch',v);self.assertIn('skimage',v);self.assertFalse(torch.cuda.is_initialized())
    def test_false_positive_increase_is_decline_not_improvement(self):
        from retinal_m0 import maples_branch_science_analysis as a
        self.assertTrue(callable(getattr(a,'change_summary',None)))
        r=a.change_summary([dict(filename='x',delta=.1),dict(filename='y',delta=-.1),dict(filename='z',delta=0.)],'fp_density')
        self.assertEqual(r['improved'],1);self.assertEqual(r['declined'],1);self.assertEqual(r['unchanged'],1)
        self.assertEqual(r['rows'][0]['judgment'],'declined')
    def test_clean_pause_keeps_committed_last_and_durable_budget_ready(self):
        from retinal_m0 import maples_branch_science_worker as w
        with tempfile.TemporaryDirectory() as d:
            model,opt,scaler=w._fixture_components(2026);data=w.FixtureDataset();pairs=w.fixture_pairs(data,2)
            b=BudgetLedger(Path(d)/'budget.json','fixture');b.initialize_isolated()
            def authorize():
                s=b.snapshot()
                if s['verified_successes']==1 and s.get('compute_pending') is None:raise w.CleanPause('test boundary')
            with self.assertRaises(w.CleanPause):w.run_segment(model=model,optimizer=opt,scaler=scaler,dataset=data,track='B1',start=0,end=2,out=Path(d)/'run',budget=b,binding_sha='fixture',expected_pairs=pairs,device='cpu',authorize=authorize)
            s=b.snapshot();last=torch.load(Path(d)/'run/last.pt',map_location='cpu',weights_only=True)
            self.assertEqual(s['status'],'READY');self.assertEqual(last['step'],1);self.assertEqual(s['forward_calls'],1)
            self.assertEqual(s,last['budget'])
    def test_detached_launcher_is_held_before_spawning(self):
        from unittest.mock import patch
        from retinal_m0 import maples_branch_science as e
        with patch.object(e.subprocess,'Popen') as popen:
            with self.assertRaises(RuntimeError):e.launch_authorized()
            popen.assert_not_called()
    def test_completed_evaluation_binding_checked_without_inference(self):
        from unittest.mock import patch
        from retinal_m0 import maples_branch_science as e
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dest=root/'runs/B1/evaluation/step_000200';pd=dest/'probabilities';pd.mkdir(parents=True)
            refs=root/'references/dev';refs.mkdir(parents=True);(refs/'manifest.json').write_text('{}')
            cp=root/'runs/B1/step_000200.pt';cp.write_bytes(b'checkpoint fixture, not a model')
            np.save(pd/'image.npy',np.ones((1,1),np.float32),allow_pickle=False)
            record=dict(method='B1',step=200,threshold=.5,reference_manifest_sha256=e.file_sha(refs/'manifest.json'),checkpoint_sha256=e.file_sha(cp),images=[dict(filename='image',probability_sha256=e.file_sha(pd/'image.npy'))])
            (dest/'metrics.json').write_text(json.dumps(record))
            with patch.object(e,'OUT',root),patch.object(e,'DEV_NAMES',['image']):
                self.assertTrue(e.verify_completed_evaluation('B1',200));self.assertFalse(e.verify_completed_evaluation('B1',400))
                np.save(pd/'image.npy',np.zeros((1,1),np.float32),allow_pickle=False)
                with self.assertRaises(RuntimeError):e.verify_completed_evaluation('B1',200)
    def test_actual_base_and_aux_shared_loop_matches_independent_one_step(self):
        from retinal_m0 import maples_branch_science_worker as w
        from retinal_m0.maples_core import weighted_masked_loss
        from retinal_m0.maples_branch_bce import normalized_positive_bce
        with tempfile.TemporaryDirectory() as d:
            expected,opt,scaler=w._fixture_components(2026);data=w.FixtureDataset('B1')
            batch={k:torch.stack([data[i][k] for i in range(4)]) for k in ('image','mask','fov','selected')}
            z=expected(batch['image']);loss=weighted_masked_loss(z,batch['mask'],batch['fov'],batch['fov'])+.03*normalized_positive_bce(z,batch['mask'],batch['fov'],batch['selected'])
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(expected.parameters(),1.)
            scaler.step(opt);scaler.update()
            actual=w.run_fixture_segment(Path(d)/'actual',start=0,end=1)
            self.assertTrue(exact_equal(actual['model'],expected.state_dict()))
            self.assertTrue(exact_equal(actual['optimizer'],opt.state_dict()))
            self.assertTrue(exact_equal(actual['scaler'],scaler.state_dict()))
    def test_cap4000_and_per_arm2000_are_hard_not_extra11(self):
        from retinal_m0.maples_g2_engineering import atomic_json
        with tempfile.TemporaryDirectory() as d:
            b=BudgetLedger(Path(d)/'budget.json','fixture');b.initialize_isolated();s=b.snapshot()
            keys=['charged_events','verified_successes','attempts','amp_retries','failed_events','uncertain_events','optimizer_events']
            for name in ['B1','B2']:
                t={k:2000 if k in ['charged_events','verified_successes','attempts','optimizer_events'] else 0 for k in keys}
                t.update(pending=None,retry_cursor=None);s['tracks'][name]=t
            for k in keys:s[k]=sum(t[k] for t in s['tracks'].values())
            atomic_json(b.path,s);before=b.snapshot()
            with self.assertRaises(RuntimeError):b.before_forward('B1',2001,1)
            with self.assertRaises(RuntimeError):b.before_forward('B2',2001,1)
            self.assertEqual(before,b.snapshot())
    def test_empty_target_exact_zero_and_no_resampling(self):
        from retinal_m0.maples_branch_bce import normalized_positive_bce
        x=torch.randn(4,1,3,3,requires_grad=True);one=torch.ones_like(x);zero=torch.zeros_like(x)
        y=.03*normalized_positive_bce(x,one,one,zero);y.backward()
        self.assertEqual(float(y),0.);self.assertEqual(int(torch.count_nonzero(x.grad)),0)
    def test_axial_d1_at_image_border_does_not_wrap(self):
        from retinal_m0.maples_branch_science_analysis import cross_dilate
        p=np.zeros((5,5),bool);p[0,0]=1;h=cross_dilate(p)
        self.assertEqual(int(h.sum()),3);self.assertFalse(h[-1,0]);self.assertFalse(h[0,-1]);self.assertFalse(h[1,1])
    def test_zero_and_full_support_and_d1_ge_d0(self):
        from retinal_m0.maples_branch_science_analysis import evaluate_image
        a=SolRegressionTests().arrays()
        z=evaluate_image(np.zeros((8,8),np.float32),a,'A0');f=evaluate_image(np.ones((8,8),np.float32),a,'B2')
        self.assertEqual(z['candidate_path_d1_macro'],0.);self.assertEqual(f['candidate_path_d1_macro'],1.)
        self.assertGreaterEqual(f['candidate_path_d1_macro'],f['candidate_path_d0_macro'])
    def test_singleton_valid_arc_is_NA_not_zero_or_connected(self):
        from retinal_m0.maples_branch_science_analysis import evaluate_image
        a=SolRegressionTests().arrays();a['Omega'][:]=0;a['Omega'][2,1]=1
        r=evaluate_image(np.ones((8,8),np.float32),a,'B2')
        self.assertEqual(r['candidate_path_d1_macro'],1.);self.assertIsNone(r['arc_recall_d1_macro']);self.assertEqual(r['path_records'][0]['longest_hit'],0.)
    def test_whole_missed_path_and_U_cut_not_bridged(self):
        from retinal_m0.maples_branch_science_analysis import evaluate_image
        a=SolRegressionTests().arrays();a['Omega'][2,3]=0
        r=evaluate_image(np.zeros((8,8),np.float32),a,'A0')
        self.assertEqual(r['whole_missed_d1']['value'],1.);self.assertEqual(r['path_records'][0]['longest_miss'],2.)
    def test_nonprimary_paths_retained_separately(self):
        from retinal_m0.maples_branch_science_analysis import evaluate_image
        a=SolRegressionTests().arrays();a['candidate_B2'][:]=0
        r=evaluate_image(np.zeros((8,8),np.float32),a,'B1')
        self.assertIsNone(r['candidate_path_d1_macro']);self.assertEqual(len(r['excluded_path_records']),1)
    def test_original_full_FOV_evaluator_matches_new_global_counts(self):
        from retinal_m0.maples_evaluate import evaluate_arrays
        from retinal_m0.maples_branch_science_analysis import evaluate_image
        a=SolRegressionTests().arrays();a['Omega'][2,3]=0
        p=np.zeros((8,8),np.float32);p[2,1:4]=1;p[3,1]=1
        old=evaluate_arrays(p,a['GT'],a['FOV'],threshold=.5)
        new=evaluate_image(p,a,'B1')
        for k in ['dice','precision','recall','fp_density']:self.assertEqual(old[k],new[k])

class FinalReviewRegressionTests(unittest.TestCase):
    def test_actual_module_entry_and_worker_share_parent_session_without_GPU(self):
        script=r'''
import os,runpy,sys,tempfile,json
from pathlib import Path
from unittest.mock import patch
import torch
from retinal_m0 import maples_branch_science as e
from retinal_m0 import maples_branch_science_worker as w
from retinal_m0 import maples_g1_runtime as resources
with tempfile.TemporaryDirectory() as d:
 root=Path(d);out=root/'science';out.mkdir();(root/'experiments').mkdir()
 def worker(method,dest):
  from retinal_m0 import maples_branch_science as runtime
  assert runtime is e
  assert runtime.verify_active_session()=={'CPU_ENTRY_FIXTURE':True}
  assert runtime._ACTIVE['pid']==os.getpid()
  assert not torch.cuda.is_initialized()
  return {'status':'CANONICAL_SESSION_SHARED_CPU_ONLY','method':method}
 sys.argv=['maples_branch_science','--run','B1']
 with patch.object(e,'ROOT',root),patch.object(e,'OUT',out),patch.object(e,'validate_science_release',return_value={'CPU_ENTRY_FIXTURE':True}),patch.object(w,'run_scientific_training',side_effect=worker),patch.object(resources,'probe_resources',return_value={'free_disk_bytes':64*1024**3}),patch.dict(os.environ,CUDA_VISIBLE_DEVICES='${GPU_UUID}'):
  runpy.run_module('retinal_m0.maples_branch_science',run_name='__main__')
 assert not torch.cuda.is_initialized()
 assert json.loads((out/'science_budget.json').read_text())['verified_successes']==0
'''
        r=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('CANONICAL_SESSION_SHARED_CPU_ONLY',r.stdout)

    def test_real_overflow_then_pause_commits_retry_before_resuming(self):
        from retinal_m0 import maples_branch_science_worker as w
        from retinal_m0 import maples_branch_science as e
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'runs/B1';b=BudgetLedger(root/'budget.json','fixture');b.initialize_isolated()
            model,opt,scaler=w._fixture_components(2026);data=w.FixtureDataset();pairs=w.fixture_pairs(data,3)
            def poison(g):
                state=b.snapshot().get('compute_pending')
                return g*float('inf') if state==dict(track='B1',step=2,attempt=1) else g
            handle=next(model.parameters()).register_hook(poison)
            def authorize():
                state=b.snapshot()
                if state['amp_retries'] and e.clean_pause_ready(state,root/'runs'):raise w.CleanPause('after committed retry')
            with self.assertRaises(w.CleanPause):w.run_segment(model=model,optimizer=opt,scaler=scaler,dataset=data,track='B1',start=0,end=3,out=out,budget=b,binding_sha='fixture',expected_pairs=pairs,device='cpu',authorize=authorize)
            handle.remove();saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
            self.assertEqual(saved['step'],2);self.assertEqual(saved['scaler']['scale'],512.)
            self.assertEqual(saved['budget'],b.snapshot());self.assertEqual(b.snapshot()['status'],'READY')
            model,opt,scaler=w._fixture_components(2026)
            w.run_segment(model=model,optimizer=opt,scaler=scaler,dataset=data,track='B1',start=2,end=3,out=out,budget=b,binding_sha='fixture',expected_pairs=pairs,device='cpu',authorize=lambda:None)
            self.assertEqual(b.snapshot()['verified_successes'],3)
    def test_first_step_signal_waits_until_real_last_exists(self):
        from retinal_m0 import maples_branch_science_worker as w
        from retinal_m0 import maples_branch_science as e
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);b=BudgetLedger(root/'budget.json','fixture');b.initialize_isolated()
            model,opt,scaler=w._fixture_components(2026);data=w.FixtureDataset();pairs=w.fixture_pairs(data,2)
            def authorize():
                if e.clean_pause_ready(b.snapshot(),root/'runs'):raise w.CleanPause('first committed boundary')
            with self.assertRaises(w.CleanPause):w.run_segment(model=model,optimizer=opt,scaler=scaler,dataset=data,track='B1',start=0,end=2,out=root/'runs/B1',budget=b,binding_sha='fixture',expected_pairs=pairs,device='cpu',authorize=authorize)
            self.assertEqual(torch.load(root/'runs/B1/last.pt',map_location='cpu',weights_only=True)['step'],1)
            self.assertEqual(b.snapshot()['verified_successes'],1)
    def test_retry_cursor_is_not_a_committed_pause_boundary(self):
        from retinal_m0 import maples_branch_science as e
        with tempfile.TemporaryDirectory() as d:
            b=BudgetLedger(Path(d)/'budget.json','fixture');b.initialize_isolated()
            b.before_forward('B1',1,1);b.before_gradient('B1',1,1);b.reserve('B1',1,1);b.resolve('B1','amp_overflow',amp_retry=True)
            self.assertFalse(e.clean_pause_ready(b.snapshot(),Path(d)/'runs'))
    def test_first_step_created_directory_requires_a_committed_last_before_pause(self):
        from retinal_m0 import maples_branch_science as e
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);b=BudgetLedger(root/'budget.json','fixture');b.initialize_isolated()
            self.assertTrue(e.clean_pause_ready(b.snapshot(),root/'runs'))
            (root/'runs/B1').mkdir(parents=True)
            self.assertFalse(e.clean_pause_ready(b.snapshot(),root/'runs'))
    def test_completed_restore_validates_logs_and_milestone_before_evaluation(self):
        from retinal_m0 import maples_branch_science_worker as w
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'fixture';saved=w.run_fixture_segment(out,start=0,end=1)
            b=BudgetLedger(root/'fixture.ledger.json','CPU_FIXTURE_ONLY')
            got=w.load_committed_checkpoint(out,'B1',1,b,'CPU_FIXTURE_ONLY')
            self.assertTrue(exact_equal(got['model'],saved['model']))
            bad=dict(saved);bad['model']={k:v.clone() for k,v in saved['model'].items()}
            next(iter(bad['model'].values())).add_(1)
            w.save_last(out/'last.pt',bad)
            with self.assertRaises(RuntimeError):w.load_committed_checkpoint(out,'B1',1,b,'CPU_FIXTURE_ONLY')
            w.save_last(out/'last.pt',saved);w.save_last(out/'step_000001.pt',bad)
            with self.assertRaisesRegex(RuntimeError,'milestone'):w.load_committed_checkpoint(out,'B1',1,b,'CPU_FIXTURE_ONLY')
    def test_completed_previous_arm_checks_own_state_while_later_arm_has_budget(self):
        from retinal_m0 import maples_branch_science_worker as w
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'fixture';saved=w.run_fixture_segment(out,start=0,end=1)
            b=BudgetLedger(root/'fixture.ledger.json','CPU_FIXTURE_ONLY')
            b.before_forward('B2',1,1);b.before_gradient('B2',1,1);b.reserve('B2',1,1);b.resolve('B2','success')
            with self.assertRaises(RuntimeError):w.load_committed_checkpoint(out,'B1',1,b,'CPU_FIXTURE_ONLY')
            got=w.load_committed_checkpoint(out,'B1',1,b,'CPU_FIXTURE_ONLY',allow_later_arm=True)
            self.assertTrue(exact_equal(got['model'],saved['model']))
            with (out/'attempts.jsonl').open('a') as f:f.write('{}\n')
            with self.assertRaises(RuntimeError):w.load_committed_checkpoint(out,'B1',1,b,'CPU_FIXTURE_ONLY',allow_later_arm=True)
    def test_200_and_2000_milestone_resume_fail_before_evaluation(self):
        from unittest.mock import patch
        from retinal_m0 import maples_branch_science_worker as w
        from retinal_m0 import maples_branch_science as e
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'configs').mkdir();(root/'configs/B1.json').write_text('{}')
            for step in (200,2000):
                class Ledger:
                    def __init__(self,*a):pass
                    def snapshot(self):return dict(status='READY',tracks={'B1':dict(verified_successes=step)})
                with patch.object(e,'OUT',root),patch.object(e,'verify_active_session',return_value={}),patch.object(e,'verify_completed_evaluation',return_value=True),patch.object(e,'evaluate_checkpoint') as evaluate,patch.object(w,'BudgetLedger',Ledger),patch.object(w,'load_committed_checkpoint',side_effect=RuntimeError('corrupt state')) as load,patch.object(torch.cuda,'set_device') as gpu:
                    with self.assertRaisesRegex(RuntimeError,'corrupt state'):w.run_scientific_training('B1',root)
                    load.assert_called_once();evaluate.assert_not_called();gpu.assert_not_called()
    def test_immutable_sources_never_replace_old_expected_hash(self):
        from retinal_m0 import maples_branch_science as e
        pins={'old':'a'};e.merge_source_pins(pins,{'old':'a','new':'b'})
        self.assertEqual(pins,{'old':'a','new':'b'})
        with self.assertRaises(RuntimeError):e.merge_source_pins(pins,{'old':'changed'})
        self.assertEqual(pins['old'],'a')
    def test_near_thin_d0_d1_are_distinct(self):
        from retinal_m0.maples_branch_science_analysis import evaluate_image
        a=SolRegressionTests().arrays();a['near']=a['FOV'];a['thin']=a['GT']
        p=np.zeros((8,8),np.float32);p[1,1:7]=1
        r=evaluate_image(p,a,'B1')
        self.assertEqual(r['near_thin_recall_d0']['value'],0.)
        self.assertEqual(r['near_thin_recall_d1']['value'],1.)
    def test_frozen_strata_include_short_complex_and_exact_length_without_new_bins(self):
        from retinal_m0 import maples_branch_science_analysis as a
        records={m:[dict(filename='im',native_shape=[8,8],path_records=[dict(branch_id=0,denominator=2,d0_numerator=1,d1_numerator=2,d1=1.,d0=.5,length_reference=20.,candidate_B2=True,junction_complex_GT_only=True)],excluded_path_records=[dict(branch_id=1,denominator=1,d0_numerator=0,d1_numerator=0,length_reference=5.,candidate_B2=False,junction_complex_GT_only=False)])] for m in ['A0','B1','B2']}
        r=a.path_strata(records)
        self.assertEqual(r['short_branch']['short_lt20']['A0']['n_paths'],1)
        self.assertEqual(r['short_branch']['short_lt20']['A0']['path_d1_macro'],0.)
        self.assertEqual(r['complex_flags']['junction_complex_GT_only=true']['B2']['n_paths'],1)
        self.assertEqual(r['length_reference_exact']['20.0']['comparisons']['B2-A0']['estimate'],0.)
        self.assertEqual(r['length_reference_exact']['20.0']['A0']['n_images'],1)

if __name__=='__main__': unittest.main()
