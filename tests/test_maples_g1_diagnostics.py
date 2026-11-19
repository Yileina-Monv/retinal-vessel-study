"""Real parameter-gradient math on tiny synthetic CPU models, NOT retinal inference."""
import unittest
import torch
from retinal_m0.maples_g1_diagnostics import diagnose,state_digest
from retinal_m0.maples_g1_safety import GateError


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026);self.model=torch.nn.Conv2d(3,1,1)
        self.x=torch.randn(4,3,4,4);self.y=torch.ones(4,1,4,4);self.f=self.y.clone()
        self.s=torch.zeros_like(self.y);self.s[:,0,1,1]=1;self.events=[]

    def evaluate(self,scale=1,precision='FP32'):
        scaler=torch.amp.GradScaler('cpu',enabled=scale!=1,init_scale=scale)
        return diagnose(self.model,self.x,self.y,self.f,self.s,precision=precision,scaler=scaler,
                        record=lambda e,v:self.events.append((e,v)))

    def test_real_parameter_gradients_and_analytic(self):
        before=state_digest(self.model);m=self.evaluate()
        self.assertTrue(m['analytic_FP32_pass']);self.assertTrue(m['zero_lambda_value_exact'])
        self.assertTrue(m['zero_lambda_gradient_exact']);self.assertTrue(m['unscaled_gradients'])
        self.assertGreater(m['grad_l2_base'],0);self.assertGreater(m['grad_l2_aux_weighted'],0)
        self.assertEqual(len(m['per_layer_grad_norms']),2)
        self.assertEqual(sum(e=='autograd_call' for e,_ in self.events),4)
        self.assertEqual(before,state_digest(self.model))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_actual_scaling_removed(self):
        one=self.evaluate();scaled=self.evaluate(scale=1024)
        for key in ('grad_l2_base','grad_l2_aux_weighted','grad_l2_combined','analytic_max_absolute_error'):
            self.assertEqual(one[key],scaled[key])
        self.assertEqual(scaled['loss_scale'],1024)
        self.assertEqual(scaled['scaler_before'],scaled['scaler_after'])

    def test_all_empty(self):
        self.s.zero_();m=self.evaluate()
        self.assertEqual(m['aux_unweighted'],0);self.assertEqual(m['grad_l2_aux_weighted'],0)
        self.assertTrue(m['empty_aux_value_zero']);self.assertTrue(m['empty_aux_logit_gradient_zero'])
        self.assertEqual(m['empty_target_crop_indices'],[0,1,2,3])
        self.assertIsNone(m['cosine_base_aux'])

    def test_partial_empty_keeps_batch_denominator(self):
        self.s[1:]=0;m=self.evaluate()
        with torch.no_grad():
            z=self.model(self.x);expected=torch.nn.functional.softplus(-z[0,0,1,1])/4
        self.assertAlmostEqual(m['aux_unweighted'],float(expected),places=7)
        self.assertTrue(m['empty_aux_logit_gradient_zero'])

    def test_dense_target_and_offtarget_support(self):
        self.s[:]=1;m=self.evaluate();self.assertTrue(m['analytic_FP32_pass'])
        self.assertEqual(m['N_target'],[16]*4)

    def test_invalid_target_stops(self):
        self.y.zero_()
        with self.assertRaises(ValueError):self.evaluate()

    def test_nonfinite_input_stops(self):
        self.x[0,0,0,0]=float('nan')
        with self.assertRaises(GateError):self.evaluate()

    def test_cpu_cannot_claim_real_amp(self):
        with self.assertRaises(GateError):self.evaluate(precision='AMP')


if __name__=='__main__':unittest.main()
