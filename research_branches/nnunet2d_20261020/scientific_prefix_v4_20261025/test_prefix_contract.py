import copy,unittest
from prefix_contract import validate_prefix,PrefixViolation

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.base={'attempt':1,'ids':['TrainA','TrainB'],'input_sha256':'x','target_sha256':'y','lr':.01,'classification':'SUCCESS','loss':.6950486302375793,'unscaled_preclip_L2':1.,'scale_before':65536.,'scale_after':65536.,'optimizer_calls':1,'model_changed':True,'optimizer_changed':True}
    def test_one_FP32_ULP_is_diagnostic_does_not_stop(self):
        a=dict(self.base,loss=.6950486898422241);r=validate_prefix([a],[self.base]);self.assertAlmostEqual(r['events'][0]['Loss_abs_difference'],2**-24);self.assertFalse(r['numeric_bitwise_equality_required'])
    def test_finite_numeric_difference_does_not_select_a_new_tolerance(self):
        a=dict(self.base,loss=.6951,unscaled_preclip_L2=1.0001);self.assertFalse(validate_prefix([a],[self.base])['new_tolerance_scan'])
    def test_wrong_input_label_id_or_LR_still_rejected(self):
        for key,value in [('input_sha256','other'),('target_sha256','other'),('ids',['Dev']),('lr',.02)]:
            with self.assertRaises(PrefixViolation):validate_prefix([dict(self.base,**{key:value})],[self.base])
    def test_nonfinite_loss_gradient_scale_or_unverified_update_still_rejected(self):
        for key,value in [('loss',float('nan')),('loss',float('inf')),('unscaled_preclip_L2',float('inf')),('scale_before',0),('scale_after',float('nan')),('optimizer_calls',0),('model_changed',False)]:
            with self.assertRaises(PrefixViolation):validate_prefix([dict(self.base,**{key:value})],[self.base])
    def test_valid_handled_AMP_skip_remains_allowed_even_if_reference_succeeded(self):
        a=dict(self.base,classification='AMP_SKIP',unscaled_preclip_L2='inf',scale_after=32768.,optimizer_calls=0,model_changed=False,optimizer_changed=False);self.assertEqual(validate_prefix([a],[self.base])['events'][0]['actual_classification'],'AMP_SKIP')
    def test_skip_with_mutation_or_no_backoff_rejected(self):
        a=dict(self.base,classification='AMP_SKIP',unscaled_preclip_L2='inf',scale_after=32768.,optimizer_calls=0,model_changed=False,optimizer_changed=False)
        for change in [{'optimizer_changed':True},{'model_changed':True},{'scale_after':65536.},{'optimizer_calls':1}]:
            with self.assertRaises(PrefixViolation):validate_prefix([dict(a,**change)],[self.base])

if __name__=='__main__':unittest.main(verbosity=2)
