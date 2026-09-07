import unittest
import numpy as np
import torch
from retinal_batchstudy.summarize import exact_signflip_p, holm, mean_interval
from retinal_batchstudy.run import read, CONFIG, schedule_lr
from retinal_m0.objectives import pixel_loss


class BatchStudyTests(unittest.TestCase):
    def test_accumulation_gradient_matches_full_batch(self):
        torch.manual_seed(3)
        inputs = torch.randn(4,1,9,7)
        targets = torch.randint(0,2,inputs.shape).float()
        gradients=[]
        for chunk in (1,2,4):
            logits=inputs.clone().requires_grad_()
            for start in range(0,4,chunk):
                loss,_=pixel_loss(logits[start:start+chunk],targets[start:start+chunk])
                (loss*chunk/4).backward()
            gradients.append(logits.grad)
        for gradient in gradients[1:]:
            torch.testing.assert_close(gradient,gradients[0],rtol=1e-6,atol=1e-8)

    def test_fp64_weighted_reduction_identity(self):
        torch.manual_seed(4)
        x=torch.randn(4,5,dtype=torch.float64)
        target=torch.randn_like(x)
        gradients=[]
        for chunk in (1,2,4):
            weight=torch.tensor(0.7,dtype=torch.float64,requires_grad=True)
            for start in range(0,4,chunk):
                (((x[start:start+chunk]*weight-target[start:start+chunk])**2).mean()*chunk/4).backward()
            gradients.append(weight.grad)
        for gradient in gradients[1:]:
            torch.testing.assert_close(gradient,gradients[0],rtol=1e-14,atol=1e-14)

    def test_exact_test_and_holm(self):
        self.assertEqual(exact_signflip_p([0.]*8),1.)
        self.assertEqual(exact_signflip_p([1.]*8),2/256)
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

    def test_sample_budget_schedule_and_interval(self):
        config=read(CONFIG)
        for arm in config['arms']:
            effective=arm['microbatch']*arm['accumulation']
            self.assertEqual(config['training_draws']%effective,0)
            for end in config['evaluation_draws']:
                self.assertEqual(end%effective,0)
        self.assertAlmostEqual(schedule_lr(config,0),config['learning_rate'])
        self.assertAlmostEqual(schedule_lr(config,config['training_draws']),config['minimum_learning_rate'])
        self.assertEqual(mean_interval([.01]*8,.01),[.01,.01])


if __name__=='__main__':
    unittest.main()
