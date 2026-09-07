from io import BytesIO
import random
import unittest

import numpy as np
import torch
from torch import nn

from retinal_m0.inference import starts, tiled_probability
from retinal_m0.objectives import binary_metrics, pixel_loss
from retinal_m0.state import restore_rng, rng_state, state_digest


class PixelwiseModel(nn.Module):
    def forward(self, image):
        return 3 * image[:, :1] - 1


class M0Tests(unittest.TestCase):
    def test_empty_positive_and_mixed_losses_have_finite_gradients(self):
        for fill in (0.0, 1.0):
            logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
            target = torch.full_like(logits, fill)
            loss, _ = pixel_loss(logits, target)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertTrue((logits.grad > 0).all() if fill == 0 else (logits.grad < 0).all())
        logits = torch.tensor([[[[-30., 30.]]]])
        target = torch.tensor([[[[0., 1.]]]])
        self.assertLess(float(pixel_loss(logits, target)[0]), 1e-5)
        self.assertGreater(float(pixel_loss(-logits, target)[0]), 1)

    def test_accumulated_microbatch_loss_matches_effective_batch(self):
        generator = torch.Generator().manual_seed(7)
        values = torch.randn(4, 1, 8, 8, generator=generator)
        targets = (torch.rand(4, 1, 8, 8, generator=generator) > .8).float()
        whole = values.clone().requires_grad_()
        pixel_loss(whole, targets)[0].backward()
        micro = values.clone().requires_grad_()
        for offset in (0, 2):
            (pixel_loss(micro[offset:offset+2], targets[offset:offset+2])[0] / 2).backward()
        self.assertTrue(torch.allclose(whole.grad, micro.grad, atol=1e-7, rtol=1e-6))

    def test_tiling_on_irregular_image_matches_pixelwise_reference(self):
        image = torch.rand(3, 73, 97, generator=torch.Generator().manual_seed(6))
        expected = (3 * image[0] - 1).sigmoid()
        actual, count = tiled_probability(PixelwiseModel(), image, patch=32, stride=16, batch_size=3, device="cpu", amp=False)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-7, rtol=1e-6))
        self.assertEqual(starts(2048, 512, 256), [0, 256, 512, 768, 1024, 1280, 1536])
        self.assertEqual(count, len(starts(73, 32, 16)) * len(starts(97, 32, 16)))

    def test_metric_empty_denominators_are_explicit(self):
        empty = torch.zeros(8, 8)
        self.assertIsNone(binary_metrics(empty, empty)["dice"])
        self.assertIsNone(binary_metrics(empty, torch.ones_like(empty))["precision"])
        self.assertEqual(binary_metrics(torch.ones_like(empty), empty)["dice"], 0)
        self.assertEqual(binary_metrics(torch.ones_like(empty), torch.ones_like(empty))["dice"], 1)

    def test_tensor_safe_checkpoint_preserves_rng_optimizer_and_digest(self):
        model = nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step()
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(), "step": 1}
        buffer = BytesIO()
        torch.save(payload, buffer)
        before = state_digest(payload)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        buffer.seek(0)
        restored = torch.load(buffer, map_location="cpu", weights_only=True)
        self.assertEqual(before, state_digest(restored))
        restore_rng(restored["rng"])
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))


if __name__ == "__main__":
    unittest.main()
