"""Device-parity probe for the candidate loss only, using synthetic full-size logits."""
import argparse
import json
import math

import torch

from near_path_aux import near_path_aux


def evaluate(device, dtype):
    batch, height, width = 2, 1024, 1536
    logits = torch.zeros((batch, 2, height, width), device=device, dtype=dtype, requires_grad=True)
    vessel = torch.zeros((batch, height, width), device=device, dtype=torch.bool)
    near = torch.zeros_like(vessel)
    near[0, 50:70, 100:200] = True
    paths = [[51 * width + x for x in range(105, 117)],
             [60 * width + x for x in range(130, 157)]]
    for path in paths:
        vessel[0].view(-1)[path] = True
    # The second image and all other pixels must contribute no auxiliary gradient.
    loss = near_path_aux(logits, vessel, near, [paths, []])
    loss.backward()
    gradient = logits.grad[:, 1]
    flat = gradient[0].flatten()
    background = near[0] & ~vessel[0]
    expected_bg = 0.25 / int(background.sum())
    if not torch.isfinite(logits.grad).all() or abs(loss.item() - math.log(2)) > 1e-5:
        raise ValueError("Non-finite gradient or wrong objective")
    for path in paths:
        expected = torch.full((len(path),), -0.125 / len(path), device=device)
        torch.testing.assert_close(flat[path].float(), expected, rtol=0.002, atol=1e-7)
    torch.testing.assert_close(gradient[0][background].float(),
                               torch.full_like(gradient[0][background], expected_bg, dtype=torch.float32),
                               rtol=0.002, atol=1e-7)
    if gradient[~near].count_nonzero() or not torch.equal(logits.grad[:,0], -gradient):
        raise ValueError("Unexpected gradient outside the target or across classes")
    return dict(device=device, dtype=str(dtype), shape=list(logits.shape), loss=loss.item(),
                verified_target_pixels=sum(map(len, paths)), verified_background_pixels=int(background.sum()),
                optimizer_updates=0, synthetic=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    selected = parser.parse_args().device
    results = [evaluate(selected, dtype) for dtype in (torch.float32, torch.float16)]
    print(json.dumps(results, indent=2))
