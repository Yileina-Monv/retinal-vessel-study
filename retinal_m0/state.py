"""Local, tensor-safe checkpoints including optimization, RNG and data position."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def state_digest(value):
    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(f"tensor:{array.dtype}:{array.shape}".encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(f"key:{key}".encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}".encode())
            for child in item:
                update(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    update(value)
    return digest.hexdigest()
