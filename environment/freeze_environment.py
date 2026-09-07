"""Record exact installed versions from this project's environment only."""
import importlib.metadata
import json
from pathlib import Path
import re
import sys

directory = Path(__file__).resolve().parent
expected_environment = (directory.parent / ".venv").resolve()
if Path(sys.prefix).resolve() != expected_environment:
    raise SystemExit("Run this using the project's .venv interpreter")

packages = {}
for distribution in importlib.metadata.distributions():
    name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
    packages[name] = distribution.version

torch_names = {"torch", "torchvision"}
if not torch_names <= packages.keys():
    raise SystemExit("PyTorch dependencies are incomplete")
for label, selected in (
    ("torch", torch_names), ("runtime", packages.keys() - torch_names)
):
    text = "# Windows x64 / Python 3.13; generated from the accepted local environment.\n"
    text += "\n".join(f"{name}=={packages[name]}" for name in sorted(selected)) + "\n"
    (directory / f"requirements-{label}.lock.txt").write_text(text, encoding="utf-8")

(directory / "installed-packages.json").write_text(
    json.dumps(dict(sorted(packages.items())), indent=2), encoding="utf-8"
)
print(f"Recorded {len(packages)} distributions in two version locks")
