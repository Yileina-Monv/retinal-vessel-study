"""Stage the two Drive folders on the local runtime disk and verify every input."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage(parent, supplement, work):
    manifest = json.loads(Path(__file__).with_name("assets.json").read_text(encoding="utf-8"))
    cache = work / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        name = item["name"]
        local = cache / name
        if not local.exists() or digest(local) != item["sha256"]:
            folder = parent if item["folder"] == "parent" else supplement
            shutil.copyfile(folder / name, local)
            if digest(local) != item["sha256"]:
                raise ValueError(f"Copy failed SHA-256: {name}")
        if "extract" in item:
            target = work / item["extract"]
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(local) as archive:
                archive.extractall(target)
    for spec in manifest["join"]:
        target = work / spec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) == spec["sha256"]:
            continue
        tmp = target.with_suffix(".assembling")
        with tmp.open("wb") as dest:
            for name in spec["parts"]:
                with (cache / name).open("rb") as part:
                    shutil.copyfileobj(part, dest)
        if digest(tmp) != spec["sha256"]:
            raise ValueError(f"Joined checkpoint mismatch: {target}")
        tmp.replace(target)
    print(f"Verified {len(manifest['files'])} input files and two original checkpoints.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for arg in ("parent", "supplement", "work"):
        ap.add_argument("--" + arg, required=True, type=Path)
    args = ap.parse_args()
    stage(args.parent, args.supplement, args.work)
