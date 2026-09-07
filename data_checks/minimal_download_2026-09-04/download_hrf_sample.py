"""Download the public HRF healthy subset for a bounded availability check."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import time
import zipfile
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parent
ARCHIVES = ROOT / "archives"
ARCHIVES.mkdir(exist_ok=True)
BASE = "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/"
SOURCES = {
    "images": "healthy.zip",
    "labels": "healthy_manualsegm.zip",
    "fov": "healthy_fovmask.zip",
}
LIMIT = 100 * 1024 * 1024
records = []

def origin_path(url):
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"

for kind, name in SOURCES.items():
    url = BASE + name
    destination = ARCHIVES / name
    record = {"kind": kind, "source": url, "local_path": str(destination.relative_to(ROOT))}
    began = time.monotonic()
    try:
        if not destination.exists():
            partial = destination.with_suffix(".zip.part")
            with requests.get(url, stream=True, timeout=(15, 45)) as response:
                record["http_status"] = response.status_code
                record["final_origin_path"] = origin_path(response.url)
                response.raise_for_status()
                length = int(response.headers.get("content-length") or 0)
                record["advertised_bytes"] = length or None
                if length > LIMIT:
                    raise ValueError("Archive exceeds the 100 MiB smoke-check limit")
                total = 0
                next_progress = 8 * 1024 * 1024
                with partial.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > LIMIT:
                            raise ValueError("Streaming size exceeds the smoke-check limit")
                        output.write(chunk)
                        if total >= next_progress:
                            print(f"{name}: {total / 1024 / 1024:.1f} MiB", flush=True)
                            next_progress += 8 * 1024 * 1024
                if length and total != length:
                    raise ValueError("Incomplete response body")
            partial.replace(destination)
        record["size_bytes"] = destination.stat().st_size
        digest = hashlib.sha256()
        with destination.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        record["sha256"] = digest.hexdigest()
        with zipfile.ZipFile(destination) as archive:
            members = archive.infolist()
            for member in members:
                path = PurePosixPath(member.filename.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts or ":" in member.filename:
                    raise ValueError("Unsafe archive member path")
                if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError("Archive contains a symbolic link")
            if sum(member.file_size for member in members) > 300 * 1024 * 1024:
                raise ValueError("Expanded archive exceeds the bounded check limit")
            damaged = archive.testzip()
            if damaged:
                raise ValueError("ZIP CRC failure")
            record["zip_crc_ok"] = True
            record["members"] = [{"name": m.filename, "bytes": m.file_size} for m in members if not m.is_dir()]
        record["status"] = "PASS"
        print(json.dumps({"kind": kind, "status": "PASS", "bytes": record["size_bytes"], "members": [m["name"] for m in record["members"]]}, ensure_ascii=False), flush=True)
    except Exception as exc:
        record["status"] = "FAIL"
        record["error_type"] = type(exc).__name__
        # Avoid logging signed redirect queries or credential-bearing headers.
        print(json.dumps({"kind": kind, "status": "FAIL", "error_type": type(exc).__name__}), flush=True)
    finally:
        record["elapsed_seconds"] = round(time.monotonic() - began, 3)
        records.append(record)
        (ROOT / "hrf_download_evidence.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if any(item["status"] != "PASS" for item in records):
    raise SystemExit(2)
