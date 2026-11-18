"""Bounded public metadata and local inventory; never downloads model/data archives."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

MODEL = "ARTPARK-IISc/Vaani-LID_v0"
LANGUAGES = ("en_us", "hi_in", "ta_in", "bn_in", "te_in")
METADATA_HOSTS = frozenset(("huggingface.co", "datasets-server.huggingface.co", "www.openslr.org"))


def validate_url(url: str, allowed_hosts=METADATA_HOSTS) -> str:
    """Only official HTTPS metadata endpoints; no arbitrary redirect destinations."""
    p = urlparse(url)
    if p.scheme != "https" or p.hostname not in allowed_hosts or p.username or p.password or p.port not in (None, 443):
        raise ValueError("URL is outside the official metadata allowlist")
    return url


class OfficialRedirects(HTTPRedirectHandler):
    def __init__(self, allowed_hosts=METADATA_HOSTS):
        super().__init__()
        self.allowed_hosts=allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl,self.allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class MetadataClient:
    """Enforce per-response, cumulative-byte and wall-time bounds."""
    def __init__(self, max_bytes: int = 8 * 1024 * 1024, max_seconds: float = 120, allowed_hosts=METADATA_HOSTS):
        self.max_bytes, self.max_seconds = max_bytes, max_seconds
        self.bytes_read = 0
        self.started = time.monotonic()
        self.allowed_hosts=allowed_hosts
        self.opener = build_opener(OfficialRedirects(allowed_hosts))

    def get(self, url: str) -> bytes:
        validate_url(url,self.allowed_hosts)
        remaining = self.max_seconds - (time.monotonic() - self.started)
        if remaining <= 0 or self.bytes_read >= self.max_bytes:
            raise RuntimeError("Metadata preflight budget exhausted")
        req = Request(url, headers={"User-Agent": "VoxLingua-preflight/0.1"})
        with self.opener.open(req, timeout=min(20, remaining)) as response:
            validate_url(response.url,self.allowed_hosts)
            budget = min(2 * 1024 * 1024, self.max_bytes - self.bytes_read)
            data = response.read(budget + 1)
            self.bytes_read += len(data)
            if len(data) > budget:
                raise RuntimeError("Metadata response exceeds byte cap")
            return data

    def json(self, url: str):
        return json.loads(self.get(url))


def command(args: list[str]) -> dict:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=45)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def memory_inventory() -> dict:
    if sys.platform != "win32":
        return {"status": "unavailable", "reason": "Windows memory API is not present"}
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
    state = Status()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        return {"status": "unavailable"}
    return {"total_bytes": state.total, "available_bytes": state.available, "load_percent": state.load}


def inventory(root: Path) -> dict:
    packages = {}
    for name in ("numpy", "scipy", "soundfile", "librosa", "scikit-learn", "xgboost", "torch", "transformers", "huggingface-hub", "pytest", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    disk = shutil.disk_usage(root)
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "virtualenv": sys.prefix != sys.base_prefix, "packages": packages,
            "memory": memory_inventory(), "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
            "nvidia": command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"]),
            "ffmpeg": shutil.which("ffmpeg"), "git": command(["git", "status", "--short"])}


def upstream(client: MetadataClient, evidence: Path) -> dict:
    evidence.mkdir(parents=True, exist_ok=True)
    result = {"sources": {}, "blockers": []}
    def checked(name, fn):
        try:
            result["sources"][name] = fn()
        except Exception as exc:
            result["sources"][name] = {"status": "blocked", "reason": str(exc)}
            result["blockers"].append(f"{name}: {exc}")

    def model():
        meta = client.json(f"https://huggingface.co/api/models/{MODEL}")
        rev = meta["sha"]
        files = client.json(f"https://huggingface.co/api/models/{MODEL}/tree/{rev}?recursive=true")
        config = client.json(f"https://huggingface.co/{MODEL}/raw/{rev}/config.json")
        code = client.get(f"https://huggingface.co/{MODEL}/raw/{rev}/modeling_whisper_lid.py")
        card = client.get(f"https://huggingface.co/{MODEL}/raw/{rev}/README.md")
        folder = evidence / rev
        folder.mkdir(exist_ok=True)
        for name, data in (("modeling_whisper_lid.py.txt", code), ("README.md", card), ("config.json", json.dumps(config, indent=2).encode())):
            (folder / name).write_bytes(data)
        base = client.json(f"https://huggingface.co/api/models/{config['whisper_model']}")
        base_files = client.json(f"https://huggingface.co/api/models/{config['whisper_model']}/tree/{base['sha']}?recursive=true")
        return {"status": "metadata_verified", "model_id": MODEL, "revision": rev, "license": meta.get("cardData", {}).get("license"),
                "files": [{"path": f["path"], "size": f.get("size"), "sha256": f.get("lfs", {}).get("oid")} for f in files if f["type"] == "file"],
                "labels": config["id2label"], "label_count": len(config["id2label"]),
                "code_sha256": hashlib.sha256(code).hexdigest(), "code_review": "PENDING: read recorded source before any execution",
                "constructor_base_model": config["whisper_model"], "base_revision": base["sha"],
                "base_files": [{"path": f["path"], "size": f.get("size")} for f in base_files if f["type"] == "file"],
                "inference_smoke": "NOT RUN", "weights_downloaded": False}

    def fleurs():
        meta = client.json("https://huggingface.co/api/datasets/google/fleurs")
        rev = meta["sha"]
        configs = {}
        for lang in LANGUAGES:
            files = client.json(f"https://huggingface.co/api/datasets/google/fleurs/tree/{rev}/data/{lang}/audio")
            configs[lang] = [{"path": f["path"], "size": f.get("size"), "sha256": f.get("lfs", {}).get("oid")} for f in files if f["type"] == "file"]
        return {"status": "metadata_verified", "revision": rev, "license": meta.get("cardData", {}).get("license"),
                "language_archives": configs, "route": "per-official-split tar.gz; subset transfer feasibility NOT RUN", "streaming_saves_bytes": "NOT ASSUMED",
                "recording_id_uniqueness": "NOT RUN", "speaker_metadata": "NOT RUN", "audio_decode": "NOT RUN"}

    def musan():
        text = client.get("https://www.openslr.org/17/").decode()
        (evidence / "musan-official-page.html").write_text(text, encoding="utf-8")
        return {"status": "official_page_accessible", "license": "CC BY 4.0" if "Attribution 4.0" in text else "UNVERIFIED",
                "full_archive_planning_bytes": 11_000_000_000, "route": "official page offers full archive; genuine small subset route NOT VERIFIED",
                "downloaded": False, "fallback": None}

    checked("vaani", model)
    checked("fleurs", fleurs)
    checked("musan", musan)
    result["metadata_bytes"] = client.bytes_read
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/preflight.json"))
    parser.add_argument("--network", action="store_true", help="Bounded official metadata only; no weights, archives or remote-code execution")
    args = parser.parse_args(argv)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "stage": "preflight", "status": "INCOMPLETE",
              "local": inventory(Path.cwd()), "acquisition_budget": {"minutes": 45, "transfer_gib": 6, "approval_threshold_mib": 500},
              "vaani_smoke": "NOT RUN", "pilot": "NOT RUN", "training": "NOT RUN"}
    if args.network:
        report["upstream"] = upstream(MetadataClient(), args.out.parent / "upstream")
    else:
        report["upstream"] = {"status": "NOT RUN", "reason": "--network not provided"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        args.out = args.out.with_name(f"{args.out.stem}-{stamp}.json")
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.out), "status": report["status"], "message": "Inventory recorded; smoke/pilot/training remain gated."}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
