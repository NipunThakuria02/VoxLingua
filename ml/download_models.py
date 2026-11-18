"""Explicitly approved, pinned, hash-verified upstream snapshots on Volume D."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler


def official_storage(url: str) -> str:
    p = urlparse(url)
    host = p.hostname or ""
    # Hugging Face origin and its managed HF/Xet content delivery domains only.
    if p.scheme != "https" or not (host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co")) or p.username or p.password or p.port not in (None,443):
        raise ValueError("Refusing redirect outside official Hugging Face storage")
    return url


class Redirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        official_storage(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def fetch(opener, url: str, target: Path, size: int, checksum: str | None, deadline: float) -> dict:
    official_storage(url)
    if target.exists():
        actual=digest(target)
        if target.stat().st_size != size or (checksum and actual != checksum):
            raise ValueError(f"Existing file mismatch: {target.name}; refusing overwrite")
        return {"path":str(target),"bytes":size,"sha256":actual,"reused":True}
    part=target.with_suffix(target.suffix+".part")
    if part.exists():
        raise ValueError(f"Incomplete file exists: {part}; inspect before retrying (no silent overwrite)")
    target.parent.mkdir(parents=True,exist_ok=True)
    count=0
    last=time.monotonic()
    h=hashlib.sha256()
    with opener.open(Request(url,headers={"User-Agent":"VoxLingua-approved-download/0.1"}),timeout=30) as response, part.open("xb") as stream:
        official_storage(response.url)
        while True:
            if time.monotonic()>deadline:
                raise TimeoutError("Approved download time budget exceeded")
            block=response.read(1024*1024)
            if not block: break
            count+=len(block)
            if count>size: raise ValueError("Response exceeds pinned file size")
            stream.write(block);h.update(block)
            if time.monotonic()-last>15:
                print(f"{target.parent.name}/{target.name}: {count/1024**2:.0f}/{size/1024**2:.0f} MiB",flush=True)
                last=time.monotonic()
    if count != size or (checksum and h.hexdigest()!=checksum):
        raise ValueError(f"Size or SHA-256 mismatch: {target.name}; partial file quarantined, not loaded")
    part.rename(target)
    return {"path":str(target),"bytes":count,"sha256":h.hexdigest(),"reused":False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("results/download-plan.json"))
    p.add_argument("--destination",type=Path,required=True)
    p.add_argument("--approved",action="store_true",help="Only use after explicit user approval of the 4.86-GB plan")
    a=p.parse_args(argv)
    if not a.approved: p.error("Explicit user approval is required; no download started")
    plan=json.loads(a.plan.read_text())
    a.destination.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.destination).free<plan["estimated_peak_disk_allowance_bytes"]: raise RuntimeError("Insufficient approved disk allowance")
    opener=build_opener(Redirects())
    deadline=time.monotonic()+45*60
    records=[]
    for source in plan["sources"]:
        repo,rev=source["repository"],source["revision"]
        # Repository and revision values are pinned in the approved plan, never user upload input.
        url=f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
        with opener.open(url,timeout=20) as response: tree=json.loads(response.read(2*1024*1024))
        allowed=(["README.md","config.json","preprocessor_config.json","modeling_whisper_lid.py","model.safetensors"] if repo.startswith("ARTPARK-IISc/") else ["config.json","model.safetensors"])
        for name in allowed:
            info=next(x for x in tree if x["path"]==name)
            checksum=info.get("lfs",{}).get("oid")
            if name=="model.safetensors" and info["size"]!=source["weight_bytes"]: raise RuntimeError("Pinned plan size mismatch")
            if repo.startswith("ARTPARK-IISc/") and name=="modeling_whisper_lid.py":
                preflight=json.loads(Path("results/preflight.json").read_text())
                checksum=preflight['upstream']['sources']['vaani']['code_sha256']
            record=fetch(opener,f"https://huggingface.co/{repo}/resolve/{rev}/{name}",a.destination/repo.split('/')[-1]/rev/name,info["size"],checksum,deadline)
            record.update(repository=repo,revision=rev,license=source['license'])
            records.append(record)
            print(json.dumps({"verified":name,"repository":repo,"bytes":record['bytes']}),flush=True)
    receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"approval":"User approved reviewed official weights, requested Volume D","files":records,"total_bytes":sum(x['bytes'] for x in records)}
    receipt_path=a.destination/'verified-downloads.json'
    if receipt_path.exists(): receipt_path=a.destination/f"verified-downloads-{time.time_ns()}.json"
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({"complete":True,"receipt":str(receipt_path)}),flush=True)


if __name__=="__main__": main()
