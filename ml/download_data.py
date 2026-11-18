"""Download one genuine FLEURS validation WAV for the language preflight smoke test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from .preflight import MetadataClient


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke-only",action="store_true",required=True,help="One English validation sample only; no training corpus acquisition")
    p.add_argument("--destination",type=Path,required=True)
    a=p.parse_args(argv)
    report=json.loads(Path('results/preflight.json').read_text())
    rev=report['upstream']['sources']['fleurs']['revision']
    c=MetadataClient(max_bytes=4*1024*1024,max_seconds=60)
    query=urlencode(dict(dataset='google/fleurs',config='en_us',split='validation',offset=0,length=1))
    row=c.json('https://datasets-server.huggingface.co/rows?'+query)['rows'][0]['row']
    url=row['audio'][0]['src']
    if '/'+rev+'/' not in url:
        raise ValueError('Viewer asset revision differs from pinned dataset revision')
    audio=c.get(url)
    a.destination.mkdir(parents=True,exist_ok=True)
    target=a.destination/'fleurs-en-validation-smoke.wav'
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=hashlib.sha256(audio).hexdigest():
            raise ValueError('Existing smoke audio differs; refusing overwrite')
    else: target.write_bytes(audio)
    provenance={'source':'google/fleurs','revision':rev,'config':'en_us','official_split':'validation','row_index':0,'upstream_id':row['id'],
                'source_relpath':Path(row['path']).name,'transcription':row['transcription'],'license':'CC-BY-4.0',
                'retrieved_utc':datetime.now(timezone.utc).isoformat(),'route':'Official Hugging Face dataset-viewer individual WAV asset',
                'viewer_documentation':'https://huggingface.co/docs/dataset-viewer/rows','asset_sha256':hashlib.sha256(audio).hexdigest(),
                'bytes':len(audio),'purpose':'Preflight language smoke only; reserve from production locked test', 'fallback':False}
    path=a.destination/'fleurs-smoke-provenance.json'
    path.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps({'audio':str(target),'bytes':len(audio),'revision':rev,'metadata_and_audio_transfer':c.bytes_read}))


if __name__=='__main__': main()
