"""Small, explicit FLEURS + user-approved ESC-10 pilot acquisition on Volume D."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import tarfile
import time
from urllib.request import build_opener, Request
from pathlib import Path
from urllib.parse import urlencode
from .preflight import MetadataClient, LANGUAGES, METADATA_HOSTS
from .download_models import Redirects, official_storage

ESC_REV='33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6'


def fleurs_table(opener, config, split, revision):
    official_split='dev' if split=='validation' else split
    base=f'https://huggingface.co/datasets/google/fleurs/resolve/{revision}/data/{config}'
    with opener.open(base+'/'+official_split+'.tsv',timeout=30) as response:
        official_storage(response.url)
        meta=response.read(6*1024*1024+1)
        if len(meta)>6*1024*1024: raise ValueError('TSV exceeds bounded metadata size')
    table={row[1]:row for row in csv.reader(StringIO(meta.decode('utf-8')),delimiter='\t') if len(row)>=7}
    return table,len(meta),base+'/'+ 'audio/'+official_split+'.tar.gz'


def speech_entry(row, config, split, revision, rel, data, route):
    return {'source':'google/fleurs','source_revision':revision,'config':config,'language':config,
            'upstream_id':row[0],'official_split':split,'source_relpath':rel.as_posix(),
            'source_sha256':hashlib.sha256(data).hexdigest(),'transcription':row[3],'gender':row[6],
            'speaker_id':None,'route':route}


def archive_subset(root, config, split, count, revision, *, cap=12*1024*1024, eligible=None, minimum=None):
    """Read a bounded gzip prefix, never extract paths supplied by the archive."""
    opener=build_opener(Redirects())
    table,metadata_bytes,url=fleurs_table(opener,config,split,revision)
    start=time.monotonic()
    total=[0]
    class LimitedReader:
        def __init__(self,response): self.response=response
        def read(self,n=-1):
            if time.monotonic()-start>90: raise TimeoutError('Archive subset time cap')
            if n<0: n=cap-total[0]+1
            remaining=cap-total[0]+(minimum is None)
            if remaining<=0: return b''
            b=self.response.read(min(n,remaining));total[0]+=len(b)
            if total[0]>cap: raise ValueError('Archive prefix byte cap reached; full archive not downloaded')
            return b
    entries=[]
    try:
        with opener.open(Request(url,headers={'Range':f'bytes=0-{cap-1}','User-Agent':'VoxLingua-bounded-subset/0.1'}),timeout=30) as response:
            official_storage(response.url)
            with tarfile.open(fileobj=LimitedReader(response),mode='r|gz') as archive:
                for member in archive:
                    name=Path(member.name).name
                    if not member.isfile() or name not in table: continue
                    row=table[name]
                    if not 48000<=int(row[5])<=240000: continue
                    if member.size>2*1024*1024: continue
                    data=archive.extractfile(member).read()
                    rel=Path('speech')/config/split/name
                    entry=speech_entry(row,config,split,revision,rel,data,'bounded official split tar.gz prefix; no full archive')
                    if eligible is not None and not eligible(entry):
                        continue
                    path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
                    if path.exists():
                        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['source_sha256']: raise ValueError('Existing acquisition content differs')
                    else: path.write_bytes(data)
                    entries.append(entry)
                    if len(entries)==count: break
    except (tarfile.ReadError,TimeoutError):
        if minimum is None or len(entries)<minimum:
            raise
    required=count if minimum is None else minimum
    if len(entries)<required: raise RuntimeError('Not enough candidates within bounded prefix')
    return entries,total[0]+metadata_bytes


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=Path('data/raw/pilot'))
    parser.add_argument('--esc10-approved',action='store_true',help='Requires explicit fallback approval; no silent corpus substitution')
    a=parser.parse_args(argv)
    if not a.esc10_approved: parser.error('The disclosed ESC-10 fallback must be approved first')
    root=a.destination
    if (root/'acquisition.json').exists(): raise ValueError('Pilot acquisition already exists; reuse it or choose a new destination')
    root.mkdir(parents=True,exist_ok=True)
    # The original author repository was inspected before adding its raw-data host.
    client=MetadataClient(max_bytes=16*1024*1024,max_seconds=8*60,allowed_hosts=METADATA_HOSTS|{'raw.githubusercontent.com'})
    pre=json.loads(Path('results/preflight.json').read_text())
    fleurs_rev=pre['upstream']['sources']['fleurs']['revision']
    entries=[];noise=[];archive_bytes=0
    for config in LANGUAGES:
        for split,count in [('train',9),('validation',2),('test',2)]:
            rows,transferred=archive_subset(root,config,split,count,fleurs_rev)
            entries.extend(rows);archive_bytes+=transferred
            if archive_bytes>180*1024*1024: raise ValueError('Pilot acquisition transfer cap exceeded')
            print(f'FLEURS {config}/{split}: {len(rows)} candidates, {transferred/1024**2:.2f} MiB transferred',flush=True)
    base=f'https://raw.githubusercontent.com/karolpiczak/ESC-50/{ESC_REV}'
    license_bytes=client.get(base+'/LICENSE');meta_bytes=client.get(base+'/meta/esc50.csv')
    (root/'ESC-50-LICENSE.txt').write_bytes(license_bytes)
    (root/'esc50-metadata.csv').write_bytes(meta_bytes)
    candidates=list(csv.DictReader(StringIO(meta_bytes.decode())))
    used=set()
    for esc_fold,count in [(1,2),(2,2),(3,2),(4,3),(5,3)]:
        eligible=[r for r in candidates if int(r['fold'])==esc_fold and r['esc10']=='True' and r['category'] in ('rain','sea_waves','crackling_fire','helicopter','chainsaw')]
        eligible.sort(key=lambda r:hashlib.sha256(('20261115:'+r['filename']).encode()).hexdigest())
        selected=0
        for row in eligible:
            if row['src_file'] in used: continue
            used.add(row['src_file'])
            data=client.get(base+'/audio/'+row['filename'])
            path=root/'noise'/row['filename'];path.parent.mkdir(exist_ok=True);path.write_bytes(data)
            noise.append({'noise_id':'esc10:'+row['filename'],'noise_source_group_id':'esc10-source:'+row['src_file'],
                          'corpus':'ESC-10','category':'noise','environment':row['category'],'source_relpath':'noise/'+row['filename'],
                          'source_sha256':hashlib.sha256(data).hexdigest(),'split':'train' if esc_fold<=3 else 'validation' if esc_fold==4 else 'test',
                          'fold_id':esc_fold-1 if esc_fold<=3 else None,'license':'CC-BY-3.0','revision':ESC_REV})
            selected+=1
            if selected==count: break
        if selected!=count: raise RuntimeError('Insufficient distinct ESC-10 parent recordings')
    report={'run_kind':'pilot','created_utc':datetime.now(timezone.utc).isoformat(),'seed':20261115,'speech':entries,'noise':noise,
            'noise_is_fallback':True,'noise_corpus':'ESC-10','fallback_approval':'User explicitly approved ESC-10 for pilot; MUSAN retained for production',
            'limitations':['Environmental noise only; no music/competing speech','Pilot candidate pool is first 9/2/2 eligible recordings from each bounded official split archive prefix, then hash-ordered in builder; not a representative population sample','No verified speaker identifiers; no speaker independence claim'],
            'transfer_bytes':client.bytes_read+archive_bytes,'licenses':{'FLEURS':'CC-BY-4.0','ESC-10':'CC-BY-3.0'},
            'citations':{'FLEURS':'https://huggingface.co/datasets/google/fleurs','ESC-10':'https://github.com/karolpiczak/ESC-50'},
            'license_attribution_file':'ESC-50-LICENSE.txt','music_used':False}
    (root/'acquisition.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'manifest':str(root/'acquisition.json'),'speech_candidates':len(entries),'noise_files':len(noise),'transfer_bytes':client.bytes_read+archive_bytes}),flush=True)


if __name__=='__main__': main()
