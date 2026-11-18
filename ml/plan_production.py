"""Offline production acquisition plan and pilot-source reservations; downloads nothing."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import unicodedata
from pathlib import Path
from voxlingua.audio import decode_audio


def fingerprint(path):
    raw=path.read_bytes();audio=decode_audio(raw)
    content=hashlib.sha256(f'{audio.sample_rate}:{audio.n_channels}:{audio.samples.shape}'.encode()+audio.samples.astype('<f4').tobytes()).hexdigest()
    return {'source_sha256':hashlib.sha256(raw).hexdigest(),'content_sha256':content}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-data',type=Path,default=Path('data/pilot'))
    parser.add_argument('--out',type=Path,default=Path('results/production-plan.json'))
    args=parser.parse_args(argv)
    provenance=json.loads((args.pilot_data/'manifests/provenance.json').read_text(encoding='utf-8'))
    raw_root=Path(provenance['raw_root'])
    speech=[];noise=[]
    for item in provenance['speech']:
        identity=fingerprint(raw_root/item['source_relpath'])
        if identity['source_sha256']!=item['source_sha256']: raise ValueError('Pilot source changed')
        speech.append({'config':item['config'],'official_split':item['official_split'],'upstream_id':item['upstream_id'],**identity})
    for item in provenance['noise']:
        identity=fingerprint(raw_root/item['source_relpath'])
        if identity['source_sha256']!=item['source_sha256']: raise ValueError('Pilot noise source changed')
        noise.append({'noise_id':item['noise_id'],'noise_source_group_id':item['noise_source_group_id'],**identity})
    with (args.pilot_data/'manifests/originals.csv').open(newline='',encoding='utf-8') as f:
        originals=list(csv.DictReader(f))
    sentences={r['config']+':'+hashlib.sha256(' '.join(unicodedata.normalize('NFC',r['transcription']).split()).encode()).hexdigest() for r in provenance['speech']}
    reservations={'speech':speech,'noise':noise,'sentence_ids':sorted(sentences|{r['sentence_id'] for r in originals}),
                  'original_ids':sorted({r['original_id'] for r in originals})}
    report={'schema_version':'production-plan-v1','created_utc':datetime.now(timezone.utc).isoformat(),'status':'PROPOSED_ACQUISITION_NOT_AUTHORIZED',
            'target':{'originals':600,'languages':['en_us','hi_in','ta_in','bn_in','te_in'],'derivative_rows':3600,'maximum_seconds':15},
            'reserved_pilot_sources':reservations,'sources':{'speech':'google/fleurs','noise':'MUSAN (production target); ESC-10 approval was pilot-only'},
            'budget':{'planning_transfer_cap_gib':6,'planning_acquisition_minutes':45,'approval_threshold_mib':500,
                      'musan_full_archive_planning_bytes':11000000000,'musan_archive_size_is_estimate':True,
                      'full_archive_exceeds_transfer_cap':True,'new_download_started':False},
            'required_before_acquisition':['Verify genuine licensed MUSAN subset route or obtain explicit revised size/time approval for full archive',
                                           'Measure free disk plus archive, extraction, caches and temporary headroom on D:',
                                           'Exclude all reserved pilot byte/content hashes, parent noise sources and sentence identities before build',
                                           'Acquire distinct production recordings in original official partitions; assign CV groups/folds before augmentation',
                                           'Keep pilot exposed test sources out of production evaluation; create a new test lock',
                                           'Implement and verify production build/train/evaluate/promotion gates; never copy pilot models into production'],
            'reuse_without_download':['Verified Vaani and underlying Whisper snapshots on D:','Existing project environment'],
            'production_artifacts_ready':False,'production_promotion_authorized':False}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'plan':str(args.out),'reserved_speech_files':len(speech),'reserved_noise_sources':len(noise),'downloads_started':False}))


if __name__=='__main__': main()
