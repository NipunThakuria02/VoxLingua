"""Isolated pilot and production dataset builder; originals, folds and noise stay separate."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from voxlingua.audio import make_feature_copy, make_language_copy
from voxlingua.features import FEATURE_NAMES, extract_features, feature_schema
from voxlingua.schemas import CLASSES
from .augment import derivative, screen_reference
from .production_gate import source_audio, speech_identity, validate_production_acquisition


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict], names=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if names is None: names=sorted({k for row in rows for k in row})
    with path.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else '' if v is None else v for k,v in row.items()})


def validate_samples(rows: list[dict], originals: list[dict]):
    by_original={}
    for row in rows: by_original.setdefault(row['original_id'],[]).append(row)
    for oid,group in by_original.items():
        if len(group)!=6 or {r['label'] for r in group}!=set(CLASSES): raise ValueError('Each original must retain all six classes')
        for key in ('split','fold_id','cv_group_id'):
            if len({r[key] for r in group})!=1: raise ValueError('Original derivatives cross split/fold')
    if len({r['sample_id'] for r in rows})!=len(rows): raise ValueError('Duplicate sample IDs')
    for key in ('source_sha256','content_sha256'):
        kept=[r[key] for r in originals if r.get('screening_status')=='kept']
        if len(kept)!=len(set(kept)): raise ValueError('Original byte/content hashes are duplicated')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path)
    p.add_argument('--out',type=Path)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pilot',action='store_true')
    mode.add_argument('--production',action='store_true')
    p.add_argument('--reservation-plan',type=Path,default=Path('results/production-plan.json'))
    a=p.parse_args(argv)
    kind='production' if a.production else 'pilot'
    a.raw=a.raw or Path('data/raw')/kind
    a.out=a.out or Path('data')/kind
    if a.production and (a.out.resolve().is_relative_to(Path('data/pilot').resolve()) or a.out.resolve()==Path('data').resolve()):
        p.error('Use a separate production dataset directory')
    smoke_files=sorted(Path('results').glob('vaani-smoke*.json'))
    if not any(json.loads(f.read_text()).get('status')=='PASSED' for f in smoke_files):
        p.error('Vaani smoke must pass before the pilot build; do not skip the implementation order')
    acquisition=json.loads((a.raw/'acquisition.json').read_text(encoding='utf-8'))
    if acquisition['run_kind']!=kind: raise ValueError('Acquisition kind does not match the build mode')
    if (a.out/'manifests').exists() or (a.out/'features').exists():
        raise ValueError('Dataset namespace already contains build artifacts')
    limits={'train':6,'validation':1,'test':1}
    if a.production:
        reservation=json.loads(a.reservation_plan.read_text(encoding='utf-8'))
        limits={'train':84,'validation':18,'test':18}
        for row in acquisition['speech']:
            row['original_id'],row['sentence_id'],_=speech_identity(row)
            _,row['content_sha256']=source_audio(a.raw,row)
        for row in acquisition['noise']:
            _,row['content_sha256']=source_audio(a.raw,row)
            if row.get('corpus')!='MUSAN' or row.get('category')!='noise':
                raise ValueError('Production requires verified MUSAN environmental noise members')
        acquisition['production_gate']=validate_production_acquisition(acquisition,reservation)
        acquisition['reservation_plan_sha256']=digest(a.reservation_plan)

    samples=[];originals=[];feature_rows={s:[] for s in ['train','validation','test']}
    noise_rows=acquisition['noise']
    source_splits={}
    for row in noise_rows:
        key=row['noise_source_group_id']
        if key in source_splits and source_splits[key]!=row['split']: raise ValueError('Noise parent source crosses splits')
        source_splits[key]=row['split']
    seen_content=set();seen_sentences=set();quota={};upstream_records={}
    candidates=sorted(acquisition['speech'],key=lambda r:hashlib.sha256(f"20261115:{r['config']}:{r['source_sha256']}".encode()).hexdigest())
    for row in candidates:
        split=row['official_split'];key=(row['config'],split)
        if quota.get(key,0)>=limits[split]: continue
        audio,content=source_audio(a.raw,row)
        oid,sentence,text_hash=speech_identity(row)
        upstream_records.setdefault(str(row['upstream_id']),[]).append(oid)
        fold=quota.get(key,0)%3 if split=='train' else None
        base={**row,'original_id':oid,'content_sha256':content,'sentence_id':sentence,'transcription_sha256':text_hash,
              'split':split,'fold_id':fold,'cv_group_id':oid,'fleurs_official_split':split,'fleurs_config':row['config'],
              'source_sample_rate':audio.sample_rate,'num_samples':len(audio.samples),'duration_s':audio.duration_s,'run_kind':kind}
        x=make_feature_copy(audio)
        reasons=screen_reference(x)
        if content in seen_content: reasons.append('duplicate_recording_content')
        if sentence in seen_sentences: reasons.append('duplicate_sentence')
        selected=[n for n in noise_rows if n['split']==split and (split!='train' or n['fold_id']==fold)]
        if not selected: raise ValueError('No split/fold-disjoint noise source')
        variants=[]
        if not reasons:
            for label_id,label in enumerate(CLASSES):
                success=False
                for attempt in range(10):
                    rng=np.random.default_rng(np.random.SeedSequence([20261115,int(hashlib.sha256(oid.encode()).hexdigest()[:8],16),label_id,attempt]))
                    noise_info=None;n=None
                    if label=='added_noise':
                        noise_info=selected[int(rng.integers(0,len(selected)))]
                        noise_audio,_=source_audio(a.raw,noise_info)
                        n=make_language_copy(noise_audio)
                    try:
                        y,meta=derivative(x,label,rng,n)
                        vector=extract_features(y)
                    except ValueError: continue
                    sample={'sample_id':oid+'|'+label,'original_id':oid,'cv_group_id':oid,'fold_id':fold,'language':row['config'],'split':split,
                            'label':label,'label_id':label_id,'seed':20261115,'attempt':attempt,'generator_version':'aug-v1','feature_schema_version':'fs-v1','run_kind':kind,**meta}
                    if noise_info: sample.update(noise_id=noise_info['noise_id'],noise_category='noise',noise_source_group_id=noise_info['noise_source_group_id'])
                    variants.append((sample,dict(zip(FEATURE_NAMES,vector.tolist()))));success=True;break
                if not success: reasons.append('derivative_failed:'+label);break
        if reasons:
            originals.append({**base,'screening_status':'excluded','exclusion_reason':';'.join(reasons)})
            continue
        for sample,features in variants:
            samples.append(sample);feature_rows[split].append({'sample_id':sample['sample_id'],**features})
        originals.append({**base,'screening_status':'kept','exclusion_reason':None})
        seen_content.add(content);seen_sentences.add(sentence);quota[key]=quota.get(key,0)+1
        print(f'Kept {row["config"]}/{split}: {quota[key]} originals',flush=True)
    kept=[o for o in originals if o['screening_status']=='kept']
    if a.pilot and not 30<=len(kept)<=60: raise RuntimeError(f'Pilot requires 30–60 originals; only {len(kept)} survived')
    validate_samples(samples,originals)
    if a.production:
        expected={(lang,split) for lang in reservation['target']['languages'] for split in limits}
        if not expected.issubset(quota) or any(quota[lang,'train']<3 for lang in reservation['target']['languages']):
            raise ValueError('Production screen must retain each language in every split and all three training folds')
    if set(feature_rows)!=set(s for s, rows in feature_rows.items() if rows):
        raise ValueError('Every official partition must have retained originals')
    a.out.mkdir(parents=True,exist_ok=True);manifest=a.out/'manifests'
    write_csv(manifest/'originals.csv',originals)
    write_csv(manifest/'noise_sources.csv',noise_rows)
    write_csv(manifest/'samples.csv',samples)
    for split,rows in feature_rows.items():
        if not rows: raise RuntimeError('Every official partition must have retained originals')
        write_csv(a.out/'features/fs-v1'/f'{split}.csv',sorted(rows,key=lambda r:r['sample_id']),['sample_id',*FEATURE_NAMES])
        write_csv(manifest/f'{split}_samples.csv',sorted([s for s in samples if s['split']==split],key=lambda r:r['sample_id']))
    lock={'run_kind':kind,'test_samples_sha256':digest(manifest/'test_samples.csv'),'test_features_sha256':digest(a.out/'features/fs-v1/test.csv')}
    (manifest/'test_lock.json').write_text(json.dumps(lock,indent=2))
    schema=feature_schema();(a.out/'features/fs-v1/schema.json').write_text(json.dumps(schema,indent=2))
    provenance={**acquisition,'raw_root':str(a.raw.resolve()),'retained_originals':len(kept),'rows':len(samples),
                'counts':{s:sum(o['split']==s for o in kept) for s in feature_rows},'speaker_independence':'NOT ESTABLISHED; no verified speaker IDs',
                'upstream_id_collision_count':sum(len(set(ids))>1 for ids in upstream_records.values()),'upstream_id_audit':upstream_records,
                'feature_schema_hash':schema['sha256'],'feature_cache_format':'CSV float32 roundtrip; Parquet deferred',
                'screening_exclusions':[{'original_id':o['original_id'],'reason':o['exclusion_reason']} for o in originals if o['screening_status']=='excluded']}
    (manifest/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps({'originals':len(kept),'samples':len(samples),'data_root':str(a.out)}))


if __name__=='__main__': main()
