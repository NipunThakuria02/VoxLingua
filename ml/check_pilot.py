"""Audit the real pilot's manifest/partition integrity without opening test features."""
import argparse
import csv
import json
from pathlib import Path
from .build_dataset import validate_samples


def check(root: Path) -> dict:
    def rows(name):
        with (root/'manifests'/name).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
    samples=rows('samples.csv');originals=rows('originals.csv');noise=rows('noise_sources.csv')
    validate_samples(samples,originals)
    kept=[r for r in originals if r['screening_status']=='kept']
    for r in kept:
        if r['split']!=r['fleurs_official_split']: raise ValueError('Official partition changed')
    for key in ('original_id','content_sha256','source_sha256','sentence_id'):
        partitions={}
        for r in kept:
            if r[key] in partitions and partitions[r[key]]!=r['split']: raise ValueError(key+' crosses partitions')
            partitions[r[key]]=r['split']
    for key in ('noise_source_group_id','source_sha256'):
        partitions={}
        for r in noise:
            if r[key] in partitions and partitions[r[key]]!=r['split']: raise ValueError(key+' crosses noise partitions')
            partitions[r[key]]=r['split']
    for group in {r['cv_group_id'] for r in samples if r['split']=='train'}:
        if len({r['fold_id'] for r in samples if r['cv_group_id']==group})!=1: raise ValueError('CV group spans folds')
    return {'status':'PASSED','originals':len(kept),'samples':len(samples),'classes':6,
            'originals_by_split':{s:sum(r['split']==s for r in kept) for s in ['train','validation','test']},
            'class_support':{s:{str(k):sum(r['split']==s and r['label_id']==str(k) for r in samples) for k in range(6)} for s in ['train','validation','test']},
            'noise_corpus':'ESC-10','noise_is_fallback':True,'test_features_opened':False,'speaker_independence':'NOT ESTABLISHED'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('data/pilot'));p.add_argument('--out',type=Path,default=Path('results/pilot-integrity.json'))
    a=p.parse_args(argv);report=check(a.data);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__': main()
