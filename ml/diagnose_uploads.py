"""Compare known-language FLEURS WAVs through the local app engine."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from voxlingua.config import RuntimeConfig
from voxlingua.runtime import AnalysisEngine

LANGUAGES = {'en_us':'English', 'hi_in':'Hindi', 'ta_in':'Tamil', 'bn_in':'Bengali', 'te_in':'Telugu'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--dataset', choices=['pilot','production-validation'], default='pilot')
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    production = args.dataset=='production-validation'
    manifests = Path('data/production/manifests' if production else 'data/pilot/manifests')
    provenance = json.loads((manifests/'provenance.json').read_text(encoding='utf-8'))
    raw = Path(provenance['raw_root']).resolve()
    config = RuntimeConfig.from_yaml('configs/runtime.yaml')
    selected = []
    if production:
        # Include screened-out references because uploads can have those recording conditions.
        selected = sorted((dict(row, split='validation') for row in provenance['speech']
                           if row['official_split']=='validation'),
                          key=lambda row:(row['config'],row['original_id']))
        selection='All acquired official validation WAVs, including screened-out references'
    else:
        with (manifests/'originals.csv').open(newline='', encoding='utf-8') as handle:
            candidates = [row for row in csv.DictReader(handle)
                          if row['screening_status']=='kept' and row['split'] in ('train','validation')]
        for code in LANGUAGES:
            for split, count in [('validation', 1), ('train', 3)]:
                matching = sorted((row for row in candidates if row['config']==code and row['split']==split),
                                  key=lambda row:row['original_id'])
                selected.extend(matching[:count])
        selection='First 1 validation and 3 train originals per language by stable ID'
    observations = []
    engine = AnalysisEngine(config)
    try:
        for row in selected:
            path = (raw/row['source_relpath']).resolve()
            if not path.is_relative_to(raw):
                raise ValueError('Diagnostic path escapes the acquisition')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=row['source_sha256']:
                raise ValueError('Diagnostic source hash changed')
            report = engine.analyze(data, path.name)
            language = report['language']
            observations.append({'expected':LANGUAGES[row['config']], 'dataset_split':row['split'],
                                 'source':row['source_relpath'], 'source_sha256':row['source_sha256'],
                                 'source_revision':row['source_revision'], 'original_id':row['original_id'],
                                 'input':report['audio_info'], 'language_status':report['language_status'],
                                 'language':language, 'warnings':report['warnings'],
                                 'language_errors':[err for err in report['errors'] if err.get('branch')=='language']})
            print(LANGUAGES[row['config']], language['top_label'] if language else report['language_status'], flush=True)
    finally:
        engine.unload()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump({'scope':'Known-language diagnostic on already-reserved FLEURS originals',
                   'dataset':args.dataset, 'selection':selection,
                   'browser_parity':'NOT RUN','observations':observations},handle,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
