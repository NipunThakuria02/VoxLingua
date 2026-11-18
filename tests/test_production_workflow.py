"""Exercise the production workflow in a disposable workspace with generated signals."""
import hashlib
from io import BytesIO
import json
import tarfile
import time

import numpy as np
import pytest
import soundfile as sf

from ml import build_dataset, evaluate, train_models
from ml.acquire_production import noise_subset
from ml.promote import promote
from voxlingua.config import RuntimeConfig
from voxlingua.research import read_production_research
from voxlingua.runtime import AnalysisEngine


def test_provisioned_archive_preserves_distinct_parents(tmp_path):
    archive_path = tmp_path / 'musan.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for index in range(16):
            signal = np.random.default_rng(max(0,index-1)).normal(0,.05,16000)
            encoded = BytesIO()
            sf.write(encoded,signal,16000,format='WAV',subtype='PCM_16')
            contents = encoded.getvalue()
            member = tarfile.TarInfo(f'musan/noise/test/noise-{index}.wav')
            member.size = len(contents)
            archive.addfile(member,BytesIO(contents))
    rows, receipt = noise_subset(tmp_path/'out',time.monotonic()+15,local_archive=archive_path)
    assert len(rows)==len({row['parent_sha256'] for row in rows})==15
    assert {row['split'] for row in rows}=={'train','validation','test'}
    assert receipt['transfer_bytes']==0 and receipt['compressed_bytes_read']>0


def test_prod_flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / 'raw'
    raw.mkdir()
    results = tmp_path / 'results'
    results.mkdir()
    (results / 'vaani-smoke-test.json').write_text('{"status":"PASSED"}')
    reservation = {
        'target': {'languages': ['en_us']},
        'reserved_pilot_sources': {'speech': [], 'noise': [], 'sentence_ids': [], 'original_ids': []},
    }
    (results / 'production-plan.json').write_text(json.dumps(reservation))
    buckets = [('train', 0), ('train', 1), ('train', 2), ('validation', None), ('test', None)]
    speech, noise = [], []
    t = np.arange(48000) / 16000
    rng = np.random.default_rng(11)
    for index, (split, fold) in enumerate(buckets):
        speech_name = f'speech-{index}.wav'
        noise_name = f'noise-{index}.wav'
        signal = (.7 + .25 * np.sin(2*np.pi*5*t)) * (
            .12*np.sin(2*np.pi*(170+index*21)*t) + .09*np.sin(2*np.pi*710*t)
            + .04*np.sin(2*np.pi*2990*t) + .007*rng.standard_normal(len(t))
        )
        sf.write(raw/speech_name, signal, 16000, subtype='PCM_16')
        sf.write(raw/noise_name, .05*rng.standard_normal(len(t)), 16000, subtype='PCM_16')
        speech.append({'source':'google/fleurs','config':'en_us','official_split':split,
                       'source_revision':'synthetic-test','source_relpath':speech_name,
                       'source_sha256':hashlib.sha256((raw/speech_name).read_bytes()).hexdigest(),
                       'transcription':f'Synthetic fixture {index}','upstream_id':str(index)})
        noise.append({'corpus':'MUSAN','category':'noise','split':split,'fold_id':fold,
                      'noise_id':noise_name,'noise_source_group_id':noise_name,'source_relpath':noise_name,
                      'source_sha256':hashlib.sha256((raw/noise_name).read_bytes()).hexdigest()})
    (raw/'acquisition.json').write_text(json.dumps({
        'run_kind':'production','noise_corpus':'MUSAN','noise_is_fallback':False,
        'licenses':{'FLEURS':'fixture','MUSAN':'fixture'},'revisions':{'FLEURS':'fixture','MUSAN':'fixture'},
        'citations':{},'limitations':['Synthetic software acceptance fixture'],'speech':speech,'noise':noise,
    }))
    build_dataset.main(['--production','--raw',str(raw)])
    train_models.main(['--production'])
    run = next((tmp_path / 'models').iterdir()).name
    assert read_production_research()['status']=='empty'
    with pytest.raises(ValueError):
        promote(run)
    evaluate.main(['--production','--run-id',run,'--timing-repeats','5'])
    with pytest.raises(ValueError, match='already exposed'):
        train_models.main(['--production'])
    pointer = promote(run)
    assert json.loads(pointer.read_text())['run_id']==run
    report = AnalysisEngine(RuntimeConfig()).analyze((raw/speech[0]['source_relpath']).read_bytes())
    assert report['distortion']['run_id']==run and report['acoustics_status']=='ok'
    assert read_production_research()['status']=='available'
    with pytest.raises(FileExistsError):
        promote(run)
    metrics = results/run/'metrics.json'
    metrics.write_bytes(metrics.read_bytes()+b'\n')
    assert read_production_research()['status']=='empty'
