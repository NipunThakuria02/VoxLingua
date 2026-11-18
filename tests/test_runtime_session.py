import hashlib
from io import BytesIO
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import soundfile as sf
from voxlingua.config import RuntimeConfig
from voxlingua.runtime import AnalysisEngine
from voxlingua.session import RecordingSession
from voxlingua.research import read_production_research
from voxlingua.schemas import ServiceError
from ml.train_models import json_metadata


def audio(frequency=200):
    stream=BytesIO();x=.2*np.sin(2*np.pi*frequency*np.arange(32000)/16000)
    sf.write(stream,x,16000,format='WAV',subtype='PCM_16')
    return stream.getvalue()


class CountingEngine:
    config=RuntimeConfig()
    def __init__(self): self.calls=0
    def analyze(self,data,filename,*,refine_upload=True):
        self.calls+=1
        return {'language':{'top_label':'test-only'},'distortion':{'run_id':'test-only'},'plots':[1,2],
                'refine_upload':refine_upload}


def test_shared_recording_cached_across_views_and_defensive_results():
    engine=CountingEngine();session=RecordingSession(engine);data=audio()
    session.set_recording(data,'recording.wav')
    language_view=session.analyze();language_view['plots'].append(3)
    voice_view=session.analyze()
    assert engine.calls==1 and voice_view['plots']==[1,2]
    session.set_recording(data,'renamed.wav');session.analyze()
    assert engine.calls==1 and session.playback_bytes()==data
    session.set_recording(audio(300));session.analyze()
    assert engine.calls==2


def test_clear_releases_session_audio_results_and_requires_new_input():
    session=RecordingSession(CountingEngine());session.set_recording(audio());session.analyze();session.clear()
    assert not session.has_recording and session.playback_bytes() is None and session.audio_info is None
    assert session._result is None
    with pytest.raises(ServiceError,match='first'): session.analyze()


def test_same_wav_from_both_sources_uses_language_refinement():
    engine = CountingEngine()
    session = RecordingSession(engine)
    data = audio()
    session.set_recording(data, 'microphone.wav')
    assert session.analyze()['refine_upload'] is True
    assert session.is_new_recording(data, microphone=True)
    session.set_recording(data, 'microphone.wav', microphone=True)
    assert session.analyze()['refine_upload'] is True
    assert engine.calls == 2


def test_invalid_replacement_does_not_destroy_valid_recording():
    session=RecordingSession(CountingEngine());data=audio();session.set_recording(data)
    with pytest.raises(ServiceError): session.set_recording(b'invalid')
    assert session.playback_bytes()==data


def test_sessions_do_not_share_user_audio():
    engine=CountingEngine();first=RecordingSession(engine);second=RecordingSession(engine)
    first.set_recording(audio())
    assert not second.has_recording


def test_explicit_retry_only_reruns_on_request():
    engine=CountingEngine();session=RecordingSession(engine);session.set_recording(audio())
    session.analyze();session.analyze();session.retry_analysis()
    assert engine.calls==2


def test_runtime_config_relative_paths_and_limits(tmp_path):
    path=tmp_path/'runtime.yaml';path.write_text('models_dir: ./local-models\nlanguage_enabled: false\naudio: {max_seconds: 12}\n')
    config=RuntimeConfig.from_yaml(path)
    assert config.models_dir==tmp_path/'local-models' and config.audio.max_seconds==12


@pytest.mark.parametrize('text',['audio: {max_seconds: 30}', 'language_enabled: "false"','audio: {max_bytes: true}','run_kind: pilot','[]'])
def test_invalid_runtime_config_is_rejected(tmp_path,text):
    path=tmp_path/'bad.yaml';path.write_text(text)
    with pytest.raises((ValueError,TypeError)): RuntimeConfig.from_yaml(path)


def test_no_production_model_is_honest_partial_and_runtime_stays_offline(tmp_path):
    engine=AnalysisEngine(RuntimeConfig(models_dir=tmp_path/'models'))
    with patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
        result=engine.analyze(audio())
    assert result['status']=='partial' and result['acoustics_status']=='ok'
    assert result['distortion_status']=='unavailable' and not result['model_context']['pilot_models_allowed']


def test_runtime_preserves_model_integrity_error(tmp_path):
    engine=AnalysisEngine(RuntimeConfig(models_dir=tmp_path))
    with patch('voxlingua.distortion.DistortionService.load',side_effect=ServiceError('E_MODEL_INTEGRITY_FAILED','Tampered test artifact')):
        result=engine.analyze(audio())
    assert any(e['code']=='E_MODEL_INTEGRITY_FAILED' for e in result['errors'])
    assert result['status']=='partial'


def test_research_missing_and_pilot_pointer_never_show_scores(tmp_path):
    models=tmp_path/'models';models.mkdir();results=tmp_path/'results'
    assert read_production_research(models,results)['status']=='empty'
    (models/'current.json').write_text(json.dumps({'run_id':'pilot/any-run'}))
    assert read_production_research(models,results)['metrics'] is None
    (models/'current.json').write_text(json.dumps({'run_id':'../outside'}))
    assert read_production_research(models,results)['status']=='empty'


def test_research_reads_real_fixture_and_detects_tampered_predictions(tmp_path):
    models=tmp_path/'models';results=tmp_path/'results';run='test-production-fixture'
    folder=models/run;output=results/run;folder.mkdir(parents=True);output.mkdir(parents=True)
    (models/'current.json').write_text(json.dumps({'run_id':run}))
    (folder/'run_config.json').write_text(json.dumps({'run_kind':'production'}))
    manifest=folder/'artifact_manifest.json';manifest.write_text('{}')
    predictions=output/'predictions.csv';predictions.write_text('test fixture only')
    metrics={'run_kind':'production','run_id':run,'provenance':{'artifact_manifest_hash':hashlib.sha256(manifest.read_bytes()).hexdigest()},
             'models':{'test-only':{'macro_f1':0.25,'predictions_path':predictions.name,'predictions_sha256':hashlib.sha256(predictions.read_bytes()).hexdigest()}}}
    (output/'metrics.json').write_text(json.dumps(metrics))
    assert read_production_research(models,results)['metrics']['models']['test-only']['macro_f1']==0.25
    predictions.write_text('tampered fixture')
    assert read_production_research(models,results)['status']=='empty'


def test_strict_json_parameter_metadata_does_not_mutate_model_defaults():
    original={'missing':float('nan'),'nested':[float('inf'),float('-inf'),3]}
    safe=json_metadata(original)
    encoded=json.dumps(safe,allow_nan=False)
    assert json.loads(encoded)=={'missing':'NaN','nested':['Infinity','-Infinity',3]}
    assert np.isnan(original['missing'])


def test_pilot_research_is_explicit_and_detects_tampering(tmp_path):
    from voxlingua.research import read_pilot_research
    models=tmp_path/'models';results=tmp_path/'results';run='test-pilot'
    folder=models/'pilot'/run;output=results/'pilot'/run
    folder.mkdir(parents=True);output.mkdir(parents=True)
    (folder/'run_config.json').write_text(json.dumps({'run_kind':'pilot'}))
    manifest=folder/'artifact_manifest.json';manifest.write_text('{}')
    predictions=output/'predictions.csv';predictions.write_text('synthetic evidence')
    ids=output/'sample_ids.csv';ids.write_text('sample_id\nsynthetic-ID\n')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    metrics={'run_kind':'pilot','run_id':run,'provenance':{'artifact_manifest_hash':digest(manifest)},
             'models':{'test-only':{'predictions_path':predictions.name,'predictions_sha256':digest(predictions)}},
             'sample_id_parity':{'sorted_ids_path':ids.name,'sorted_ids_sha256':hashlib.sha256(json.dumps(['synthetic-ID'],separators=(',',':')).encode()).hexdigest()}}
    (output/'metrics.json').write_text(json.dumps(metrics))
    assert read_production_research(models,results)['status']=='empty'
    assert read_pilot_research(run,models,results)['status']=='available'
    assert not (models/'current.json').exists()
    predictions.write_text('tampered')
    assert read_pilot_research(run,models,results)['status']=='empty'
    assert read_pilot_research('../escape',models,results)['status']=='empty'
