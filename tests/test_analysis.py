from io import BytesIO
from unittest.mock import patch
import numpy as np
import soundfile as sf
from voxlingua.analysis import analyze_recording
from voxlingua.schemas import ServiceError
from ml.download_models import official_storage
import pytest


def fixture(silent=False):
    stream=BytesIO()
    x=np.zeros(32000) if silent else 0.3*np.sin(2*np.pi*200*np.arange(32000)/16000)
    sf.write(stream,x,16000,format='WAV',subtype='PCM_16')
    return stream.getvalue()


class FakeLanguage:
    def identify(self,x): return {'stub':True}


class FakeDistortion:
    def predict(self,x): return {'stub':True}


def test_all_branches_and_offline_runtime():
    with patch('socket.socket.connect',side_effect=AssertionError('No runtime network allowed')):
        report=analyze_recording(fixture(),language_service=FakeLanguage(),distortion_service=FakeDistortion())
    assert report['status']=='success'
    assert set(['decode','acoustics','features','classify','language']).issubset(report['timings_ms'])


def test_missing_weights_or_models_produce_honest_partial():
    report=analyze_recording(fixture())
    assert report['status']=='partial' and report['acoustics_status']=='ok'
    assert report['language_status']==report['distortion_status']=='unavailable'
    assert report['language'] is None and report['distortion'] is None


def test_silent_skipped_language_is_partial():
    report=analyze_recording(fixture(True),distortion_service=FakeDistortion())
    assert report['status']=='partial' and report['language_status']=='skipped'
    assert report['distortion_status']=='ok'


def test_close_language_results_are_reported_as_tentative():
    class CloseScores:
        def identify(self, _):
            return {'top_label':'Urdu','top5':[{'label':'Urdu','score':.249},{'label':'Hindi','score':.234}]}

    report=analyze_recording(fixture(),language_service=CloseScores())
    assert report['language_status']=='ok' and report['language']['top_label']=='Urdu'
    assert any('tentative' in warning for warning in report['warnings'])


def test_failure_isolation():
    class Failing:
        def identify(self,x): raise ServiceError('E_LANG_OOM','Test OOM')
    report=analyze_recording(fixture(),language_service=Failing(),distortion_service=FakeDistortion())
    assert report['status']=='partial' and report['language_status']=='error'
    assert report['distortion_status']==report['acoustics_status']=='ok'


@pytest.mark.parametrize('url',['http://huggingface.co/file','https://huggingface.co.attacker.example/file','https://user:secret@huggingface.co/file','https://random-mirror.example/file'])
def test_weight_downloads_reject_untrusted_redirects(url):
    with pytest.raises(ValueError): official_storage(url)
