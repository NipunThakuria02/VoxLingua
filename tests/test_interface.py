"""Actual Streamlit navigation/state tests with synthetic audio and a stub language service."""
from io import BytesIO
from pathlib import Path
import base64
import json
import numpy as np
import pytest
import soundfile as sf
import streamlit as st
from streamlit.testing.v1 import AppTest
from voxlingua.analysis import CLOSE_LANGUAGE_SCORES_WARNING, analyze_recording
from voxlingua.config import RuntimeConfig
from voxlingua.language import VAANI_REVISION
from voxlingua.session import RecordingSession
from voxlingua.schemas import ServiceError
from voxlingua.language_regions import language_region
from ui.components import microphone_audio

APP=Path(__file__).parents[1]/'app.py'


def wav():
    b=BytesIO();sf.write(b,.2*np.sin(2*np.pi*200*np.arange(32000)/16000),16000,format='WAV',subtype='PCM_16');return b.getvalue()


class Language:
    def identify(self,x): return {'top_label':'Hindi','top_score':.7,'top5':[{'label':'Hindi','score':.7},{'label':'English','score':.3}],
                                 'inference_ms':1.0,'device':'test-only','model_id':'synthetic-test-stub','revision':'test-only','warnings':[]}


class CountingEngine:
    config=RuntimeConfig()
    def __init__(self): self.calls=0
    def analyze(self,data,filename):
        self.calls+=1
        return analyze_recording(data,filename,language_service=Language())


@pytest.mark.parametrize('page',['Home','Language Lab','Voice Lab','Research Lab','Methodology'])
def test_five_pages_render_without_errors(page, monkeypatch, tmp_path):
    if page=='Research Lab':
        import voxlingua.research as research
        read_production = research.read_production_research
        monkeypatch.setattr(research,'read_production_research',
                            lambda *_: read_production(tmp_path/'models',tmp_path/'results'))
    at=AppTest.from_file(str(APP));at.session_state['page']=page;at.run(timeout=20)
    assert not at.exception
    if page=='Research Lab': assert any('Evaluation not available' in m.value for m in at.markdown)


def test_home_start_and_methodology_navigation():
    at=AppTest.from_file(str(APP)).run(timeout=20)
    next(b for b in at.button if 'Start analysis' in b.label).click().run(timeout=20)
    assert at.session_state['page']=='Language Lab' and not at.exception
    at.button(key='nav_4').click().run(timeout=20)
    assert at.button(key='nav_4').label == 'Methodology'
    assert at.session_state['page'] == 'Methodology'
    at.button(key='method_01').click().run(timeout=20)
    assert at.session_state['method_section']=='Architecture' and not at.exception
    at.button(key='method_back').click().run(timeout=20)
    assert at.session_state['method_section'] is None


@pytest.mark.parametrize('key,page',[
    ('home_language','Language Lab'),
    ('home_acoustics','Voice Lab'),
    ('home_intelligence','Research Lab'),
])
def test_home_branches_open_their_labs(key,page):
    at=AppTest.from_file(str(APP)).run(timeout=20)
    at.button(key=key).click().run(timeout=20)
    assert at.session_state['page']==page and not at.exception


@pytest.mark.parametrize('page',['Language Lab','Voice Lab'])
def test_idle_lab_audio_is_a_closed_disclosure(page):
    at=AppTest.from_file(str(APP));at.session_state['page']=page;at.run(timeout=20)
    audio=next(e for e in at.expander if 'Add audio' in e.label)
    assert not audio.proto.expanded
    assert at.radio(key='capture_source').value=='Upload'
    assert not at.exception


def test_analyse_once_switch_labs_and_clear_widget_epoch():
    at=AppTest.from_file(str(APP)).run(timeout=20)
    engine=CountingEngine();session=RecordingSession(engine);session.set_recording(wav())
    at.session_state['recording']=session;at.session_state['page']='Language Lab'
    at.run(timeout=20);at.button(key='analyse_audio').click().run(timeout=20)
    assert not at.exception and engine.calls==1 and at.session_state['report']['language']['top_label']=='Hindi'
    at.button(key='nav_2').click().run(timeout=20)
    assert not at.exception and engine.calls==1 and at.session_state['recording'].has_recording
    assert at.session_state['report']['acoustics']['pitch_median_hz']==pytest.approx(200,rel=0.02)
    assert any('VOICED PITCH' in m.value and '200' in m.value for m in at.markdown)
    detail = next(m.value for m in at.markdown if '<div class="detail-grid">' in m.value)
    measured_pitch = at.session_state['report']['acoustics']['pitch_median_hz']
    assert all(value in detail for value in ('Recording', 'Level &amp; clipping', 'Spectral characteristics',
                                             f'{measured_pitch:.1f} Hz', '16,000 Hz', 'Unavailable'))
    assert not at.json
    at.button(key='clear_audio').click().run(timeout=20)
    assert not at.session_state['recording'].has_recording and at.session_state['report'] is None
    assert at.session_state['capture_epoch']==1


def test_labs_hide_explanatory_captions_and_keep_actionable_warnings():
    class ReportEngine(CountingEngine):
        def analyze(self, data, filename):
            report = super().analyze(data, filename)
            report['warnings'].append(CLOSE_LANGUAGE_SCORES_WARNING)
            scores = {'reference': 1.0}
            report['distortion'] = {
                'predicted_class': 'reference', 'class_scores': scores, 'voting_scores': scores,
                'base_predictions': {'cart': {'top': 'reference', 'scores': scores}},
                'final_model': 'stacking', 'latency_ms': 1.0, 'run_id': 'synthetic-test',
            }
            return report

    at = AppTest.from_file(str(APP)).run(timeout=20)
    session = RecordingSession(ReportEngine())
    session.set_recording(wav())
    at.session_state['recording'] = session
    at.session_state['page'] = 'Language Lab'
    at.run(timeout=20)
    at.button(key='analyse_audio').click().run(timeout=20)
    assert not at.exception
    assert CLOSE_LANGUAGE_SCORES_WARNING in at.session_state['report']['warnings']
    assert all('highest language scores' not in c.value and 'Model scores' not in c.value for c in at.caption)
    assert any('Short input' in c.value for c in at.caption)
    assert at.warning

    at.button(key='nav_2').click().run(timeout=20)
    assert not at.exception
    assert any(h.value == 'Reference' for h in at.subheader)
    for removed in ('highest language scores', 'Final method:', 'Multiple problems may coexist'):
        assert all(removed not in c.value for c in at.caption)


def test_recorder_payload_roundtrip_and_size_guards():
    data=wav();value={'state':'ready','audio_base64':base64.b64encode(data).decode()}
    assert microphone_audio(value,len(data))==data
    with pytest.raises(ServiceError): microphone_audio(value,10)
    with pytest.raises(ServiceError): microphone_audio({'state':'ready','audio_base64':'not-base64!'},100)
    assert microphone_audio({'state':'recording'},100) is None


def test_region_association_is_curated_not_geolocation():
    assert language_region('Hindi')['native']=='हिन्दी'
    assert 'not the speaker' in language_region('Hindi')['meaning']
    assert language_region('Unmapped test label') is None


def test_short_microphone_capture_is_recoverable(monkeypatch):
    import ui.components
    b=BytesIO();sf.write(b,np.zeros(4000),16000,format='WAV',subtype='PCM_16')
    monkeypatch.setattr(ui.components,'microphone',lambda **kwargs:{'state':'ready','audio_base64':base64.b64encode(b.getvalue()).decode()})
    at=AppTest.from_file(str(APP));at.session_state['page']='Language Lab';at.run(timeout=20)
    at.radio(key='capture_source').set_value('Record').run(timeout=20)
    assert not at.exception
    assert any('at least one second' in error.value for error in at.error)
    assert not at.session_state['recording'].has_recording
    assert not at.session_state['recording_active']
    at.radio(key='capture_source').set_value('Upload').run(timeout=20)
    assert not at.exception and at.get('file_uploader')


@pytest.mark.parametrize('constant_level', [0, .02])
def test_microphone_without_signal_preserves_previous_upload(monkeypatch, constant_level):
    import ui.components
    buffer = BytesIO()
    sf.write(buffer, np.full(32000, constant_level, dtype=np.float32), 16000, format='WAV', subtype='PCM_16')
    payload = {'state': 'ready', 'audio_base64': base64.b64encode(buffer.getvalue()).decode()}
    monkeypatch.setattr(ui.components, 'microphone', lambda **kwargs: payload)
    at = AppTest.from_file(str(APP)).run(timeout=20)
    session = RecordingSession(CountingEngine())
    uploaded = wav()
    session.set_recording(uploaded, 'upload.wav')
    at.session_state['recording'] = session
    at.session_state['page'] = 'Language Lab'
    at.run(timeout=20)
    at.radio(key='capture_source').set_value('Record').run(timeout=20)
    assert not at.exception
    assert any('no usable signal' in error.value for error in at.error)
    assert session.playback_bytes() == uploaded
    assert not at.session_state['recording_active']


def test_microphone_payload_can_be_analysed_replayed_and_cleared(monkeypatch):
    import ui.components
    data = wav()
    payload = {'state': 'ready', 'audio_base64': base64.b64encode(data).decode()}
    monkeypatch.setattr(ui.components, 'microphone', lambda **kwargs: payload)
    at = AppTest.from_file(str(APP)).run(timeout=20)
    engine = CountingEngine()
    session = RecordingSession(engine)
    at.session_state['recording'] = session
    at.session_state['page'] = 'Language Lab'
    at.run(timeout=20)
    at.radio(key='capture_source').set_value('Record').run(timeout=20)
    assert not at.exception
    assert session.playback_bytes() == data
    assert at.get('audio')
    at.button(key='analyse_audio').click().run(timeout=20)
    assert engine.calls == 1 and at.session_state['report']['language']['top_label'] == 'Hindi'
    at.button(key='nav_2').click().run(timeout=20)
    assert not at.exception and engine.calls == 1
    payload.clear()
    at.button(key='clear_audio').click().run(timeout=20)
    assert not at.exception and not session.has_recording and at.session_state['report'] is None


def test_switching_audio_source_preserves_report_after_recorder_error(monkeypatch):
    import ui.components
    monkeypatch.setattr(ui.components,'microphone',lambda **kwargs:{'state':'error','message':'Microphone permission denied.'})
    at=AppTest.from_file(str(APP));at.session_state['page']='Language Lab';at.run(timeout=20)
    engine=CountingEngine();session=RecordingSession(engine);session.set_recording(wav())
    at.session_state['recording']=session
    at.button(key='analyse_audio').click().run(timeout=20)
    at.radio(key='capture_source').set_value('Record').run(timeout=20)
    assert any('experimental' in c.value for c in at.caption)
    assert any('Microphone permission denied' in c.value for c in at.caption)
    at.radio(key='capture_source').set_value('Upload').run(timeout=20)
    assert not at.exception and at.get('file_uploader')
    assert session.has_recording and engine.calls==1
    assert at.session_state['report']['language']['top_label']=='Hindi'


def test_architecture_changes_and_ensemble_has_distinct_content():
    from ui.methodology import ARCHITECTURE_BRANCHES, architecture_html
    at = AppTest.from_file(str(APP))
    at.session_state['page'] = 'Methodology'
    at.session_state['method_section'] = 'Architecture'
    at.run(timeout=20)
    for branch, (heading, _, _, path) in ARCHITECTURE_BRANCHES.items():
        at.selectbox(key='architecture_branch').set_value(branch).run(timeout=20)
        assert not at.exception and not at.info
        note = next(m.value for m in at.markdown if '<section class="branch-explanation">' in m.value)
        assert heading in note
        assert all(other == branch or values[0] not in note for other, values in ARCHITECTURE_BRANCHES.items())
        assert f'path="{path}"' in architecture_html(False, branch)
        assert '<animateMotion' not in architecture_html(True, branch)
    at.session_state['method_section'] = 'Ensemble learning'
    at.run(timeout=20)
    assert not at.exception
    assert [t.label for t in at.tabs]==['Soft voting','Grouped stacking']
    assert any('flow-steps' in m.value and '25%' in m.value for m in at.tabs[0].markdown)
    assert any('Offline training' in m.value and 'Refitted base models' in m.value for m in at.tabs[1].markdown)


def test_supported_languages_grid_uses_sorted_model_metadata(tmp_path, monkeypatch):
    folder = tmp_path / 'Vaani-LID_v0' / VAANI_REVISION
    folder.mkdir(parents=True)
    (folder / 'config.json').write_text(json.dumps({'id2label': {'0': 'Tamil', '1': 'Hindi', '2': 'English'}}))
    monkeypatch.setattr(RuntimeConfig, 'from_yaml', lambda _: RuntimeConfig(upstream_models=tmp_path))
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(str(APP))
        at.session_state['page'] = 'Language Lab'
        at.run(timeout=20)
    finally:
        st.cache_resource.clear()
    assert not at.exception
    assert any('3 supported labels' == c.value for c in at.caption)
    catalogue = next(element for element in at.get('html') if 'language-grid' in element.proto.body)
    assert '<li>English</li><li>Hindi</li><li>Tamil</li>' in catalogue.proto.body
