"""VoxLingua local interface. All audio/model computation stays in services."""
from pathlib import Path
from html import escape
import json
import sys

FRONTEND = Path(__file__).resolve().parent / 'frontend'
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))
import streamlit as st
import streamlit.components.v1 as components
from voxlingua.analysis import CLOSE_LANGUAGE_SCORES_WARNING
from voxlingua.config import RuntimeConfig
from voxlingua.runtime import AnalysisEngine
from voxlingua.session import RecordingSession
from voxlingua.schemas import ServiceError
from voxlingua.language import VAANI_REVISION
from voxlingua.language_regions import language_region
from voxlingua.research import read_production_research,read_pilot_research
from ui.components import microphone,microphone_audio
from ui.visuals import scene
from ui.player import waveform_player
from ui import charts,methodology
from ui.readouts import acoustic_details, language_grid

ROOT=Path(__file__).resolve().parent
PAGES=['Home','Language Lab','Voice Lab','Research Lab','Methodology']
st.set_page_config(page_title='VoxLingua · A new perspective on sound',page_icon='◌',layout='wide',initial_sidebar_state='expanded')


@st.cache_resource
def engine():
    configuration = ROOT / 'configs/runtime.local.yaml'
    if not configuration.exists():
        configuration = ROOT / 'configs/runtime.yaml'
    return AnalysisEngine(RuntimeConfig.from_yaml(configuration))


def navigate(page): st.session_state.page=page


def clear_audio():
    st.session_state.recording.clear()
    st.session_state.report=None
    st.session_state.capture_epoch+=1
    st.session_state.recording_active=False
    st.session_state.pop('capture_error',None)


def initialize():
    st.session_state.setdefault('page','Home')
    if st.session_state.page not in PAGES:
        st.session_state.page='Methodology'
    st.session_state.setdefault('report',None)
    st.session_state.setdefault('capture_epoch',0)
    st.session_state.setdefault('recording_active',False)
    if 'recording' not in st.session_state: st.session_state.recording=RecordingSession(engine())


def change_audio_source():
    """Discard the previous input widget while retaining the accepted recording."""
    st.session_state.capture_epoch+=1
    st.session_state.recording_active=False


def source_inputs(session,epoch,compact):
    source=st.radio('Audio source',['Upload','Record'],horizontal=True,key='capture_source',
                    format_func=lambda option: 'Upload WAV' if option=='Upload' else 'Record (experimental)',
                    on_change=change_audio_source)
    if source=='Record':
        st.caption('Live recording is experimental. Use WAV upload for the supported workflow.')
        value=microphone(key='microphone_'+str(epoch),compact=compact,reduced=st.session_state.reduced_motion,default=None)
        st.session_state.recording_active=isinstance(value,dict) and value.get('state')=='recording'
        if isinstance(value,dict) and value.get('state')=='error': st.caption(value.get('message','Microphone unavailable. WAV upload remains available.'))
        try:
            data=microphone_audio(value,engine().config.audio.max_bytes)
            if data is not None:
                # Session validates, preserves same-input cache and replaces changed input atomically.
                changed=session.is_new_recording(data,microphone=True)
                session.set_recording(data,'microphone.wav',microphone=True)
                if changed: st.session_state.report=None;st.rerun()
        except ValueError as exc:
            st.session_state.recording_active=False
            st.error('Audio could not be accepted: '+str(exc)+' Record again for 5–10 seconds, or upload a WAV file.')
    else:
        value=st.file_uploader('Drop a WAV recording',type=['wav'],accept_multiple_files=False,key='upload_'+str(epoch),help='1–15 seconds · maximum 20 MiB · original amplitude preserved')
        if value is not None:
            try:
                data=value.getvalue();changed=session.is_new_recording(data);session.set_recording(data,value.name)
                if changed: st.session_state.report=None;st.rerun()
            except ValueError as exc: st.error('Audio could not be accepted: '+str(exc))
        st.session_state.recording_active=False
def capture(compact=False):
    session=st.session_state.recording;epoch=st.session_state.capture_epoch
    with st.expander('＋ Add audio',expanded=False):
        source_inputs(session,epoch,compact)
        has_waveform=compact and st.session_state.page=='Voice Lab' and st.session_state.report and st.session_state.report.get('acoustics')
        if session.has_recording and not has_waveform:
            info=session.audio_info
            st.audio(session.playback_bytes(),format='audio/wav')
            st.caption(f'{info["duration_s"]:.2f} s · {info["sample_rate"]:,} Hz · {info["n_channels"]} channel(s) · Shared across both labs')
        a,b,c=st.columns([1.4,1,1])
        analyse=a.button('Analyse audio  ↗',type='primary',disabled=not session.has_recording or st.session_state.recording_active,use_container_width=True,key='analyse_audio')
        b.button('Replace',disabled=st.session_state.recording_active,on_click=clear_audio,use_container_width=True,key='replace_audio')
        c.button('Clear',disabled=st.session_state.recording_active,on_click=clear_audio,use_container_width=True,key='clear_audio')
        st.markdown('<div class="fine">5–10 s recommended · 15 s maximum · Local processing<br>Browser gain, noise suppression or echo cancellation may affect measurements. WAV upload offers better measurement control.</div>',unsafe_allow_html=True)
    return analyse


def warnings(report):
    for error in report.get('errors',[]): st.warning(error['message'])
    for warning in report.get('warnings',[]):
        if warning != CLOSE_LANGUAGE_SCORES_WARNING:
            st.caption(warning)


def language_results(report,reduced):
    language=report.get('language');region=language_region(language['top_label']) if language else None
    with st.container(key='result_language'):
        left,right=st.columns([1,1.15],gap='large')
        with left:
            st.markdown('<div class="eyebrow">LANGUAGE / '+escape(report['language_status'].upper())+'</div>',unsafe_allow_html=True)
            if language:
                if region and region['native']!=language['top_label']: st.markdown('<div class="native-language">'+escape(region['native'])+'</div>',unsafe_allow_html=True)
                st.markdown('<div class="language-name">'+escape(language['top_label'])+'</div>',unsafe_allow_html=True)
                st.caption(f'{language["inference_ms"]:.0f} ms · {language["device"]} · Pretrained Vaani-LID')
                refinement=language.get('upload_refinement')
                if refinement and refinement['applied']:
                    st.caption(f'Whisper supported {language["top_label"]}. Vaani ranked {refinement["vaani_first"]} first.')
                if language.get('upload_refinement_error'):
                    st.caption('Whisper second opinion unavailable. Showing the Vaani ranking.')
                st.markdown('<div class="result-rule"></div>',unsafe_allow_html=True)
                if refinement: st.caption('Original Vaani-LID ranking')
                for item in language['top5']:
                    score=max(0,min(1,float(item['score'])))
                    st.markdown(f'<div class="score-row"><span>{escape(item["label"])}</span><span>{score:.3f}</span></div><div class="score-track"><div class="score-fill" style="width:{score*100:.3f}%"></div></div>',unsafe_allow_html=True)
                if not region: st.caption('No curated region illustration is available for this label.')
            else:
                st.subheader('No language result.')
                st.caption('Language analysis may be skipped for insufficient signal or unavailable when the local model cannot run.')
        with right: components.html(scene('globe','result',reduced,region),height=460,scrolling=False)
    warnings(report)


def voice_results(report,reduced):
    acoustics=report.get('acoustics');distortion=report.get('distortion')
    if not acoustics:
        st.warning('Acoustic analysis is unavailable for this recording.');warnings(report);return
    tabs=st.tabs(['Signal analysis','Model comparison','Details'])
    with tabs[0]:
        components.html(waveform_player(st.session_state.recording.playback_bytes(),acoustics['plots'],reduced),height=215,scrolling=False)
        values=[('ESTIMATED SNR',acoustics['est_snr_db'],'dB'),('VOICED PITCH',acoustics['pitch_median_hz'],'Hz'),('RMS LEVEL',acoustics['rms_dbfs'],'dBFS'),('CLIPPING',acoustics['clipping_fraction'],'fraction'),('DURATION',acoustics['duration_s'],'s'),('SPEECH ACTIVITY',acoustics['speech_activity_ratio'],'estimate')]
        for col,(label,value,unit) in zip(st.columns(6),values):
            reading='—' if value is None else f'{value:.3f}' if unit in ('fraction','estimate') else f'{value:.1f}'
            col.markdown(f'<div class="reading"><div class="label">{label}</div><span class="value">{reading}</span><span class="unit">{unit}</span></div>',unsafe_allow_html=True)
        st.caption('Voiced pitch is the median estimated fundamental frequency. The pitch field is unavailable when fewer than ten voiced frames are detected. dBFS measures digital level.')
        spectral_mode=st.radio('Spectral view',['2D spectrogram','3D waterfall'],horizontal=True,key='spectral_view')
        st.plotly_chart(charts.spectrogram(acoustics['plots'],spectral_mode=='3D waterfall'),use_container_width=True,key='spectrogram',config={'displaylogo':False,'scrollZoom':True})
        st.caption('Actual input signal · time, frequency and window-normalized spectral power. The 3D view decimates measured bins for interactive performance; use 2D on lower-powered devices.')
    with tabs[1]:
        if distortion:
            st.markdown('<div class="eyebrow">MOST LIKELY MODELLED CONDITION</div>',unsafe_allow_html=True)
            st.subheader(distortion['predicted_class'].replace('_',' ').title() if distortion['predicted_class'] else 'Uncertain / abstained')
            st.plotly_chart(charts.class_scores(distortion['class_scores']),use_container_width=True)
            st.dataframe([{'Model':name.replace('_',' ').title(),'Predicted condition':value['top'].replace('_',' '),'Top score':max(value['scores'].values())} for name,value in distortion['base_predictions'].items()],hide_index=True,use_container_width=True)
            with st.expander('Equal-weight voting'):
                st.plotly_chart(charts.class_scores(distortion['voting_scores']),key='voting_scores',use_container_width=True)
        else: st.markdown('<div class="empty-panel"><h2>Recording condition unavailable.</h2><p>No evaluated production distortion model has been promoted. Acoustic measurements remain available. Pilot models are kept separate.</p></div>',unsafe_allow_html=True)
    with tabs[2]:
        st.markdown(acoustic_details(acoustics),unsafe_allow_html=True)
        for note in acoustics.get('warnings', ()):
            st.caption(note)
    warnings(report)


def lab(language=True):
    reduced=st.session_state.reduced_motion;report=st.session_state.report
    title='Language Lab' if language else 'Voice Lab'
    st.markdown('<div class="eyebrow">IDENTIFY THE LANGUAGE</div>' if language else '<div class="eyebrow">UNDERSTAND THE RECORDING</div>',unsafe_allow_html=True)
    st.title('A voice. A world of language.' if language else 'Listen beneath the surface.')
    st.markdown('<p class="lede">Upload a WAV file to analyse your speech. Your audio stays on this machine.</p>',unsafe_allow_html=True)
    slot=None
    if report is None:
        slot=st.empty()
        with slot.container(): components.html(scene('globe' if language else 'bars','idle',reduced),height=520 if language else 420,scrolling=False)
    with st.container(key='controlbar' if report is not None else 'input_language' if language else 'input_voice'):
        clicked=capture(compact=report is not None)
    if clicked:
        if slot is None: slot=st.empty()
        with slot.container(): components.html(scene('globe' if language else 'bars','processing',reduced),height=420,scrolling=False)
        with st.status('Analysing your recording locally…',expanded=False) as status:
            st.session_state.report=st.session_state.recording.analyze()
            status.update(label='Analysis complete',state='complete')
        st.rerun()
    report=st.session_state.report
    if report is not None:
        if language: language_results(report,reduced)
        else: voice_results(report,reduced)
    if language:
        with st.expander('Supported languages & model attribution'):
            path=engine().config.upstream_models/'Vaani-LID_v0'/VAANI_REVISION/'config.json'
            try:
                labels=list(json.loads(path.read_text(encoding='utf-8'))['id2label'].values())
                st.caption(f'{len(labels)} supported labels')
                st.html(language_grid(labels),width='stretch')
            except (OSError,ValueError,KeyError): st.caption('Language catalogue unavailable until the local model metadata is accessible.')
            st.markdown('[Vaani-LID_v0 · ARTPARK-IISc · MIT](https://huggingface.co/ARTPARK-IISc/Vaani-LID_v0) · Pretrained upstream model, not trained by this project. Pulikodan et al., VAANI. Code-switching is unsupported; unknown languages are not reliably rejected.')
            st.markdown('WAV uploads and microphone recordings can use [OpenAI Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) as a local second opinion for selected Hindi and Tamil confusions. The original Vaani-LID scores remain visible.')


def research():
    st.markdown('<div class="eyebrow">EVIDENCE OVER ASSUMPTION</div>',unsafe_allow_html=True);st.title('Research Lab')
    st.markdown('<p class="lede">A measured comparison of independent models, voting, stacking and transparent signal rules.</p>',unsafe_allow_html=True)
    evidence=st.radio('Evidence to view',['Production evaluation','Development pilot'],horizontal=True,key='research_evidence')
    if evidence=='Development pilot':
        st.warning('DEVELOPMENT PILOT ONLY · FLEURS + ESC-10 fallback · not a production evaluation or real-world reliability claim.')
        runs=sorted(p.parent.name for p in (ROOT/'results/pilot').glob('*/metrics.json'))
        if not runs: st.info('No saved pilot evaluation is available.');return
        run=st.selectbox('Saved pilot run',runs)
        data=read_pilot_research(run,ROOT/'models',ROOT/'results')
    else:
        data=read_production_research(ROOT/'models',ROOT/'results')
        st.caption('Language Lab can use a local Whisper large-v3-turbo second opinion for selected Hindi and Tamil results. The benchmark below measures distortion classification.')
    if data['status']!='available':
        st.markdown('<div class="empty-panel"><h2>Evaluation not available.</h2><p>'+escape(data['reason'])+'<br>Production training, held-out evaluation and artifact promotion are still pending. Choose Development pilot above to inspect the completed small experiment, clearly labelled separately.</p></div>',unsafe_allow_html=True)
        return
    metrics=data['metrics'];models=metrics['models'];st.caption(metrics['run_kind'].upper()+' RUN '+data['run_id'])
    if metrics['run_kind']=='pilot':
        st.caption('Test support: '+str(metrics['sample_id_parity']['original_count'])+' original recordings. Pilot findings do not establish production reliability. Stacking remains the primary method even when another model scores higher.')
    counts=metrics['sample_id_parity'];cols=st.columns(4)
    for col,label,value in zip(cols,['TEST ORIGINALS','TEST SAMPLES','CLASSES','METHODS'],[counts['original_count'],counts['sample_count'],len(next(iter(models.values()))['per_class']),len(models)]): col.metric(label,value)
    chosen=st.multiselect('Comparison metrics',['accuracy','macro_f1','macro_precision','macro_recall'],default=['macro_f1','accuracy'])
    if chosen: st.plotly_chart(charts.comparison(models,chosen),use_container_width=True,config={'displaylogo':False})
    selected=st.selectbox('Inspect model',list(models),format_func=lambda x:x.replace('_',' ').title())
    a,b=st.columns(2)
    with a: st.subheader('Confusion matrix');st.plotly_chart(charts.confusion(models[selected]['confusion_matrix_counts'],list(models[selected]['per_class'])),use_container_width=True)
    with b:
        st.subheader('Class support');st.bar_chart({name:v['support'] for name,v in models[selected]['per_class'].items()})
        st.dataframe([{'Condition':name.replace('_',' '),**values}
                      for name,values in models[selected]['per_class'].items()],hide_index=True)
    st.subheader('Measured inference time')
    st.dataframe([{'Model':n.replace('_',' ').title(),
                   'Classifier median (ms)':v['latency_ms']['classifier_median'],
                   'Classifier p95 (ms)':v['latency_ms']['classifier_p95'],
                   'Features + classifier median (ms)':v['latency_ms'].get('feature_classify_median'),
                   'Features + classifier p95 (ms)':v['latency_ms'].get('feature_classify_p95'),
                   'Training (s)':v.get('train_seconds')} for n,v in models.items()],hide_index=True,use_container_width=True)
    with st.expander('Run settings and dataset provenance'):
        for label,key in [('Evaluation date','created_utc'),('Seed','seed'),('Feature schema','feature_schema_version')]:
            if key in metrics: st.text(f'{label} · {metrics[key]}')
        st.json(metrics['dataset'],expanded=False)
        st.json(metrics['provenance'],expanded=False)
        st.json(metrics['timing_protocol'],expanded=False)
        if data.get('settings'):
            st.json(data['settings'],expanded=False)
    sensitivity=metrics.get('robustness',{}).get('unseen_severity')
    if sensitivity:
        with st.expander('Unseen noise severity probe'):
            if sensitivity['status']=='measured':
                st.write(f"{sensitivity['correct_added_noise']} of {sensitivity['support']} recordings were classified as added noise at {sensitivity['target_snr_db']:g} dB injected SNR.")
                st.dataframe(sensitivity['predictions'],hide_index=True)
            else:
                st.caption(sensitivity['reason'])
    for limitation in metrics['limitations']: st.caption('• '+limitation)


def main():
    initialize();st.markdown('<style>'+ (ROOT/'frontend/ui/theme.css').read_text(encoding='utf-8')+'</style>',unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-orbit"></span>VOXLINGUA</div><div class="eyebrow">A CLOSER LISTEN</div>',unsafe_allow_html=True)
        st.markdown('<div style="height:42px"></div>',unsafe_allow_html=True)
        for i,page in enumerate(PAGES): st.button(page,key='nav_'+str(i),type='primary' if st.session_state.page==page else 'secondary',use_container_width=True,disabled=st.session_state.recording_active,on_click=navigate,args=(page,))
        st.markdown('<div style="height:80px"></div>',unsafe_allow_html=True)
        st.toggle('Reduce motion',key='reduced_motion',value=False)
    if st.session_state.reduced_motion: st.markdown('<style>*,*:before,*:after{animation:none!important;transition:none!important}</style>',unsafe_allow_html=True)
    st.markdown('<div class="page-top"><span>VOXLINGUA / VOICE INTELLIGENCE</span><span class="status-dot">● &nbsp; ON DEVICE</span></div>',unsafe_allow_html=True)
    if st.session_state.page=='Home':
        components.html(scene('orbit',reduced=st.session_state.reduced_motion),height=570,scrolling=False)
        with st.container(key='home_action'):
            _,col,_=st.columns([1,1,1]);col.button('Start analysis  ↗',type='primary',use_container_width=True,on_click=navigate,args=('Language Lab',))
        for col,title,target,body in zip(st.columns(3,gap='large'),['Language','Acoustics','Intelligence'],['Language Lab','Voice Lab','Research Lab'],['Find the language within a voice.','See the measurable shape of sound.','Explore how independent models compare.']):
            with col:
                with st.container(key='branch_'+title.lower()):
                    st.button(title,key='home_'+title.lower(),on_click=navigate,args=(target,),use_container_width=True)
                    st.markdown(f'<p class="branch-description">{body}</p>',unsafe_allow_html=True)
    elif st.session_state.page=='Language Lab': lab(True)
    elif st.session_state.page=='Voice Lab': lab(False)
    elif st.session_state.page=='Research Lab': research()
    else: methodology.render(st.session_state.reduced_motion)


if __name__=='__main__': main()
