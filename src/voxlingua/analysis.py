"""Three independent service branches; partial results survive individual failures."""
from __future__ import annotations
import time
from .audio import decode_audio, make_feature_copy, make_language_copy
from .acoustics import analyze_acoustics
from .features import extract_features
from .schemas import InputLimits, ServiceError


CLOSE_LANGUAGE_SCORES_WARNING = 'The two highest language scores are close. Treat the ranking as tentative. This is a heuristic based on the score gap.'


def analyze_recording(data: bytes, filename: str | None = None, *, language_service=None, distortion_service=None, limits: InputLimits | None = None, refine_upload: bool = False) -> dict:
    """Analyze one recording locally, cheap branches first; no network or retention."""
    result={'status':'error','audio_info':{},'acoustics':None,'distortion':None,'language':None,
            'acoustics_status':'skipped','distortion_status':'skipped','language_status':'skipped','errors':[],'warnings':[],'timings_ms':{}}
    def failure(branch, exc):
        code=getattr(exc,'code',f'E_{branch.upper()}_FAILED')
        result[f'{branch}_status']='unavailable' if code in ('E_LANG_UNAVAILABLE','E_MODEL_ARTIFACT_MISSING','E_MODEL_SCHEMA_MISMATCH','E_MODEL_INTEGRITY_FAILED','E_MODEL_CLASS_ORDER_MISMATCH') else 'error'
        result['errors'].append({'branch':branch,'code':code,'message':str(exc)})
    start=time.perf_counter()
    try: audio=decode_audio(data,filename,limits)
    except ServiceError as exc:
        result['errors'].append({'code':exc.code,'message':str(exc)})
        return result
    result['timings_ms']['decode']=(time.perf_counter()-start)*1000
    result['audio_info']={'sample_rate':audio.sample_rate,'n_channels':audio.n_channels,'duration_s':audio.duration_s,'source_format':audio.source_format}
    start=time.perf_counter()
    try:
        result['acoustics']=analyze_acoustics(audio)
        result['acoustics_status']='ok'
    except Exception as exc: failure('acoustics',exc)
    result['timings_ms']['acoustics']=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    try:
        if distortion_service is None:
            raise ServiceError('E_MODEL_ARTIFACT_MISSING','No trained production distortion artifacts are available.')
        features=extract_features(make_feature_copy(audio))
        result['timings_ms']['features']=(time.perf_counter()-start)*1000
        infer_start=time.perf_counter()
        result['distortion']=distortion_service.predict(features)
        result['timings_ms']['classify']=(time.perf_counter()-infer_start)*1000
        result['distortion_status']='ok'
    except Exception as exc: failure('distortion',exc)
    result['timings_ms']['distortion']=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    try:
        # Signal level is a conservative gate, not a reliable speech/OOD detector.
        import numpy as np
        mono=make_language_copy(audio)
        if np.sqrt(np.mean(mono.astype(np.float64)**2))<1e-5:
            result['language_status']='skipped'
            result['warnings'].append('Language analysis skipped: no speech or insufficient signal; this level gate is heuristic.')
        elif language_service is None:
            raise ServiceError('E_LANG_UNAVAILABLE','Language analysis was not enabled or local weights are unavailable.')
        else:
            language=language_service.identify(mono, refine_upload=True) if refine_upload else language_service.identify(mono)
            result['language']=language
            result['language_status']='ok'
            if len(mono)<3*16000:
                result['warnings'].append('Short input can make the language ranking unreliable. Try 5–10 seconds of clear speech.')
            top=language.get('top5',[])
            if len(top)>=2 and top[0]['score']-top[1]['score']<0.05:
                result['warnings'].append(CLOSE_LANGUAGE_SCORES_WARNING)
    except Exception as exc: failure('language',exc)
    result['timings_ms']['language']=(time.perf_counter()-start)*1000
    states=[result[k+'_status'] for k in ('acoustics','distortion','language')]
    result['status']='success' if all(s=='ok' for s in states) else 'partial' if 'ok' in states else 'error'
    return result
