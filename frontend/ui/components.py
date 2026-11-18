"""Local Streamlit components and input adapters; never transmits audio externally."""
import base64
from pathlib import Path
import streamlit.components.v1 as components
from voxlingua.schemas import ServiceError

ROOT=Path(__file__).resolve().parent
microphone=components.declare_component('voxlingua_local_microphone',path=str(ROOT/'recorder'))


def microphone_audio(value: dict, maximum_bytes: int) -> bytes | None:
    if not isinstance(value,dict) or value.get('state')!='ready': return None
    encoded=value.get('audio_base64')
    if not isinstance(encoded,str): raise ServiceError('E_AUDIO_CORRUPT','Recorder returned invalid audio.')
    if len(encoded)>4*((maximum_bytes+2)//3): raise ServiceError('E_AUDIO_TOO_LARGE','Recording exceeds the allowed size.')
    try: data=base64.b64decode(encoded,validate=True)
    except (ValueError,base64.binascii.Error) as exc: raise ServiceError('E_AUDIO_CORRUPT','Recorder returned invalid WAV data.') from exc
    if len(data)>maximum_bytes: raise ServiceError('E_AUDIO_TOO_LARGE','Recording exceeds the allowed size.')
    return data
