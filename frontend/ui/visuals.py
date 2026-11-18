"""Locally served scenes; decorative motion is separate from analytical data."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def scene(kind,state='idle',reduced=False,region=None):
    options=json.dumps({'kind':kind,'state':state,'reduced':reduced,'region':region},ensure_ascii=True).replace('<','\\u003c')
    template=(ROOT/'scene.html').read_text(encoding='utf-8')
    copy=''
    if kind=='orbit':
        copy='<div class="hero-copy"><small>VOXLINGUA / LOCAL VOICE INTELLIGENCE</small><h1>A NEW PERSPECTIVE<br>ON SOUND</h1><p>Every voice tells more.<br>Discover the language, quality and hidden characteristics of speech.</p></div>'
    captions={'orbit':'THREE INDEPENDENT PERSPECTIVES. ONE RECORDING.','globe':'LANGUAGE REGIONS · NOT SPEAKER LOCATION','bars':'ILLUSTRATIVE ACTIVITY · NOT A MEASUREMENT'}
    return template.replace('__OPTIONS__',options).replace('__LAND__',(ROOT/'assets/land-points.json').read_text() if kind=='globe' else '[]').replace('__COPY__',copy).replace('__CAPTION__',captions[kind]).replace('__HINT__','PROCESSING · NO ESTIMATED PERCENTAGE' if state=='processing' else '').replace('__ARIA__',{'orbit':'White orbital particles with an empty centre','globe':'Rotating digital globe illustrating language regions','bars':'Illustrative circular activity bars'}[kind])
