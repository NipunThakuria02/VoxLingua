"""Original-waveform playback with a cursor driven by the actual audio element."""
import base64
import json


def waveform_player(data: bytes, plots: dict, reduced=False):
    payload=json.dumps({'x':plots['waveform_time_s'],'y':plots['waveform_amplitude'],'reduced':reduced},allow_nan=False).replace('<','\\u003c')
    audio=base64.b64encode(data).decode('ascii')
    return '''<!doctype html><html><head><style>
    @font-face{font-family:Montserrat;src:url('/app/static/fonts/Montserrat.ttf')}
    body{margin:0;background:#090909;color:#969696;font:10px Montserrat,sans-serif}canvas{width:100%;height:135px;display:block;cursor:crosshair}audio{width:100%;height:36px;margin-top:12px}small{display:block;margin-top:8px;font-size:9px;letter-spacing:.06em}
    </style></head><body><canvas id="wave" tabindex="0" role="slider" aria-label="Waveform playback position; use left and right arrows to seek" aria-valuemin="0" aria-valuenow="0"></canvas>
    <audio id="audio" controls preload="metadata" src="data:audio/wav;base64,'''+audio+'''"></audio><small>ORIGINAL WAVEFORM · FULL-SCALE AMPLITUDE · TIME IN SECONDS</small><script>
    const data='''+payload+''',audio=document.getElementById('audio'),canvas=document.getElementById('wave'),ctx=canvas.getContext('2d');let timer=0;
    function draw(){const W=innerWidth,H=135,dpr=Math.min(devicePixelRatio||1,2);if(canvas.width!==Math.round(W*dpr)){canvas.width=W*dpr;canvas.height=H*dpr}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);const maximum=Math.max(1,...data.y.map(Math.abs));const scale=(H/2-18)/maximum;
    ctx.strokeStyle='#303030';ctx.lineWidth=.6;ctx.beginPath();ctx.moveTo(0,H/2);ctx.lineTo(W,H/2);ctx.stroke();ctx.strokeStyle='#b9c7b3';ctx.lineWidth=1;ctx.beginPath();for(let i=0;i<data.y.length;i++){const x=i/Math.max(1,data.y.length-1)*W,y=H/2-data.y[i]*scale;i?ctx.lineTo(x,y):ctx.moveTo(x,y)}ctx.stroke();const duration=Number.isFinite(audio.duration)?audio.duration:data.x[data.x.length-1];const x=duration?audio.currentTime/duration*W:0;ctx.strokeStyle='#ededeb';ctx.beginPath();ctx.moveTo(x,8);ctx.lineTo(x,H-14);ctx.stroke();ctx.fillStyle='#a0a0a0';ctx.font='9px Montserrat';ctx.fillText(audio.currentTime.toFixed(2)+'s',Math.min(W-50,x+6),12);ctx.fillText('-'+maximum.toFixed(1),2,H-2);ctx.fillText('+'+maximum.toFixed(1),2,11);canvas.setAttribute('aria-valuenow',audio.currentTime.toFixed(2));canvas.setAttribute('aria-valuemax',String(duration||0))}
    canvas.addEventListener('click',e=>{if(Number.isFinite(audio.duration)){audio.currentTime=Math.max(0,Math.min(audio.duration,e.offsetX/innerWidth*audio.duration));draw()}});
    canvas.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)&&Number.isFinite(audio.duration)){e.preventDefault();audio.currentTime=Math.max(0,Math.min(audio.duration,audio.currentTime+(e.key==='ArrowRight'?.5:-.5)));draw()}});
    audio.addEventListener('timeupdate',draw);audio.addEventListener('loadedmetadata',draw);addEventListener('resize',draw);draw();addEventListener('pagehide',()=>audio.pause());
    </script></body></html>'''
