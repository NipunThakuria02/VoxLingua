"""Explicit offline development smoke of all three branches using a sealed pilot."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
from voxlingua.analysis import analyze_recording
from voxlingua.distortion import DistortionService
from voxlingua.language import LanguageService


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-run-id',required=True)
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--upstream-models',type=Path,default=Path('models/upstream'))
    parser.add_argument('--out',type=Path,default=Path('results/three-branch-pilot-smoke.json'))
    args=parser.parse_args(argv)
    if args.pilot_run_id in ('.','..') or any(c in args.pilot_run_id for c in '/\\:'): raise ValueError('Invalid pilot ID')
    if args.out.exists(): raise ValueError('Choose a new report path; smoke evidence is not overwritten')
    model=DistortionService.load_run(Path('models/pilot')/args.pilot_run_id,allow_pilot=True)
    language=LanguageService(args.upstream_models)
    try:
        # Enforce actual offline inference, including custom model loading, not just HF flags.
        with patch('socket.socket.connect',side_effect=RuntimeError('Runtime network connection forbidden')):
            report=analyze_recording(args.audio.read_bytes(),args.audio.name,language_service=language,distortion_service=model)
        report['run_kind']='pilot'
        report['production_promotion']=False
        report['runtime_network_blocked']=True
        report['evaluation_scope']='One functional integration recording; no accuracy or generalization claim'
        if report['acoustics']: report['acoustics'].pop('plots',None)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2,allow_nan=False)
        print(json.dumps({'report':str(args.out),'status':report['status'],'branches':{n:report[n+'_status'] for n in ['language','distortion','acoustics']}}))
        return 0 if report['status']=='success' else 1
    finally: language.unload()


if __name__=='__main__': raise SystemExit(main())
