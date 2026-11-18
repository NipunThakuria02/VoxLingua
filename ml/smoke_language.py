"""Run a real, offline Vaani smoke with verified snapshots; never trains a model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--models',type=Path,default=Path('models/upstream'))
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--out',type=Path,default=Path('results/vaani-smoke.json'))
    a=p.parse_args(argv)
    from voxlingua.audio import decode_audio, make_language_copy
    from voxlingua.language import LanguageService
    service=LanguageService(a.models,prefer_gpu=not a.cpu)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'FAILED','audio':str(a.audio),'device_requested':'cpu' if a.cpu else 'cuda-fp16','code_review':'docs/VAANI_CODE_REVIEW.md'}
    start=time.perf_counter()
    try:
        import torch
        x=make_language_copy(decode_audio(a.audio.read_bytes()))
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        first=service.identify(x)
        second=service.identify(x)
        report.update(status='PASSED',cold_result=first,warm_result=second,labels=service.labels,
                      peak_vram_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                      note='One known-language clip demonstrates inference, not accuracy/generalization.')
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error_code=getattr(exc,'code',None),error=str(exc))
    finally:
        service.unload()
        report['total_seconds']=time.perf_counter()-start
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists(): a.out=a.out.with_name(a.out.stem+f'-{time.time_ns()}.json')
    a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'report':str(a.out),'status':report['status'],'error':report.get('error')}))
    return 0 if report['status']=='PASSED' else 1


if __name__=='__main__': raise SystemExit(main())
