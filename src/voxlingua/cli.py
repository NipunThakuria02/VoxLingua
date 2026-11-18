"""Local command-line core; interface design is deliberately deferred."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from .config import RuntimeConfig
from .runtime import AnalysisEngine


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audio',type=Path)
    p.add_argument('--config',type=Path,help='Runtime YAML only; paths resolve relative to the YAML file')
    p.add_argument('--models-dir',type=Path,help='Trusted production models directory; pilot artifacts remain forbidden')
    p.add_argument('--language',action='store_true',help='Load verified local Vaani snapshots only (no downloads)')
    p.add_argument('--upstream-models',type=Path)
    p.add_argument('--plots',action='store_true',help='Include waveform/spectrogram data in stdout')
    p.add_argument('--out',type=Path,help='Explicitly save report JSON; otherwise stdout only')
    args=p.parse_args(argv)
    config=RuntimeConfig.from_yaml(args.config) if args.config else RuntimeConfig()
    if args.language: config=replace(config,language_enabled=True)
    if args.models_dir: config=replace(config,models_dir=args.models_dir)
    if args.upstream_models: config=replace(config,upstream_models=args.upstream_models)
    engine=AnalysisEngine(config)
    try:
        result=engine.analyze(args.audio.read_bytes(),args.audio.name)
        if not args.plots and result.get('acoustics'): result['acoustics'].pop('plots',None)
        text=json.dumps(result,indent=2,allow_nan=False)
        if args.out:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            with args.out.open('x',encoding='utf-8') as f: f.write(text)
            print(f'Report saved: {args.out}')
        else: print(text)
        return 1 if result['status']=='error' else 0
    finally:
        engine.unload()


if __name__=='__main__': raise SystemExit(main())
