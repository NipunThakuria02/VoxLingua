"""Validated runtime configuration; offline training keeps its separate contract."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from .schemas import InputLimits


@dataclass(frozen=True)
class RuntimeConfig:
    audio: InputLimits = field(default_factory=InputLimits)
    models_dir: Path = Path('models')
    upstream_models: Path = Path('models/upstream')
    language_enabled: bool = False

    @classmethod
    def from_yaml(cls, path: Path | str):
        """Read only explicit runtime settings; reject unsupported runtime options."""
        path=Path(path)
        value=yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(value,dict): raise ValueError('Runtime configuration must be a mapping')
        if set(value)-{'audio','models_dir','upstream_models','language_enabled'}:
            raise ValueError('Unknown runtime setting; use configs/runtime.yaml, not training defaults')
        if 'language_enabled' in value and not isinstance(value['language_enabled'],bool):
            raise ValueError('language_enabled must be true or false')
        audio=value.get('audio',{})
        if not isinstance(audio,dict): raise ValueError('audio settings must be a mapping')
        for key in ('max_seconds','min_seconds','max_bytes'):
            if key in audio and (isinstance(audio[key],bool) or not isinstance(audio[key],(float,int))):
                raise ValueError('Audio limits must be numeric')
        if 'max_bytes' in audio and not isinstance(audio['max_bytes'],int): raise ValueError('max_bytes must be an integer')
        def resolve(key,default):
            raw=value.get(key,default)
            if not isinstance(raw,str) or not raw: raise ValueError(key+' must be a nonempty local path')
            target=Path(raw)
            return target if target.is_absolute() else (path.parent/target).resolve()
        return cls(audio=InputLimits(**audio),models_dir=resolve('models_dir','../models'),
                   upstream_models=resolve('upstream_models','models/upstream'),
                   language_enabled=value.get('language_enabled',False))
