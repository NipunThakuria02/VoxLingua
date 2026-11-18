"""Application-facing engine: lazy local services, production artifacts only."""
from .analysis import analyze_recording
from .config import RuntimeConfig
from .schemas import ServiceError


class _UnavailableDistortion:
    def __init__(self,error): self.error=error
    def predict(self,features): raise self.error


class AnalysisEngine:
    """One engine per app process; sessions retain their own recordings and results."""
    def __init__(self,config: RuntimeConfig):
        self.config=config
        self._distortion=None
        self._language=None

    def _services(self):
        if self._distortion is None:
            from .distortion import DistortionService
            try: self._distortion=DistortionService.load(self.config.models_dir)
            except ServiceError as exc: self._distortion=_UnavailableDistortion(exc)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                self._distortion=_UnavailableDistortion(ServiceError('E_MODEL_ARTIFACT_MISSING','Production artifact metadata is invalid: '+str(exc)))
        if self.config.language_enabled and self._language is None:
            from .language import LanguageService
            self._language=LanguageService(self.config.upstream_models)

    def analyze(self,data: bytes,filename: str | None = None, *, refine_upload: bool = True) -> dict:
        self._services()
        result=analyze_recording(data,filename,distortion_service=self._distortion,language_service=self._language,limits=self.config.audio,refine_upload=refine_upload)
        result['model_context']={'mode':'production','pilot_models_allowed':False,
                                 'run_id':result['distortion']['run_id'] if result['distortion'] else None}
        return result

    def unload(self):
        if self._language is not None: self._language.unload()
        self._language=self._distortion=None
