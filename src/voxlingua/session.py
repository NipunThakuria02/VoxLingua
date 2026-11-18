"""In-memory recording shared by future Language Lab and Voice Lab views."""
from copy import deepcopy
import numpy as np
from .audio import decode_audio
from .schemas import ServiceError


class RecordingSession:
    """Per-user state. No files, network, global cache or cross-session audio sharing."""
    def __init__(self,engine):
        self.engine=engine
        self._data=None
        self._filename=None
        self._identity=None
        self._audio_info=None
        self._result=None

    @property
    def audio_info(self): return deepcopy(self._audio_info)

    @property
    def has_recording(self): return self._data is not None

    def is_new_recording(self, data: bytes, *, microphone: bool = False) -> bool:
        return data != self._data or microphone != self._microphone

    def set_recording(self,data: bytes,filename: str | None = None, *, microphone: bool = False):
        """Validate before replacing; repeat bytes preserve the completed analysis cache."""
        audio=decode_audio(data,filename,self.engine.config.audio)
        if microphone and np.ptp(audio.samples, axis=0).max() < 0.002:
            raise ServiceError(
                'E_MICROPHONE_NO_SIGNAL',
                'The microphone recording has no usable signal. Check the selected input and mute switch.',
            )
        if audio.sha256!=self._identity or microphone!=self._microphone:
            self.clear()
            self._data=bytes(data)
            self._identity=audio.sha256
            self._microphone=microphone
            self._audio_info={'duration_s':audio.duration_s,'sample_rate':audio.sample_rate,'n_channels':audio.n_channels,'source_format':audio.source_format}
        self._filename=filename
        return self.audio_info

    def playback_bytes(self):
        """Bytes for the active user's local playback widget; never logged or written."""
        return self._data

    def analyze(self):
        if self._data is None: raise ServiceError('E_AUDIO_EMPTY','Record or upload audio first.')
        if self._result is None:
            self._result=self.engine.analyze(self._data,self._filename)
        return deepcopy(self._result)

    def retry_analysis(self):
        """Explicit retry, never an automatic page-navigation action."""
        self._result=None
        return self.analyze()

    def clear(self):
        """Release this session's references to input audio, plots and analysis results."""
        self._data=self._filename=self._identity=self._audio_info=self._result=None
        self._microphone=False
