"""Native-amplitude WAV decoding and explicit mono/model copies."""
from hashlib import sha256
from io import BytesIO
from math import gcd
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
from .schemas import DecodedAudio, InputLimits, ServiceError


def quantize_int16(x: np.ndarray) -> np.ndarray:
    """Round ties to even, saturate to signed PCM16, return float32 full-scale units."""
    return (np.clip(np.rint(np.asarray(x, dtype=np.float64) * 32768), -32768, 32767) / 32768).astype(np.float32)


def decode_audio(data: bytes, filename: str | None = None, limits: InputLimits | None = None) -> DecodedAudio:
    """Decode WAV to read-only native-rate float32 [samples, channels], never normalize."""
    limits = limits or InputLimits()
    if not data:
        raise ServiceError("E_AUDIO_EMPTY", "The recording is empty.")
    if len(data) > limits.max_bytes:
        raise ServiceError("E_AUDIO_TOO_LARGE", "Audio exceeds the configured upload size.")
    try:
        with sf.SoundFile(BytesIO(data)) as f:
            if f.format not in ("WAV", "WAVEX", "RF64"):
                raise ServiceError("E_AUDIO_UNSUPPORTED_FORMAT", "This core build supports WAV only.")
            sr, channels, frames, fmt = f.samplerate, f.channels, f.frames, f.format
            if not 8000 <= sr <= 192000 or not 1 <= channels <= 8:
                raise ServiceError("E_AUDIO_UNSUPPORTED_FORMAT", "Supported WAV rates: 8–192 kHz; channels: 1–8.")
            duration = frames / sr
            if duration < limits.min_seconds:
                raise ServiceError("E_AUDIO_TOO_SHORT", "Provide at least one second of audio.")
            if duration > limits.max_seconds:
                raise ServiceError("E_AUDIO_TOO_LONG", "Maximum recording or upload duration is 15 seconds.")
            samples = f.read(dtype="float32", always_2d=True)
            if len(samples) != frames:
                raise ServiceError("E_AUDIO_CORRUPT", "Audio data is truncated.")
    except ServiceError:
        raise
    except (RuntimeError, ValueError, sf.LibsndfileError) as exc:
        raise ServiceError("E_AUDIO_CORRUPT", "The audio cannot be decoded as WAV.") from exc
    if not np.isfinite(samples).all():
        raise ServiceError("E_AUDIO_NONFINITE", "Audio contains invalid sample values.")
    samples.setflags(write=False)
    return DecodedAudio(samples, sr, channels, duration, fmt.lower(), sha256(data).hexdigest())


def make_language_copy(a: DecodedAudio) -> np.ndarray:
    """Mean downmix then explicit polyphase resampling, with no gain compensation."""
    x = a.samples.mean(axis=1, dtype=np.float64)
    if a.sample_rate != 16000:
        g = gcd(a.sample_rate, 16000)
        x = resample_poly(x, 16000 // g, a.sample_rate // g, window=("kaiser", 5.0), padtype="constant")
    return np.asarray(x, dtype=np.float32)


def make_feature_copy(a: DecodedAudio, max_seconds: float = 15.0) -> np.ndarray:
    """Produce full accepted recording, mono/16kHz/PCM16-grid; never truncate silently."""
    if a.duration_s > min(15, max_seconds):
        raise ServiceError("E_AUDIO_TOO_LONG", "Feature copy cannot silently truncate audio.")
    return quantize_int16(make_language_copy(a))
