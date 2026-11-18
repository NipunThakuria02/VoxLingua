"""Fixed class order and audio contracts."""
from dataclasses import dataclass
import numpy as np

CLASSES = ("reference", "added_noise", "clipping", "low_volume", "frequency_filtering", "interruption")


class ServiceError(ValueError):
    """Typed, user-displayable service failure."""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class InputLimits:
    max_seconds: float = 15.0
    min_seconds: float = 1.0
    max_bytes: int = 20 * 1024 * 1024

    def __post_init__(self):
        if not 0 < self.min_seconds <= self.max_seconds <= 15 or self.max_bytes <= 0:
            raise ValueError("Limits must satisfy 0 < minimum <= maximum <= 15 seconds and positive bytes")


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray
    sample_rate: int
    n_channels: int
    duration_s: float
    source_format: str
    sha256: str
