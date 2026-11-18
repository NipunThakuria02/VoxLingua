"""The single fs-v1 extractor: 84 finite, ordered, amplitude-sensitive features."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from scipy.signal import windows
from scipy.stats import kurtosis
from .schemas import ServiceError

FEATURE_SCHEMA_VERSION = "fs-v1"
FEATURE_NAMES = (
    "rms_dbfs", "peak_dbfs", "abs_p99_dbfs", "crest_factor_db", "clip_frac_ge_0p99", "clip_frac_ge_0p999", "flat_top_run_frac", "near_peak_density", "amp_kurtosis",
    "frame_rms_dbfs_mean", "frame_rms_dbfs_std", "frame_rms_dbfs_p10", "frame_rms_dbfs_p50", "frame_rms_dbfs_p90", "dynamic_range_db",
) + tuple(f"{stem}_{stat}" for stem in ("centroid_hz", "bandwidth_hz", "rolloff85_hz", "flatness", "zcr") for stat in ("mean", "std", "p10", "p90")) + tuple(f"mfcc_{i:02d}_{stat}" for stat in ("mean", "std") for i in range(13)) + (
    "band_db_0_300", "band_db_300_1000", "band_db_1000_2000", "band_db_2000_3000", "band_db_3000_4000", "band_db_4000_6000", "band_db_6000_8000", "spectral_cutoff_99_hz", "hf_lf_ratio_db", "spectral_slope_db_per_khz",
    "zero_run_frac", "longest_zero_run_ms", "low_energy_frac_abs", "low_energy_frac_rel", "n_short_gaps_rel", "max_gap_ms_rel", "mean_gap_ms_rel", "max_drop_db", "max_rise_db", "n_drops_gt20db", "n_rises_gt20db", "noise_floor_flatness", "noise_floor_centroid_hz",
)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open maximal true runs of a 1-D mask."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def frames(x: np.ndarray, length: int, hop: int) -> np.ndarray:
    """Uncentered/unpadded frames [time, sample], a read-only view."""
    return np.lib.stride_tricks.sliding_window_view(x, length)[::hop]


def db_amp(value):
    return 20 * np.log10(np.maximum(value, 1e-6))


def spectral(fr: np.ndarray, sr: int, n_fft: int) -> tuple[np.ndarray, dict]:
    """One-sided, window-energy-normalized power [frame,bin] and frame descriptors."""
    w = windows.hann(fr.shape[1], sym=False)
    raw = np.abs(np.fft.rfft(fr * w, n=n_fft, axis=1)) ** 2
    s = raw / (n_fft * np.sum(w ** 2))
    s[:, 1:-1] *= 2
    freq = np.fft.rfftfreq(n_fft, 1 / sr)
    total = s.sum(axis=1)
    silent = total < 1e-12
    denom = np.maximum(total, 1e-300)
    centroid = (s * freq).sum(axis=1) / denom
    bandwidth = np.sqrt((s * (freq - centroid[:, None]) ** 2).sum(axis=1) / denom)
    rolloff = freq[np.argmax(np.cumsum(s, axis=1) >= total[:, None] * 0.85, axis=1)]
    flat = np.exp(np.log(s + 1e-12).mean(axis=1)) / (s + 1e-12).mean(axis=1)
    centroid[silent] = bandwidth[silent] = rolloff[silent] = 0
    flat[silent] = 1
    zcr = np.count_nonzero(np.diff(np.signbit(fr), axis=1), axis=1) / (fr.shape[1] - 1)
    return s, {"centroid_hz": centroid, "bandwidth_hz": bandwidth, "rolloff85_hz": rolloff, "flatness": flat, "zcr": zcr}


def extract_features(x16k: np.ndarray) -> np.ndarray:
    """Input mono/16kHz/PCM16-grid, 1–15s; output finite float32[84]. No metadata accepted."""
    x = np.asarray(x16k, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ServiceError("E_AUDIO_NONFINITE", "Feature input must be a finite mono array.")
    if len(x) < 16000:
        raise ServiceError("E_AUDIO_TOO_SHORT", "Features require at least one second.")
    if len(x) > 15 * 16000:
        raise ServiceError("E_AUDIO_TOO_LONG", "Features accept at most 15 seconds.")
    if np.any(x < -1) or np.any(x > 32767 / 32768) or not np.array_equal(x * 32768, np.rint(x * 32768)):
        raise ValueError("Feature input must use the signed PCM16 grid; prepare it with make_feature_copy")
    fr = frames(x, 400, 160)
    levels = db_amp(np.sqrt(np.mean(fr ** 2, axis=1)))
    peak, rms = np.max(np.abs(x)), np.sqrt(np.mean(x ** 2))
    occupancy = np.zeros(len(x), dtype=bool)
    if peak >= 0.2:
        for sign in (1, -1):
            for a, b in runs((x * sign) >= 0.98 * peak):
                if b - a >= 3:
                    occupancy[a:b] = True
    vals = [db_amp(rms), db_amp(peak), db_amp(np.quantile(np.abs(x), 0.99)), db_amp(peak) - db_amp(rms), np.mean(np.abs(x) >= 0.99), np.mean(np.abs(x) >= 0.999), occupancy.mean(), np.mean(np.abs(x) >= 0.95 * peak) if peak else 0, float(kurtosis(x, fisher=True, bias=True)) if np.var(x) > 0 else 0]
    vals += [levels.mean(), levels.std(), *np.quantile(levels, [0.1, 0.5, 0.9]), np.quantile(levels, 0.95) - np.quantile(levels, 0.05)]
    s, descriptors = spectral(fr, 16000, 512)
    for arr in descriptors.values():
        vals += [arr.mean(), arr.std(), *np.quantile(arr, [0.1, 0.9])]
    import librosa
    w = windows.hann(400, sym=False)
    raw = np.abs(np.fft.rfft(fr * w, n=512, axis=1)) ** 2
    mel = librosa.filters.mel(sr=16000, n_fft=512, n_mels=40, fmin=20, fmax=8000, htk=False, norm="slaney", dtype=np.float32) @ raw.T
    logmel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=13, dct_type=2, norm="ortho", lifter=0)
    vals += mfcc.mean(axis=1).tolist() + mfcc.std(axis=1).tolist()
    f = np.fft.rfftfreq(512, 1 / 16000)
    mean_s = s.mean(axis=0)
    edges = [0, 300, 1000, 2000, 3000, 4000, 6000, 8000]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (f >= lo) & ((f <= hi) if hi == 8000 else (f < hi))
        vals.append(10 * np.log10(mean_s[mask].sum() + 1e-12))
    cutoff = f[np.searchsorted(np.cumsum(mean_s), mean_s.sum() * 0.99)] if mean_s.sum() else 0
    ratio = 10 * np.log10((mean_s[(f >= 4000) & (f <= 8000)].sum() + 1e-12) / (mean_s[(f >= 0) & (f < 1000)].sum() + 1e-12))
    use = (f >= 100) & (f <= 8000)
    slope = np.polyfit(f[use] / 1000, 10 * np.log10(mean_s[use] + 1e-12), 1)[0]
    vals += [cutoff, ratio, slope]
    zero_runs = [(a, b) for a, b in runs(x == 0) if a > 0 and b < len(x) and b - a >= 16]
    gaps = [(b - a) * 10 for a, b in runs(levels < np.quantile(levels, 0.95) - 35) if a > 0 and b < len(levels) and 2 <= b - a <= 30]
    delta = np.diff(levels)
    vals += [sum(b - a for a, b in zero_runs) / len(x), max((b - a for a, b in zero_runs), default=0) / 16,
             np.mean(levels < -70), np.mean(levels < np.quantile(levels, 0.95) - 35), len(gaps), max(gaps, default=0), np.mean(gaps) if gaps else 0,
             max(0, -delta.min()), max(0, delta.max()), np.sum(delta < -20), np.sum(delta > 20)]
    low = np.argsort(levels, kind="stable")[:max(1, int(np.floor(0.2 * len(levels))))]
    vals += [descriptors["flatness"][low].mean(), descriptors["centroid_hz"][low].mean()]
    output = np.asarray(vals, dtype=np.float32)
    if output.shape != (84,) or not np.isfinite(output).all():
        raise RuntimeError("Feature contract violated")
    return output


def feature_schema() -> dict:
    """Serialize exact names/parameters/implementation identity for artifact validation."""
    parameters = {"sample_rate": 16000, "frame": 400, "hop": 160, "n_fft": 512, "window": "periodic_hann", "center": False,
                  "quantization": "PCM16 ties-even saturating", "mfcc": "40 Slaney Mel, 20–8000 Hz, raw rfft power, ref1 amin1e-10 no top_db, DCT-II ortho 13, 400/160 aligned",
                  "std_ddof": 0, "quantiles": "linear", "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    contract = {"version": FEATURE_SCHEMA_VERSION, "names": list(FEATURE_NAMES), "parameters": parameters}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**contract, "sha256": digest, "versions": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "librosa")}}


def validate_feature_columns(columns: list[str]) -> None:
    """Reject missing, extra, reordered or metadata columns; sample_id is never in X."""
    if list(columns) != list(FEATURE_NAMES):
        raise ValueError("Feature columns must exactly match the ordered fs-v1 allowlist")
