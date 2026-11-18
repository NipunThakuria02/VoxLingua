"""Deterministic native-rate acoustics; no classifier or GUI dependency."""
from math import ceil
import numpy as np
from .audio import make_language_copy
from .features import frames, runs, spectral
from .schemas import DecodedAudio


def measure_clipping(a: DecodedAudio) -> dict:
    """Count channels separately; native-rate duration criteria avoid rate-biased runs."""
    x = a.samples
    near = flat = 0
    minimum = ceil(3 * a.sample_rate / 16000)
    for ch in x.T:
        peak = np.max(np.abs(ch))
        if peak < 0.2:
            continue
        for sign in (1, -1):
            for start, end in runs(ch * sign >= 0.98 * peak):
                if end - start < minimum:
                    continue
                near += end - start
                # A plateau is a segment of samples, hence one more sample than adjacent edges.
                for i, j in runs(np.abs(np.diff(ch[start:end].astype(np.float64))) <= 1 / 32768):
                    if j - i + 1 >= minimum:
                        flat += j - i + 1
    return {"clipping_fraction": float(np.mean(np.abs(x) >= 0.99)), "near_peak_run_fraction": near / x.size, "plateau_fraction": flat / x.size}


def estimated_snr(frame_power: np.ndarray) -> float | None:
    """Level-contrast estimate only; return null when reliable contrast is absent."""
    p = np.asarray(frame_power, dtype=np.float64)
    count = int(np.floor(0.1 * len(p)))
    if count < 10 or not np.isfinite(p).all():
        return None
    order = np.argsort(p, kind="stable")
    noise = p[order[:count]].mean()
    top = p[order[-ceil(0.5 * len(p)):]].mean()
    speech = top - noise
    if noise <= 0 or speech <= noise or 10 * np.log10(top / noise) < 15:
        return None
    return float(10 * np.log10(speech / noise))


def measure_pitch(a: DecodedAudio) -> tuple[float | None, float | None]:
    """Return median fundamental frequency in Hz and fraction of voiced pitch frames."""
    x = make_language_copy(a)
    frame_length, hop = 640, 160  # 40 and 10 ms at 16 kHz
    pitch_frames = frames(x, frame_length, hop)
    audible = np.mean(pitch_frames ** 2, axis=1) >= 1e-8
    if np.count_nonzero(audible) < 10:
        return None, None

    from librosa import pyin

    f0, voiced, _ = pyin(x, fmin=65, fmax=400, sr=16000, frame_length=frame_length,
                         hop_length=hop, center=False)
    voiced = voiced & audible & np.isfinite(f0)
    if np.count_nonzero(voiced) < 10:
        return None, None
    return float(np.median(f0[voiced])), float(np.mean(voiced))


def analyze_acoustics(a: DecodedAudio) -> dict:
    """Measure original mono RMS/spectra and all-channel peak/clipping; no normalization."""
    x = a.samples.mean(axis=1, dtype=np.float64)
    length, hop = round(0.025 * a.sample_rate), round(0.010 * a.sample_rate)
    fr = frames(x, length, hop)
    p = np.mean(fr ** 2, axis=1)
    level = 10 * np.log10(np.maximum(p, 1e-12))
    s, descriptors = spectral(fr, a.sample_rate, 1 << (length - 1).bit_length())
    active = p > 1e-8
    rms, peak = np.sqrt(np.mean(x ** 2)), np.max(np.abs(a.samples))
    activity = float(np.mean((level >= np.quantile(level, 0.95) - 30) & (level >= -60)))
    warnings = ["Estimated SNR and speech activity are heuristics, not calibrated measurements."]
    if rms < 1e-5:
        warnings.append("No speech or insufficient signal; language analysis should be skipped.")
    if peak > 1:
        warnings.append("Float WAV exceeds full scale; the feature copy will saturate, original measurements are preserved.")
    pitch, voiced_fraction = measure_pitch(a)
    report = {"rms_dbfs": float(20 * np.log10(rms)) if rms else None, "peak_dbfs": float(20 * np.log10(peak)) if peak else None,
              **measure_clipping(a), "duration_s": a.duration_s, "sample_rate": a.sample_rate, "n_channels": a.n_channels,
              "est_snr_db": estimated_snr(p), "speech_activity_ratio": activity, "silence_ratio": 1 - activity,
              "pitch_median_hz": pitch, "voiced_fraction": voiced_fraction, "recommendations": [], "warnings": warnings}
    for key, source in (("spectral_centroid_hz", "centroid_hz"), ("spectral_bandwidth_hz", "bandwidth_hz"), ("spectral_flatness", "flatness"), ("zero_crossing_rate", "zcr")):
        report[key] = float(descriptors[source][active].mean()) if active.any() else None
    # Plots contain actual signal summaries; no generated/simulated measurements.
    stride = max(1, len(x) // 4000)
    report["plots"] = {"waveform_time_s": (np.arange(len(x))[::stride] / a.sample_rate).tolist(), "waveform_amplitude": x[::stride].tolist(),
                       "spectrogram_time_s": ((np.arange(len(fr)) * hop + length / 2) / a.sample_rate).tolist(),
                       "spectrogram_frequency_hz": np.fft.rfftfreq((s.shape[1] - 1) * 2, 1 / a.sample_rate).tolist(),
                       "spectrogram_dbfs": (10 * np.log10(s.T + 1e-12)).tolist()}
    return report
