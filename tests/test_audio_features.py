"""Synthetic correctness tests; these are not model evaluation or a real-data pilot."""
from hashlib import sha256
from io import BytesIO
import numpy as np
import pytest
import soundfile as sf
from scipy.fft import dct
from scipy.signal import windows
from voxlingua.audio import decode_audio, make_feature_copy, make_language_copy, quantize_int16
from voxlingua.acoustics import analyze_acoustics, estimated_snr
from voxlingua.features import FEATURE_NAMES, extract_features, feature_schema, validate_feature_columns
from voxlingua.schemas import InputLimits, ServiceError


def wav(x, sr=16000, subtype="FLOAT"):
    out = BytesIO()
    sf.write(out, x, sr, format="WAV", subtype=subtype)
    return out.getvalue()


def sine(sr=16000, amplitude=0.5, seconds=2, frequency=440):
    return (amplitude * np.sin(2 * np.pi * frequency * np.arange(round(sr * seconds)) / sr)).astype(np.float32)


@pytest.mark.parametrize("sr", [16000,44100,48000])
def test_original_amplitude_clipping_and_hash(sr):
    a = decode_audio(wav(np.clip(sine(sr, 2, frequency=100), -1, 1), sr))
    before = sha256(a.samples.tobytes()).hexdigest()
    report = analyze_acoustics(a)
    assert report['clipping_fraction'] == pytest.approx(2/3, abs=0.01)
    assert report['plateau_fraction'] == pytest.approx(2/3, abs=0.01)
    assert report['peak_dbfs'] == 0
    copy = make_feature_copy(a)
    assert copy.dtype == np.float32 and copy.shape == (32000,)
    assert sha256(a.samples.tobytes()).hexdigest() == before
    assert not a.samples.flags.writeable


def test_ordinary_sine_is_not_flat_plateau():
    report = analyze_acoustics(decode_audio(wav(sine(frequency=100))))
    assert report['plateau_fraction'] < 0.01
    assert report['peak_dbfs'] == pytest.approx(-6.0206, abs=0.001)
    assert report['rms_dbfs'] == pytest.approx(-9.0309, abs=0.001)


@pytest.mark.parametrize("seconds,code", [(0.5,'E_AUDIO_TOO_SHORT'), (15.001,'E_AUDIO_TOO_LONG')])
def test_duration_errors(seconds, code):
    with pytest.raises(ServiceError) as error:
        decode_audio(wav(sine(seconds=seconds)))
    assert error.value.code == code


@pytest.mark.parametrize("seconds", [1,15])
def test_duration_boundaries(seconds):
    assert decode_audio(wav(sine(seconds=seconds))).duration_s == seconds


@pytest.mark.parametrize("data,code", [(b'','E_AUDIO_EMPTY'), (b'not a WAV','E_AUDIO_CORRUPT')])
def test_invalid_bytes(data,code):
    with pytest.raises(ServiceError) as error:
        decode_audio(data)
    assert error.value.code == code


def test_nonfinite_and_size():
    with pytest.raises(ServiceError, match="invalid") as error:
        decode_audio(wav(np.full(16000,np.nan)))
    assert error.value.code == 'E_AUDIO_NONFINITE'
    with pytest.raises(ServiceError) as error:
        decode_audio(b'12345',limits=InputLimits(max_bytes=4))
    assert error.value.code == 'E_AUDIO_TOO_LARGE'
    with pytest.raises(ValueError): InputLimits(max_seconds=30)


def test_float_out_of_range_preserved_and_pcm_grid_saturates():
    a = decode_audio(wav(sine(amplitude=1.2)))
    assert a.samples.max() > 1
    assert make_feature_copy(a).max() <= 32767/32768
    assert np.array_equal(quantize_int16(np.array([0.5/32768,1.5/32768,-1.5/32768,2])), np.array([0,2/32768,-2/32768,32767/32768],dtype=np.float32))


def test_stereo_cancellation_and_resampling():
    x=sine(48000)
    a=decode_audio(wav(np.column_stack([x,-x]),48000))
    assert a.n_channels==2 and not make_language_copy(a).any()
    b=decode_audio(wav(x,48000))
    assert np.max(np.abs(make_language_copy(b)[100:-100])) == pytest.approx(0.5,abs=0.01)


def test_silence_nulls_and_finite_features():
    a=decode_audio(wav(np.zeros(32000)))
    report=analyze_acoustics(a)
    for key in ['rms_dbfs','peak_dbfs','est_snr_db','pitch_median_hz','voiced_fraction']:
        assert report[key] is None
    x=extract_features(make_feature_copy(a))
    assert x.shape==(84,) and x.dtype==np.float32 and np.isfinite(x).all()
    assert x[FEATURE_NAMES.index('near_peak_density')]==0


@pytest.mark.parametrize("sr, fundamental", [(8000, 95), (16000, 175), (48000, 265)])
def test_pitch_measures_voiced_harmonic_frames(sr, fundamental):
    voiced = sine(sr, 0.025, 1, fundamental) + sine(sr, 0.012, 1, 2 * fundamental)
    report = analyze_acoustics(decode_audio(wav(np.r_[voiced, np.zeros(sr)], sr)))
    assert report["pitch_median_hz"] == pytest.approx(fundamental, rel=0.02)
    assert report["voiced_fraction"] == pytest.approx(0.5, abs=0.06)


def test_pitch_is_undefined_for_broadband_noise():
    noise = np.random.default_rng(20261116).normal(0, 0.04, 32000)
    report = analyze_acoustics(decode_audio(wav(noise)))
    assert report["pitch_median_hz"] is None
    assert report["voiced_fraction"] is None


def test_feature_identity_and_exact_allowlist():
    x=make_feature_copy(decode_audio(wav(sine())))
    assert np.array_equal(extract_features(x),extract_features(x))
    assert len(FEATURE_NAMES)==len(set(FEATURE_NAMES))==84
    validate_feature_columns(list(FEATURE_NAMES))
    assert 'noise_floor_flatness' in FEATURE_NAMES
    with pytest.raises(ValueError): validate_feature_columns(list(FEATURE_NAMES)+['noise_id'])
    with pytest.raises(ValueError): validate_feature_columns(list(reversed(FEATURE_NAMES)))
    assert feature_schema()['sha256']==feature_schema()['sha256']


def test_mfcc_matches_independent_explicit_frames():
    import librosa
    x=quantize_int16(sine(seconds=1.07)).astype(np.float64)
    count=1+(len(x)-400)//160
    power=np.column_stack([np.abs(np.fft.rfft(x[t*160:t*160+400]*windows.hann(400,sym=False),512))**2 for t in range(count)])
    mel=librosa.filters.mel(sr=16000,n_fft=512,n_mels=40,fmin=20,fmax=8000,htk=False,norm='slaney')@power
    coefficients=dct(10*np.log10(np.maximum(mel,1e-10)),type=2,axis=0,norm='ortho')[:13]
    actual=extract_features(x)
    expected=np.r_[coefficients.mean(axis=1),coefficients.std(axis=1)].astype(np.float32)
    assert np.allclose(actual[35:61],expected,atol=1e-5,rtol=0)


def test_snr_invalid_may_be_null_and_valid_monotonic():
    assert estimated_snr(np.ones(200)) is None
    assert estimated_snr(np.zeros(200)) is None
    values=[estimated_snr(np.r_[np.full(40,n),np.ones(160)]) for n in [0.001,0.01,0.1]]
    valid=[v for v in values if v is not None]
    assert len(valid)>=2 and all(a>b for a,b in zip(valid,valid[1:]))
    assert values[-1] is None


def test_gap_and_no_end_zero_false_positive():
    x=sine()
    x[16000:17600]=0
    v=extract_features(quantize_int16(x))
    assert v[FEATURE_NAMES.index('longest_zero_run_ms')]==pytest.approx(100,abs=1)
    assert v[FEATURE_NAMES.index('n_short_gaps_rel')]>=1
    x=sine();x[:1600]=0;x[-1600:]=0
    v=extract_features(quantize_int16(x))
    assert v[FEATURE_NAMES.index('zero_run_frac')]==0
