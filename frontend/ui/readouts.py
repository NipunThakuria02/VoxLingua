"""Presentation helpers for language labels and acoustic readouts."""
from html import escape


ACOUSTIC_GROUPS = (
    ('Recording', (
        ('Duration', 'duration_s', '.2f', ' s'),
        ('Sample rate', 'sample_rate', ',d', ' Hz'),
        ('Channels', 'n_channels', 'd', ''),
    )),
    ('Level & clipping', (
        ('RMS level', 'rms_dbfs', '.1f', ' dBFS'),
        ('Peak level', 'peak_dbfs', '.1f', ' dBFS'),
        ('Clipped samples', 'clipping_fraction', '.2%', ''),
        ('Near-peak runs', 'near_peak_run_fraction', '.2%', ''),
        ('Flat-topped samples', 'plateau_fraction', '.2%', ''),
    )),
    ('Pitch & activity', (
        ('Median voiced pitch', 'pitch_median_hz', '.1f', ' Hz'),
        ('Voiced frames', 'voiced_fraction', '.1%', ''),
        ('Estimated speech activity', 'speech_activity_ratio', '.1%', ''),
        ('Estimated silence', 'silence_ratio', '.1%', ''),
        ('Estimated SNR', 'est_snr_db', '.1f', ' dB'),
    )),
    ('Spectral characteristics', (
        ('Spectral centroid', 'spectral_centroid_hz', '.1f', ' Hz'),
        ('Spectral bandwidth', 'spectral_bandwidth_hz', '.1f', ' Hz'),
        ('Spectral flatness', 'spectral_flatness', '.4f', ''),
        ('Zero-crossing rate', 'zero_crossing_rate', '.2%', ''),
    )),
)


def language_grid(labels: list[str]) -> str:
    """Show every label in alphabetic order, as selectable browser text."""
    items = ''.join(f'<li>{escape(label)}</li>' for label in sorted(labels, key=str.casefold))
    return f'<ul class="language-grid" role="list" aria-label="Supported languages">{items}</ul>'


def acoustic_details(acoustics: dict) -> str:
    """Format measured values and explicit unavailable states for the Details tab."""
    panels = []
    for title, fields in ACOUSTIC_GROUPS:
        rows = []
        for label, key, spec, unit in fields:
            value = acoustics.get(key)
            reading = 'Unavailable' if value is None else format(value, spec) + unit
            rows.append(f'<div><dt>{label}</dt><dd>{escape(reading)}</dd></div>')
        panels.append(f'<section class="detail-card"><h3>{escape(title)}</h3><dl>{"".join(rows)}</dl></section>')
    return f'<div class="detail-grid">{"".join(panels)}</div>'
