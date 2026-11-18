from ui.player import waveform_player


def test_waveform_player_uses_real_audio_and_actual_time_cursor():
    html=waveform_player(b'test-only',{'waveform_time_s':[0,.5,1],'waveform_amplitude':[0,.2,0]})
    assert 'dGVzdC1vbmx5' in html
    assert 'audio.currentTime' in html and 'audio.duration' in html
    assert 'ArrowLeft' in html and 'ArrowRight' in html
    assert 'navigator.mediaDevices' not in html
