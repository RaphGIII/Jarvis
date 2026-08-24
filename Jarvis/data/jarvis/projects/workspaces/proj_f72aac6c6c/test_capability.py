# Tests for this capability. Replace these with real ones.

import main


def test_no_audio_files_found():
    result = main.run({'dry_run': False})
    assert 'error' in result and result['error'] == 'No audio files found'


def test_dry_run_reported_correctly():
    result = main.run({'dry_run': True})
    assert 'message' in result and result['message'] == 'Would play audio with media player'