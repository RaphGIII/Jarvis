import main

def test_play_pause_command():
    result = main.run({'dry_run': True})
    assert 'folders' in result and result['audio_files_count'] > 0, 'Expected non-zero audio files count'

    result = main.check_media_keys({'dry_run': True})
    assert 'message' in result and 'Media keys test successful.' in result['message'], 'Expected media keys test message not found'