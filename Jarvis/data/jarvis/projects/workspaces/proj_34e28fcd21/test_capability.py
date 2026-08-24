import main

def test_play_pause_command():
    result = main.run({'dry_run': True, 'play_pause_test': False})
    assert 'folders' in result and len(result['folders']) > 0, 'Expected folders to be found'

    result = main.run({'dry_run': True, 'play_pause_test': True})
    assert 'message' in result and 'Play/Pause command executed.' in result['message'], 'Expected play/pause message not found'