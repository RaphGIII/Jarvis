import main

def test_media_keys_play_pause():
    payload = {'dry_run': True}
    result = main.run(payload)
    assert 'message' in result, 'Expected a message field in the response'
    assert result['message'] == 'Dry run: Attempting to play/pause media keys', f'Unexpected message: {result['message']}'


def test_media_keys_play_pause_real():
    payload = {'dry_run': False}
    result = main.run(payload)
    assert 'message' in result, f'Unexpected message: {result['message']}'