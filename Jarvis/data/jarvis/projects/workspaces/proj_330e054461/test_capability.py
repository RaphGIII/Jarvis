from __future__ import annotations

from unittest.mock import patch

import main


def test_run_dry_run_true():
    result = main.run({'dry_run': True})
    assert 'would_use' in result, 'Dry run should return a path'
    assert isinstance(result['ok'], bool), 'Dry run should return a boolean'


def test_run_dry_run_false_no_player_found():
    with patch('shutil.which', return_value=None):
        result = main.run({'dry_run': False})
        assert not result['ok'], 'Should indicate no player found'
        assert isinstance(result['error'], str), 'Error message should be a string'


def test_run_dry_run_false_player_found():
    with patch('shutil.which', return_value='C:\Program Files\VideoLAN\VLC\vlc.exe'):
        result = main.run({'dry_run': False})
        assert 'message' in result, 'The response should contain a message key'
        assert isinstance(result['message'], str), 'Message should be a string'