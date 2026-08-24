# Capability implementation. Replace the body of run().

from __future__ import annotations

from typing import Any

# Every payload key run() accepts. A caller cannot pass what is not declared.
INPUT_SCHEMA = {"type": "object", "properties": {"dry_run": {"type": "boolean"}}, "required": []}


def run(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get('dry_run'):
        return {'message': 'Dry run: Attempting to play/pause media keys'}

    import shutil

    vlc_path = shutil.which('vlc')
    spotify_path = shutil.which('spotify')
    if vlc_path:
        return {'message': 'Dry run: VLC is installed and would be used to play music'}
    elif spotify_path:
        return {'message': 'Dry run: Spotify is installed and would be used to play music'}
    else:
        media_control('playpause', dry_run=True)
        return {'message': 'Dry run: System media keys would be used to play/pause music'}