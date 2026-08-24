# Capability implementation. Replace the body of run().

from __future__ import annotations

from typing import Any

# Every payload key run() accepts. A caller cannot pass what is not declared.
INPUT_SCHEMA = {"type": "object", "properties": {"dry_run": {"type": "boolean"}}, "required": []}


def run(payload: dict[str, Any]) -> dict[str, Any]:
    dry_run = payload.get('dry_run', False)
    if dry_run:
        return {
            "message": "Would play audio with media player"
        }
    else:
        local_audio_files = find_media(paths=[r'C:\Users\rapha\Music', r'C:\Users\rapha\Downloads'], limit=1)
        if local_audio_files['audio_files'] > 0:
            import os
            return {
                "message": "Playing audio with media player"
            }
        else:
            return {
                "error": "No audio files found"
            }
        local_audio_files = find_media(paths=[r'C:\Users\rapha\Music', r'C:\Users\rapha\Downloads'], limit=1)
        if local_audio_files['audio_files'] > 0:
            import os
            return {
                "message": "Playing audio with media player"
            }
        else:
            return {
                "error": "No audio files found"
            }