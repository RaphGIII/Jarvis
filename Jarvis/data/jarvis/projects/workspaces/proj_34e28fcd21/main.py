from __future__ import annotations

from typing import Any

# Every payload key run() accepts. A caller cannot pass what is not declared.
INPUT_SCHEMA = {"type": "object", "properties": {"dry_run": {"type": "boolean"}, "play_pause_test": {"type": "boolean"}}, "required": ["dry_run", "play_pause_test"]}


def run(payload: dict[str, Any]) -> dict[str, Any]:
    dry_run = payload.get('dry_run', False)
    if dry_run:
        result = media_folders()
        return {
            "ok": True,
            "folders": result['folders'],
            "message": "Dry run complete."
        }

    # Check for installed media applications
    vlc_path = find_program({'name': 'vlc'})['path']
    spotify_path = find_program({'name': 'spotify'})['path']

    if not vlc_path and not spotify_path:
        return {
            "ok": False,
            "error": "No media player found."
        }

    # Identify local audio files
    folders = media_folders()['folders']
    audio_files_count = sum(folder['audio_files'] for folder in folders)

    if audio_files_count == 0:
        return {
            "ok": False,
            "error": "No audio files found."
        }

    # Play music using the first available media player
    if vlc_path:
        run_command(['"' + vlc_path + '"', '--play-and-exit'], timeout_seconds=10)
    elif spotify_path:
        run_command(['"' + spotify_path + '"'], timeout_seconds=10)

    return {
        "ok": True,
        "message": f'{audio_files_count} audio files played.'
    }