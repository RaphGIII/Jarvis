import json

import pytest

from brain.json_utils import lenient_json_loads


def test_lenient_json_loads_parses_well_formed_json_normally():
    assert lenient_json_loads('{"a": 1, "b": [1, 2, 3]}') == {"a": 1, "b": [1, 2, 3]}


def test_lenient_json_loads_tolerates_literal_control_characters_in_strings():
    """Small local models served via guided-JSON decoding sometimes emit a
    literal newline byte inside a JSON string value instead of the escaped
    \\n sequence. That is otherwise well-formed, semantically valid output;
    strict JSON parsing rejects it entirely, which was observed to waste a
    full implementation attempt during a live multi-file capability build."""
    raw = '{"path": "main.py", "content": "def run(payload):\n    return {}\n"}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    assert lenient_json_loads(raw) == {"path": "main.py", "content": "def run(payload):\n    return {}\n"}


def test_lenient_json_loads_still_raises_on_genuinely_malformed_json():
    with pytest.raises(json.JSONDecodeError):
        lenient_json_loads("{not json at all")
