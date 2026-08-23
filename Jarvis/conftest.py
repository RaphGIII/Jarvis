"""Repository-wide pytest configuration.

The only job of this file is to make the test suite runnable on hosts where
the default pytest temp root (``<system temp>/pytest-of-<user>``) is not
readable by the current user.  That situation is not hypothetical: on the
primary Windows development machine that directory carries an ACL that denies
even ``os.scandir``, which makes every ``tmp_path`` based test error out
during fixture setup.

``PYTEST_DEBUG_TEMPROOT`` is the documented hook for relocating that root, and
it has to be set before ``TempPathFactory`` resolves the base temp directory,
which happens lazily on first use -- importing the root ``conftest`` is early
enough.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _usable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".jarvis_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        list(os.scandir(path))
        return True
    except OSError:
        return False


def _select_temp_root() -> Path | None:
    default_root = Path(tempfile.gettempdir())
    if _usable(default_root / f"pytest-of-{os.environ.get('USERNAME') or os.environ.get('USER') or 'user'}"):
        return None

    for candidate in (default_root / "jarvis-pytest", Path(__file__).resolve().parent / ".pytest_basetemp"):
        if _usable(candidate):
            return candidate
    return None


if not os.environ.get("PYTEST_DEBUG_TEMPROOT"):
    _root = _select_temp_root()
    if _root is not None:
        os.environ["PYTEST_DEBUG_TEMPROOT"] = str(_root)
