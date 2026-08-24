"""Runtime services: deadlines, heartbeats, cost policy, the older runtimes.

This package's ``__init__`` is deliberately lazy, and that is load-bearing
rather than a style preference.

``runtime.deadline`` and ``runtime.cost_policy`` are leaf utilities imported by
``projects.engine`` and ``development.repository_engineer``.  Importing them
initialises this package first -- and when this file eagerly imported
``jarvis_runtime`` -> ``capability_runtime`` -> ``development`` ->
``repository_engineer``, that landed straight back on ``runtime.deadline``, a
cycle through the entire development stack to reach a module with no
dependencies of its own.

Single-threaded, Python hides this: the half-initialised ``runtime`` module is
already in ``sys.modules``, so the second import finds it and moves on.  Under
two threads it does not hide at all, and the observed symptoms were a bare
``KeyError: 'runtime'`` in one thread and ``_DeadlockError`` on
``_ModuleLock('runtime.deadline')`` in the other -- which surfaced as the
service reporting OFFLINE while the very same check passed when run by hand.

PEP 562 module ``__getattr__`` keeps ``from runtime import JarvisRuntime``
working for existing callers while making ``import runtime.deadline`` cost
nothing but the leaf module.
"""

from typing import Any

_LAZY = {
    "JarvisRuntime": "runtime.jarvis_runtime",
    "JarvisRuntimeConfig": "runtime.jarvis_runtime",
    "CapabilityAcquisitionRuntime": "runtime.capability_runtime",
    "CapabilityRuntimeConfig": "runtime.capability_runtime",
    "RuntimeMode": "runtime.runtime_state",
    "RuntimeState": "runtime.runtime_state",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
