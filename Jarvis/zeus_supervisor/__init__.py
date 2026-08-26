"""The ZEUS supervisor: the small process that is *not* ZEUS.

ZEUS rewrites itself.  That is the point of the system, and it is also the one
thing that makes an ordinary "run the server" loop unsafe: a candidate that
passed its tests can still leave the installation unable to start, and the
process that would notice is the one that just died.  So the thing that starts
ZEUS, checks it, and puts it back when a promotion goes wrong has to live
outside the code ZEUS is allowed to change.

This package is that thing.  It is deliberately small, imports nothing from the
application at runtime, and is one of the protected paths that self-development
may not write to (see :mod:`owner.protected`).  When frozen into ``ZEUS.exe`` it
is immutable to the running system by construction.

    ZEUS.exe / python -m zeus_supervisor
        |
        preflight   -- Python, repository, Ollama, models, one real generation
        |
        known-good  -- the revision that last passed a health check
        |
        core        -- python -m jarvis.serve, as a child process
        |
        healthcheck -- /api/health says READY, not "a port opened"
        |
        keep / roll back / hold
"""

from __future__ import annotations

__version__ = "1.0.0"

#: Exit code the core uses to say "restart me" (a promotion wants to be tried).
EXIT_RESTART_REQUESTED = 75
#: Exit code the core uses to say "stay down" (the owner asked for a shutdown).
EXIT_SHUTDOWN_REQUESTED = 0
