"""The ZEUS Catalog: what exists, where, with which contracts, depending on what.

An engineer -- Codex, a metered API role, a person -- should not have to read
the repository to find out how it is put together.  The catalog answers the
cheap questions first:

* which modules exist and what each one owns (``ZEUS_MAP.yaml``)
* which public contracts each module offers (``INTERFACES.yaml``)
* what depends on what, and what is affected by a change (``DEPENDENCY_GRAPH.json``)
* which tests belong to a module, which capabilities exist, which HTTP entry
  points matter, which UI modules the page loads

Everything in it is *derived* from the code by :mod:`catalog.build`, so it
cannot disagree with the code for long: :mod:`catalog.check` rebuilds it in
memory and fails when the committed catalog is stale, and the
self-development verification regenerates it inside the candidate worktree
so a promoted change carries its own catalog update.

:mod:`catalog.context` turns the catalog into the narrow engineering context
the brief asks for -- the impacted modules' manifests, their contracts, their
direct dependencies and their tests -- instead of the whole repository.
"""

# Submodules are imported explicitly (catalog.build, catalog.check,
# catalog.context); importing them here would re-import under ``python -m``.
