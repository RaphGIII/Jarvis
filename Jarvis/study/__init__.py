"""Studium: the owner's study material as a searchable, locatable library.

A namespace of its own -- not personal memory (preferences, habits, facts about
the owner), not project knowledge, not the knowledge graph.  Documents go
through one pipeline, whatever their format:

    file -> parser -> NormalizedDocument (units: pages / slides / sections, each
    with blocks that carry their exact location) -> chunks -> SQLite index
    (FTS5 word index + FTS5 trigram index) -> hybrid retrieval

and every search result carries the location it came from -- page, slide,
heading path and paragraph -- so "Zeig mir die Seite zum Frank-Starling-
Mechanismus" ends at the page, not at a summary.
"""
