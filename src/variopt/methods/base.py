"""Semantic root hierarchy for variopt search methods."""

from abc import ABC


class SearchMethod(ABC):
    """Semantic root for search methods over canonical runtime artifacts.

    Notes
    -----
    Concrete search-method families refine this root with run-scoped or
    one-shot transition laws while preserving the same canonical artifact
    vocabulary.
    """
