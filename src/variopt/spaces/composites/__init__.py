"""Composite container search spaces facade."""

from .adapters import CompositeChildSpace
from .array_space import ArraySpace
from .record_space import RecordSpace
from .records import RecordCandidate
from .tuple_space import TupleSpace

__all__ = [
    "ArraySpace",
    "CompositeChildSpace",
    "RecordCandidate",
    "RecordSpace",
    "TupleSpace",
]
