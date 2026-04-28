"""Canonical recursive value types for heterogeneous search spaces."""

from collections.abc import Mapping, Sequence
from typing import TypeAlias

SpaceScalarValue: TypeAlias = bool | int | float | str | bytes | bytearray

SpaceBoundaryValue: TypeAlias = (
    SpaceScalarValue
    | Sequence["SpaceBoundaryValue"]
    | Mapping[str, "SpaceBoundaryValue"]
)

SpaceCandidateValue: TypeAlias = (
    SpaceScalarValue
    | tuple["SpaceCandidateValue", ...]
    | Mapping[str, "SpaceCandidateValue"]
)
