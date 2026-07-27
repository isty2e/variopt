"""Closed taxonomy for built-in structured-space geometry."""

from typing import TypeAlias, TypeGuard

from ..composites.array_space import ArraySpace
from ..composites.record_space import RecordSpace
from ..composites.tuple_space import TupleSpace
from ..permutation import PermutationSpace
from ..scalar import CategoricalSpace, IntegerSpace, RealSpace
from ..types import SpaceBoundaryValue, SpaceCandidateValue, SpaceScalarValue

BuiltinGeometrySpace: TypeAlias = (
    RealSpace
    | IntegerSpace
    | CategoricalSpace[SpaceScalarValue]
    | PermutationSpace
    | TupleSpace
    | RecordSpace
    | ArraySpace[SpaceBoundaryValue, SpaceCandidateValue]
)

BUILTIN_GEOMETRY_SPACE_TYPES = (
    RealSpace,
    IntegerSpace,
    CategoricalSpace,
    PermutationSpace,
    TupleSpace,
    RecordSpace,
    ArraySpace,
)


def is_builtin_geometry_space(space: object) -> TypeGuard[BuiltinGeometrySpace]:
    """Return whether ``space`` belongs to the built-in geometry taxonomy."""
    return isinstance(space, BUILTIN_GEOMETRY_SPACE_TYPES)


def is_exact_builtin_geometry_space(
    space: object,
) -> TypeGuard[BuiltinGeometrySpace]:
    """Return whether ``space`` is an exact built-in geometry realization."""
    return type(space) in BUILTIN_GEOMETRY_SPACE_TYPES
