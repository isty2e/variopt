"""Geometry compilation and generic structured-distance helpers."""

from typing import TypeAlias, TypeGuard, TypeVar

from ..composites.array_space import ArraySpace
from ..composites.record_space import RecordSpace
from ..composites.tuple_space import TupleSpace
from ..permutation import PermutationSpace
from ..scalar import CategoricalSpace, IntegerSpace, RealSpace
from ..structured import LeafPath, StructuredLeafSpace, StructuredSearchSpace
from ..types import SpaceBoundaryValue, SpaceCandidateValue, SpaceScalarValue
from .composites import (
    ArraySpaceGeometry,
    BinaryArraySpaceGeometry,
    IntegerArraySpaceGeometry,
    RealArraySpaceGeometry,
    RecordSpaceGeometry,
    TupleSpaceGeometry,
    collect_child_geometries,
    collect_field_geometries,
)
from .contracts import CompiledStructuredGeometryProvider, StructuredSpaceGeometry
from .leaf import is_categorical_leaf_space
from .parts import StructuredDistanceParts
from .permutation import PermutationSpaceGeometry
from .scalar import CategoricalSpaceGeometry, IntegerSpaceGeometry, RealSpaceGeometry

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)
BuiltinGeometrySpace: TypeAlias = (
    RealSpace
    | IntegerSpace
    | CategoricalSpace[SpaceScalarValue]
    | PermutationSpace
    | TupleSpace
    | RecordSpace
    | ArraySpace[SpaceBoundaryValue, SpaceCandidateValue]
)
_ArrayGeometrySpace: TypeAlias = ArraySpace[SpaceBoundaryValue, SpaceCandidateValue]


def _is_array_geometry_space(space: object) -> TypeGuard[_ArrayGeometrySpace]:
    """Return whether ``space`` is an array geometry owner."""
    return isinstance(space, ArraySpace)


def _is_builtin_geometry_space(
    space: object,
) -> TypeGuard[BuiltinGeometrySpace]:
    """Return whether one space belongs to the built-in geometry family."""
    return isinstance(
        space,
        (
            RealSpace,
            IntegerSpace,
            CategoricalSpace,
            PermutationSpace,
            TupleSpace,
            RecordSpace,
            ArraySpace,
        ),
    )


def distance_parts(
    space: StructuredSearchSpace[BoundaryT, CandidateT],
    left: CandidateT,
    right: CandidateT,
) -> StructuredDistanceParts:
    """Return canonical structured distance parts for two candidates.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured search space shared by ``left`` and ``right``.
    left : CandidateT
        Left canonical candidate.
    right : CandidateT
        Right canonical candidate.

    Returns
    -------
    StructuredDistanceParts
        Structured distance decomposition between the two candidates.
    """
    geometry = compile_structured_geometry(space)
    if geometry is not None:
        return geometry.distance_parts(left, right)

    return generic_distance_parts(space, left, right)


def generic_distance_parts(
    space: StructuredSearchSpace[BoundaryT, CandidateT],
    left: CandidateT,
    right: CandidateT,
) -> StructuredDistanceParts:
    """Return canonical distance parts through the generic leaf traversal path.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured search space shared by ``left`` and ``right``.
    left : CandidateT
        Left canonical candidate.
    right : CandidateT
        Right canonical candidate.

    Returns
    -------
    StructuredDistanceParts
        Structured distance decomposition produced by generic leaf traversal.

    Raises
    ------
    ValueError
        If the space exposes no active or mismatched leaves.
    """
    space.validate(left)
    space.validate(right)

    declared_leaf_paths = space.leaf_paths()
    left_active_leaf_paths = set(space.active_leaf_paths_for_validated_candidate(left))
    right_active_leaf_paths = set(
        space.active_leaf_paths_for_validated_candidate(right)
    )

    shared_leaf_count = 0
    topology_mismatch_leaf_count = 0
    squared_distance = 0.0

    for path in declared_leaf_paths:
        left_active = path in left_active_leaf_paths
        right_active = path in right_active_leaf_paths
        if left_active and right_active:
            shared_leaf_count += 1
            squared_distance += _normalized_squared_leaf_distance_at_path(
                space=space,
                path=path,
                left=space.leaf_value_at_validated_path(left, path),
                right=space.leaf_value_at_validated_path(right, path),
            )
            continue

        if left_active or right_active:
            topology_mismatch_leaf_count += 1

    if shared_leaf_count + topology_mismatch_leaf_count == 0:
        msg = "structured diversity metric requires at least one leaf path"
        raise ValueError(msg)
    return StructuredDistanceParts(
        overlap_squared_distance=squared_distance,
        shared_leaf_count=shared_leaf_count,
        topology_mismatch_leaf_count=topology_mismatch_leaf_count,
    )


def _normalized_squared_leaf_distance_at_path(
    *,
    space: StructuredSearchSpace[BoundaryT, CandidateT],
    path: LeafPath,
    left: SpaceCandidateValue,
    right: SpaceCandidateValue,
) -> float:
    """Return a leaf distance using any stronger owner-space law at ``path``."""
    if _path_has_permutation_owner(space, path):
        if left == right:
            return 0.0
        return 1.0

    return normalized_squared_leaf_distance(
        space=space.leaf_space_at_path(path),
        left=left,
        right=right,
    )


def normalized_squared_leaf_distance(
    *,
    space: StructuredLeafSpace,
    left: SpaceCandidateValue,
    right: SpaceCandidateValue,
) -> float:
    """Return one normalized squared distance between canonical leaf values.

    Parameters
    ----------
    space : StructuredLeafSpace
        Leaf space shared by ``left`` and ``right``.
    left : SpaceCandidateValue
        Left canonical leaf value.
    right : SpaceCandidateValue
        Right canonical leaf value.

    Returns
    -------
    float
        Normalized squared leaf distance.

    Raises
    ------
    TypeError
        If ``space`` is not a supported built-in leaf space.
    """
    if isinstance(space, RealSpace):
        return RealSpaceGeometry(space).squared_distance(
            left,
            right,
        )

    if isinstance(space, IntegerSpace):
        return IntegerSpaceGeometry(space).squared_distance(
            left,
            right,
        )

    if is_categorical_leaf_space(space):
        return CategoricalSpaceGeometry(space).squared_distance(
            left,
            right,
        )

    msg = f"unsupported structured leaf space for diversity: {type(space)!r}"
    raise TypeError(msg)


def _path_has_permutation_owner(space: object, path: LeafPath) -> bool:
    """Return whether ``path`` is owned by a nested permutation space."""
    if isinstance(space, PermutationSpace):
        _ = space.leaf_space_at_path(path)
        return True

    if isinstance(space, TupleSpace):
        if len(path) == 0 or type(path[0]) is not int:
            return False
        child_index = path[0]
        child_spaces = space.child_spaces
        if child_index < 0 or child_index >= len(child_spaces):
            return False
        return _path_has_permutation_owner(child_spaces[child_index], path[1:])

    if isinstance(space, RecordSpace):
        if len(path) == 0 or not isinstance(path[0], str):
            return False
        field_name = path[0]
        for name, child_space in space.fields:
            if name == field_name:
                return _path_has_permutation_owner(child_space, path[1:])
        return False

    if _is_array_geometry_space(space):
        if len(path) == 0 or type(path[0]) is not int:
            return False
        element_index = path[0]
        if element_index < 0 or element_index >= space.length:
            return False
        element_space: object = space.element_space
        return _path_has_permutation_owner(element_space, path[1:])

    return False


def compile_builtin_structured_geometry(
    space: StructuredSearchSpace[BoundaryT, CandidateT],
) -> StructuredSpaceGeometry | None:
    """Compile one fast geometry for built-in structured spaces.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured space to compile.

    Returns
    -------
    StructuredSpaceGeometry | None
        Compiled built-in geometry, or ``None`` when ``space`` is not part of
        the built-in geometry family.
    """
    if not _is_builtin_geometry_space(space):
        return None
    return compile_builtin_child_space_geometry(space)


def compile_structured_geometry(
    space: StructuredSearchSpace[BoundaryT, CandidateT],
) -> StructuredSpaceGeometry | None:
    """Return one compiled geometry from sidecar providers or built-ins.

    Third-party structured spaces may opt into compiled geometry through the
    sidecar ``CompiledStructuredGeometryProvider`` protocol. Built-in
    realizations remain the fallback when no provider is present.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured space to compile.

    Returns
    -------
    StructuredSpaceGeometry | None
        Provider-supplied geometry when available, otherwise a built-in
        geometry, or ``None`` if neither path can compile one.
    """
    provider_geometry = compile_provider_structured_geometry(space)
    if provider_geometry is not None:
        return provider_geometry
    return compile_builtin_structured_geometry(space)


def compile_provider_structured_geometry(
    space: StructuredSearchSpace[BoundaryT, CandidateT],
) -> StructuredSpaceGeometry | None:
    """Return one compiled geometry from an optional sidecar provider.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured space to inspect for a provider implementation.

    Returns
    -------
    StructuredSpaceGeometry | None
        Provider-compiled geometry, or ``None`` when the space has no provider
        or declines compilation.
    """
    if not isinstance(space, CompiledStructuredGeometryProvider):
        return None

    geometry = space.compile_structured_geometry()
    if geometry is None:
        return None
    return geometry


def compile_builtin_child_space_geometry(
    space: BuiltinGeometrySpace,
) -> StructuredSpaceGeometry | None:
    """Compile one fast geometry for one built-in child space.

    Parameters
    ----------
    space : BuiltinGeometrySpace
        Built-in child space to compile.

    Returns
    -------
    StructuredSpaceGeometry | None
        Compiled geometry for ``space``, or ``None`` when one of its nested
        child spaces cannot be compiled.
    """
    if isinstance(space, RealSpace):
        return RealSpaceGeometry(space)

    if isinstance(space, IntegerSpace):
        return IntegerSpaceGeometry(space)

    if isinstance(space, CategoricalSpace):
        return CategoricalSpaceGeometry(space)

    if isinstance(space, PermutationSpace):
        return PermutationSpaceGeometry(space)

    if isinstance(space, TupleSpace):
        child_geometries = collect_child_geometries(
            tuple(
                compile_builtin_child_space_geometry(child_space)
                if _is_builtin_geometry_space(child_space)
                else None
                for child_space in space.child_spaces
            ),
        )
        if child_geometries is None:
            return None
        return TupleSpaceGeometry(
            arity=len(space.child_spaces),
            child_geometries=child_geometries,
        )

    if isinstance(space, RecordSpace):
        field_geometries = collect_field_geometries(
            tuple(
                (
                    name,
                    (
                        compile_builtin_child_space_geometry(child_space)
                        if _is_builtin_geometry_space(child_space)
                        else None
                    ),
                )
                for name, child_space in space.fields
            ),
        )
        if field_geometries is None:
            return None
        return RecordSpaceGeometry(field_geometries=field_geometries)

    element_space: object = space.element_space
    if isinstance(element_space, IntegerSpace):
        if (
            element_space.low == 0
            and element_space.high == 1
            and element_space.scale == "linear"
        ):
            return BinaryArraySpaceGeometry(
                length=space.length,
                element_space=element_space,
            )
        return IntegerArraySpaceGeometry(
            length=space.length,
            element_space=element_space,
        )

    if isinstance(element_space, RealSpace):
        return RealArraySpaceGeometry(
            length=space.length,
            element_space=element_space,
        )

    if not _is_builtin_geometry_space(element_space):
        return None

    element_geometry = compile_builtin_child_space_geometry(element_space)
    if element_geometry is None:
        return None
    return ArraySpaceGeometry(length=space.length, element_geometry=element_geometry)
