"""Numeric coordinate projections for structured search spaces."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..scalar import IntegerSpace, RealSpace
from ..structured import LeafPath, StructuredSearchSpace
from ..types import SpaceCandidateValue

BoundaryT = TypeVar("BoundaryT")
StructuredCandidateT = TypeVar("StructuredCandidateT", bound=SpaceCandidateValue)
NumericLeafSpace = RealSpace | IntegerSpace


@dataclass(frozen=True, slots=True)
class HomogeneousNumericSubspaceDescriptor(
    FrozenGenericSlotsCompat,
    Generic[BoundaryT, StructuredCandidateT],
):
    """Coordinate-space descriptor for one homogeneous numeric leaf family.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, StructuredCandidateT]
        Structured search space owning the numeric leaf family.
    leaf_paths : tuple[LeafPath, ...]
        Leaf paths in canonical coordinate order.
    leaf_spaces : tuple[NumericLeafSpace, ...]
        Numeric leaf spaces aligned with ``leaf_paths``.
    """

    space: StructuredSearchSpace[BoundaryT, StructuredCandidateT]
    leaf_paths: tuple[LeafPath, ...]
    leaf_spaces: tuple[NumericLeafSpace, ...]

    def __post_init__(self) -> None:
        """Reject invalid descriptor construction."""
        normalized_leaf_paths = tuple(tuple(path) for path in self.leaf_paths)
        object.__setattr__(self, "leaf_paths", normalized_leaf_paths)
        object.__setattr__(self, "leaf_spaces", tuple(self.leaf_spaces))
        if len(self.leaf_paths) == 0:
            msg = "homogeneous numeric subspace descriptors require at least one leaf path"
            raise ValueError(msg)
        if len(self.leaf_paths) != len(self.leaf_spaces):
            msg = "leaf_paths and leaf_spaces must have matching lengths"
            raise ValueError(msg)

    @property
    def coordinate_bounds(self) -> tuple[tuple[float, float], ...]:
        """Return coordinate-space bounds in canonical leaf order."""
        return tuple(
            leaf_space.coordinate_bounds()
            for leaf_space in self.leaf_spaces
        )

    @property
    def coordinate_spans(self) -> tuple[float, ...]:
        """Return coordinate-space spans in canonical leaf order."""
        return tuple(
            coordinate_high - coordinate_low
            for coordinate_low, coordinate_high in self.coordinate_bounds
        )

    def coordinates_from_candidate(
        self,
        candidate: StructuredCandidateT,
    ) -> tuple[float, ...]:
        """Return one canonical candidate in descriptor coordinate space.

        Parameters
        ----------
        candidate : StructuredCandidateT
            Candidate to project into coordinate space.

        Returns
        -------
        tuple[float, ...]
            Coordinate vector aligned with ``leaf_paths``.

        Raises
        ------
        TypeError
            If a numeric leaf is not represented with canonical Python scalar
            types.
        """
        self.space.validate(candidate)
        coordinates: list[float] = []
        for path, leaf_space in zip(self.leaf_paths, self.leaf_spaces, strict=True):
            leaf_value = self.space.leaf_value_at_path(candidate, path)
            if isinstance(leaf_space, RealSpace):
                if type(leaf_value) is not float:
                    msg = "real-valued numeric subspaces require canonical float leaf values"
                    raise TypeError(msg)
                coordinates.append(leaf_space.to_coordinate(leaf_value))
                continue

            if type(leaf_value) is not int:
                msg = "integer-valued numeric subspaces require canonical integer leaf values"
                raise TypeError(msg)
            coordinates.append(leaf_space.to_coordinate(leaf_value))

        return tuple(coordinates)

    def candidate_from_coordinates(
        self,
        template_candidate: StructuredCandidateT,
        coordinates: Sequence[float],
    ) -> StructuredCandidateT:
        """Return one canonical candidate projected from coordinate space.

        Parameters
        ----------
        template_candidate : StructuredCandidateT
            Candidate supplying the non-numeric and inactive leaf structure.
        coordinates : Sequence[float]
            Coordinate vector aligned with ``leaf_paths``.

        Returns
        -------
        StructuredCandidateT
            Candidate with numeric leaf values projected from ``coordinates``.

        Raises
        ------
        ValueError
            If ``coordinates`` has the wrong dimensionality.
        """
        self.space.validate(template_candidate)
        normalized_coordinates = tuple(float(coordinate) for coordinate in coordinates)
        if len(normalized_coordinates) != len(self.leaf_paths):
            msg = "coordinate vector length must match the numeric subspace dimension"
            raise ValueError(msg)

        replacements = {
            path: leaf_space.project_coordinate(coordinate)
            for path, leaf_space, coordinate in zip(
                self.leaf_paths,
                self.leaf_spaces,
                normalized_coordinates,
                strict=True,
            )
        }
        candidate = self.space.replace_leaf_values(template_candidate, replacements)
        self.space.validate(candidate)
        return candidate

    def clip_coordinate_deltas(
        self,
        deltas: Sequence[float],
        *,
        max_coordinate_fraction: float,
    ) -> tuple[float, ...]:
        """Return coordinate deltas clipped to one per-leaf fraction bound.

        Parameters
        ----------
        deltas : Sequence[float]
            Coordinate deltas aligned with ``leaf_paths``.
        max_coordinate_fraction : float
            Maximum fraction of each coordinate span that any delta may take.

        Returns
        -------
        tuple[float, ...]
            Clipped coordinate deltas.

        Raises
        ------
        ValueError
            If ``deltas`` has the wrong dimensionality.
        """
        normalized_deltas = tuple(float(delta) for delta in deltas)
        if len(normalized_deltas) != len(self.leaf_paths):
            msg = "delta vector length must match the numeric subspace dimension"
            raise ValueError(msg)

        clipped: list[float] = []
        for delta, coordinate_span in zip(
            normalized_deltas,
            self.coordinate_spans,
            strict=True,
        ):
            max_delta = coordinate_span * max_coordinate_fraction
            clipped.append(min(max_delta, max(-max_delta, delta)))
        return tuple(clipped)

    def changed_leaf_paths(
        self,
        source_candidate: StructuredCandidateT,
        candidate: StructuredCandidateT,
    ) -> tuple[LeafPath, ...]:
        """Return the leaf paths whose canonical values differ.

        Parameters
        ----------
        source_candidate : StructuredCandidateT
            Reference candidate.
        candidate : StructuredCandidateT
            Candidate compared against ``source_candidate``.

        Returns
        -------
        tuple[LeafPath, ...]
            Leaf paths whose canonical values differ between the two
            candidates.
        """
        self.space.validate(source_candidate)
        self.space.validate(candidate)
        return tuple(
            path
            for path in self.leaf_paths
            if self.space.leaf_value_at_path(source_candidate, path)
            != self.space.leaf_value_at_path(candidate, path)
        )


def compile_homogeneous_numeric_subspace(
    space: StructuredSearchSpace[BoundaryT, StructuredCandidateT],
    *,
    leaf_paths: Sequence[LeafPath] | None = None,
) -> HomogeneousNumericSubspaceDescriptor[BoundaryT, StructuredCandidateT] | None:
    """Return one homogeneous numeric descriptor when the leaf family is valid.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, StructuredCandidateT]
        Structured search space whose leaves should be compiled.
    leaf_paths : Sequence[LeafPath] | None, default=None
        Optional explicit leaf family. ``None`` uses all leaf paths from
        ``space``.

    Returns
    -------
    HomogeneousNumericSubspaceDescriptor[BoundaryT, StructuredCandidateT] | None
        Numeric subspace descriptor when the selected leaves are non-empty,
        static, numeric, and homogeneous. Otherwise ``None``.
    """
    if not space.has_static_topology():
        return None

    normalized_leaf_paths = tuple(space.leaf_paths() if leaf_paths is None else tuple(tuple(path) for path in leaf_paths))
    if len(normalized_leaf_paths) == 0:
        return None

    leaf_spaces: list[NumericLeafSpace] = []
    for path in normalized_leaf_paths:
        leaf_space = space.leaf_space_at_path(path)
        if not isinstance(leaf_space, (RealSpace, IntegerSpace)):
            return None
        leaf_spaces.append(leaf_space)

    first_leaf_space = leaf_spaces[0]
    if any(type(leaf_space) is not type(first_leaf_space) for leaf_space in leaf_spaces[1:]):
        return None

    return HomogeneousNumericSubspaceDescriptor(
        space=space,
        leaf_paths=normalized_leaf_paths,
        leaf_spaces=tuple(leaf_spaces),
    )
