"""Continuous coordinate projections for structured search spaces."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..base import SearchSpace
from ..scalar import RealSpace
from ..structured import LeafPath, StructuredSearchSpace
from ..types import SpaceCandidateValue

BoundaryT = TypeVar("BoundaryT")
StructuredCandidateT = TypeVar("StructuredCandidateT", bound=SpaceCandidateValue)


@dataclass(frozen=True, slots=True)
class ContinuousStructuredSpaceCodec(FrozenGenericSlotsCompat, Generic[BoundaryT, StructuredCandidateT]):
    """Coordinate-space codec for continuous structured candidates.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, StructuredCandidateT]
        Structured search space owning the continuous leaf family.
    leaf_paths : tuple[LeafPath, ...]
        Real-valued leaf paths in canonical coordinate order.
    leaf_spaces : tuple[RealSpace, ...]
        Real-valued leaf spaces aligned with ``leaf_paths``.
    """

    space: StructuredSearchSpace[BoundaryT, StructuredCandidateT]
    leaf_paths: tuple[LeafPath, ...]
    leaf_spaces: tuple[RealSpace, ...]

    @classmethod
    def from_space(
        cls,
        space: SearchSpace[BoundaryT, StructuredCandidateT],
    ) -> "ContinuousStructuredSpaceCodec[BoundaryT, StructuredCandidateT]":
        """Normalize one search space into the continuous structured codec.

        Parameters
        ----------
        space : SearchSpace[BoundaryT, StructuredCandidateT]
            Search space supplied to a continuous optimizer.

        Returns
        -------
        ContinuousStructuredSpaceCodec[BoundaryT, StructuredCandidateT]
            Codec over the structured real-valued leaf family.

        Raises
        ------
        TypeError
            If ``space`` is not a static structured space with only
            :class:`RealSpace` leaves.
        ValueError
            If ``space`` does not expose any editable leaves.
        """
        if not isinstance(space, StructuredSearchSpace):
            msg = (
                "continuous optimizer codecs require a structured search space "
                "with RealSpace leaves"
            )
            raise TypeError(msg)

        structured_space = space
        if not structured_space.has_static_topology():
            msg = (
                "continuous optimizer codecs require a structured search space "
                "with static topology"
            )
            raise TypeError(msg)
        leaf_paths = structured_space.leaf_paths()
        if len(leaf_paths) == 0:
            msg = "continuous optimizer codecs require at least one editable RealSpace leaf"
            raise ValueError(msg)

        leaf_spaces: list[RealSpace] = []
        for path in leaf_paths:
            leaf_space = structured_space.leaf_space_at_path(path)
            if not isinstance(leaf_space, RealSpace):
                msg = (
                    "continuous optimizer codecs require every structured leaf "
                    "to be a RealSpace"
                )
                raise TypeError(msg)
            leaf_spaces.append(leaf_space)

        return cls(
            space=structured_space,
            leaf_paths=leaf_paths,
            leaf_spaces=tuple(leaf_spaces),
        )

    @property
    def coordinate_bounds(self) -> tuple[tuple[float, float], ...]:
        """Return coordinate-space bounds in canonical leaf order."""
        return tuple(
            leaf_space.coordinate_bounds()
            for leaf_space in self.leaf_spaces
        )

    def coordinates_from_candidate(
        self,
        candidate: StructuredCandidateT,
    ) -> tuple[float, ...]:
        """Return one canonical candidate in optimizer coordinate space.

        Parameters
        ----------
        candidate : StructuredCandidateT
            Candidate to project into optimizer coordinates.

        Returns
        -------
        tuple[float, ...]
            Coordinate vector aligned with ``leaf_paths``.

        Raises
        ------
        TypeError
            If any real leaf is not represented with canonical Python ``float``
            values.
        """
        self.space.validate(candidate)
        coordinates: list[float] = []
        for path, leaf_space in zip(self.leaf_paths, self.leaf_spaces, strict=True):
            leaf_value = self.space.leaf_value_at_path(candidate, path)
            if type(leaf_value) is not float:
                msg = "continuous structured codec requires canonical float leaf values"
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
            Candidate supplying the non-real and inactive leaf structure.
        coordinates : Sequence[float]
            Coordinate vector aligned with ``leaf_paths``.

        Returns
        -------
        StructuredCandidateT
            Candidate with real leaves projected from ``coordinates``.

        Raises
        ------
        ValueError
            If ``coordinates`` has the wrong dimensionality.
        """
        self.space.validate(template_candidate)
        normalized_coordinates = tuple(float(coordinate) for coordinate in coordinates)
        if len(normalized_coordinates) != len(self.leaf_paths):
            msg = "coordinate vector length does not match the structured leaf count"
            raise ValueError(msg)

        replacements = {
            path: leaf_space.project_coordinate(float(coordinate))
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
