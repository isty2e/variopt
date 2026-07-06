"""Structured search-space diversity metrics derived from space semantics."""

import math
from dataclasses import dataclass, field
from typing import Generic, TypeGuard, TypeVar

from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..distance import require_valid_distance
from ..spaces import StructuredSearchSpace
from ..spaces.geometry.compile import (
    compile_structured_geometry,
    generic_distance_parts,
)
from ..spaces.geometry.composites import (
    DistancePartValuesGeometry,
    ValidatedDistancePartValuesGeometry,
    geometry_has_distance_part_values,
    geometry_has_validated_distance_part_values,
)
from ..spaces.geometry.contracts import StructuredSpaceGeometry
from ..spaces.types import SpaceBoundaryValue, SpaceCandidateValue
from .base import DiversityMetric

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)
MetricCandidateT = TypeVar("MetricCandidateT")


@dataclass(frozen=True, slots=True)
class StructuredSpaceDiversityMetric(FrozenGenericSlotsCompat,
    DiversityMetric[CandidateT],
    Generic[BoundaryT, CandidateT],
):
    """Leaf-wise normalized diversity metric over one structured search space.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured search space whose geometry defines the diversity metric.
    """

    space: StructuredSearchSpace[BoundaryT, CandidateT]
    geometry: StructuredSpaceGeometry | None = field(init=False, repr=False)
    part_values_geometry: DistancePartValuesGeometry | None = field(
        init=False,
        repr=False,
    )
    validated_part_values_geometry: ValidatedDistancePartValuesGeometry | None = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Compile and cache any built-in structured geometry once."""
        geometry = compile_structured_geometry(self.space)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(
            self,
            "part_values_geometry",
            (
                geometry
                if geometry is not None and geometry_has_distance_part_values(geometry)
                else None
            ),
        )
        object.__setattr__(
            self,
            "validated_part_values_geometry",
            (
                geometry
                if geometry is not None
                and geometry_has_validated_distance_part_values(geometry)
                else None
            ),
        )

    @override
    def distance(self, left: CandidateT, right: CandidateT) -> float:
        """Return the RMS normalized leaf distance between two candidates.

        Parameters
        ----------
        left : CandidateT
            Left canonical candidate.
        right : CandidateT
            Right canonical candidate.

        Returns
        -------
        float
            RMS normalized structured distance.
        """
        part_values_geometry = self.part_values_geometry
        if part_values_geometry is not None:
            (
                overlap_squared_distance,
                shared_leaf_count,
                topology_mismatch_leaf_count,
            ) = part_values_geometry.distance_part_values(left, right)
            return _distance_from_part_values(
                overlap_squared_distance=overlap_squared_distance,
                shared_leaf_count=shared_leaf_count,
                topology_mismatch_leaf_count=topology_mismatch_leaf_count,
            )
        geometry = self.geometry
        if geometry is None:
            parts = generic_distance_parts(
                self.space,
                left,
                right,
            )
            return _distance_from_part_values(
                overlap_squared_distance=parts.overlap_squared_distance,
                shared_leaf_count=parts.shared_leaf_count,
                topology_mismatch_leaf_count=parts.topology_mismatch_leaf_count,
            )
        parts = geometry.distance_parts(left, right)
        return _distance_from_part_values(
            overlap_squared_distance=parts.overlap_squared_distance,
            shared_leaf_count=parts.shared_leaf_count,
            topology_mismatch_leaf_count=parts.topology_mismatch_leaf_count,
        )


def supports_validated_structured_distance(
    metric: DiversityMetric[MetricCandidateT],
) -> TypeGuard[StructuredSpaceDiversityMetric[SpaceBoundaryValue, SpaceCandidateValue]]:
    """Return whether ``metric`` exposes the internal validated path."""
    return isinstance(metric, StructuredSpaceDiversityMetric)


def structured_distance_between_validated_candidates(
    metric: StructuredSpaceDiversityMetric[BoundaryT, CandidateT],
    left: CandidateT,
    right: CandidateT,
) -> float:
    """Return structured distance for candidates validated by ``metric.space``.

    This internal algebra is intentionally not part of the facade-level
    diversity contract. Callers must own evidence that both candidates have
    already crossed the matching space validation boundary.
    """
    validated_part_values_geometry = metric.validated_part_values_geometry
    if validated_part_values_geometry is not None:
        (
            overlap_squared_distance,
            shared_leaf_count,
            topology_mismatch_leaf_count,
        ) = validated_part_values_geometry.distance_part_values_for_validated_candidates(
            left,
            right,
        )
        return _distance_from_part_values(
            overlap_squared_distance=overlap_squared_distance,
            shared_leaf_count=shared_leaf_count,
            topology_mismatch_leaf_count=topology_mismatch_leaf_count,
        )

    return metric.distance(left, right)


def _distance_from_part_values(
    *,
    overlap_squared_distance: float,
    shared_leaf_count: int,
    topology_mismatch_leaf_count: int,
) -> float:
    """Return the RMS structured distance from raw distance-part values."""
    total_leaf_count = shared_leaf_count + topology_mismatch_leaf_count
    if total_leaf_count == 0:
        msg = "structured diversity metric requires at least one leaf path"
        raise ValueError(msg)
    return require_valid_distance(
        math.sqrt(
            require_valid_distance(
                overlap_squared_distance + float(topology_mismatch_leaf_count)
            )
            / total_leaf_count,
        ),
    )
