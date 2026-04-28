"""Structured search-space diversity metrics derived from space semantics."""

import math
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..distance import require_valid_distance
from ..spaces import StructuredSearchSpace
from ..spaces.geometry.compile import (
    compile_structured_geometry,
    generic_distance_parts,
)
from ..spaces.geometry.contracts import StructuredSpaceGeometry
from ..spaces.geometry.parts import StructuredDistanceParts
from ..spaces.types import SpaceCandidateValue
from .base import DiversityMetric

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)


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

    def __post_init__(self) -> None:
        """Compile and cache any built-in structured geometry once."""
        object.__setattr__(
            self,
            "geometry",
            compile_structured_geometry(self.space),
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
        if self.geometry is None:
            parts = generic_distance_parts(
                self.space,
                left,
                right,
            )
        else:
            parts = self.geometry.distance_parts(left, right)
        if parts.total_leaf_count == 0:
            msg = "structured diversity metric requires at least one leaf path"
            raise ValueError(msg)
        return require_valid_distance(
            math.sqrt(
                _collapse_distance_parts_to_full_penalty_squared_distance(
                    parts,
                )
                / parts.total_leaf_count,
            ),
        )


def _collapse_distance_parts_to_full_penalty_squared_distance(
    distance_parts: StructuredDistanceParts,
) -> float:
    """Collapse structured distance parts under the current RMS penalty policy."""
    return require_valid_distance(
        distance_parts.overlap_squared_distance
        + float(distance_parts.topology_mismatch_leaf_count),
    )
