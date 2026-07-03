"""Built-in permutation structured-space geometry implementations."""

from dataclasses import dataclass

from ..permutation import PermutationSpace, normalize_permutation_values
from ..types import SpaceCandidateValue
from .leaf import require_candidate_tuple
from .parts import StructuredDistanceParts


@dataclass(frozen=True, slots=True)
class PermutationSpaceGeometry:
    """Fast geometry for one permutation space.

    Parameters
    ----------
    space : PermutationSpace
        Permutation space whose mismatch-count geometry is exposed.
    """

    space: PermutationSpace

    def distance_parts(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> StructuredDistanceParts:
        """Return mismatch-count distance parts for two permutation candidates.

        Parameters
        ----------
        left : SpaceCandidateValue
            Left permutation candidate.
        right : SpaceCandidateValue
            Right permutation candidate.

        Returns
        -------
        StructuredDistanceParts
            Structured distance decomposition based on permutation mismatches.
        """
        overlap_squared_distance, shared_leaf_count, topology_mismatch_leaf_count = (
            self.distance_part_values(left, right)
        )
        return StructuredDistanceParts(
            overlap_squared_distance=overlap_squared_distance,
            shared_leaf_count=shared_leaf_count,
            topology_mismatch_leaf_count=topology_mismatch_leaf_count,
        )

    def distance_part_values(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw distance-part values for two permutation candidates."""
        left_tuple = require_candidate_tuple(
            value=left,
            message="permutation-space diversity requires canonical tuple candidates",
        )
        right_tuple = require_candidate_tuple(
            value=right,
            message="permutation-space diversity requires canonical tuple candidates",
        )
        left_permutation = normalize_permutation_values(left_tuple, size=self.space.size)
        right_permutation = normalize_permutation_values(right_tuple, size=self.space.size)

        mismatch_count = 0.0
        for index in range(self.space.size):
            if left_permutation[index] != right_permutation[index]:
                mismatch_count += 1.0
        return (mismatch_count, self.space.size, 0)
