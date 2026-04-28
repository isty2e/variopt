"""Distance-part payloads for structured-space geometry."""

from dataclasses import dataclass

from ...distance import require_valid_distance


@dataclass(frozen=True, slots=True)
class StructuredDistanceParts:
    """Canonical structured distance decomposition before metric collapse.

    Parameters
    ----------
    overlap_squared_distance : float
        Squared distance accumulated over shared leaves.
    shared_leaf_count : int
        Number of leaves compared under shared topology.
    topology_mismatch_leaf_count : int, default=0
        Number of leaves counted as full-penalty topology mismatches.
    """

    overlap_squared_distance: float
    shared_leaf_count: int
    topology_mismatch_leaf_count: int = 0

    def __post_init__(self) -> None:
        """Reject invalid distance-part payloads."""
        _ = require_valid_distance(self.overlap_squared_distance)
        if self.shared_leaf_count < 0:
            msg = "shared_leaf_count must be non-negative"
            raise ValueError(msg)
        if self.topology_mismatch_leaf_count < 0:
            msg = "topology_mismatch_leaf_count must be non-negative"
            raise ValueError(msg)
        if self.shared_leaf_count == 0 and self.overlap_squared_distance != 0.0:
            msg = "overlap_squared_distance requires at least one shared leaf"
            raise ValueError(msg)

    @property
    def total_leaf_count(self) -> int:
        """Return the total compared leaf count after topology decomposition."""
        return self.shared_leaf_count + self.topology_mismatch_leaf_count
