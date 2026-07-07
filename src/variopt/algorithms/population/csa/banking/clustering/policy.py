"""Public CSA clustering-policy configuration objects."""

from dataclasses import dataclass
from typing import Literal

CSAClusterUpdateMode = Literal["largest_cluster", "current_cluster"]


@dataclass(frozen=True, slots=True)
class CSAClusteringPolicy:
    """CSA-specific cluster-aware update and cutoff configuration.

    Parameters
    ----------
    enabled : bool, default=False
        Whether cluster-aware admission logic is enabled.
    cluster_cutoff_ratio : float, default=3.0
        Ratio used to derive the cluster-specific cutoff from the base cutoff.
    cluster_distance_ratio : float, default=1.5
        Ratio used to derive clustering distance from the base cutoff.
    update_mode : CSAClusterUpdateMode, default="largest_cluster"
        Cluster-aware replacement mode.
    """

    enabled: bool = False
    cluster_cutoff_ratio: float = 3.0
    cluster_distance_ratio: float = 1.5
    update_mode: CSAClusterUpdateMode = "largest_cluster"

    def __post_init__(self) -> None:
        """Reject invalid clustering-policy configuration."""
        if self.cluster_cutoff_ratio <= 0.0:
            msg = "cluster_cutoff_ratio must be positive"
            raise ValueError(msg)

        if self.cluster_distance_ratio <= 0.0:
            msg = "cluster_distance_ratio must be positive"
            raise ValueError(msg)

        if self.update_mode not in {"largest_cluster", "current_cluster"}:
            msg = "update_mode must be one of 'largest_cluster' or 'current_cluster'"
            raise ValueError(msg)
