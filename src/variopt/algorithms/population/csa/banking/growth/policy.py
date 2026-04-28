"""Public CSA bank-growth policy objects."""

from dataclasses import dataclass
from typing import Literal

CSAEnergyGapUpdateMode = Literal[
    "fixed",
    "max_score_ratio",
    "multiplicative_decay",
]


@dataclass(frozen=True, slots=True)
class CSABankGrowthPolicy:
    """CSA-specific adaptive bank growth and shrink policy.

    Parameters
    ----------
    enabled : bool, default=False
        Whether adaptive bank growth is enabled.
    maximum_capacity : int | None, optional
        Maximum bank capacity reachable under growth.
    initial_energy_gap_limit : float, default=300.0
        Initial energy-gap threshold used by the growth logic.
    energy_gap_update_mode : CSAEnergyGapUpdateMode, default="fixed"
        Update rule for the energy-gap threshold.
    energy_gap_update_factor : float, default=1.0
        Update factor used by the selected threshold rule.
    maximum_growth_per_generation : int, default=9999
        Maximum number of growth events allowed per generation.
    require_distance_cutoff : bool, default=True
        Whether growth requires an active distance cutoff.
    """

    enabled: bool = False
    maximum_capacity: int | None = None
    initial_energy_gap_limit: float = 300.0
    energy_gap_update_mode: CSAEnergyGapUpdateMode = "fixed"
    energy_gap_update_factor: float = 1.0
    maximum_growth_per_generation: int = 9999
    require_distance_cutoff: bool = True

    def __post_init__(self) -> None:
        """Reject invalid growth-policy definitions."""
        if self.maximum_capacity is not None and self.maximum_capacity <= 0:
            msg = "maximum_capacity must be positive"
            raise ValueError(msg)

        if self.initial_energy_gap_limit < 0.0:
            msg = "initial_energy_gap_limit must be non-negative"
            raise ValueError(msg)

        if self.energy_gap_update_mode not in {
            "fixed",
            "max_score_ratio",
            "multiplicative_decay",
        }:
            msg = (
                "energy_gap_update_mode must be one of 'fixed', "
                "'max_score_ratio', or 'multiplicative_decay'"
            )
            raise ValueError(msg)

        if self.energy_gap_update_factor <= 0.0:
            msg = "energy_gap_update_factor must be positive"
            raise ValueError(msg)

        if self.maximum_growth_per_generation < 0:
            msg = "maximum_growth_per_generation must be non-negative"
            raise ValueError(msg)

        if self.enabled and self.maximum_capacity is None:
            msg = "enabled growth policies must declare maximum_capacity"
            raise ValueError(msg)
