"""Public CSA bank-update policy objects."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

CSALocalUpdateMode = Literal["disabled", "normal"]
CSAFarUpdateMode = Literal["worst", "crowded_worst", "crowding_aware"]
CSANicheQualityMode = Literal["disabled", "mean", "best_mean"]


@dataclass(frozen=True, slots=True)
class CSANicheQualityPolicy:
    """Optional local niche-quality signal used by crowding-aware far updates.

    Parameters
    ----------
    mode : CSANicheQualityMode, default="disabled"
        Niche-quality statistic used by crowding-aware far updates.
    ratio : float, default=0.0
        Relative weight assigned to the niche-quality term.
    """

    mode: CSANicheQualityMode = "disabled"
    ratio: float = 0.0

    def __post_init__(self) -> None:
        """Reject invalid niche-quality policy configuration."""
        if self.mode not in ("disabled", "mean", "best_mean"):
            msg = "mode must be one of 'disabled', 'mean', or 'best_mean'"
            raise ValueError(msg)

        if self.ratio < 0.0:
            msg = "ratio must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CSABankUpdatePolicy:
    """CSA-specific near/far bank admission policy.

    Parameters
    ----------
    minimum_significant_score_gap_ratio : float, default=0.0
        Minimum score change relative to the bank score span for an update to
        count as significant. Appended entries are always significant; when
        both bank snapshots have zero score span, any nonzero score change is
        significant.
    local_update_mode : CSALocalUpdateMode, default="normal"
        Policy used for near-bank local updates.
    far_update_mode : CSAFarUpdateMode, default="worst"
        Policy used for far-bank replacement.
    crowding_penalty_ratio : float, default=0.75
        Weight assigned to the crowding penalty in crowding-aware modes.
    niche_quality_policy : CSANicheQualityPolicy, default=CSANicheQualityPolicy()
        Optional niche-quality policy used by crowding-aware far updates.
    """

    minimum_significant_score_gap_ratio: float = 0.0
    local_update_mode: CSALocalUpdateMode = "normal"
    far_update_mode: CSAFarUpdateMode = "worst"
    crowding_penalty_ratio: float = 0.75
    niche_quality_policy: CSANicheQualityPolicy = field(
        default_factory=CSANicheQualityPolicy,
    )

    def __post_init__(self) -> None:
        """Reject invalid update-policy configuration."""
        if isinstance(self.minimum_significant_score_gap_ratio, bool):
            msg = "minimum_significant_score_gap_ratio must be numeric"
            raise TypeError(msg)
        try:
            significance_ratio_is_finite = isfinite(
                self.minimum_significant_score_gap_ratio,
            )
        except TypeError as error:
            msg = "minimum_significant_score_gap_ratio must be numeric"
            raise TypeError(msg) from error
        if not significance_ratio_is_finite:
            msg = "minimum_significant_score_gap_ratio must be finite"
            raise ValueError(msg)
        if self.minimum_significant_score_gap_ratio < 0.0:
            msg = "minimum_significant_score_gap_ratio must be non-negative"
            raise ValueError(msg)

        if self.local_update_mode not in ("disabled", "normal"):
            msg = "local_update_mode must be one of 'disabled' or 'normal'"
            raise ValueError(msg)

        if self.far_update_mode not in ("worst", "crowded_worst", "crowding_aware"):
            msg = (
                "far_update_mode must be one of "
                "'worst', 'crowded_worst', or 'crowding_aware'"
            )
            raise ValueError(msg)

        if self.crowding_penalty_ratio < 0.0:
            msg = "crowding_penalty_ratio must be non-negative"
            raise ValueError(msg)
