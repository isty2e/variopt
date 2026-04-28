"""Boundary policy for CSA proposal adaptation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CSAProposalPolicy:
    """Boundary-level policy for history-aware CSA proposal adaptation.

    The proposal subdomain governs adaptive memory used to bias child
    generation. It does not alter banking, cutoff, or selection semantics.

    Parameters
    ----------
    enabled : bool, default=False
        Whether proposal adaptation is active.
    family_bias_strength : float, default=1.0
        Strength of family-level weighting.
    leaf_bias_strength : float, default=1.0
        Strength of leaf-level weighting.
    local_displacement_leaf_bias_strength : float, default=0.0
        Strength of local-displacement leaf weighting.
    score_decay : float, default=0.95
        Exponential decay applied to adaptive proposal credit.
    minimum_family_weight : float, default=1e-3
        Minimum family weight after adaptation.
    minimum_leaf_weight : float, default=1e-3
        Minimum leaf weight after adaptation.
    numeric_covariance_strength : float, default=0.0
        Strength of numeric covariance adaptation.
    numeric_covariance_min_observations : int, default=4
        Minimum observations required before covariance adaptation activates.
    numeric_covariance_ridge : float, default=1e-6
        Ridge term used for covariance regularization.
    local_search_base_budget : int, default=2
        Base per-proposal local-search budget.
    local_search_max_budget : int, default=8
        Maximum per-proposal local-search budget.
    local_search_disable_failure_streak : int, default=3
        Failure streak threshold that disables local search temporarily.
    local_search_failure_cooldown_updates : int, default=2
        Cooldown update count after local-search disablement.
    """

    enabled: bool = False
    family_bias_strength: float = 1.0
    leaf_bias_strength: float = 1.0
    local_displacement_leaf_bias_strength: float = 0.0
    score_decay: float = 0.95
    minimum_family_weight: float = 1e-3
    minimum_leaf_weight: float = 1e-3
    numeric_covariance_strength: float = 0.0
    numeric_covariance_min_observations: int = 4
    numeric_covariance_ridge: float = 1e-6
    local_search_base_budget: int = 2
    local_search_max_budget: int = 8
    local_search_disable_failure_streak: int = 3
    local_search_failure_cooldown_updates: int = 2

    def __post_init__(self) -> None:
        """Reject invalid proposal-adaptation boundary settings."""
        if self.family_bias_strength < 0.0:
            msg = "family_bias_strength must be non-negative"
            raise ValueError(msg)

        if self.leaf_bias_strength < 0.0:
            msg = "leaf_bias_strength must be non-negative"
            raise ValueError(msg)

        if self.local_displacement_leaf_bias_strength < 0.0:
            msg = "local_displacement_leaf_bias_strength must be non-negative"
            raise ValueError(msg)

        if self.score_decay <= 0.0 or self.score_decay > 1.0:
            msg = "score_decay must lie in (0.0, 1.0]"
            raise ValueError(msg)

        if self.minimum_family_weight <= 0.0:
            msg = "minimum_family_weight must be positive"
            raise ValueError(msg)

        if self.minimum_leaf_weight <= 0.0:
            msg = "minimum_leaf_weight must be positive"
            raise ValueError(msg)

        if self.numeric_covariance_strength < 0.0:
            msg = "numeric_covariance_strength must be non-negative"
            raise ValueError(msg)

        if self.numeric_covariance_min_observations <= 0:
            msg = "numeric_covariance_min_observations must be positive"
            raise ValueError(msg)

        if self.numeric_covariance_ridge < 0.0:
            msg = "numeric_covariance_ridge must be non-negative"
            raise ValueError(msg)

        if self.local_search_base_budget <= 0:
            msg = "local_search_base_budget must be positive"
            raise ValueError(msg)

        if self.local_search_max_budget <= 0:
            msg = "local_search_max_budget must be positive"
            raise ValueError(msg)

        if self.local_search_max_budget < self.local_search_base_budget:
            msg = "local_search_max_budget must be at least local_search_base_budget"
            raise ValueError(msg)

        if self.local_search_disable_failure_streak <= 0:
            msg = "local_search_disable_failure_streak must be positive"
            raise ValueError(msg)

        if self.local_search_failure_cooldown_updates < 0:
            msg = "local_search_failure_cooldown_updates must be non-negative"
            raise ValueError(msg)
