"""Named CSA preset defaults for boundary-level profile normalization."""

from dataclasses import dataclass

from ..banking.clustering import CSAClusteringPolicy
from ..banking.growth import CSABankGrowthPolicy
from ..banking.update import CSABankUpdatePolicy
from ..progression.cutoff.policy import CSACutoffSchedule
from ..progression.refresh import CSARefreshPolicy
from ..scoring.acceptance import CSAAcceptancePolicy
from ..scoring.model import CSAScoreModelDefaults


@dataclass(frozen=True, slots=True)
class CSAProfileDefaults:
    """Resolved default semantic knobs for one named CSA profile.

    Parameters
    ----------
    seed_count : int
        Number of active seeds used by the profile.
    initial_new_bank_cut : int
        Initial new-bank cutoff used when seeding the run.
    random_seed_mode : int
        Legacy-compatible random-seed mode selector.
    weighted_partner_selection : bool
        Whether parent selection is weighted by score.
    max_bank_capacity : int | None
        Optional maximum bank capacity for adaptive bank growth.
    cutoff_schedule : CSACutoffSchedule
        Cutoff schedule defaults for progression control.
    acceptance_policy : CSAAcceptancePolicy
        Acceptance defaults for score-based admissions.
    clustering_policy : CSAClusteringPolicy
        Clustering defaults for bank update logic.
    growth_policy : CSABankGrowthPolicy
        Bank growth defaults.
    refresh_policy : CSARefreshPolicy
        Refresh defaults used during progression.
    restart_lite : bool
        Whether lightweight restart behavior is enabled.
    cycle_limit : int
        Progression cycle limit for the profile.
    update_policy : CSABankUpdatePolicy
        Bank update defaults for near/far admission logic.
    score_model_defaults : CSAScoreModelDefaults
        Score-model defaults for acceptance and adaptation.
    """

    seed_count: int
    initial_new_bank_cut: int
    random_seed_mode: int
    weighted_partner_selection: bool
    max_bank_capacity: int | None
    cutoff_schedule: CSACutoffSchedule
    acceptance_policy: CSAAcceptancePolicy
    clustering_policy: CSAClusteringPolicy
    growth_policy: CSABankGrowthPolicy
    refresh_policy: CSARefreshPolicy
    restart_lite: bool
    cycle_limit: int
    update_policy: CSABankUpdatePolicy
    score_model_defaults: CSAScoreModelDefaults


def profile_defaults_for_preset(preset: str) -> CSAProfileDefaults:
    """Resolve the default semantic knobs for one named preset.

    Parameters
    ----------
    preset : str
        Public preset name.

    Returns
    -------
    CSAProfileDefaults
        Fully resolved defaults implied by ``preset``.

    Raises
    ------
    ValueError
        If ``preset`` is not supported.
    """
    if preset == "variopt":
        return CSAProfileDefaults(
            seed_count=5,
            initial_new_bank_cut=1,
            random_seed_mode=0,
            weighted_partner_selection=False,
            max_bank_capacity=24,
            cutoff_schedule=CSACutoffSchedule(
                initial_distance_divisor=2.0,
                minimum_distance_divisor=5.0,
                reduction_factor=0.983912,
                stagnation_update_limit=10,
                cycle_increment_requires_minimum_cutoff=False,
            ),
            acceptance_policy=CSAAcceptancePolicy(),
            clustering_policy=CSAClusteringPolicy(),
            growth_policy=CSABankGrowthPolicy(),
            refresh_policy=CSARefreshPolicy(),
            restart_lite=False,
            cycle_limit=3,
            update_policy=CSABankUpdatePolicy(
                minimum_significant_score_gap=0.001,
                local_update_mode="normal",
                far_update_mode="crowding_aware",
            ),
            score_model_defaults=CSAScoreModelDefaults(),
        )

    if preset == "joung_2018":
        return CSAProfileDefaults(
            seed_count=6,
            initial_new_bank_cut=1,
            random_seed_mode=0,
            weighted_partner_selection=False,
            max_bank_capacity=None,
            cutoff_schedule=CSACutoffSchedule(
                initial_distance_divisor=2.0,
                minimum_distance_divisor=5.0,
                reduction_factor=0.983912,
                stagnation_update_limit=10,
                cycle_increment_requires_minimum_cutoff=False,
            ),
            acceptance_policy=CSAAcceptancePolicy(),
            clustering_policy=CSAClusteringPolicy(),
            growth_policy=CSABankGrowthPolicy(),
            refresh_policy=CSARefreshPolicy(),
            restart_lite=False,
            cycle_limit=3,
            update_policy=CSABankUpdatePolicy(
                minimum_significant_score_gap=0.001,
                local_update_mode="normal",
            ),
            score_model_defaults=CSAScoreModelDefaults(),
        )

    msg = f"unsupported CSA preset: {preset}"
    raise ValueError(msg)
