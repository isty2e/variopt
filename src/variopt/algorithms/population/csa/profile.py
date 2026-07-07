"""Public CSA configuration profiles."""

from dataclasses import dataclass
from typing import Generic, Literal

from typing_extensions import Self, override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....typevars import CandidateT
from ...profile import AlgorithmProfile
from .banking.clustering import CSAClusteringPolicy
from .banking.growth import CSABankGrowthPolicy
from .banking.update import CSABankUpdatePolicy
from .generation.perturbation import CSAPerturbationSchedule
from .generation.proposal import CSAProposalPolicy
from .presets.profile_defaults import profile_defaults_for_preset
from .progression.cutoff.policy import CSACutoffSchedule
from .progression.refresh import CSARefreshPolicy
from .scoring.acceptance import CSAAcceptancePolicy
from .scoring.model import CSAScoreModel
from .selection.policy import validate_random_seed_mode

CSAPresetName = Literal[
    "variopt",
    "joung_2018",
]


@dataclass(frozen=True, slots=True)
class CSAResolvedProfile(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Canonical CSA configuration used by optimizer internals.

    Parameters
    ----------
    perturbation_schedule : CSAPerturbationSchedule[CandidateT]
        Family schedule used to generate new CSA proposals.
    proposal_policy : CSAProposalPolicy
        Adaptive proposal-weighting policy applied across generations.
    seed_count : int
        Number of seeds tracked in each CSA generation.
    initial_new_bank_cut : int
        Initial cutoff applied before adaptive cutoff updates begin.
    random_seed_mode : int
        Legacy-compatible seed-selection mode identifier.
    weighted_partner_selection : bool
        Whether partner sampling is weighted by CSA scores.
    max_bank_capacity : int | None
        Optional staged bank ceiling. ``None`` keeps the bank fixed.
    cutoff_schedule : CSACutoffSchedule
        Schedule object that initializes and updates the CSA cutoff state.
    acceptance_policy : CSAAcceptancePolicy
        Acceptance policy used when scoring and admitting observations.
    clustering_policy : CSAClusteringPolicy
        Optional cluster-aware admission policy for the bank.
    growth_policy : CSABankGrowthPolicy
        Optional adaptive bank-growth policy.
    refresh_policy : CSARefreshPolicy
        Refresh and restart policy applied at run boundaries.
    restart_lite : bool
        Whether lightweight restarts are enabled after convergence events.
    cycle_limit : int
        Maximum number of cycles before staged lifecycle actions fire.
    update_policy : CSABankUpdatePolicy
        Bank-update policy used to admit or reject observations.
    score_model : CSAScoreModel[CandidateT]
        Score model used to map objective values to CSA scores.
    """

    perturbation_schedule: CSAPerturbationSchedule[CandidateT]
    proposal_policy: CSAProposalPolicy
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
    score_model: CSAScoreModel[CandidateT]


@dataclass(frozen=True, slots=True)
class CSAProfile(
    FrozenGenericSlotsCompat,
    AlgorithmProfile[CSAResolvedProfile[CandidateT]],
    Generic[CandidateT],
):
    """Boundary-level CSA configuration with named CSA presets.

    Parameters
    ----------
    perturbation_schedule : CSAPerturbationSchedule[CandidateT] | None, default=None
        Explicit perturbation schedule override. Required before the profile can
        resolve.
    proposal_policy : CSAProposalPolicy | None, default=None
        Optional proposal-weighting policy override.
    preset : {"variopt", "joung_2018"}, default="variopt"
        Baseline preset from which omitted fields inherit defaults.
    seed_count : int | None, default=None
        Optional override for the number of tracked seeds.
    initial_new_bank_cut : int | None, default=None
        Optional override for the initial cutoff threshold.
    random_seed_mode : int | None, default=None
        Optional override for legacy-compatible seed-selection mode.
    weighted_partner_selection : bool | None, default=None
        Optional override for weighted partner selection.
    max_bank_capacity : int | None, default=None
        Optional staged bank-capacity ceiling.
    cutoff_schedule : CSACutoffSchedule | None, default=None
        Optional override for cutoff initialization and updates.
    acceptance_policy : CSAAcceptancePolicy | None, default=None
        Optional override for CSA acceptance semantics.
    clustering_policy : CSAClusteringPolicy | None, default=None
        Optional override for cluster-aware bank admission.
    growth_policy : CSABankGrowthPolicy | None, default=None
        Optional override for adaptive bank growth.
    refresh_policy : CSARefreshPolicy | None, default=None
        Optional override for refresh and restart handling.
    restart_lite : bool | None, default=None
        Optional override for lightweight restart behavior.
    cycle_limit : int | None, default=None
        Optional override for the staged lifecycle cycle limit.
    update_policy : CSABankUpdatePolicy | None, default=None
        Optional override for bank-update semantics.
    score_model : CSAScoreModel[CandidateT] | None, default=None
        Optional override for score computation.
    """

    perturbation_schedule: CSAPerturbationSchedule[CandidateT] | None = None
    proposal_policy: CSAProposalPolicy | None = None
    preset: CSAPresetName = "variopt"
    seed_count: int | None = None
    initial_new_bank_cut: int | None = None
    random_seed_mode: int | None = None
    weighted_partner_selection: bool | None = None
    max_bank_capacity: int | None = None
    cutoff_schedule: CSACutoffSchedule | None = None
    acceptance_policy: CSAAcceptancePolicy | None = None
    clustering_policy: CSAClusteringPolicy | None = None
    growth_policy: CSABankGrowthPolicy | None = None
    refresh_policy: CSARefreshPolicy | None = None
    restart_lite: bool | None = None
    cycle_limit: int | None = None
    update_policy: CSABankUpdatePolicy | None = None
    score_model: CSAScoreModel[CandidateT] | None = None

    def __post_init__(self) -> None:
        """Validate boundary-level CSA profile fields.

        Raises
        ------
        ValueError
            Raised when a numeric override is outside the supported range.
        """
        _ = profile_defaults_for_preset(self.preset)

        if self.seed_count is not None and self.seed_count <= 0:
            msg = "seed_count must be positive"
            raise ValueError(msg)

        if self.initial_new_bank_cut is not None and self.initial_new_bank_cut < 0:
            msg = "initial_new_bank_cut must be non-negative"
            raise ValueError(msg)

        if self.random_seed_mode is not None:
            validate_random_seed_mode(self.random_seed_mode)

        if self.max_bank_capacity is not None and self.max_bank_capacity <= 0:
            msg = "max_bank_capacity must be positive"
            raise ValueError(msg)

        if self.cycle_limit is not None and self.cycle_limit < 0:
            msg = "cycle_limit must be non-negative"
            raise ValueError(msg)

    @classmethod
    def variopt(
        cls,
        *,
        perturbation_schedule: CSAPerturbationSchedule[CandidateT],
        proposal_policy: CSAProposalPolicy | None = None,
        seed_count: int | None = None,
        initial_new_bank_cut: int | None = None,
        random_seed_mode: int | None = None,
        weighted_partner_selection: bool | None = None,
        max_bank_capacity: int | None = None,
        cutoff_schedule: CSACutoffSchedule | None = None,
        acceptance_policy: CSAAcceptancePolicy | None = None,
        clustering_policy: CSAClusteringPolicy | None = None,
        growth_policy: CSABankGrowthPolicy | None = None,
        refresh_policy: CSARefreshPolicy | None = None,
        restart_lite: bool | None = None,
        cycle_limit: int | None = None,
        update_policy: CSABankUpdatePolicy | None = None,
        score_model: CSAScoreModel[CandidateT] | None = None,
    ) -> Self:
        """Build the current variopt house CSA profile.

        Parameters
        ----------
        perturbation_schedule : CSAPerturbationSchedule[CandidateT]
            Explicit perturbation schedule used by the resulting profile.
        proposal_policy : CSAProposalPolicy | None, default=None
            Optional proposal-weighting policy override.
        seed_count : int | None, default=None
            Optional override for the number of tracked seeds.
        initial_new_bank_cut : int | None, default=None
            Optional override for the initial cutoff threshold.
        random_seed_mode : int | None, default=None
            Optional override for legacy-compatible seed-selection mode.
        weighted_partner_selection : bool | None, default=None
            Optional override for weighted partner selection.
        max_bank_capacity : int | None, default=None
            Optional staged bank-capacity ceiling.
        cutoff_schedule : CSACutoffSchedule | None, default=None
            Optional override for cutoff initialization and updates.
        acceptance_policy : CSAAcceptancePolicy | None, default=None
            Optional override for acceptance semantics.
        clustering_policy : CSAClusteringPolicy | None, default=None
            Optional override for cluster-aware bank admission.
        growth_policy : CSABankGrowthPolicy | None, default=None
            Optional override for adaptive bank growth.
        refresh_policy : CSARefreshPolicy | None, default=None
            Optional override for refresh and restart semantics.
        restart_lite : bool | None, default=None
            Optional override for lightweight restart behavior.
        cycle_limit : int | None, default=None
            Optional override for the staged lifecycle cycle limit.
        update_policy : CSABankUpdatePolicy | None, default=None
            Optional override for bank-update semantics.
        score_model : CSAScoreModel[CandidateT] | None, default=None
            Optional override for score computation.

        Returns
        -------
        Self
            CSA profile rooted in the ``variopt`` preset.
        """
        return cls(
            perturbation_schedule=perturbation_schedule,
            proposal_policy=proposal_policy,
            preset="variopt",
            seed_count=seed_count,
            initial_new_bank_cut=initial_new_bank_cut,
            random_seed_mode=random_seed_mode,
            weighted_partner_selection=weighted_partner_selection,
            max_bank_capacity=max_bank_capacity,
            cutoff_schedule=cutoff_schedule,
            acceptance_policy=acceptance_policy,
            clustering_policy=clustering_policy,
            growth_policy=growth_policy,
            refresh_policy=refresh_policy,
            restart_lite=restart_lite,
            cycle_limit=cycle_limit,
            update_policy=update_policy,
            score_model=score_model,
        )

    @classmethod
    def joung_2018(
        cls,
        *,
        perturbation_schedule: CSAPerturbationSchedule[CandidateT],
        proposal_policy: CSAProposalPolicy | None = None,
        seed_count: int | None = None,
        initial_new_bank_cut: int | None = None,
        random_seed_mode: int | None = None,
        weighted_partner_selection: bool | None = None,
        max_bank_capacity: int | None = None,
        cutoff_schedule: CSACutoffSchedule | None = None,
        acceptance_policy: CSAAcceptancePolicy | None = None,
        clustering_policy: CSAClusteringPolicy | None = None,
        growth_policy: CSABankGrowthPolicy | None = None,
        refresh_policy: CSARefreshPolicy | None = None,
        restart_lite: bool | None = None,
        cycle_limit: int | None = None,
        update_policy: CSABankUpdatePolicy | None = None,
        score_model: CSAScoreModel[CandidateT] | None = None,
    ) -> Self:
        """Build a profile aligned to Joung 2018 generic CSA settings.

        Parameters
        ----------
        perturbation_schedule : CSAPerturbationSchedule[CandidateT]
            Explicit perturbation schedule used by the resulting profile.
        proposal_policy : CSAProposalPolicy | None, default=None
            Optional proposal-weighting policy override.
        seed_count : int | None, default=None
            Optional override for the number of tracked seeds.
        initial_new_bank_cut : int | None, default=None
            Optional override for the initial cutoff threshold.
        random_seed_mode : int | None, default=None
            Optional override for legacy-compatible seed-selection mode.
        weighted_partner_selection : bool | None, default=None
            Optional override for weighted partner selection.
        max_bank_capacity : int | None, default=None
            Optional staged bank-capacity ceiling.
        cutoff_schedule : CSACutoffSchedule | None, default=None
            Optional override for cutoff initialization and updates.
        acceptance_policy : CSAAcceptancePolicy | None, default=None
            Optional override for acceptance semantics.
        clustering_policy : CSAClusteringPolicy | None, default=None
            Optional override for cluster-aware bank admission.
        growth_policy : CSABankGrowthPolicy | None, default=None
            Optional override for adaptive bank growth.
        refresh_policy : CSARefreshPolicy | None, default=None
            Optional override for refresh and restart semantics.
        restart_lite : bool | None, default=None
            Optional override for lightweight restart behavior.
        cycle_limit : int | None, default=None
            Optional override for the staged lifecycle cycle limit.
        update_policy : CSABankUpdatePolicy | None, default=None
            Optional override for bank-update semantics.
        score_model : CSAScoreModel[CandidateT] | None, default=None
            Optional override for score computation.

        Returns
        -------
        Self
            CSA profile rooted in the ``joung_2018`` preset.
        """
        return cls(
            perturbation_schedule=perturbation_schedule,
            proposal_policy=proposal_policy,
            preset="joung_2018",
            seed_count=seed_count,
            initial_new_bank_cut=initial_new_bank_cut,
            random_seed_mode=random_seed_mode,
            weighted_partner_selection=weighted_partner_selection,
            max_bank_capacity=max_bank_capacity,
            cutoff_schedule=cutoff_schedule,
            acceptance_policy=acceptance_policy,
            clustering_policy=clustering_policy,
            growth_policy=growth_policy,
            refresh_policy=refresh_policy,
            restart_lite=restart_lite,
            cycle_limit=cycle_limit,
            update_policy=update_policy,
            score_model=score_model,
        )

    @override
    def resolve(self) -> CSAResolvedProfile[CandidateT]:
        """Materialize the canonical CSA configuration.

        Returns
        -------
        CSAResolvedProfile[CandidateT]
            Immutable optimizer-ready configuration derived from the boundary
            profile and selected preset defaults.

        Raises
        ------
        ValueError
            Raised when required fields such as ``perturbation_schedule`` are
            still missing.
        """
        perturbation_schedule = self.perturbation_schedule
        if perturbation_schedule is None:
            msg = "perturbation_schedule must be provided"
            raise ValueError(msg)

        defaults = profile_defaults_for_preset(self.preset)
        proposal_policy = (
            CSAProposalPolicy()
            if self.proposal_policy is None
            else self.proposal_policy
        )
        seed_count = defaults.seed_count if self.seed_count is None else self.seed_count
        initial_new_bank_cut = (
            defaults.initial_new_bank_cut
            if self.initial_new_bank_cut is None
            else self.initial_new_bank_cut
        )
        random_seed_mode = (
            defaults.random_seed_mode
            if self.random_seed_mode is None
            else self.random_seed_mode
        )
        weighted_partner_selection = (
            defaults.weighted_partner_selection
            if self.weighted_partner_selection is None
            else self.weighted_partner_selection
        )
        max_bank_capacity = (
            defaults.max_bank_capacity
            if self.max_bank_capacity is None
            else self.max_bank_capacity
        )
        cutoff_schedule = (
            defaults.cutoff_schedule
            if self.cutoff_schedule is None
            else self.cutoff_schedule
        )
        acceptance_policy = (
            defaults.acceptance_policy
            if self.acceptance_policy is None
            else self.acceptance_policy
        )
        clustering_policy = (
            defaults.clustering_policy
            if self.clustering_policy is None
            else self.clustering_policy
        )
        growth_policy = (
            defaults.growth_policy if self.growth_policy is None else self.growth_policy
        )
        refresh_policy = (
            defaults.refresh_policy
            if self.refresh_policy is None
            else self.refresh_policy
        )
        restart_lite = (
            defaults.restart_lite if self.restart_lite is None else self.restart_lite
        )
        cycle_limit = (
            defaults.cycle_limit if self.cycle_limit is None else self.cycle_limit
        )
        update_policy = (
            defaults.update_policy if self.update_policy is None else self.update_policy
        )
        if self.score_model is None:
            score_model: CSAScoreModel[CandidateT] = CSAScoreModel(
                biased_potential=defaults.score_model_defaults.biased_potential,
                adaptive_potential=None,
            )
        else:
            score_model = self.score_model
        validate_random_seed_mode(random_seed_mode)

        return CSAResolvedProfile(
            perturbation_schedule=perturbation_schedule,
            proposal_policy=proposal_policy,
            seed_count=seed_count,
            initial_new_bank_cut=initial_new_bank_cut,
            random_seed_mode=random_seed_mode,
            weighted_partner_selection=weighted_partner_selection,
            max_bank_capacity=max_bank_capacity,
            cutoff_schedule=cutoff_schedule,
            acceptance_policy=acceptance_policy,
            clustering_policy=clustering_policy,
            growth_policy=growth_policy,
            refresh_policy=refresh_policy,
            restart_lite=restart_lite,
            cycle_limit=cycle_limit,
            update_policy=update_policy,
            score_model=score_model,
        )
