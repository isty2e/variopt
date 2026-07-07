"""CSA bank-update logic for batch shadow-bank reduction."""

from collections.abc import Callable, Sequence

import numpy as np

from ......artifacts import Observation
from ......distance import require_valid_distance
from ......diversity import DiversityMetric
from ......typevars import CandidateT
from ...progression.cutoff.logic import initialize_cutoff_state
from ...progression.cutoff.policy import CSACutoffSchedule
from ...progression.state import CSAProgressionState
from ...scoring.acceptance_state import CSAAcceptanceState
from ...scoring.model_state import CSAScoreModelState
from ...trace.events.state import CSAEventTraceState
from ..bank import Bank, BankEntry
from ..clustering import CSAClusteringState
from ..growth import CSABankGrowthState
from ..growth.logic import (
    reduce_bank_by_energy_cut,
    should_attempt_remove_top,
    try_append_growth_entry,
)
from ..queries import (
    BankDistanceWorkspace,
    crowded_indices,
    crowding_aware_scores,
    validated_candidate_distance,
)
from .admission import admit_observation, replace_bank_entry
from .policy import CSABankUpdatePolicy
from .result import BankUpdateResult, changed_indices, significant_update_indices


def apply_bank_update_batch(
    *,
    bank: Bank[CandidateT],
    state: CSAProgressionState,
    observations: Sequence[Observation[CandidateT]],
    diversity_metric: DiversityMetric[CandidateT],
    infer_average_distance: Callable[
        [Sequence[BankEntry[CandidateT]]],
        float,
    ],
    infer_score_gap: Callable[
        [Sequence[BankEntry[CandidateT]]],
        float | None,
    ],
    cutoff_schedule: CSACutoffSchedule,
    update_policy: CSABankUpdatePolicy,
    acceptance_state: CSAAcceptanceState,
    score_model_state: CSAScoreModelState[CandidateT],
    growth_state: CSABankGrowthState[CandidateT],
    clustering_state: CSAClusteringState[CandidateT],
    base_bank_capacity: int,
    masked_seed_indices: frozenset[int],
    random_state: np.random.RandomState | None,
    trace_state: CSAEventTraceState[CandidateT] | None = None,
) -> BankUpdateResult[CandidateT]:
    """Reduce observations against a shadow bank and summarize the batch delta.

    Parameters
    ----------
    bank : Bank[CandidateT]
        Current bank snapshot used to initialize the shadow bank.
    state : CSAProgressionState
        Current progression state, including cutoff metadata.
    observations : Sequence[Observation[CandidateT]]
        Observations to reduce against the shadow bank.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used for distances and crowding.
    infer_average_distance : Callable[[Sequence[BankEntry[CandidateT]]], float]
        Callback used to summarize average bank distance.
    infer_score_gap : Callable[[Sequence[BankEntry[CandidateT]]], float | None]
        Callback used to summarize the bank score gap.
    cutoff_schedule : CSACutoffSchedule
        Schedule used to initialize the cutoff when needed.
    update_policy : CSABankUpdatePolicy
        Bank-update policy controlling local, far, and crowding-aware updates.
    acceptance_state : CSAAcceptanceState
        Acceptance-state runtime used for stochastic acceptance decisions.
    score_model_state : CSAScoreModelState[CandidateT]
        Score-model runtime used to shape bank and trial scores.
    growth_state : CSABankGrowthState[CandidateT]
        Growth-state runtime used for staged bank growth.
    clustering_state : CSAClusteringState[CandidateT]
        Clustering runtime used for cluster-aware updates.
    base_bank_capacity : int
        Minimum bank capacity preserved by energy-cut reduction.
    masked_seed_indices : frozenset[int]
        Seed indices that must be excluded from some update paths.
    random_state : numpy.random.RandomState | None
        Random-state instance used by stochastic acceptance, when required.
    trace_state : CSAEventTraceState[CandidateT] | None, default=None
        Optional trace state that records each bank-update step.

    Returns
    -------
    BankUpdateResult[CandidateT]
        Final bank-update result after applying the full observation batch.
    """
    previous_bank = bank
    shadow_bank = bank
    shadow_state = state
    shadow_score_model_state = score_model_state
    shadow_growth_state = growth_state
    shadow_clustering_state = clustering_state
    distance_workspace: BankDistanceWorkspace[CandidateT] | None = None
    if shadow_clustering_state.requires_initialization(entries=shadow_bank.entries):
        # Guard before calling ensure_initialized: Python evaluates arguments
        # eagerly, so this branch owns average-distance inference laziness.
        shadow_clustering_state = shadow_clustering_state.ensure_initialized(
            entries=shadow_bank.entries,
            reference_average_distance=infer_average_distance(shadow_bank.entries),
            diversity_metric=diversity_metric,
        )

    for observation in observations:
        bank_before_step = shadow_bank
        shadow_state = initialize_cutoff_if_needed(
            bank=shadow_bank,
            state=shadow_state,
            infer_average_distance=infer_average_distance,
            infer_score_gap=infer_score_gap,
            cutoff_schedule=cutoff_schedule,
        )
        active_distance_cutoff = 0.0
        if shadow_state.distance_cutoff is not None:
            active_distance_cutoff = shadow_state.distance_cutoff

        if not shadow_bank.is_full:
            shadow_bank = admit_observation(
                policy=update_policy,
                bank=shadow_bank,
                observation=observation,
                diversity_metric=diversity_metric,
                distance_cutoff=active_distance_cutoff,
            )
        else:
            (
                shadow_bank,
                shadow_score_model_state,
                shadow_growth_state,
                shadow_clustering_state,
                distance_workspace,
            ) = admit_full_bank_observation(
                bank=shadow_bank,
                observation=observation,
                diversity_metric=diversity_metric,
                distance_cutoff=active_distance_cutoff,
                minimum_distance_cutoff=shadow_state.minimum_distance_cutoff,
                update_policy=update_policy,
                acceptance_state=acceptance_state,
                score_model_state=shadow_score_model_state,
                growth_state=shadow_growth_state,
                clustering_state=shadow_clustering_state,
                base_bank_capacity=base_bank_capacity,
                masked_seed_indices=masked_seed_indices,
                random_state=random_state,
                distance_workspace=distance_workspace,
            )
        shadow_state = initialize_cutoff_if_needed(
            bank=shadow_bank,
            state=shadow_state,
            infer_average_distance=infer_average_distance,
            infer_score_gap=infer_score_gap,
            cutoff_schedule=cutoff_schedule,
        )
        batch_changed_indices = changed_indices(
            previous_bank=bank_before_step,
            next_bank=shadow_bank,
        )
        batch_significant_update_indices = significant_update_indices(
            previous_bank=bank_before_step,
            next_bank=shadow_bank,
            minimum_significant_score_gap=update_policy.minimum_significant_score_gap,
        )
        if trace_state is not None:
            trace_state = trace_state.record_bank_update_step(
                candidate=observation.candidate,
                value=observation.score,
                bank_before=bank_before_step,
                bank_after=shadow_bank,
                distance_cutoff_before=active_distance_cutoff,
                distance_cutoff_after=shadow_state.distance_cutoff,
                changed_indices=batch_changed_indices,
                significant_update_indices=batch_significant_update_indices,
            )

    final_changed_indices = changed_indices(
        previous_bank=previous_bank,
        next_bank=shadow_bank,
    )
    final_significant_update_indices = significant_update_indices(
        previous_bank=previous_bank,
        next_bank=shadow_bank,
        minimum_significant_score_gap=update_policy.minimum_significant_score_gap,
    )
    removed_indices: frozenset[int] = frozenset()
    (
        shadow_bank,
        removed_indices,
        shadow_score_model_state,
    ) = reduce_bank_by_energy_cut(
        state=shadow_growth_state,
        bank=shadow_bank,
        minimum_capacity=base_bank_capacity,
        score_model_state=shadow_score_model_state,
        diversity_metric=diversity_metric,
        distance_cutoff=(
            0.0
            if shadow_state.distance_cutoff is None
            else shadow_state.distance_cutoff
        ),
        minimum_distance_cutoff=shadow_state.minimum_distance_cutoff,
        distance_workspace=distance_workspace,
    )
    if final_changed_indices or removed_indices:
        shadow_clustering_state = shadow_clustering_state.recluster(
            entries=shadow_bank.entries,
            diversity_metric=diversity_metric,
        )
    return BankUpdateResult(
        bank=shadow_bank,
        state=shadow_state,
        score_model_state=shadow_score_model_state,
        growth_state=shadow_growth_state,
        clustering_state=shadow_clustering_state,
        trace_state=trace_state,
        changed_indices=final_changed_indices,
        significant_update_indices=final_significant_update_indices,
        removed_indices=removed_indices,
    )


def admit_full_bank_observation(
    *,
    bank: Bank[CandidateT],
    observation: Observation[CandidateT],
    diversity_metric: DiversityMetric[CandidateT],
    distance_cutoff: float,
    minimum_distance_cutoff: float | None,
    update_policy: CSABankUpdatePolicy,
    acceptance_state: CSAAcceptanceState,
    score_model_state: CSAScoreModelState[CandidateT],
    growth_state: CSABankGrowthState[CandidateT],
    clustering_state: CSAClusteringState[CandidateT],
    base_bank_capacity: int,
    masked_seed_indices: frozenset[int],
    random_state: np.random.RandomState | None,
    distance_workspace: BankDistanceWorkspace[CandidateT] | None,
) -> tuple[
    Bank[CandidateT],
    CSAScoreModelState[CandidateT],
    CSABankGrowthState[CandidateT],
    CSAClusteringState[CandidateT],
    BankDistanceWorkspace[CandidateT] | None,
]:
    """Admit or reject one observation when the bank is already full.

    Parameters
    ----------
    bank : Bank[CandidateT]
        Full bank snapshot to update.
    observation : Observation[CandidateT]
        Observation being considered for admission.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used for distances and crowding.
    distance_cutoff : float
        Active cutoff distance used by local and cluster updates.
    minimum_distance_cutoff : float | None
        Optional lower bound used by score shaping.
    update_policy : CSABankUpdatePolicy
        Bank-update policy controlling local, far, and crowding-aware updates.
    acceptance_state : CSAAcceptanceState
        Acceptance-state runtime used for stochastic acceptance decisions.
    score_model_state : CSAScoreModelState[CandidateT]
        Score-model runtime used to shape bank and trial scores.
    growth_state : CSABankGrowthState[CandidateT]
        Growth-state runtime used for staged bank growth.
    clustering_state : CSAClusteringState[CandidateT]
        Clustering runtime used for cluster-aware updates.
    base_bank_capacity : int
        Minimum bank capacity preserved by energy-cut reduction.
    masked_seed_indices : frozenset[int]
        Seed indices that must be excluded from some update paths.
    random_state : numpy.random.RandomState | None
        Random-state instance used by stochastic acceptance, when required.
    distance_workspace : BankDistanceWorkspace[CandidateT] | None
        Batch-local distance workspace aligned to ``bank.entries``, when one
        has already been created.

    Returns
    -------
    tuple[Bank[CandidateT], CSAScoreModelState[CandidateT], CSABankGrowthState[CandidateT], CSAClusteringState[CandidateT], BankDistanceWorkspace[CandidateT] | None]
        Updated bank, associated CSA runtime states, and the next batch-local
        distance workspace.
    """
    new_entry = BankEntry(
        candidate=observation.candidate,
        value=observation.score,
        proposal_id=observation.proposal.proposal_id,
    )

    def get_distance_workspace() -> BankDistanceWorkspace[CandidateT]:
        nonlocal distance_workspace
        workspace = distance_workspace
        if workspace is None:
            workspace = BankDistanceWorkspace(
                entries=bank.entries,
                diversity_metric=diversity_metric,
            )
            distance_workspace = workspace
        return workspace

    def rebase_distance_workspace(
        *,
        next_bank: Bank[CandidateT],
        invalidated_indices: frozenset[int],
        admitted_index: int | None = None,
    ) -> BankDistanceWorkspace[CandidateT] | None:
        workspace = distance_workspace
        if workspace is None:
            return None

        rebased_workspace = workspace.rebase(
            entries=next_bank.entries,
            invalidated_indices=invalidated_indices,
        )
        if admitted_index is None:
            return rebased_workspace

        admitted_distances = entry_distances
        if len(next_bank.entries) == len(entry_distances) + 1:
            admitted_distances = (*entry_distances, 0.0)
        # The workspace already stores symmetric pair keys; these validated
        # trial-to-bank distances become bank-pair distances after admission.
        rebased_workspace.seed_entry_distances(
            entry_index=admitted_index,
            distances=admitted_distances,
        )
        return rebased_workspace

    entry_distances = tuple(
        require_valid_distance(
            validated_candidate_distance(
                diversity_metric,
                observation.candidate,
                entry.candidate,
            ),
        )
        for entry in bank.entries
    )
    scored_bank, score_model_state = score_model_state.score_bank(
        entries=bank.entries,
        diversity_metric=diversity_metric,
        distance_cutoff=distance_cutoff,
        minimum_distance_cutoff=minimum_distance_cutoff,
        masked_entry_indices=masked_seed_indices,
        distance_workspace=(
            get_distance_workspace()
            if score_model_state.score_model.has_biased_potential
            else None
        ),
    )
    trial = score_model_state.score_trial(
        observation=observation,
        bank_real_scores=scored_bank.real_scores,
        entry_distances=entry_distances,
        diversity_metric=diversity_metric,
        distance_cutoff=distance_cutoff,
        minimum_distance_cutoff=minimum_distance_cutoff,
    )

    nearest_index = min(range(len(entry_distances)), key=entry_distances.__getitem__)
    nearest_distance = entry_distances[nearest_index]
    adaptive_potential_active = score_model_state.adaptive_potential_state is not None
    local_update_allowed = (
        max(scored_bank.real_scores) >= trial.real_score or adaptive_potential_active
    )

    if nearest_distance < distance_cutoff:
        if update_policy.local_update_mode != "disabled" and local_update_allowed:
            comparison_score = score_model_state.comparison_score_for_entry(
                base_score=scored_bank.shaped_scores[nearest_index],
                entry_real_score=scored_bank.real_scores[nearest_index],
                trial_real_score=trial.real_score,
                entry_distance=nearest_distance,
                biased_sigma2=scored_bank.biased_sigma2,
            )
            if acceptance_state.should_accept(
                trial_score=trial.shaped_score,
                reference_score=comparison_score,
                random_state=random_state,
            ):
                next_bank = replace_bank_entry(
                    bank=bank,
                    index=nearest_index,
                    new_entry=new_entry,
                )
                return (
                    next_bank,
                    score_model_state.bump_trial(trial),
                    growth_state,
                    clustering_state.register_admission(
                        admitted_index=nearest_index,
                        nearest_index=nearest_index,
                        nearest_distance=nearest_distance,
                        appended=False,
                    ),
                    rebase_distance_workspace(
                        next_bank=next_bank,
                        invalidated_indices=frozenset({nearest_index}),
                        admitted_index=nearest_index,
                    ),
                )

            if adaptive_potential_active:
                score_model_state = score_model_state.bump_candidate(
                    candidate=bank.entries[nearest_index].candidate,
                    diversity_metric=diversity_metric,
                )

    pre_growth_bank = bank
    bank, growth_state, did_grow = try_append_growth_entry(
        state=growth_state,
        bank=bank,
        observation=observation,
        scored_bank=scored_bank,
        trial=trial,
        nearest_distance=nearest_distance,
        active_distance_cutoff=distance_cutoff,
        adaptive_potential_active=adaptive_potential_active,
    )
    if did_grow:
        appended_index = len(bank.entries) - 1
        return (
            bank,
            score_model_state,
            growth_state,
            clustering_state.register_admission(
                admitted_index=appended_index,
                nearest_index=nearest_index,
                nearest_distance=nearest_distance,
                appended=True,
            ),
            rebase_distance_workspace(
                next_bank=bank,
                invalidated_indices=frozenset(),
                admitted_index=appended_index,
            ),
        )

    # The scored-bank view, trial-bank distances, and distance workspace above
    # are all aligned to this pre-growth bank snapshot.
    if bank is not pre_growth_bank:
        msg = (
            "try_append_growth_entry must not replace the bank unless did_grow is true"
        )
        raise RuntimeError(msg)

    if clustering_state.should_attempt_cluster_update(
        nearest_distance=nearest_distance,
        distance_cutoff=distance_cutoff,
    ):
        cluster_update = clustering_state.select_cluster_update(
            shaped_scores=scored_bank.shaped_scores,
            nearest_index=nearest_index,
        )
        if cluster_update is not None:
            if acceptance_state.should_accept(
                trial_score=trial.shaped_score,
                reference_score=cluster_update.comparison_score,
                random_state=random_state,
            ):
                next_bank = replace_bank_entry(
                    bank=bank,
                    index=cluster_update.remove_index,
                    new_entry=new_entry,
                )
                return (
                    next_bank,
                    score_model_state.bump_trial(trial),
                    growth_state,
                    clustering_state.register_admission(
                        admitted_index=cluster_update.remove_index,
                        nearest_index=nearest_index,
                        nearest_distance=nearest_distance,
                        appended=False,
                    ),
                    rebase_distance_workspace(
                        next_bank=next_bank,
                        invalidated_indices=frozenset({cluster_update.remove_index}),
                        admitted_index=cluster_update.remove_index,
                    ),
                )

            if adaptive_potential_active:
                score_model_state = score_model_state.bump_candidate(
                    candidate=bank.entries[cluster_update.comparison_index].candidate,
                    diversity_metric=diversity_metric,
                )

    remove_top_cutoff = clustering_state.remove_top_cutoff(
        distance_cutoff=distance_cutoff,
    )
    if not should_attempt_remove_top(
        state=growth_state,
        bank=bank,
        scored_bank=scored_bank,
        trial=trial,
        nearest_distance=nearest_distance,
        active_distance_cutoff=remove_top_cutoff,
        minimum_capacity=base_bank_capacity,
        adaptive_potential_active=adaptive_potential_active,
    ):
        return (
            bank,
            score_model_state,
            growth_state,
            clustering_state,
            distance_workspace,
        )

    adjusted_bank_scores = score_model_state.trial_adjusted_bank_scores(
        scored_bank=scored_bank,
        trial_real_score=trial.real_score,
        entry_distances=entry_distances,
    )
    removable_indices = tuple(range(len(adjusted_bank_scores)))
    if update_policy.far_update_mode == "crowded_worst":
        crowded_removable_indices = tuple(
            crowded_indices(
                entries=bank.entries,
                diversity_metric=diversity_metric,
                distance_cutoff=distance_cutoff,
                distance_workspace=get_distance_workspace(),
            )
        )
        if crowded_removable_indices:
            removable_indices = crowded_removable_indices
        worst_index = max(removable_indices, key=adjusted_bank_scores.__getitem__)
    elif update_policy.far_update_mode == "crowding_aware":
        removal_scores = crowding_aware_scores(
            base_scores=adjusted_bank_scores,
            entries=bank.entries,
            diversity_metric=diversity_metric,
            distance_cutoff=distance_cutoff,
            penalty_ratio=update_policy.crowding_penalty_ratio,
            niche_quality_policy=update_policy.niche_quality_policy,
            distance_workspace=get_distance_workspace(),
        )
        worst_index = max(range(len(removal_scores)), key=removal_scores.__getitem__)
    else:
        worst_index = max(removable_indices, key=adjusted_bank_scores.__getitem__)
    comparison_score = score_model_state.comparison_score_for_entry(
        base_score=adjusted_bank_scores[worst_index],
        entry_real_score=scored_bank.real_scores[worst_index],
        trial_real_score=trial.real_score,
        entry_distance=entry_distances[worst_index],
        biased_sigma2=scored_bank.biased_sigma2,
    )
    if acceptance_state.should_accept(
        trial_score=trial.shaped_score,
        reference_score=comparison_score,
        random_state=random_state,
    ):
        next_bank = replace_bank_entry(
            bank=bank,
            index=worst_index,
            new_entry=new_entry,
        )
        return (
            next_bank,
            score_model_state.bump_trial(trial),
            growth_state,
            clustering_state.register_admission(
                admitted_index=worst_index,
                nearest_index=nearest_index,
                nearest_distance=nearest_distance,
                appended=False,
            ),
            rebase_distance_workspace(
                next_bank=next_bank,
                invalidated_indices=frozenset({worst_index}),
                admitted_index=worst_index,
            ),
        )

    if adaptive_potential_active:
        score_model_state = score_model_state.bump_candidate(
            candidate=bank.entries[worst_index].candidate,
            diversity_metric=diversity_metric,
        )
    return (
        bank,
        score_model_state,
        growth_state,
        clustering_state,
        distance_workspace,
    )


def initialize_cutoff_if_needed(
    *,
    bank: Bank[CandidateT],
    state: CSAProgressionState,
    infer_average_distance: Callable[
        [Sequence[BankEntry[CandidateT]]],
        float,
    ],
    infer_score_gap: Callable[
        [Sequence[BankEntry[CandidateT]]],
        float | None,
    ],
    cutoff_schedule: CSACutoffSchedule,
) -> CSAProgressionState:
    """Initialize cutoff state once the bank first reaches capacity.

    Parameters
    ----------
    bank : Bank[CandidateT]
        Current bank snapshot.
    state : CSAProgressionState
        Current progression state.
    infer_average_distance : Callable[[Sequence[BankEntry[CandidateT]]], float]
        Callback used to summarize average bank distance.
    infer_score_gap : Callable[[Sequence[BankEntry[CandidateT]]], float | None]
        Callback used to summarize the bank score gap.
    cutoff_schedule : CSACutoffSchedule
        Schedule used to initialize the cutoff.

    Returns
    -------
    CSAProgressionState
        Progression state with cutoff initialized when appropriate.
    """
    if state.cutoff_is_initialized or not bank.is_full:
        return state

    average_distance = (
        infer_average_distance(bank.entries)
        if cutoff_schedule.requires_average_distance_for_initialization
        else None
    )
    return initialize_cutoff_state(
        state=state,
        schedule=cutoff_schedule,
        average_distance=average_distance,
        score_gap=infer_score_gap(bank.entries),
    )


def infer_score_gap(entries: Sequence[BankEntry[CandidateT]]) -> float | None:
    """Infer the min-max score gap across bank entries.

    Parameters
    ----------
    entries : Sequence[BankEntry[CandidateT]]
        Entries whose score spread is summarized.

    Returns
    -------
    float | None
        Difference between maximum and minimum objective value, or ``None``
        when no entries are available.
    """
    if not entries:
        return None

    values = tuple(entry.value for entry in entries)
    return max(values) - min(values)
