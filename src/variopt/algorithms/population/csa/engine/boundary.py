"""CSA boundary progression and refresh logic."""

from collections.abc import Callable, Sequence
from dataclasses import replace

from .....diversity import DiversityMetric
from .....typevars import CandidateT
from ..banking.bank import BankEntry
from ..banking.reference import (
    ReferenceRefreshState,
    build_reference_bank_from_bank,
    build_sorted_bank_from_bank,
    sort_entries_by_value,
)
from ..generation.state import GenerationRuntimeState
from ..progression.cutoff.policy import CSACutoffSchedule
from ..progression.refresh import CSARefreshPolicy
from ..progression.stage import CSAStageState
from ..selection.state import SeedSelectionState
from .state import CSAEngineState

InferAverageDistance = Callable[[Sequence[BankEntry[CandidateT]]], float]
InferScoreGap = Callable[[Sequence[BankEntry[CandidateT]]], float | None]


def begin_refresh(
    engine_state: CSAEngineState[CandidateT],
    *,
    refresh_policy: CSARefreshPolicy | None = None,
) -> CSAEngineState[CandidateT]:
    """Return an engine state that has entered refresh mode.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Engine state immediately before the refresh boundary action.
    refresh_policy : CSARefreshPolicy | None, default=None
        Refresh policy override. ``None`` uses the default refresh policy.

    Returns
    -------
    CSAEngineState[CandidateT]
        Engine state with refresh bookkeeping initialized and generation
        runtime reset.
    """
    effective_refresh_policy = (
        CSARefreshPolicy() if refresh_policy is None else refresh_policy
    )
    engine_state = replace(
        engine_state,
        progression_state=engine_state.progression_state.record_refresh(),
    )
    return replace(
        engine_state,
        banking_state=replace(
            engine_state.banking_state,
            refresh_state=build_refresh_state(
                engine_state=engine_state,
                target_capacity=engine_state.progression_state.stage_state.current_capacity,
                refresh_policy=effective_refresh_policy,
                preserve_existing_entries=False,
            ),
            clustering_state=engine_state.banking_state.clustering_state.reset(),
        ),
        selection_state=SeedSelectionState(),
        generation_state=GenerationRuntimeState(),
        progression_state=engine_state.progression_state.begin_refresh(),
    )


def begin_stage_transition(
    engine_state: CSAEngineState[CandidateT],
    transition: tuple[CSAStageState, bool],
    *,
    refresh_policy: CSARefreshPolicy | None = None,
) -> CSAEngineState[CandidateT]:
    """Return an engine state after one stage-transition boundary action.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Engine state immediately before the boundary action.
    transition : tuple[CSAStageState, bool]
        Stage-transition payload produced by the progression state. The boolean
        flag indicates whether the transition triggers bank growth.
    refresh_policy : CSARefreshPolicy | None, default=None
        Refresh policy override. ``None`` uses the default refresh policy.

    Returns
    -------
    CSAEngineState[CandidateT]
        Engine state after applying the stage transition and any required
        follow-up refresh initialization.
    """
    effective_refresh_policy = (
        CSARefreshPolicy() if refresh_policy is None else refresh_policy
    )
    _, grows_bank = transition
    engine_state = replace(
        engine_state,
        progression_state=engine_state.progression_state.apply_stage_transition(transition),
        selection_state=SeedSelectionState(),
        generation_state=GenerationRuntimeState(),
    )
    if not grows_bank:
        return engine_state

    engine_state = replace(
        engine_state,
        progression_state=engine_state.progression_state.record_refresh(),
    )
    return replace(
        engine_state,
        banking_state=replace(
            engine_state.banking_state,
            refresh_state=build_refresh_state(
                engine_state=engine_state,
                target_capacity=engine_state.progression_state.stage_state.current_capacity,
                refresh_policy=effective_refresh_policy,
                preserve_existing_entries=True,
            ),
        ),
        progression_state=engine_state.progression_state.begin_refresh(),
    )


def complete_refresh(
    engine_state: CSAEngineState[CandidateT],
    refresh_state: ReferenceRefreshState[CandidateT],
    *,
    refresh_policy: CSARefreshPolicy | None = None,
    diversity_metric: DiversityMetric[CandidateT],
    cutoff_schedule: CSACutoffSchedule,
    infer_average_distance: InferAverageDistance[CandidateT],
    infer_score_gap: InferScoreGap[CandidateT],
) -> CSAEngineState[CandidateT]:
    """Return an engine state that has completed refresh with rebuilt banks.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Engine state immediately before refresh completion.
    refresh_state : ReferenceRefreshState[CandidateT]
        Refresh payload containing preserved entries and admitted newcomers.
    refresh_policy : CSARefreshPolicy | None, default=None
        Refresh policy override. ``None`` uses the default refresh policy.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to initialize clustering and adaptive state.
    cutoff_schedule : CSACutoffSchedule
        Schedule used to resolve the rebuilt cutoff values.
    infer_average_distance : InferAverageDistance[CandidateT]
        Callback that computes the refreshed reference-bank average distance.
    infer_score_gap : InferScoreGap[CandidateT]
        Callback that computes the refreshed bank score gap.

    Returns
    -------
    CSAEngineState[CandidateT]
        Engine state with rebuilt bank/reference-bank snapshots and refreshed
        progression bookkeeping.
    """
    effective_refresh_policy = (
        CSARefreshPolicy() if refresh_policy is None else refresh_policy
    )
    refreshed_bank = refresh_state.build_bank()
    refreshed_reference_bank = refresh_state.build_reference_bank()
    average_distance = infer_average_distance(refreshed_reference_bank.entries)
    distance_cutoff, minimum_distance_cutoff = cutoff_schedule.resolve_initial_cutoffs(
        average_distance=average_distance,
    )
    score_gap = infer_score_gap(refreshed_bank.entries)

    clustering_state = engine_state.banking_state.clustering_state.ensure_initialized(
        entries=refreshed_bank.entries,
        reference_average_distance=average_distance,
        diversity_metric=diversity_metric,
    )
    progression_state = engine_state.progression_state.complete_refresh(
        distance_cutoff=distance_cutoff,
        minimum_distance_cutoff=minimum_distance_cutoff,
        previous_score_gap=score_gap,
    )
    preserved_entry_count = len(refresh_state.preserved_bank_entries)
    if effective_refresh_policy.mode == "adaptive_refresh":
        newcomer_mask = frozenset(range(preserved_entry_count))
        stage_state = progression_state.stage_state
        if stage_state.seed_mask or stage_state.partner_mask:
            progression_state = replace(
                progression_state,
                stage_state=stage_state.with_masks(
                    seed_mask=newcomer_mask,
                    partner_mask=newcomer_mask,
                ),
            )
        elif effective_refresh_policy.newcomer_first_round:
            progression_state = progression_state.with_refresh_mask(newcomer_mask)

    return replace(
        engine_state,
        banking_state=replace(
            engine_state.banking_state,
            bank=refreshed_bank,
            reference_bank=refreshed_reference_bank,
            refresh_state=None,
            clustering_state=clustering_state,
        ),
        selection_state=SeedSelectionState(),
        generation_state=GenerationRuntimeState(),
        progression_state=progression_state,
    )


def apply_pending_boundary_action(
    engine_state: CSAEngineState[CandidateT],
    *,
    refresh_policy: CSARefreshPolicy | None = None,
    diversity_metric: DiversityMetric[CandidateT],
    cutoff_schedule: CSACutoffSchedule,
    infer_average_distance: InferAverageDistance[CandidateT],
    infer_score_gap: InferScoreGap[CandidateT],
) -> CSAEngineState[CandidateT]:
    """Consume and apply the pending run-boundary action.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Engine state holding a pending run-boundary action.
    refresh_policy : CSARefreshPolicy | None, default=None
        Refresh policy override. ``None`` uses the default refresh policy.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used when a refresh completes immediately.
    cutoff_schedule : CSACutoffSchedule
        Cutoff schedule used when a refresh completes immediately.
    infer_average_distance : InferAverageDistance[CandidateT]
        Callback that computes the refreshed reference-bank average distance.
    infer_score_gap : InferScoreGap[CandidateT]
        Callback that computes the refreshed bank score gap.

    Returns
    -------
    CSAEngineState[CandidateT]
        Engine state after consuming the pending boundary action.
    """
    effective_refresh_policy = (
        CSARefreshPolicy() if refresh_policy is None else refresh_policy
    )
    pending_action, progression_state = engine_state.progression_state.consume_pending_action()
    engine_state = replace(engine_state, progression_state=progression_state)
    if pending_action.kind == "refresh":
        return begin_refresh(engine_state, refresh_policy=effective_refresh_policy)

    assert pending_action.stage_transition is not None
    next_engine_state = begin_stage_transition(
        engine_state,
        pending_action.stage_transition,
        refresh_policy=effective_refresh_policy,
    )
    refresh_state = next_engine_state.banking_state.refresh_state
    if refresh_state is None or not refresh_state.has_enough_entries:
        return next_engine_state

    return complete_refresh(
        next_engine_state,
        refresh_state,
        refresh_policy=effective_refresh_policy,
        diversity_metric=diversity_metric,
        cutoff_schedule=cutoff_schedule,
        infer_average_distance=infer_average_distance,
        infer_score_gap=infer_score_gap,
    )


def sync_reference_bank_if_uninitialized(
    engine_state: CSAEngineState[CandidateT],
    *,
    diversity_metric: DiversityMetric[CandidateT],
    infer_average_distance: InferAverageDistance[CandidateT],
) -> CSAEngineState[CandidateT]:
    """Return an engine state with sorted bank/reference bank once full.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Current engine state.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used when initializing clustering state.
    infer_average_distance : InferAverageDistance[CandidateT]
        Callback that computes the reference-bank average distance.

    Returns
    -------
    CSAEngineState[CandidateT]
        Sorted engine state when the bank first becomes full, otherwise the
        original state.
    """
    if not engine_state.banking_state.bank.is_full:
        return engine_state

    if engine_state.banking_state.reference_bank.initialized:
        return engine_state

    bank = build_sorted_bank_from_bank(engine_state.banking_state.bank)
    reference_bank = build_reference_bank_from_bank(bank)
    clustering_state = engine_state.banking_state.clustering_state.ensure_initialized(
        entries=bank.entries,
        reference_average_distance=infer_average_distance(reference_bank.entries),
        diversity_metric=diversity_metric,
    )
    return replace(
        engine_state,
        banking_state=replace(
            engine_state.banking_state,
            bank=bank,
            reference_bank=reference_bank,
            clustering_state=clustering_state,
        ),
    )


def build_refresh_state(
    *,
    engine_state: CSAEngineState[CandidateT],
    target_capacity: int,
    refresh_policy: CSARefreshPolicy,
    preserve_existing_entries: bool,
) -> ReferenceRefreshState[CandidateT]:
    """Build one refresh-state payload from the configured refresh policy.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Engine state whose bank snapshots seed the refresh payload.
    target_capacity : int
        Capacity that the refreshed bank should target.
    refresh_policy : CSARefreshPolicy
        Refresh policy determining whether entries are preserved and how many.
    preserve_existing_entries : bool
        Whether existing entries should be copied into the refresh payload.

    Returns
    -------
    ReferenceRefreshState[CandidateT]
        Refresh payload aligned with the configured policy.
    """
    if refresh_policy.mode == "legacy":
        if not preserve_existing_entries:
            return ReferenceRefreshState(target_capacity=target_capacity)

        return ReferenceRefreshState(
            target_capacity=target_capacity,
            preserved_bank_entries=engine_state.banking_state.bank.entries,
            preserved_reference_entries=engine_state.banking_state.reference_bank.entries,
        )

    preserved_entry_count = refresh_policy.resolve_preserved_entry_count(
        entry_count=len(engine_state.banking_state.bank.entries),
        target_capacity=target_capacity,
    )
    preserved_entries = sort_entries_by_value(engine_state.banking_state.bank.entries)[
        :preserved_entry_count
    ]
    return ReferenceRefreshState(
        target_capacity=target_capacity,
        preserved_bank_entries=preserved_entries,
        preserved_reference_entries=preserved_entries,
    )
