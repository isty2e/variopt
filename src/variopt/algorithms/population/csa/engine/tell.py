"""CSA tell-side reducer logic."""

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np

from .....artifacts import Observation
from .....diversity import DiversityMetric
from .....spaces import LeafPath
from .....typevars import CandidateT
from ..banking.bank import BankEntry
from ..banking.growth.logic import advance_growth_state
from ..banking.update import CSABankUpdatePolicy
from ..banking.update.logic import apply_bank_update_batch
from ..generation.proposal.logic import update_proposal_state
from ..generation.proposal.state.attribution import (
    NumericSubspaceDisplacement,
    ProposalAttribution,
)
from ..progression.cutoff.logic import advance_cutoff_state
from ..progression.cutoff.policy import CSACutoffSchedule
from ..progression.refresh import CSARefreshPolicy
from ..trace.events.artifacts import CSAProposalFamilyTrace
from .boundary import (
    apply_pending_boundary_action,
    complete_refresh,
    sync_reference_bank_if_uninitialized,
)
from .state import CSAEngineState

InferAverageDistance = Callable[[Sequence[BankEntry[CandidateT]]], float]
InferScoreGap = Callable[[Sequence[BankEntry[CandidateT]]], float | None]
InferLocalDisplacementLeafPaths = Callable[
    [CandidateT, CandidateT], tuple[LeafPath, ...]
]
InferNumericSubspaceDisplacement = Callable[
    [ProposalAttribution, CandidateT],
    NumericSubspaceDisplacement | None,
]


def apply_tell(
    engine_state: CSAEngineState[CandidateT],
    observations: Sequence[Observation[CandidateT]],
    *,
    bank_capacity: int,
    diversity_metric: DiversityMetric[CandidateT],
    cutoff_schedule: CSACutoffSchedule,
    refresh_policy: CSARefreshPolicy,
    update_policy: CSABankUpdatePolicy,
    random_state: np.random.RandomState | None = None,
    infer_average_distance: InferAverageDistance[CandidateT],
    infer_score_gap: InferScoreGap[CandidateT],
    infer_local_displacement_leaf_paths: InferLocalDisplacementLeafPaths[CandidateT]
    | None = None,
    explicit_local_displacement_leaf_paths: Sequence[tuple[LeafPath, ...] | None]
    | None = None,
    infer_numeric_subspace_displacement: InferNumericSubspaceDisplacement[CandidateT]
    | None = None,
) -> CSAEngineState[CandidateT]:
    """Apply one tell batch to the CSA engine state.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Current CSA engine state.
    observations : Sequence[Observation[CandidateT]]
        Evaluator observations to assimilate.
    bank_capacity : int
        Base bank capacity used by growth logic.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used by banking and cutoff logic.
    cutoff_schedule : CSACutoffSchedule
        Cutoff schedule used for iteration advancement and refresh.
    refresh_policy : CSARefreshPolicy
        Refresh policy used when refresh collection is active or triggered.
    update_policy : CSABankUpdatePolicy
        Bank update policy controlling admission and replacement logic.
    random_state : np.random.RandomState | None, default=None
        Optional random state for stochastic banking decisions.
    infer_average_distance : InferAverageDistance[CandidateT]
        Callback that computes average distance over bank entries.
    infer_score_gap : InferScoreGap[CandidateT]
        Callback that computes score gap over bank entries.
    infer_local_displacement_leaf_paths : InferLocalDisplacementLeafPaths[CandidateT] | None, default=None
        Optional callback that infers leaf-path displacement for proposal
        attribution.
    explicit_local_displacement_leaf_paths : Sequence[tuple[LeafPath, ...] | None] | None, default=None
        Optional observation-aligned local-displacement paths supplied by
        candidate-refinement metadata. Entries set to ``None`` use fallback
        inference when available.
    infer_numeric_subspace_displacement : InferNumericSubspaceDisplacement[CandidateT] | None, default=None
        Optional callback that infers numeric-subspace displacement for
        covariance attribution.

    Returns
    -------
    CSAEngineState[CandidateT]
        Engine state after assimilating ``observations``.

    Raises
    ------
    ValueError
        If observations do not align with the pending proposal set.
    """
    if engine_state.banking_state.refresh_state is not None:
        return apply_refresh_tell(
            engine_state,
            observations,
            diversity_metric=diversity_metric,
            cutoff_schedule=cutoff_schedule,
            refresh_policy=refresh_policy,
            infer_average_distance=infer_average_distance,
            infer_score_gap=infer_score_gap,
            infer_numeric_subspace_displacement=infer_numeric_subspace_displacement,
        )

    bank_was_full = (
        engine_state.banking_state.bank.is_full
        and engine_state.progression_state.cutoff_is_initialized
    )
    committed_generation = False
    consumed_ids: set[str] = set()
    validated_observations: list[Observation[CandidateT]] = []

    for observation in observations:
        proposal_id = observation.proposal.proposal_id
        if proposal_id is None:
            msg = (
                "observations supplied to CSAOptimizer.tell must reference proposal ids"
            )
            raise ValueError(msg)

        if proposal_id in consumed_ids:
            msg = "observations supplied to tell must have distinct proposal ids"
            raise ValueError(msg)

        pending_proposal = engine_state.pending_proposals.get(proposal_id)
        if pending_proposal is None:
            msg = "observation does not correspond to a pending proposal"
            raise ValueError(msg)

        if pending_proposal != observation.proposal:
            msg = "observation proposal does not match the pending proposal"
            raise ValueError(msg)

        consumed_ids.add(proposal_id)
        validated_observations.append(observation)

    engine_state = engine_state.consume_pending_proposals(consumed_ids)
    engine_state = replace(
        engine_state,
        proposal_state=update_proposal_state(
            engine_state.proposal_state,
            validated_observations,
            explicit_local_displacement_leaf_paths=explicit_local_displacement_leaf_paths,
            infer_local_displacement_leaf_paths=infer_local_displacement_leaf_paths,
            infer_numeric_subspace_displacement=infer_numeric_subspace_displacement,
        ),
    )

    if engine_state.generation_state.is_active:
        next_generation_state = engine_state.generation_state.buffer_observations(
            validated_observations,
        )
        engine_state = replace(engine_state, generation_state=next_generation_state)
        if not engine_state.generation_state.ready_to_commit:
            return engine_state

        generation_observations, next_generation_state = (
            engine_state.generation_state.release_buffer()
        )
        validated_observations = list(generation_observations)
        engine_state = replace(engine_state, generation_state=next_generation_state)
        committed_generation = True

    batch_result = apply_bank_update_batch(
        bank=engine_state.banking_state.bank,
        state=engine_state.progression_state,
        observations=validated_observations,
        diversity_metric=diversity_metric,
        infer_average_distance=infer_average_distance,
        infer_score_gap=infer_score_gap,
        cutoff_schedule=cutoff_schedule,
        update_policy=update_policy,
        acceptance_state=engine_state.scoring_state.acceptance_state,
        score_model_state=engine_state.scoring_state.model_state,
        growth_state=engine_state.banking_state.growth_state,
        clustering_state=engine_state.banking_state.clustering_state,
        base_bank_capacity=bank_capacity,
        masked_seed_indices=engine_state.progression_state.seed_mask,
        random_state=random_state,
        trace_state=engine_state.trace_state if committed_generation else None,
    )
    updated_indices = batch_result.changed_indices
    significant_update_indices = batch_result.significant_update_indices
    entry_count_before_removal = len(batch_result.bank.entries) + len(
        batch_result.removed_indices
    )
    progression_state = batch_result.state
    if updated_indices:
        progression_state = progression_state.without_updated_seed_mask(
            updated_indices,
        )
    if batch_result.removed_indices:
        progression_state = progression_state.remove_indices(
            removed_indices=batch_result.removed_indices,
            entry_count=len(batch_result.bank.entries),
        )

    selection_state = engine_state.selection_state
    if significant_update_indices:
        selection_state = selection_state.invalidate_for_bank_update(
            updated_indices=significant_update_indices,
            entry_count=entry_count_before_removal,
        )
    if batch_result.removed_indices:
        selection_state = selection_state.remove_indices(
            removed_indices=batch_result.removed_indices,
            entry_count=len(batch_result.bank.entries),
        )

    engine_state = replace(
        engine_state,
        banking_state=replace(
            engine_state.banking_state,
            bank=batch_result.bank,
            growth_state=batch_result.growth_state,
            clustering_state=batch_result.clustering_state,
        ),
        progression_state=progression_state,
        selection_state=selection_state,
        scoring_state=replace(
            engine_state.scoring_state,
            model_state=batch_result.score_model_state,
        ),
        trace_state=(
            engine_state.trace_state
            if batch_result.trace_state is None
            else batch_result.trace_state
        ),
    )

    engine_state = sync_reference_bank_if_uninitialized(
        engine_state,
        diversity_metric=diversity_metric,
        infer_average_distance=infer_average_distance,
    )

    if validated_observations:
        engine_state = replace(
            engine_state,
            banking_state=replace(
                engine_state.banking_state,
                growth_state=advance_growth_state(
                    state=engine_state.banking_state.growth_state,
                    bank=engine_state.banking_state.bank,
                    diversity_metric=diversity_metric,
                    score_model_state=engine_state.scoring_state.model_state,
                    distance_cutoff=engine_state.progression_state.distance_cutoff,
                    minimum_distance_cutoff=engine_state.progression_state.minimum_distance_cutoff,
                ),
            ),
        )

    if engine_state.progression_state.has_pending_action:
        if engine_state.pending_proposals.is_empty:
            engine_state = apply_pending_boundary_action(
                engine_state,
                refresh_policy=refresh_policy,
                diversity_metric=diversity_metric,
                cutoff_schedule=cutoff_schedule,
                infer_average_distance=infer_average_distance,
                infer_score_gap=infer_score_gap,
            )
        if committed_generation:
            engine_state = record_generation_completion(engine_state)
        return engine_state

    if (
        consumed_ids
        and bank_was_full
        and engine_state.progression_state.cutoff_is_initialized
    ):
        unused_entry_count = engine_state.selection_state.count_unused_entries(
            entry_count=len(engine_state.banking_state.bank.entries),
            ignored_indices=engine_state.progression_state.seed_mask,
        )
        next_progression_state, cycle_increment = advance_cutoff_state(
            state=engine_state.progression_state,
            schedule=cutoff_schedule,
            score_gap=infer_score_gap(engine_state.banking_state.bank.entries),
            unused_entry_count=unused_entry_count,
        )
        engine_state = replace(engine_state, progression_state=next_progression_state)
        if cycle_increment:
            engine_state = replace(
                engine_state,
                selection_state=engine_state.selection_state.reset_bank_status(
                    entry_count=len(engine_state.banking_state.bank.entries),
                ),
                progression_state=engine_state.progression_state.clear_refresh_mask().request_boundary(),
            )
            if (
                engine_state.progression_state.has_pending_action
                and engine_state.pending_proposals.is_empty
            ):
                engine_state = apply_pending_boundary_action(
                    engine_state,
                    refresh_policy=refresh_policy,
                    diversity_metric=diversity_metric,
                    cutoff_schedule=cutoff_schedule,
                    infer_average_distance=infer_average_distance,
                    infer_score_gap=infer_score_gap,
                )

    if committed_generation:
        engine_state = replace(
            engine_state,
            scoring_state=replace(
                engine_state.scoring_state,
                acceptance_state=engine_state.scoring_state.acceptance_state.advance(),
            ),
        )
        engine_state = record_generation_completion(engine_state)

    return engine_state


def apply_refresh_tell(
    engine_state: CSAEngineState[CandidateT],
    observations: Sequence[Observation[CandidateT]],
    *,
    diversity_metric: DiversityMetric[CandidateT],
    cutoff_schedule: CSACutoffSchedule,
    refresh_policy: CSARefreshPolicy,
    infer_average_distance: InferAverageDistance[CandidateT],
    infer_score_gap: InferScoreGap[CandidateT],
    infer_local_displacement_leaf_paths: InferLocalDisplacementLeafPaths[CandidateT]
    | None = None,
    infer_numeric_subspace_displacement: InferNumericSubspaceDisplacement[CandidateT]
    | None = None,
) -> CSAEngineState[CandidateT]:
    """Apply one tell batch while refresh collection is active.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Current CSA engine state with refresh collection active.
    observations : Sequence[Observation[CandidateT]]
        Evaluator observations to assimilate into the refresh payload.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used if refresh completes immediately.
    cutoff_schedule : CSACutoffSchedule
        Cutoff schedule used if refresh completes immediately.
    refresh_policy : CSARefreshPolicy
        Refresh policy applied when refresh completion is triggered.
    infer_average_distance : InferAverageDistance[CandidateT]
        Callback that computes refreshed average distance.
    infer_score_gap : InferScoreGap[CandidateT]
        Callback that computes refreshed score gap.
    infer_local_displacement_leaf_paths : InferLocalDisplacementLeafPaths[CandidateT] | None, default=None
        Optional callback that infers leaf-path displacement for proposal
        attribution.
    infer_numeric_subspace_displacement : InferNumericSubspaceDisplacement[CandidateT] | None, default=None
        Optional callback that infers numeric-subspace displacement for
        covariance attribution.

    Returns
    -------
    CSAEngineState[CandidateT]
        Engine state after assimilating refresh observations.

    Raises
    ------
    ValueError
        If observations do not align with the pending proposal set.
    """
    assert engine_state.banking_state.refresh_state is not None
    consumed_ids: set[str] = set()

    for observation in observations:
        proposal_id = observation.proposal.proposal_id
        if proposal_id is None:
            msg = (
                "observations supplied to CSAOptimizer.tell must reference proposal ids"
            )
            raise ValueError(msg)

        if proposal_id in consumed_ids:
            msg = "observations supplied to tell must have distinct proposal ids"
            raise ValueError(msg)

        pending_proposal = engine_state.pending_proposals.get(proposal_id)
        if pending_proposal is None:
            msg = "observation does not correspond to a pending proposal"
            raise ValueError(msg)

        if pending_proposal != observation.proposal:
            msg = "observation proposal does not match the pending proposal"
            raise ValueError(msg)

        refresh_state = engine_state.banking_state.refresh_state
        assert refresh_state is not None
        engine_state = replace(
            engine_state,
            banking_state=replace(
                engine_state.banking_state,
                refresh_state=refresh_state.append_observation(observation),
            ),
        )
        consumed_ids.add(proposal_id)

    engine_state = engine_state.consume_pending_proposals(consumed_ids)
    engine_state = replace(
        engine_state,
        proposal_state=update_proposal_state(
            engine_state.proposal_state,
            observations,
            infer_local_displacement_leaf_paths=infer_local_displacement_leaf_paths,
            infer_numeric_subspace_displacement=infer_numeric_subspace_displacement,
        ),
    )
    final_refresh_state = engine_state.banking_state.refresh_state
    if (
        final_refresh_state is not None
        and final_refresh_state.has_enough_entries
        and engine_state.pending_proposals.is_empty
    ):
        return complete_refresh(
            engine_state,
            final_refresh_state,
            refresh_policy=refresh_policy,
            diversity_metric=diversity_metric,
            cutoff_schedule=cutoff_schedule,
            infer_average_distance=infer_average_distance,
            infer_score_gap=infer_score_gap,
        )

    return engine_state


def record_generation_completion(
    engine_state: CSAEngineState[CandidateT],
) -> CSAEngineState[CandidateT]:
    """Return an engine state with one completed generation trace, if enabled.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Engine state immediately after generation completion.

    Returns
    -------
    CSAEngineState[CandidateT]
        Engine state with the trace reducer finalized when tracing is enabled.
    """
    trace_state = engine_state.trace_state
    if trace_state is None:
        return engine_state

    pending_boundary_action_after = None
    if engine_state.progression_state.pending_action is not None:
        pending_boundary_action_after = (
            engine_state.progression_state.pending_action.kind
        )

    proposal_state = engine_state.proposal_state
    family_traces_after = tuple(
        CSAProposalFamilyTrace(
            family_key=family_stat.family_key,
            observation_count=family_stat.observation_count,
            effective_score_credit=family_stat.effective_score_credit(
                current_update_index=proposal_state.update_index,
                score_decay=proposal_state.policy.score_decay,
            ),
            mutation_weight=None,
        )
        for family_stat in proposal_state.family_stats
    )

    return replace(
        engine_state,
        trace_state=trace_state.finish_generation(
            cycle_count=engine_state.progression_state.cycle_count,
            bank=engine_state.banking_state.bank,
            reference_bank=engine_state.banking_state.reference_bank,
            bank_status_after=engine_state.selection_state.resize_bank_status(
                entry_count=len(engine_state.banking_state.bank.entries),
            ),
            proposal_families_after=family_traces_after,
            pending_boundary_action_after=pending_boundary_action_after,
            refresh_active_after=engine_state.banking_state.refresh_state is not None,
        ),
    )
