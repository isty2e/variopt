"""CSA tell-side reducer logic."""

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np

from .....diversity import DiversityMetric
from .....spaces import LeafPath
from .....typevars import CandidateT
from ..banking.bank import BankEntry
from ..banking.growth.logic import advance_growth_state
from ..banking.update import CSABankUpdatePolicy
from ..banking.update.logic import apply_bank_update_batch
from ..generation.proposal.evidence import CSAProposalEvaluation
from ..generation.proposal.logic import (
    collect_proposal_outcome_evidence,
    consume_refresh_proposal_provenance,
    update_proposal_state,
)
from ..generation.proposal.state.attribution import (
    NumericSubspaceDisplacement,
    ProposalAttribution,
)
from ..progression.cutoff.logic import advance_cutoff_state
from ..progression.cutoff.observation import CSACutoffObservation
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
InferPairwiseDistances = Callable[
    [Sequence[BankEntry[CandidateT]]],
    tuple[float, ...],
]
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
    evaluations: Sequence[CSAProposalEvaluation[CandidateT]],
    *,
    bank_capacity: int,
    diversity_metric: DiversityMetric[CandidateT],
    cutoff_schedule: CSACutoffSchedule,
    refresh_policy: CSARefreshPolicy,
    update_policy: CSABankUpdatePolicy,
    random_state: np.random.RandomState | None = None,
    infer_average_distance: InferAverageDistance[CandidateT],
    infer_pairwise_distances: InferPairwiseDistances[CandidateT],
    infer_score_gap: InferScoreGap[CandidateT],
    infer_local_displacement_leaf_paths: InferLocalDisplacementLeafPaths[CandidateT]
    | None = None,
    infer_numeric_subspace_displacement: InferNumericSubspaceDisplacement[CandidateT]
    | None = None,
) -> CSAEngineState[CandidateT]:
    """Apply one tell batch to the CSA engine state.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Current CSA engine state.
    evaluations : Sequence[CSAProposalEvaluation[CandidateT]]
        Successful feedback to assimilate without discarding logical evaluation
        cost or explicit refinement metadata.
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
    infer_pairwise_distances : InferPairwiseDistances[CandidateT]
        Callback that materializes validated pairwise bank distances when the
        configured cutoff schedule requests them.
    infer_score_gap : InferScoreGap[CandidateT]
        Callback that computes score gap over bank entries.
    infer_local_displacement_leaf_paths : InferLocalDisplacementLeafPaths[CandidateT] | None, default=None
        Optional callback that infers leaf-path displacement for proposal
        attribution.
    infer_numeric_subspace_displacement : InferNumericSubspaceDisplacement[CandidateT] | None, default=None
        Optional callback that infers numeric-subspace displacement for
        covariance attribution.

    Returns
    -------
    CSAEngineState[CandidateT]
        Engine state after assimilating ``evaluations``.

    Raises
    ------
    ValueError
        If observations do not align with the pending proposal set.
    """
    if engine_state.banking_state.refresh_state is not None:
        return apply_refresh_tell(
            engine_state,
            evaluations,
            diversity_metric=diversity_metric,
            cutoff_schedule=cutoff_schedule,
            refresh_policy=refresh_policy,
            infer_average_distance=infer_average_distance,
            infer_score_gap=infer_score_gap,
        )

    bank_was_full = (
        engine_state.banking_state.bank.is_full
        and engine_state.progression_state.cutoff_is_initialized
    )
    committed_generation = False
    consumed_ids: set[str] = set()
    validated_evaluations: list[CSAProposalEvaluation[CandidateT]] = []

    for evaluation in evaluations:
        observation = evaluation.observation
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
        validated_evaluations.append(evaluation)

    engine_state = engine_state.consume_pending_proposals(consumed_ids)

    if engine_state.generation_state.is_active:
        next_generation_state = engine_state.generation_state.buffer_evaluations(
            validated_evaluations,
        )
        engine_state = replace(engine_state, generation_state=next_generation_state)
        if not engine_state.generation_state.ready_to_commit:
            return engine_state

        generation_evaluations, next_generation_state = (
            engine_state.generation_state.release_buffer()
        )
        validated_evaluations = list(generation_evaluations)
        engine_state = replace(engine_state, generation_state=next_generation_state)
        committed_generation = True

    validated_observations = tuple(
        evaluation.observation for evaluation in validated_evaluations
    )

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
    outcome_evidence, proposal_state = collect_proposal_outcome_evidence(
        engine_state.proposal_state,
        validated_evaluations,
        batch_result.transitions,
    )
    proposal_state = update_proposal_state(
        proposal_state,
        outcome_evidence,
        infer_local_displacement_leaf_paths=infer_local_displacement_leaf_paths,
        infer_numeric_subspace_displacement=infer_numeric_subspace_displacement,
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
        proposal_state=proposal_state,
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

    if validated_evaluations:
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
        entries = engine_state.banking_state.bank.entries
        entry_count = len(entries)
        ignored_entry_count = sum(
            1
            for index in engine_state.progression_state.seed_mask
            if 0 <= index < entry_count
        )
        eligible_entry_count = entry_count - ignored_entry_count
        unused_entry_count = engine_state.selection_state.count_unused_entries(
            entry_count=entry_count,
            ignored_indices=engine_state.progression_state.seed_mask,
        )
        observation = CSACutoffObservation(
            score_gap=infer_score_gap(entries),
            eligible_entry_count=eligible_entry_count,
            unused_entry_count=unused_entry_count,
            pairwise_distances=(
                infer_pairwise_distances(entries)
                if cutoff_schedule.requires_pairwise_distances
                else None
            ),
        )
        next_progression_state, cycle_increment = advance_cutoff_state(
            state=engine_state.progression_state,
            schedule=cutoff_schedule,
            observation=observation,
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
    evaluations: Sequence[CSAProposalEvaluation[CandidateT]],
    *,
    diversity_metric: DiversityMetric[CandidateT],
    cutoff_schedule: CSACutoffSchedule,
    refresh_policy: CSARefreshPolicy,
    infer_average_distance: InferAverageDistance[CandidateT],
    infer_score_gap: InferScoreGap[CandidateT],
) -> CSAEngineState[CandidateT]:
    """Apply one tell batch while refresh collection is active.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Current CSA engine state with refresh collection active.
    evaluations : Sequence[CSAProposalEvaluation[CandidateT]]
        Successful feedback to assimilate into the refresh payload.
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

    for evaluation in evaluations:
        observation = evaluation.observation
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
        proposal_state=consume_refresh_proposal_provenance(
            engine_state.proposal_state,
            evaluations,
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
            effective_survival_efficiency=family_stat.effective_survival_efficiency(
                current_update_index=proposal_state.update_index,
                adaptation_decay=proposal_state.policy.adaptation_decay,
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
