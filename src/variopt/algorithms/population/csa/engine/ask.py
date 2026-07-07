"""CSA ask-side planning and materialization logic."""

from dataclasses import dataclass, replace
from typing import Generic, Literal, TypeVar, cast

import numpy as np

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....distance import require_valid_distance
from .....diversity import DiversityMetric
from .....operators import VariationOperator
from .....spaces import SearchSpace
from .....spaces.projections import compile_homogeneous_numeric_subspace
from .....spaces.structured import require_space_candidate_value
from .....spaces.types import SpaceCandidateValue
from .....typevars import CandidateT
from ..banking.bank import BankEntry
from ..banking.queries import BankDistanceWorkspace
from ..generation.proposal import CSAProposalState
from ..generation.proposal.covariance import (
    build_numeric_subspace_attribution,
    sample_covariance_guided_candidate,
)
from ..generation.proposal.logic import (
    mutation_family_key,
    mutation_family_weights,
    plan_mutated_leaf_paths,
    planned_mutation_attribution,
    sample_mutation_family_indices,
)
from ..generation.state import GeneratedCandidate, GenerationQueue
from ..operators.editing import sample_exchange_count
from ..operators.structured import (
    StructuredPathMutationOperator,
    is_covariance_guided_structured_mutation_operator,
    is_structured_path_mutation_operator,
    is_validated_parent_variation_operator,
)
from ..profile import CSAResolvedProfile
from ..selection.policy import prepare_seed_batch, select_partner_indices
from ..selection.routing import should_use_reference_primary
from ..selection.state import SeedSelectionState
from ..trace.events.artifacts import CSAPrimarySource, CSAProposalFamilyTrace
from ..trace.events.state import CSAEventTraceState
from .state import CSAEngineState

BoundaryT = TypeVar("BoundaryT")
CSAAskPlanKind = Literal[
    "space_sample",
    "refresh_sample",
    "materialize_generation",
    "dequeue_generation",
]


@dataclass(frozen=True, slots=True)
class CSAAskPlan:
    """Pure ask-side plan for the next candidate request.

    Parameters
    ----------
    kind : CSAAskPlanKind
        Ask-side action that should run next.
    """

    kind: CSAAskPlanKind


@dataclass(frozen=True, slots=True)
class CSAMaterializedGeneration(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Impurely materialized CSA child pool ready for pure commit.

    Parameters
    ----------
    selection_state : SeedSelectionState
        Selection state after consuming the seed batch.
    generation_queue : GenerationQueue[CandidateT]
        Materialized child pool ready for pure commit and dequeue.
    trace_state : CSAEventTraceState[CandidateT] | None
        Updated trace state, when tracing is enabled.
    """

    selection_state: SeedSelectionState
    generation_queue: GenerationQueue[CandidateT]
    trace_state: CSAEventTraceState[CandidateT] | None


def emit_structured_mutation_candidate(
    *,
    operator: StructuredPathMutationOperator,
    proposal_state: CSAProposalState,
    seed_candidate: SpaceCandidateValue,
    seed_score: float,
    mutation_family_index: int,
    random_state: np.random.RandomState,
) -> GeneratedCandidate[SpaceCandidateValue]:
    """Return one structured mutation-family child from explicit leaf-editing capability.

    Parameters
    ----------
    operator : StructuredPathMutationOperator
        Structured leaf-editing operator selected from the mutation family.
    proposal_state : CSAProposalState
        Proposal-adaptation state used for path weighting and covariance
        guidance.
    seed_candidate : SpaceCandidateValue
        Canonical structured seed candidate taken from the active bank.
    seed_score : float
        Scalar score associated with ``seed_candidate``.
    mutation_family_index : int
        Mutation-family index used for attribution keys.
    random_state : np.random.RandomState
        Random state used for path selection and operator sampling.

    Returns
    -------
    GeneratedCandidate[SpaceCandidateValue]
        Structured mutation child together with optional planned attribution.
        Returns the seed unchanged when no active leaf paths are available.
    """
    structured_space = operator.structured_candidate_space
    editable_paths = structured_space.active_leaf_paths_for_validated_candidate(
        seed_candidate,
    )
    if len(editable_paths) == 0:
        return GeneratedCandidate(
            candidate=seed_candidate,
            planned_attribution=planned_mutation_attribution(
                source_score=seed_score,
                proposal_family_key=mutation_family_key(mutation_family_index),
                mutated_leaf_paths=(),
            ),
        )

    exchange_count = sample_exchange_count(
        leaf_count=len(editable_paths),
        max_exchange_fraction=operator.max_selected_path_fraction,
        random_state=random_state,
    )
    selected_paths = plan_mutated_leaf_paths(
        state=proposal_state,
        leaf_paths=editable_paths,
        exchange_count=exchange_count,
        random_state=random_state,
    )

    numeric_subspace_attribution = None
    covariance_candidate = None
    if is_covariance_guided_structured_mutation_operator(operator):
        covariance_descriptor = compile_homogeneous_numeric_subspace(
            structured_space,
            leaf_paths=editable_paths,
        )
        if (
            covariance_descriptor is not None
            and proposal_state.policy.numeric_covariance_strength > 0.0
        ):
            numeric_subspace_attribution = build_numeric_subspace_attribution(
                descriptor=covariance_descriptor,
                source_candidate=seed_candidate,
            )
            covariance_candidate = sample_covariance_guided_candidate(
                descriptor=covariance_descriptor,
                source_candidate=seed_candidate,
                selected_paths=selected_paths,
                proposal_state=proposal_state,
                max_coordinate_fraction=operator.max_coordinate_fraction,
                random_state=random_state,
            )

    if covariance_candidate is None:
        candidate = operator.apply_validated_space_candidate_on_paths(
            candidate=seed_candidate,
            selected_paths=selected_paths,
            random_state=random_state,
        )
    else:
        candidate, selected_paths = covariance_candidate

    return GeneratedCandidate(
        candidate=candidate,
        planned_attribution=planned_mutation_attribution(
            source_score=seed_score,
            proposal_family_key=mutation_family_key(mutation_family_index),
            mutated_leaf_paths=selected_paths,
            numeric_subspace_attribution=numeric_subspace_attribution,
        ),
    )


def apply_variation_operator_from_validated_parents(
    *,
    operator: VariationOperator[CandidateT],
    parents: tuple[CandidateT, ...],
    random_state: np.random.RandomState,
) -> CandidateT:
    """Apply one operator to parents already validated by CSA bank state."""
    if not is_validated_parent_variation_operator(operator):
        return operator.apply(parents, random_state)

    # Built-in structured operators are reached here only from CSA bank entries,
    # which are validated on admission and checkpoint restore. Revalidating the
    # recursive candidate shape here would duplicate the boundary invariant this
    # fast path exists to avoid.
    parent_values = cast(tuple[SpaceCandidateValue, ...], parents)
    return cast(
        CandidateT,
        operator.apply_from_validated_parents(parent_values, random_state),
    )


def plan_next_ask(engine_state: CSAEngineState[CandidateT]) -> CSAAskPlan:
    """Return a pure ask-side plan for the current engine state.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Current CSA engine state.

    Returns
    -------
    CSAAskPlan
        Next ask-side action implied by ``engine_state``.

    Raises
    ------
    RuntimeError
        If the current generation pool is active but not fully observed.
    """
    if engine_state.banking_state.refresh_state is not None:
        return CSAAskPlan(kind="refresh_sample")

    if not engine_state.banking_state.bank.is_full:
        return CSAAskPlan(kind="space_sample")

    if engine_state.generation_state.queue.is_empty:
        if engine_state.generation_state.is_active:
            msg = "cannot start a new CSA child pool before the current pool is fully observed"
            raise RuntimeError(msg)

        return CSAAskPlan(kind="materialize_generation")

    return CSAAskPlan(kind="dequeue_generation")


def materialize_generation(
    *,
    engine_state: CSAEngineState[CandidateT],
    resolved_profile: CSAResolvedProfile[CandidateT],
    space: SearchSpace[BoundaryT, CandidateT],
    diversity_metric: DiversityMetric[CandidateT],
    random_state: np.random.RandomState,
) -> CSAMaterializedGeneration[CandidateT]:
    """Materialize one CSA child pool across the impure sampling seam.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Current CSA engine state.
    resolved_profile : CSAResolvedProfile[CandidateT]
        Fully resolved CSA profile controlling operator schedules.
    space : SearchSpace[BoundaryT, CandidateT]
        Search space used to validate emitted children.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used for partner selection and covariance guidance.
    random_state : np.random.RandomState
        Random state used for seed selection and operator sampling.

    Returns
    -------
    CSAMaterializedGeneration[CandidateT]
        Materialized generation queue together with committed selection and
        trace state.

    Raises
    ------
    RuntimeError
        If the resolved schedule would produce an empty child pool.
    """
    bank = engine_state.banking_state.bank
    reference_bank = engine_state.banking_state.reference_bank
    progression_state = engine_state.progression_state
    selection_state = engine_state.selection_state
    trace_state = engine_state.trace_state

    def snapshot_proposal_families_before() -> tuple[CSAProposalFamilyTrace, ...]:
        mutation_family = resolved_profile.perturbation_schedule.mutation_family
        mutation_weights_by_key = {
            mutation_family_key(index): weight
            for index, weight in enumerate(
                mutation_family_weights(
                    state=engine_state.proposal_state,
                    family=mutation_family,
                )
            )
        }
        family_stats_by_key = {
            family_stat.family_key: family_stat
            for family_stat in engine_state.proposal_state.family_stats
        }
        family_keys = tuple(
            sorted(
                set((*mutation_weights_by_key.keys(), *family_stats_by_key.keys())),
            )
        )
        return tuple(
            CSAProposalFamilyTrace(
                family_key=family_key,
                observation_count=(
                    0
                    if family_key not in family_stats_by_key
                    else family_stats_by_key[family_key].observation_count
                ),
                effective_score_credit=(
                    0.0
                    if family_key not in family_stats_by_key
                    else family_stats_by_key[family_key].effective_score_credit(
                        current_update_index=engine_state.proposal_state.update_index,
                        score_decay=engine_state.proposal_state.policy.score_decay,
                    )
                ),
                mutation_weight=mutation_weights_by_key.get(family_key),
            )
            for family_key in family_keys
        )

    def select_partner_indices_from_entries(
        *,
        entries: tuple[BankEntry[CandidateT], ...],
        seed_index: int,
        partner_count: int,
        partner_mask: frozenset[int],
        weighted_partner_selection: bool,
    ) -> tuple[int, ...]:
        if partner_count <= 0:
            return ()

        return select_partner_indices(
            entries=entries,
            seed_index=seed_index,
            partner_count=partner_count,
            partner_mask=partner_mask,
            distance_between_indices=lambda left_index, right_index: (
                require_valid_distance(
                    diversity_metric.distance(
                        entries[left_index].candidate,
                        entries[right_index].candidate,
                    )
                )
            ),
            weighted_partner_selection=weighted_partner_selection,
            random_state=random_state,
        )

    def emit_candidates_for_seed(
        *,
        seed_index: int,
        active_seed_count: int,
    ) -> tuple[GeneratedCandidate[CandidateT], ...]:
        nonlocal trace_state
        candidates: list[GeneratedCandidate[CandidateT]] = []
        empty_partner_indices: tuple[int, ...] = ()

        for spec in resolved_profile.perturbation_schedule.regular_family:
            for _ in range(spec.count):
                partner_indices = select_partner_indices_from_entries(
                    entries=bank.entries,
                    seed_index=seed_index,
                    partner_count=max(0, spec.operator.arity - 1),
                    partner_mask=progression_state.partner_mask,
                    weighted_partner_selection=resolved_profile.weighted_partner_selection,
                )
                parents = (bank.entries[seed_index].candidate,) + tuple(
                    bank.entries[index].candidate for index in partner_indices
                )
                candidate = apply_variation_operator_from_validated_parents(
                    operator=spec.operator,
                    parents=parents,
                    random_state=random_state,
                )
                space.validate(candidate)
                if trace_state is not None:
                    trace_state = trace_state.record_emitted_child(
                        family="regular",
                        proposal_family_key=None,
                        seed_index=seed_index,
                        primary_source="bank",
                        partner_indices=partner_indices,
                        candidate=candidate,
                    )
                candidates.append(GeneratedCandidate(candidate=candidate))

        should_emit_initial_family = (
            len(resolved_profile.perturbation_schedule.initial_family) > 0
            and reference_bank.is_full
            and len(reference_bank.entries) == len(bank.entries)
            and bank.is_full
        )
        if should_emit_initial_family:
            unused_entry_count = selection_state.count_unused_entries(
                entry_count=len(bank.entries),
                ignored_indices=progression_state.seed_mask,
            )
            use_reference_primary = should_use_reference_primary(
                cycle_count=progression_state.cycle_count,
                entry_count=len(bank.entries),
                active_seed_count=active_seed_count,
                unused_entry_count=unused_entry_count,
                new_bank_cut=resolved_profile.initial_new_bank_cut,
            )

            for spec in resolved_profile.perturbation_schedule.initial_family:
                for _ in range(spec.count):
                    if use_reference_primary:
                        primary_source: CSAPrimarySource = "reference"
                        primary_entries = reference_bank.entries
                    else:
                        primary_source = "bank"
                        primary_entries = bank.entries

                    partner_indices = select_partner_indices_from_entries(
                        entries=reference_bank.entries,
                        seed_index=seed_index,
                        partner_count=max(0, spec.operator.arity - 1),
                        partner_mask=progression_state.partner_mask,
                        weighted_partner_selection=False,
                    )
                    parents = (primary_entries[seed_index].candidate,) + tuple(
                        reference_bank.entries[index].candidate
                        for index in partner_indices
                    )
                    candidate = apply_variation_operator_from_validated_parents(
                        operator=spec.operator,
                        parents=parents,
                        random_state=random_state,
                    )
                    space.validate(candidate)
                    if trace_state is not None:
                        trace_state = trace_state.record_emitted_child(
                            family="initial",
                            proposal_family_key=None,
                            seed_index=seed_index,
                            primary_source=primary_source,
                            partner_indices=partner_indices,
                            candidate=candidate,
                        )
                    candidates.append(GeneratedCandidate(candidate=candidate))

        seed_candidate = bank.entries[seed_index].candidate
        seed_score = bank.entries[seed_index].value
        mutation_family = resolved_profile.perturbation_schedule.mutation_family
        mutation_family_indices = sample_mutation_family_indices(
            state=engine_state.proposal_state,
            family=mutation_family,
            random_state=random_state,
        )
        for mutation_family_index in mutation_family_indices:
            spec = mutation_family[mutation_family_index]
            generated_candidate = GeneratedCandidate(candidate=seed_candidate)
            mutation_operator = spec.operator
            if (
                engine_state.proposal_state.policy.enabled
                and is_structured_path_mutation_operator(mutation_operator)
            ):
                seed_candidate_value = require_space_candidate_value(
                    seed_candidate,
                    operation="CSA structured mutation emission",
                )
                structured_generated_candidate = emit_structured_mutation_candidate(
                    operator=mutation_operator,
                    proposal_state=engine_state.proposal_state,
                    seed_candidate=seed_candidate_value,
                    seed_score=seed_score,
                    mutation_family_index=mutation_family_index,
                    random_state=random_state,
                )
                candidate = cast(CandidateT, structured_generated_candidate.candidate)
                generated_candidate = GeneratedCandidate(
                    candidate=candidate,
                    planned_attribution=structured_generated_candidate.planned_attribution,
                )
            else:
                candidate = apply_variation_operator_from_validated_parents(
                    operator=mutation_operator,
                    parents=(seed_candidate,),
                    random_state=random_state,
                )
                generated_candidate = GeneratedCandidate(
                    candidate=candidate,
                    planned_attribution=planned_mutation_attribution(
                        source_score=seed_score,
                        proposal_family_key=mutation_family_key(mutation_family_index),
                        mutated_leaf_paths=(),
                    ),
                )

            space.validate(candidate)
            if trace_state is not None:
                trace_state = trace_state.record_emitted_child(
                    family="mutation",
                    proposal_family_key=mutation_family_key(mutation_family_index),
                    seed_index=seed_index,
                    primary_source="bank",
                    partner_indices=empty_partner_indices,
                    candidate=candidate,
                )
            candidates.append(generated_candidate)

        return tuple(candidates)

    if not selection_state.has_active_seed:
        bank_distance_workspace = BankDistanceWorkspace(
            entries=bank.entries,
            diversity_metric=diversity_metric,
        )

        def distance_between_entry_indices(left_index: int, right_index: int) -> float:
            return bank_distance_workspace.distance(left_index, right_index)

        selection_state_before = selection_state
        selection_state = prepare_seed_batch(
            current_state=selection_state,
            entries=bank.entries,
            seed_count=resolved_profile.seed_count,
            random_seed_mode=resolved_profile.random_seed_mode,
            masked_seed_indices=progression_state.seed_mask,
            distance_between_indices=distance_between_entry_indices,
            random_state=random_state,
        )
        if trace_state is not None:
            trace_state = trace_state.start_generation(
                stage_index=progression_state.stage_state.stage_index,
                stage_round=progression_state.stage_state.stage_round,
                cycle_count=progression_state.cycle_count,
                bank=bank,
                reference_bank=reference_bank,
                bank_status_before=selection_state_before.resize_bank_status(
                    entry_count=len(bank.entries),
                ),
                seed_mask=progression_state.seed_mask,
                partner_mask=progression_state.partner_mask,
                seed_batch=selection_state.active_seed_indices,
                proposal_families_before=snapshot_proposal_families_before(),
            )

    active_seed_count = len(selection_state.active_seed_indices)
    generated_candidates: list[GeneratedCandidate[CandidateT]] = []
    while selection_state.has_active_seed:
        seed_index, selection_state = selection_state.consume_seed()
        generated_candidates.extend(
            emit_candidates_for_seed(
                seed_index=seed_index,
                active_seed_count=active_seed_count,
            )
        )

    generation_queue = GenerationQueue[CandidateT].from_candidates(
        generated_candidates,
        shuffle=resolved_profile.perturbation_schedule.shuffle_children,
        random_state=random_state,
    )
    if generation_queue.is_empty:
        msg = "cannot materialize an empty CSA child pool"
        raise RuntimeError(msg)

    if trace_state is not None:
        trace_state = trace_state.record_shuffled_pool(
            tuple(
                generated_candidate.candidate
                for generated_candidate in generation_queue.candidates
            ),
        )

    return CSAMaterializedGeneration(
        selection_state=selection_state,
        generation_queue=generation_queue,
        trace_state=trace_state,
    )


def commit_materialized_generation(
    engine_state: CSAEngineState[CandidateT],
    materialized_generation: CSAMaterializedGeneration[CandidateT],
) -> tuple[GeneratedCandidate[CandidateT], CSAEngineState[CandidateT]]:
    """Commit one materialized child pool and dequeue its first candidate.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Current CSA engine state.
    materialized_generation : CSAMaterializedGeneration[CandidateT]
        Materialized child pool ready for commit.

    Returns
    -------
    tuple[GeneratedCandidate[CandidateT], CSAEngineState[CandidateT]]
        First generated candidate together with the committed engine state.
    """
    next_engine_state = replace(
        engine_state,
        selection_state=materialized_generation.selection_state,
        generation_state=engine_state.generation_state.begin(
            materialized_generation.generation_queue,
        ),
        trace_state=materialized_generation.trace_state,
    )
    return dequeue_generation_candidate(next_engine_state)


def dequeue_generation_candidate(
    engine_state: CSAEngineState[CandidateT],
) -> tuple[GeneratedCandidate[CandidateT], CSAEngineState[CandidateT]]:
    """Purely dequeue one already-generated candidate from the active pool.

    Parameters
    ----------
    engine_state : CSAEngineState[CandidateT]
        Current CSA engine state with an active generation queue.

    Returns
    -------
    tuple[GeneratedCandidate[CandidateT], CSAEngineState[CandidateT]]
        Next generated candidate and the updated engine state.
    """
    candidate, next_generation_state = engine_state.generation_state.dequeue_candidate()
    return candidate, replace(
        engine_state,
        generation_state=next_generation_state,
    )
