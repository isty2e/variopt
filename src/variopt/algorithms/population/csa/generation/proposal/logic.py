"""Pure reducer logic for CSA proposal adaptation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import fsum
from typing import TypeVar

import numpy as np

from ......kernel import ProposalLocalSearchContext
from ......randomness import (
    random_state_choice_index,
    random_state_choice_indices_without_replacement,
)
from ......spaces import LeafPath, StructuredSearchSpace
from ......spaces.types import SpaceCandidateValue
from ......typevars import CandidateT
from ...banking.update.transition import CSABankTransition
from ..perturbation import CSAPerturbationSpec
from .evidence import (
    CSAProposalEvaluation,
    CSAProposalOutcomeEvidence,
    derive_proposal_credits,
)
from .state.aggregate import CSAProposalState
from .state.attribution import (
    AdaptiveProposalGeneratorKind,
    NonAdaptiveProposalAttribution,
    NumericSubspaceAttribution,
    NumericSubspaceDisplacement,
    PlannedProposalAttribution,
    ProposalAttribution,
)
from .state.credit import (
    ProposalFamilyCreditSummary,
    ProposalGenerationCreditBatch,
    ProposalLeafCreditSummary,
    ProposalNumericDisplacementCredit,
)

BoundaryT = TypeVar("BoundaryT")
StructuredCandidateT = TypeVar("StructuredCandidateT", bound=SpaceCandidateValue)


@dataclass(frozen=True, slots=True)
class _LeafLocalSearchSignal:
    """File-local local-search signal for one structured leaf path."""

    path: LeafPath
    credit_rate: float
    local_displacement_credit_rate: float
    recent_failure_streak: int
    last_update_index: int

    @property
    def total_credit_rate(self) -> float:
        """Return the positive local-search support credit for one path."""
        return self.credit_rate + self.local_displacement_credit_rate


def _is_in_failure_cooldown(
    *,
    signal: _LeafLocalSearchSignal,
    state: CSAProposalState,
) -> bool:
    """Return whether one leaf is in the reactive post-failure cooldown window."""
    if signal.recent_failure_streak <= 0:
        return False

    cooldown_updates = state.policy.local_search_failure_cooldown_updates
    if cooldown_updates == 0:
        return False

    elapsed_updates = state.update_index - signal.last_update_index
    if elapsed_updates < 0:
        msg = "leaf signal update index must not lie in the future"
        raise ValueError(msg)

    return elapsed_updates < cooldown_updates


def _leaf_local_search_signals(
    *,
    state: CSAProposalState,
    leaf_paths: Sequence[LeafPath],
) -> tuple[_LeafLocalSearchSignal, ...]:
    """Return local-search support signals for one structured leaf family."""
    leaf_stats_by_path = {leaf_stat.path: leaf_stat for leaf_stat in state.leaf_stats}
    local_displacement_leaf_stats_by_path = {
        leaf_stat.path: leaf_stat for leaf_stat in state.local_displacement_leaf_stats
    }
    signals: list[_LeafLocalSearchSignal] = []
    for path in leaf_paths:
        normalized_path = tuple(path)
        leaf_stat = leaf_stats_by_path.get(normalized_path)
        credit_rate = 0.0
        if leaf_stat is not None:
            credit_rate = max(
                0.0,
                leaf_stat.effective_credit_rate(
                    current_update_index=state.update_index,
                    credit_decay=state.policy.credit_decay,
                ),
            )

        local_displacement_leaf_stat = local_displacement_leaf_stats_by_path.get(
            normalized_path,
        )
        local_displacement_credit_rate = 0.0
        if local_displacement_leaf_stat is not None:
            local_displacement_credit_rate = max(
                0.0,
                local_displacement_leaf_stat.effective_credit_rate(
                    current_update_index=state.update_index,
                    credit_decay=state.policy.credit_decay,
                ),
            )

        recent_failure_streak = 0
        last_update_index = 0
        if leaf_stat is not None:
            recent_failure_streak = max(
                recent_failure_streak,
                leaf_stat.recent_failure_streak,
            )
            last_update_index = max(last_update_index, leaf_stat.last_update_index)
        if local_displacement_leaf_stat is not None:
            recent_failure_streak = max(
                recent_failure_streak,
                local_displacement_leaf_stat.recent_failure_streak,
            )
            last_update_index = max(
                last_update_index,
                local_displacement_leaf_stat.last_update_index,
            )

        signals.append(
            _LeafLocalSearchSignal(
                path=normalized_path,
                credit_rate=credit_rate,
                local_displacement_credit_rate=local_displacement_credit_rate,
                recent_failure_streak=recent_failure_streak,
                last_update_index=last_update_index,
            )
        )

    return tuple(signals)


def plan_mutated_leaf_paths(
    *,
    state: CSAProposalState,
    leaf_paths: Sequence[LeafPath],
    exchange_count: int,
    random_state: np.random.RandomState,
) -> tuple[LeafPath, ...]:
    """Return one weighted mutation-path selection from proposal history.

    Parameters
    ----------
    state : CSAProposalState
        Proposal adaptation state providing leaf-level weights.
    leaf_paths : Sequence[LeafPath]
        Structured leaf paths available for mutation.
    exchange_count : int
        Number of distinct leaf paths to select.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    tuple[LeafPath, ...]
        Selected mutated leaf paths.

    Raises
    ------
    ValueError
        If ``exchange_count`` is non-positive or exceeds the number of
        available leaf paths.
    """
    if exchange_count <= 0:
        msg = "exchange_count must be positive"
        raise ValueError(msg)

    if exchange_count > len(leaf_paths):
        msg = "exchange_count must not exceed the number of leaf paths"
        raise ValueError(msg)

    weights = mutation_leaf_weights(state=state, leaf_paths=leaf_paths)
    selected_indices = random_state_choice_indices_without_replacement(
        random_state,
        len(leaf_paths),
        exchange_count,
        weights=weights,
    )
    return tuple(leaf_paths[index] for index in selected_indices)


def mutation_leaf_weights(
    *,
    state: CSAProposalState,
    leaf_paths: Sequence[LeafPath],
) -> tuple[float, ...]:
    """Return normalized mutation-path weights for one structured candidate.

    Parameters
    ----------
    state : CSAProposalState
        Proposal adaptation state providing accumulated leaf credit.
    leaf_paths : Sequence[LeafPath]
        Structured leaf paths available for mutation.

    Returns
    -------
    tuple[float, ...]
        Normalized per-leaf mutation weights aligned with ``leaf_paths``.

    Raises
    ------
    ValueError
        If ``leaf_paths`` is empty.
    """
    if len(leaf_paths) == 0:
        msg = "leaf_paths must not be empty"
        raise ValueError(msg)

    if not state.policy.enabled:
        uniform_weight = 1.0 / float(len(leaf_paths))
        return tuple(uniform_weight for _ in leaf_paths)

    signals = _leaf_local_search_signals(state=state, leaf_paths=leaf_paths)
    raw_weights: list[float] = []
    for signal in signals:
        raw_weights.append(
            state.policy.minimum_leaf_weight
            + (state.policy.leaf_bias_strength * signal.credit_rate)
            + (
                state.policy.local_displacement_leaf_bias_strength
                * signal.local_displacement_credit_rate
            )
        )

    weight_sum = sum(raw_weights)
    return tuple(weight / weight_sum for weight in raw_weights)


def proposal_local_search_context(
    *,
    state: CSAProposalState,
    leaf_paths: Sequence[LeafPath],
    attribution: ProposalAttribution | None = None,
) -> ProposalLocalSearchContext | None:
    """Return one history-conditioned local-search context for a structured proposal.

    Parameters
    ----------
    state : CSAProposalState
        Proposal adaptation state providing leaf-level support signals.
    leaf_paths : Sequence[LeafPath]
        Structured leaf paths available to local search.
    attribution : ProposalAttribution | None, default=None
        Proposal attribution associated with the candidate, when available.

    Returns
    -------
    ProposalLocalSearchContext | None
        Prioritized local-search context, or ``None`` when adaptation is
        disabled or no useful signal is available.
    """
    if not state.policy.enabled or len(leaf_paths) == 0:
        return None

    normalized_leaf_paths = tuple(tuple(path) for path in leaf_paths)
    signals = _leaf_local_search_signals(
        state=state,
        leaf_paths=normalized_leaf_paths,
    )
    signals_by_path = {signal.path: signal for signal in signals}
    if attribution is None and not any(
        signal.total_credit_rate > 0.0 for signal in signals
    ):
        return None

    prioritized_leaf_paths: list[LeafPath] = []
    seen_leaf_paths: set[LeafPath] = set()
    available_leaf_paths = set(normalized_leaf_paths)
    weights = mutation_leaf_weights(state=state, leaf_paths=normalized_leaf_paths)
    weights_by_path = {
        path: weights[index] for index, path in enumerate(normalized_leaf_paths)
    }
    mutated_signals: list[_LeafLocalSearchSignal] = []
    if attribution is not None:
        unique_mutated_paths = tuple(
            normalized_path
            for normalized_path in (
                tuple(path) for path in attribution.mutated_leaf_paths
            )
            if normalized_path in available_leaf_paths
        )
        mutated_signals = sorted(
            (signals_by_path[path] for path in dict.fromkeys(unique_mutated_paths)),
            key=lambda signal: (
                _is_in_failure_cooldown(signal=signal, state=state),
                signal.recent_failure_streak,
                -weights_by_path[signal.path],
                normalized_leaf_paths.index(signal.path),
            ),
        )
        for signal in mutated_signals:
            prioritized_leaf_paths.append(signal.path)
            seen_leaf_paths.add(signal.path)

    supporting_mutated_signal_count = sum(
        1
        for signal in mutated_signals
        if (
            signal.total_credit_rate > 0.0
            and not _is_in_failure_cooldown(signal=signal, state=state)
        )
    )
    eligible_mutated_signal_count = sum(
        1
        for signal in mutated_signals
        if (
            signal.recent_failure_streak
            < state.policy.local_search_disable_failure_streak
            and not _is_in_failure_cooldown(signal=signal, state=state)
        )
    )
    weighted_remaining_leaf_paths = sorted(
        (
            (
                _is_in_failure_cooldown(signal=signals[index], state=state),
                -weights[index],
                signals[index].recent_failure_streak,
                index,
                path,
            )
            for index, path in enumerate(normalized_leaf_paths)
            if path not in seen_leaf_paths
        ),
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    prioritized_leaf_paths.extend(
        path for _, _, _, _, path in weighted_remaining_leaf_paths
    )

    local_search_enabled = True
    local_budget: int | None = None
    if len(mutated_signals) > 0:
        local_search_enabled = (
            supporting_mutated_signal_count > 0 or eligible_mutated_signal_count > 0
        )
        if local_search_enabled:
            local_budget = min(
                state.policy.local_search_max_budget,
                (
                    state.policy.local_search_base_budget
                    + supporting_mutated_signal_count
                    + max(0, eligible_mutated_signal_count - 1)
                ),
            )

    return ProposalLocalSearchContext(
        enabled=local_search_enabled,
        local_budget=local_budget,
        prioritized_leaf_paths=tuple(prioritized_leaf_paths),
    )


def planned_mutation_attribution(
    *,
    proposal_family_key: str | None = None,
    mutated_leaf_paths: Sequence[LeafPath],
    numeric_subspace_attribution: NumericSubspaceAttribution | None = None,
    generator_kind: AdaptiveProposalGeneratorKind = "mutation",
) -> PlannedProposalAttribution:
    """Return one pre-id attribution record for a generated mutation child.

    Parameters
    ----------
    proposal_family_key : str | None, default=None
        Canonical proposal family key for the generated child, if any.
    mutated_leaf_paths : Sequence[LeafPath]
        Structured leaf paths mutated to produce the child.
    numeric_subspace_attribution : NumericSubspaceAttribution | None, default=None
        Numeric subspace metadata attached to the proposal, if any.
    generator_kind : AdaptiveProposalGeneratorKind, default="mutation"
        Whether the mutation operator changed the source candidate or passed
        it through unchanged.

    Returns
    -------
    PlannedProposalAttribution
        Pre-id attribution record ready to be bound to a generated proposal id.
    """
    return PlannedProposalAttribution(
        proposal_family_key=proposal_family_key,
        mutated_leaf_paths=tuple(mutated_leaf_paths),
        numeric_subspace_attribution=numeric_subspace_attribution,
        generator_kind=generator_kind,
    )


def mutation_family_key(index: int) -> str:
    """Return the canonical proposal-family key for one mutation-family entry.

    Parameters
    ----------
    index : int
        Zero-based mutation family index.

    Returns
    -------
    str
        Canonical proposal family key.

    Raises
    ------
    ValueError
        If ``index`` is negative.
    """
    if index < 0:
        msg = "index must be non-negative"
        raise ValueError(msg)

    return f"mutation:{index}"


def mutation_family_weights(
    *,
    state: CSAProposalState,
    family: Sequence[CSAPerturbationSpec[CandidateT]],
) -> tuple[float, ...]:
    """Return normalized mutation-family weights for adaptive proposal planning.

    Parameters
    ----------
    state : CSAProposalState
        Proposal adaptation state providing family-level reward statistics.
    family : Sequence[CSAPerturbationSpec[CandidateT]]
        Mutation family specification for the current ask step.

    Returns
    -------
    tuple[float, ...]
        Normalized weights aligned with ``family``.

    Raises
    ------
    ValueError
        If ``family`` is empty.
    """
    if len(family) == 0:
        msg = "family must not be empty"
        raise ValueError(msg)

    base_weights = tuple(float(spec.count) for spec in family)
    if not state.policy.enabled:
        weight_sum = sum(base_weights)
        return tuple(weight / weight_sum for weight in base_weights)

    family_stats_by_key = {
        family_stat.family_key: family_stat for family_stat in state.family_stats
    }
    raw_weights: list[float] = []
    for index, spec in enumerate(family):
        family_key = mutation_family_key(index)
        family_stat = family_stats_by_key.get(family_key)
        credit_rate = 0.0
        if family_stat is not None:
            credit_rate = max(
                0.0,
                family_stat.effective_credit_rate(
                    current_update_index=state.update_index,
                    credit_decay=state.policy.credit_decay,
                ),
            )

        raw_weights.append(
            float(spec.count)
            + state.policy.minimum_family_weight
            + (state.policy.family_bias_strength * credit_rate),
        )

    weight_sum = sum(raw_weights)
    return tuple(weight / weight_sum for weight in raw_weights)


def sample_mutation_family_indices(
    *,
    state: CSAProposalState,
    family: Sequence[CSAPerturbationSpec[CandidateT]],
    random_state: np.random.RandomState,
) -> tuple[int, ...]:
    """Return one adaptive mutation-family schedule for one active seed.

    Parameters
    ----------
    state : CSAProposalState
        Proposal adaptation state providing family-level weights.
    family : Sequence[CSAPerturbationSpec[CandidateT]]
        Mutation family specification for the current ask step.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    tuple[int, ...]
        Mutation family indices to draw for the active seed.
    """
    total_child_count = sum(spec.count for spec in family)
    if total_child_count <= 0:
        return ()

    if not state.policy.enabled:
        return tuple(
            index for index, spec in enumerate(family) for _ in range(spec.count)
        )

    weights = mutation_family_weights(state=state, family=family)
    return tuple(
        random_state_choice_index(
            random_state,
            len(family),
            weights=weights,
        )
        for _ in range(total_child_count)
    )


def record_proposal_attribution(
    state: CSAProposalState,
    attribution: ProposalAttribution | NonAdaptiveProposalAttribution,
) -> CSAProposalState:
    """Return a state with one additional pending proposal attribution.

    Disabled proposal policies keep the state unchanged so that future ask-side
    wiring can call this function without branching on policy at the call site.

    Parameters
    ----------
    state : CSAProposalState
        Proposal adaptation state to update.
    attribution : ProposalAttribution | NonAdaptiveProposalAttribution
        Explicit adaptive or non-adaptive provenance to append to the pending
        queue.

    Returns
    -------
    CSAProposalState
        Updated state, or the original state when proposal adaptation is
        disabled.
    """
    if not state.policy.enabled:
        return state

    return state.register_pending_attribution(attribution)


def collect_proposal_outcome_evidence(
    state: CSAProposalState,
    evaluations: Sequence[CSAProposalEvaluation[CandidateT]],
    bank_transitions: Sequence[CSABankTransition],
) -> tuple[tuple[CSAProposalOutcomeEvidence[CandidateT], ...], CSAProposalState]:
    """Join successful evaluations to bank transitions and pending provenance.

    Parameters
    ----------
    state : CSAProposalState
        Proposal state owning pending provenance.
    evaluations : Sequence[CSAProposalEvaluation[CandidateT]]
        Successful feedback records in bank-reduction order.
    bank_transitions : Sequence[CSABankTransition]
        Bank transitions aligned one-to-one with ``evaluations``.

    Returns
    -------
    tuple[tuple[CSAProposalOutcomeEvidence[CandidateT], ...], CSAProposalState]
        Adaptive evidence records and the state after every aligned provenance
        entry has been consumed exactly once.

    Raises
    ------
    ValueError
        If evaluation and transition counts differ or a proposal has no pending
        provenance while adaptation is enabled.
    TypeError
        If pending provenance has an unsupported runtime variant.
    """
    if not state.policy.enabled:
        return (), state

    evaluation_tuple = tuple(evaluations)
    transition_tuple = tuple(bank_transitions)
    if len(evaluation_tuple) != len(transition_tuple):
        msg = "bank transitions must align one-to-one with proposal evaluations"
        raise ValueError(msg)

    proposal_ids: list[str] = []
    for evaluation, bank_transition in zip(
        evaluation_tuple,
        transition_tuple,
        strict=True,
    ):
        proposal_id = evaluation.observation.proposal.proposal_id
        if proposal_id is None:
            msg = "proposal evaluations must reference proposal ids"
            raise ValueError(msg)
        if bank_transition.proposal_id != proposal_id:
            msg = "bank transition must align with evaluation proposal id"
            raise ValueError(msg)
        proposal_ids.append(proposal_id)

    provenances, next_state = state.consume_pending_attributions(proposal_ids)
    outcome_evidence: list[CSAProposalOutcomeEvidence[CandidateT]] = []
    for evaluation, bank_transition, provenance in zip(
        evaluation_tuple,
        transition_tuple,
        provenances,
        strict=True,
    ):
        if type(provenance) is NonAdaptiveProposalAttribution:
            continue
        if type(provenance) is not ProposalAttribution:
            msg = "pending provenance must be a canonical proposal variant"
            raise TypeError(msg)

        outcome_evidence.append(
            CSAProposalOutcomeEvidence(
                attribution=provenance,
                evaluation=evaluation,
                bank_transition=bank_transition,
            ),
        )

    return tuple(outcome_evidence), next_state


def consume_refresh_proposal_provenance(
    state: CSAProposalState,
    evaluations: Sequence[CSAProposalEvaluation[CandidateT]],
) -> CSAProposalState:
    """Consume explicit non-adaptive provenance for accepted refresh feedback.

    Refresh samples do not pass through ordinary bank admission and therefore do
    not produce adaptive outcome evidence. When adaptation is enabled, each one
    must still carry the explicit ``refresh_sample`` classification established
    at ask time.
    """
    if not state.policy.enabled:
        return state

    proposal_ids: list[str] = []
    for evaluation in evaluations:
        proposal_id = evaluation.observation.proposal.proposal_id
        if proposal_id is None:
            msg = "refresh evaluations must reference proposal ids"
            raise ValueError(msg)
        proposal_ids.append(proposal_id)

    provenances, next_state = state.consume_pending_attributions(proposal_ids)
    for provenance in provenances:
        if (
            type(provenance) is not NonAdaptiveProposalAttribution
            or provenance.reason != "refresh_sample"
        ):
            msg = "refresh evaluations require refresh_sample provenance"
            raise ValueError(msg)

    return next_state


def update_proposal_state(
    state: CSAProposalState,
    outcome_evidence: Sequence[CSAProposalOutcomeEvidence[CandidateT]],
    *,
    infer_local_displacement_leaf_paths: (
        Callable[[CandidateT, CandidateT], tuple[LeafPath, ...]] | None
    ) = None,
    infer_numeric_subspace_displacement: (
        Callable[[ProposalAttribution, CandidateT], NumericSubspaceDisplacement | None]
        | None
    ) = None,
) -> CSAProposalState:
    """Reduce completed proposal outcome evidence into adaptive statistics.

    The evidence has already joined proposal provenance, successful evaluation
    metadata, and the conclusive bank transition. The reducer derives bounded
    final-survival credit, canonicalizes proposal order, and applies one state
    update for the completed generation.

    Parameters
    ----------
    state : CSAProposalState
        Current proposal adaptation state.
    outcome_evidence : Sequence[CSAProposalOutcomeEvidence[CandidateT]]
        Completed adaptive evidence records to reduce.
    infer_local_displacement_leaf_paths : Callable[[CandidateT, CandidateT], tuple[LeafPath, ...]] | None, default=None
        Optional callback that infers structured leaf paths changed by local
        post-processing.
    infer_numeric_subspace_displacement : Callable[[ProposalAttribution, CandidateT], NumericSubspaceDisplacement | None] | None, default=None
        Optional callback that infers successful numeric subspace displacements.

    Returns
    -------
    CSAProposalState
        Reduced proposal adaptation state after consuming the observation batch.
    """
    if not state.policy.enabled or len(outcome_evidence) == 0:
        return state

    proposal_credits = derive_proposal_credits(outcome_evidence)
    family_credits_by_key: dict[str, list[float]] = {}
    mutation_leaf_credits_by_path: dict[LeafPath, list[float]] = {}
    local_leaf_credits_by_path: dict[LeafPath, list[float]] = {}
    numeric_displacement_credits: list[ProposalNumericDisplacementCredit] = []
    for proposal_credit in proposal_credits:
        evidence = proposal_credit.outcome_evidence
        attribution = evidence.attribution
        evaluation = evidence.evaluation
        observation = evaluation.observation
        local_displacement_leaf_paths: tuple[LeafPath, ...] = ()
        numeric_displacement: NumericSubspaceDisplacement | None = None
        explicit_leaf_paths = evaluation.refinement_changed_leaf_paths
        if explicit_leaf_paths is not None:
            local_displacement_leaf_paths = explicit_leaf_paths
        elif infer_local_displacement_leaf_paths is not None:
            local_displacement_leaf_paths = infer_local_displacement_leaf_paths(
                observation.proposal.candidate,
                observation.candidate,
            )
        if (
            proposal_credit.pipeline_credit > 0.0
            and infer_numeric_subspace_displacement is not None
        ):
            numeric_displacement = infer_numeric_subspace_displacement(
                attribution,
                observation.candidate,
            )

        family_key = attribution.proposal_family_key
        if family_key is not None:
            family_credits_by_key.setdefault(family_key, []).append(
                proposal_credit.pipeline_credit
            )
        for leaf_credit in proposal_credit.leaf_association_credits(
            local_displacement_leaf_paths=local_displacement_leaf_paths,
        ):
            credits_by_path = (
                mutation_leaf_credits_by_path
                if leaf_credit.source == "mutation"
                else local_leaf_credits_by_path
            )
            credits_by_path.setdefault(leaf_credit.path, []).append(leaf_credit.credit)
        if numeric_displacement is not None:
            numeric_displacement_credits.append(
                ProposalNumericDisplacementCredit(
                    displacement=numeric_displacement,
                    credit=proposal_credit.pipeline_credit,
                ),
            )

    generation_credit = ProposalGenerationCreditBatch(
        evidence_count=len(proposal_credits),
        family_summaries=tuple(
            ProposalFamilyCreditSummary(
                family_key=family_key,
                observation_count=len(credits),
                total_credit=fsum(credits),
            )
            for family_key, credits in family_credits_by_key.items()
        ),
        mutation_leaf_summaries=tuple(
            ProposalLeafCreditSummary(
                path=path,
                observation_count=len(credits),
                total_credit=fsum(credits),
            )
            for path, credits in mutation_leaf_credits_by_path.items()
        ),
        local_displacement_leaf_summaries=tuple(
            ProposalLeafCreditSummary(
                path=path,
                observation_count=len(credits),
                total_credit=fsum(credits),
            )
            for path, credits in local_leaf_credits_by_path.items()
        ),
        numeric_displacement_credits=tuple(numeric_displacement_credits),
    )
    return state.record_generation_credit(
        generation_credit,
    )


def infer_structured_local_displacement_leaf_paths(
    *,
    space: StructuredSearchSpace[BoundaryT, StructuredCandidateT],
    proposal_candidate: StructuredCandidateT,
    observed_candidate: StructuredCandidateT,
) -> tuple[LeafPath, ...]:
    """Return structured leaf paths changed by local post-processing.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, StructuredCandidateT]
        Structured search space shared by the proposal and observed candidate.
    proposal_candidate : StructuredCandidateT
        Candidate proposed before local post-processing.
    observed_candidate : StructuredCandidateT
        Candidate observed after local post-processing.

    Returns
    -------
    tuple[LeafPath, ...]
        Structured leaf paths whose active status or canonical value changed.
    """
    space.validate(proposal_candidate)
    space.validate(observed_candidate)
    proposal_active_leaf_paths = set(
        space.active_leaf_paths_for_validated_candidate(proposal_candidate),
    )
    observed_active_leaf_paths = set(
        space.active_leaf_paths_for_validated_candidate(observed_candidate),
    )
    return tuple(
        path
        for path in space.leaf_paths()
        if (
            (path in proposal_active_leaf_paths) != (path in observed_active_leaf_paths)
        )
        or (
            path in proposal_active_leaf_paths
            and path in observed_active_leaf_paths
            and space.leaf_value_at_validated_path(proposal_candidate, path)
            != space.leaf_value_at_validated_path(observed_candidate, path)
        )
    )
