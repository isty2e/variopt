"""Iterated local-search kernel for structured discrete spaces."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

import numpy as np
from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....artifacts import Observation, Proposal, ProposalEvaluationSpec
from ....kernel import (
    Kernel,
    KernelDiagnostics,
    KernelStatus,
    ProposalBatchQuery,
    ProposalLocalSearchContext,
)
from ....outcomes import EvaluationOutcome
from ....randomness import RandomSeed, normalize_random_state
from ....spaces import LeafPath
from .neighborhood import (
    BoundaryT,
    DiscreteLeafSpace,
    StructuredCandidateT,
    StructuredKickPolicy,
)
from .runtime.kicks import (
    accepts_strict_improvement,
    sample_structured_kick_candidate,
)
from .runtime.prepared import (
    PreparedStructuredLocalSearchRuntime,
    prepare_structured_local_search_runtime,
)
from .runtime.search import run_leafwise_local_search_episode


@dataclass(frozen=True, slots=True)
class StructuredIteratedLocalSearchKernel(
    FrozenGenericSlotsCompat,
    Kernel[
        ProposalBatchQuery[
            BoundaryT,
            StructuredCandidateT,
            Observation[StructuredCandidateT],
        ],
        tuple[
            EvaluationOutcome[
                StructuredCandidateT,
                Observation[StructuredCandidateT],
            ],
            ...,
        ],
    ],
    Generic[BoundaryT, StructuredCandidateT],
):
    """Structured iterated local-search wrapper with an explicit kick policy.

    Parameters
    ----------
    max_steps : int, default=8
        Maximum number of accepted improvement steps for each proposal.
    max_kicks : int, default=2
        Maximum number of kicks attempted after the initial local optimum.
    kick_policy : StructuredKickPolicy, optional
        Policy that determines how kicks perturb the incumbent.
    random_state : RandomSeed, optional
        Seed or random-state object used to sample kicks.

    Notes
    -----
    The current implementation intentionally fixes the inner local improvement
    to deterministic leafwise hill climbing while keeping the kick and
    acceptance logic in separate private seams for future generalization.
    """

    max_steps: int = 8
    max_kicks: int = 2
    kick_policy: StructuredKickPolicy = StructuredKickPolicy()
    random_state: RandomSeed = 0

    def __post_init__(self) -> None:
        """Validate iterated local-search budgets.

        Raises
        ------
        ValueError
            Raised when ``max_steps`` or ``max_kicks`` is invalid.
        """
        if self.max_steps <= 0:
            msg = "max_steps must be positive"
            raise ValueError(msg)

        if self.max_kicks <= 0:
            msg = "max_kicks must be positive"
            raise ValueError(msg)

    def _evaluate_original_proposal(
        self,
        *,
        runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
        proposal: Proposal[StructuredCandidateT],
        proposal_evaluation_spec: ProposalEvaluationSpec | None,
    ) -> EvaluationOutcome[StructuredCandidateT]:
        """Evaluate one original proposal once without local search."""
        return runtime.evaluate_original_proposal(
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
            method="iterated_local_search",
        )

    def _proposal_context(
        self,
        *,
        runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
        proposal_index: int,
    ) -> ProposalLocalSearchContext | None:
        """Return the canonical local-search context for one proposal index."""
        return runtime.proposal_context(proposal_index=proposal_index)

    def _episode_max_steps(
        self,
        *,
        runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
        context: ProposalLocalSearchContext | None,
    ) -> int:
        """Return the per-episode step budget after context overrides."""
        return runtime.episode_max_steps(
            default_max_steps=self.max_steps,
            context=context,
        )

    def _ordered_leaf_schedule(
        self,
        *,
        runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
        context: ProposalLocalSearchContext | None,
    ) -> tuple[tuple[LeafPath, DiscreteLeafSpace], ...]:
        """Return the leaf traversal order after one proposal-specific reordering."""
        return runtime.ordered_leaf_schedule(context=context)

    def _optimize_proposal(
        self,
        *,
        runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
        proposal_index: int,
        proposal: Proposal[StructuredCandidateT],
        random_state: np.random.RandomState,
        reserved_count: int,
    ) -> EvaluationOutcome[StructuredCandidateT]:
        """Run one iterated local-search episode for one original proposal."""
        runtime.neighborhood.space.validate(proposal.candidate)
        context = self._proposal_context(runtime=runtime, proposal_index=proposal_index)
        proposal_evaluation_spec = runtime.proposal_evaluation_spec(
            proposal_index=proposal_index,
        )
        if context is not None and not context.enabled:
            return self._evaluate_original_proposal(
                runtime=runtime,
                proposal=proposal,
                proposal_evaluation_spec=proposal_evaluation_spec,
            )

        episode_max_steps = self._episode_max_steps(runtime=runtime, context=context)
        leaf_schedule = self._ordered_leaf_schedule(
            runtime=runtime,
            context=context,
        )
        incumbent_result = run_leafwise_local_search_episode(
            runtime=runtime,
            initial_candidate=proposal.candidate,
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
            leaf_schedule=leaf_schedule,
            max_steps=episode_max_steps,
            reserved_count=reserved_count,
        )
        incumbent_record = incumbent_result.record
        incumbent_candidate = incumbent_record.candidate
        incumbent_value = incumbent_record.value
        incumbent_score = incumbent_record.score
        evaluation_count = incumbent_result.evaluation_count
        completed_steps = incumbent_result.completed_steps
        accepted_refinement = completed_steps > 0
        kick_count = 0
        terminal_message = "max_kicks reached before iterated local-search termination"
        budget_exhausted = incumbent_result.budget_exhausted

        while (
            not budget_exhausted
            and completed_steps < episode_max_steps
            and kick_count < self.max_kicks
        ):
            kicked_candidate = sample_structured_kick_candidate(
                runtime=runtime,
                candidate=incumbent_candidate,
                leaf_schedule=leaf_schedule,
                kick_policy=self.kick_policy,
                random_state=random_state,
            )
            if kicked_candidate is None:
                terminal_message = (
                    "no admissible kick found for the configured kick policy"
                )
                break

            if not runtime.can_evaluate(reserved_count=reserved_count):
                budget_exhausted = True
                break

            kick_count += 1
            kicked_result = run_leafwise_local_search_episode(
                runtime=runtime,
                initial_candidate=kicked_candidate,
                proposal=proposal,
                proposal_evaluation_spec=proposal_evaluation_spec,
                leaf_schedule=leaf_schedule,
                max_steps=episode_max_steps - completed_steps,
                reserved_count=reserved_count,
            )
            evaluation_count += kicked_result.evaluation_count
            completed_steps += kicked_result.completed_steps
            budget_exhausted = kicked_result.budget_exhausted
            kicked_record = kicked_result.record
            if accepts_strict_improvement(
                incumbent_score=incumbent_score,
                candidate_score=kicked_record.score,
            ):
                incumbent_candidate = kicked_record.candidate
                incumbent_value = kicked_record.value
                incumbent_score = kicked_record.score
                accepted_refinement = True

        status = KernelStatus.STOPPED
        if budget_exhausted:
            terminal_message = "evaluation budget exhausted before local convergence"
        if completed_steps >= episode_max_steps:
            terminal_message = (
                "max_steps reached before iterated local-search termination"
            )

        refinement = None
        if accepted_refinement:
            refinement = runtime.candidate_refinement(
                source_candidate=proposal.candidate,
                refined_candidate=incumbent_candidate,
            )

        return EvaluationOutcome(
            record=Observation.from_objective_value(
                proposal=proposal,
                proposal_evaluation_spec=proposal_evaluation_spec,
                candidate=incumbent_candidate,
                value=incumbent_value,
                direction=runtime.query.problem.direction,
            ),
            evaluation_count=evaluation_count,
            kernel_diagnostics=KernelDiagnostics(
                backend="structured.local_search",
                method="iterated_local_search",
                status=status,
                message=terminal_message,
            ),
            refinement=refinement,
            candidate_equal=runtime.query.problem.space.candidates_equal,
        )

    @override
    def run(
        self,
        query: ProposalBatchQuery[BoundaryT, StructuredCandidateT],
        runner: Callable[
            [ProposalBatchQuery[BoundaryT, StructuredCandidateT]],
            tuple[EvaluationOutcome[StructuredCandidateT], ...],
        ],
    ) -> tuple[EvaluationOutcome[StructuredCandidateT], ...]:
        """Run iterated local search for each proposal in a batch.

        Parameters
        ----------
        query : ProposalBatchQuery[BoundaryT, StructuredCandidateT]
            Proposal batch and evaluation context to optimize.
        runner : Callable[[ProposalBatchQuery[BoundaryT, StructuredCandidateT]], tuple[EvaluationOutcome[StructuredCandidateT], ...]]
            Evaluator runner used to score local-search and kick candidates.

        Returns
        -------
        tuple[EvaluationOutcome[StructuredCandidateT], ...]
            Local-search outcomes aligned to ``query.proposals``.
        """
        runtime: PreparedStructuredLocalSearchRuntime[
            BoundaryT,
            StructuredCandidateT,
        ] = prepare_structured_local_search_runtime(
            query=query,
            runner=runner,
        )
        random_state = normalize_random_state(self.random_state)
        return tuple(
            self._optimize_proposal(
                runtime=runtime,
                proposal_index=proposal_index,
                proposal=proposal,
                random_state=random_state,
                reserved_count=len(query.proposals) - proposal_index - 1,
            )
            for proposal_index, proposal in enumerate(query.proposals)
        )
