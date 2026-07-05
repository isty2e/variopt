"""Iterated local-search kernel for structured discrete spaces."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic

import numpy as np
from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....artifacts import (
    EvaluationAttemptBatch,
    KernelDiagnostics,
    KernelStatus,
    ObservationPayload,
    Proposal,
    ProposalEvaluationSpec,
)
from ....kernel import (
    Kernel,
    ProposalBatchQuery,
    ProposalLocalSearchContext,
)
from ....randomness import RandomSeed, RandomStateSnapshot
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
    structured_episode_attempt_batch,
)
from .runtime.search import run_leafwise_local_search_episode


@dataclass(frozen=True, slots=True)
class StructuredIteratedLocalSearchKernel(
    FrozenGenericSlotsCompat,
    Kernel[
        ProposalBatchQuery[
            BoundaryT,
            StructuredCandidateT,
            ObservationPayload,
        ],
        EvaluationAttemptBatch[
            StructuredCandidateT,
            ObservationPayload,
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
        Seed used to initialize the fallback kick-sampling stream when a
        proposal does not provide an episode-local random-state snapshot.

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
    _random_state_snapshot: RandomStateSnapshot = field(
        init=False,
        repr=False,
        compare=False,
    )

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

        object.__setattr__(
            self,
            "_random_state_snapshot",
            RandomStateSnapshot.from_seed(self.random_state),
        )

    def _evaluate_original_proposal(
        self,
        *,
        runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
        proposal: Proposal[StructuredCandidateT],
        proposal_evaluation_spec: ProposalEvaluationSpec | None,
    ) -> EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]:
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
    ) -> EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]:
        """Run one iterated local-search episode for one original proposal."""
        runtime.neighborhood.space.validate(proposal.candidate)
        failed_attempts: list[
            EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]
        ] = []
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
        episode_random_state = random_state
        if context is not None and context.random_state_snapshot is not None:
            episode_random_state = context.random_state_snapshot.materialize()

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
        failed_attempts.extend(incumbent_result.failed_attempts)
        incumbent_success = incumbent_result.success
        if incumbent_success is None:
            return structured_episode_attempt_batch(
                success=None,
                failed_attempts=failed_attempts,
            )

        incumbent_candidate = incumbent_success.candidate
        incumbent_value = incumbent_success.payload.value
        incumbent_score = incumbent_success.payload.score
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
                random_state=episode_random_state,
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
            failed_attempts.extend(kicked_result.failed_attempts)
            kicked_success = kicked_result.success
            if kicked_success is None:
                continue
            if accepts_strict_improvement(
                incumbent_score=incumbent_score,
                candidate_score=kicked_success.payload.score,
            ):
                incumbent_candidate = kicked_success.candidate
                incumbent_value = kicked_success.payload.value
                incumbent_score = kicked_success.payload.score
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

        success = runtime.scalar_success(
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
            candidate=incumbent_candidate,
            value=incumbent_value,
            evaluation_count=evaluation_count,
            kernel_diagnostics=KernelDiagnostics(
                backend="structured.local_search",
                method="iterated_local_search",
                status=status,
                message=terminal_message,
            ),
            refinement=refinement,
        )
        return structured_episode_attempt_batch(
            success=success,
            failed_attempts=failed_attempts,
        )

    @override
    def run(
        self,
        query: ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload],
        runner: Callable[
            [ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]],
            EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]:
        """Run iterated local search for each proposal in a batch.

        Parameters
        ----------
        query : ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]
            Proposal batch and evaluation context to optimize.
        runner : Callable[[ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]], EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]]
            Evaluator runner used to score local-search and kick candidates.

        Returns
        -------
        EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]
            Locally improved top-level attempts with inner failures summarized in diagnostics.
        """
        runtime: PreparedStructuredLocalSearchRuntime[
            BoundaryT,
            StructuredCandidateT,
        ] = prepare_structured_local_search_runtime(
            query=query,
            runner=runner,
        )

        def optimize_batch(
            random_state: np.random.RandomState,
        ) -> EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]:
            return EvaluationAttemptBatch[
                StructuredCandidateT,
                ObservationPayload,
            ].concatenate(
                tuple(
                    self._optimize_proposal(
                        runtime=runtime,
                        proposal_index=proposal_index,
                        proposal=proposal,
                        random_state=random_state,
                        reserved_count=len(query.proposals) - proposal_index - 1,
                    )
                    for proposal_index, proposal in enumerate(query.proposals)
                )
            )

        attempts, next_random_state_snapshot = self._random_state_snapshot.advance(
            optimize_batch,
        )
        object.__setattr__(
            self,
            "_random_state_snapshot",
            next_random_state_snapshot,
        )
        return attempts
