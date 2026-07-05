"""Variable-neighborhood kernel for structured discrete local search."""

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
    StructuredVariableNeighborhoodStage,
)
from .runtime.prepared import (
    PreparedStructuredLocalSearchRuntime,
    prepare_structured_local_search_runtime,
    structured_episode_attempt_batch,
)
from .runtime.search import run_structured_variable_neighborhood_stage_once


@dataclass(frozen=True, slots=True)
class StructuredVariableNeighborhoodKernel(
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
    """True variable-neighborhood wrapper over structured discrete kernels.

    Parameters
    ----------
    max_steps : int, default=8
        Maximum number of accepted improvements for each proposal.
    stages : tuple[StructuredVariableNeighborhoodStage, ...], optional
        Ordered neighborhood stages attempted during each episode.
    random_state : RandomSeed, optional
        Seed used to initialize the fallback stochastic-stage stream when a
        proposal does not provide an episode-local random-state snapshot.

    Notes
    -----
    Each stage attempts one neighborhood family once against the current
    incumbent. When a stage finds an improvement, the incumbent is updated and
    the search resets to the first configured stage. When a stage fails to
    improve, the wrapper advances to the next stage.
    """

    max_steps: int = 8
    stages: tuple[StructuredVariableNeighborhoodStage, ...] = (
        StructuredVariableNeighborhoodStage.leafwise_first_improvement(),
        StructuredVariableNeighborhoodStage.scheduled_single_then_pair(
            pair_move_leaf_limit=3,
        ),
    )
    random_state: RandomSeed = 0
    _random_state_snapshot: RandomStateSnapshot = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate variable-neighborhood episode metadata.

        Raises
        ------
        ValueError
            Raised when the step budget or stage list is invalid.
        """
        if self.max_steps <= 0:
            msg = "max_steps must be positive"
            raise ValueError(msg)

        if len(self.stages) == 0:
            msg = "stages must contain at least one neighborhood stage"
            raise ValueError(msg)

        object.__setattr__(
            self,
            "_random_state_snapshot",
            RandomStateSnapshot.from_seed(self.random_state),
        )

    def _evaluate_candidate_attempt(
        self,
        *,
        runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
        candidate: StructuredCandidateT,
        proposal: Proposal[StructuredCandidateT] | None = None,
        proposal_evaluation_spec: ProposalEvaluationSpec | None,
    ) -> EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]:
        """Evaluate one candidate through the supplied evaluator runner."""
        return runtime.evaluate_candidate_attempt(
            candidate=candidate,
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
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
            method="variable_neighborhood_search",
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
        """Run one variable-neighborhood local-search episode for one proposal."""
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

        current_attempt = self._evaluate_candidate_attempt(
            runtime=runtime,
            candidate=proposal.candidate,
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
        )
        current_success = current_attempt.single_success_or_none()
        if current_success is None:
            failed_attempts.append(current_attempt)
            return structured_episode_attempt_batch(
                success=None,
                failed_attempts=failed_attempts,
            )

        current_candidate = current_success.candidate
        current_value = current_success.payload.value
        current_score = current_success.payload.score
        evaluation_count = current_success.evaluation_count
        completed_steps = 0
        episode_max_steps = self._episode_max_steps(runtime=runtime, context=context)
        leaf_schedule = self._ordered_leaf_schedule(
            runtime=runtime,
            context=context,
        )
        current_stage_index = 0

        while completed_steps < episode_max_steps:
            stage_attempt = run_structured_variable_neighborhood_stage_once(
                stage=self.stages[current_stage_index],
                runtime=runtime,
                candidate=current_candidate,
                current_score=current_score,
                leaf_schedule=leaf_schedule,
                proposal_evaluation_spec=proposal_evaluation_spec,
                random_state=episode_random_state,
                reserved_count=reserved_count,
            )
            evaluation_count += stage_attempt.evaluation_count
            failed_attempts.extend(stage_attempt.failed_attempts)

            if stage_attempt.improved_success is not None:
                current_candidate = stage_attempt.improved_success.candidate
                current_value = stage_attempt.improved_success.payload.value
                current_score = stage_attempt.improved_success.payload.score
                completed_steps += 1
                current_stage_index = 0
                continue

            if (
                stage_attempt.budget_exhausted
                or current_stage_index == len(self.stages) - 1
            ):
                refinement = None
                if completed_steps > 0:
                    refinement = runtime.candidate_refinement(
                        source_candidate=proposal.candidate,
                        refined_candidate=current_candidate,
                    )
                terminal_message = stage_attempt.terminal_message
                if not stage_attempt.budget_exhausted:
                    terminal_message = (
                        terminal_message
                        + " after exhausting the configured variable-neighborhood stages"
                    )

                success = runtime.scalar_success(
                    proposal=proposal,
                    proposal_evaluation_spec=proposal_evaluation_spec,
                    candidate=current_candidate,
                    value=current_value,
                    evaluation_count=evaluation_count,
                    kernel_diagnostics=KernelDiagnostics(
                        backend="structured.local_search",
                        method="variable_neighborhood_search",
                        status=stage_attempt.terminal_status,
                        message=terminal_message,
                    ),
                    refinement=refinement,
                )
                return structured_episode_attempt_batch(
                    success=success,
                    failed_attempts=failed_attempts,
                )

            current_stage_index += 1

        refinement = None
        if completed_steps > 0:
            refinement = runtime.candidate_refinement(
                source_candidate=proposal.candidate,
                refined_candidate=current_candidate,
            )

        success = runtime.scalar_success(
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
            candidate=current_candidate,
            value=current_value,
            evaluation_count=evaluation_count,
            kernel_diagnostics=KernelDiagnostics(
                backend="structured.local_search",
                method="variable_neighborhood_search",
                status=KernelStatus.STOPPED,
                message="max_steps reached before variable-neighborhood termination",
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
        """Run variable-neighborhood search for each proposal in a batch.

        Parameters
        ----------
        query : ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]
            Proposal batch and evaluation context to optimize.
        runner : Callable[[ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]], EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]]
            Evaluator runner used to score candidate moves.

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
