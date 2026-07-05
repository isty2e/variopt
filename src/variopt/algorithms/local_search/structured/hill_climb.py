"""Deterministic hill-climb kernel for structured discrete local search."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

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
from .neighborhood import (
    BoundaryT,
    DiscreteLeafSpace,
    StructuredCandidateT,
    discrete_leaf_neighbors,
)
from .runtime.prepared import (
    PreparedStructuredLocalSearchRuntime,
    prepare_structured_local_search_runtime,
    structured_episode_attempt_batch,
)


@dataclass(frozen=True, slots=True)
class StructuredHillClimbKernel(
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
    """Deterministic leafwise first-improvement local-search kernel.

    Parameters
    ----------
    max_steps : int, default=8
        Maximum number of improving steps accepted for each proposal.
    """

    max_steps: int = 8

    def __post_init__(self) -> None:
        """Validate local-search budgets.

        Raises
        ------
        ValueError
            Raised when ``max_steps`` is not positive.
        """
        if self.max_steps <= 0:
            msg = "max_steps must be positive"
            raise ValueError(msg)

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
            method="leafwise_first_improvement",
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
    ) -> tuple[tuple[tuple[int | str, ...], DiscreteLeafSpace], ...]:
        """Return the leaf traversal order after one proposal-specific reordering."""
        return runtime.ordered_leaf_schedule(context=context)

    def _optimize_proposal(
        self,
        *,
        runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
        proposal_index: int,
        proposal: Proposal[StructuredCandidateT],
        reserved_count: int,
    ) -> EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]:
        """Run one local hill-climb episode for one original proposal."""
        space = runtime.neighborhood.space
        space.validate(proposal.candidate)
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
        converged = False
        episode_max_steps = self._episode_max_steps(runtime=runtime, context=context)
        leaf_schedule = self._ordered_leaf_schedule(
            runtime=runtime,
            context=context,
        )

        budget_exhausted = False
        while completed_steps < episode_max_steps:
            improved = False
            space.validate(current_candidate)
            for path, leaf_space in leaf_schedule:
                current_leaf_value = space.leaf_value_at_validated_path(
                    current_candidate,
                    path,
                )
                for replacement in discrete_leaf_neighbors(
                    leaf_space,
                    current_leaf_value,
                ):
                    if not runtime.can_evaluate(reserved_count=reserved_count):
                        budget_exhausted = True
                        break
                    proposed_candidate = space.replace_leaf_values_in_validated_candidate(
                        current_candidate,
                        {path: replacement},
                    )
                    proposed_attempt = self._evaluate_candidate_attempt(
                        runtime=runtime,
                        candidate=proposed_candidate,
                        proposal_evaluation_spec=proposal_evaluation_spec,
                    )
                    evaluation_count += proposed_attempt.evaluation_count
                    proposed_success = proposed_attempt.single_success_or_none()
                    if proposed_success is None:
                        failed_attempts.append(proposed_attempt)
                        continue
                    proposed_score = proposed_success.payload.score
                    if proposed_score < current_score:
                        current_candidate = proposed_success.candidate
                        current_value = proposed_success.payload.value
                        current_score = proposed_score
                        completed_steps += 1
                        improved = True
                        break
                if improved or budget_exhausted:
                    break

            if budget_exhausted:
                break
            if not improved:
                converged = True
                break

        status = KernelStatus.STOPPED
        message = "max_steps reached before local convergence"
        if budget_exhausted:
            message = "evaluation budget exhausted before local convergence"
        if converged:
            status = KernelStatus.CONVERGED
            message = "no improving leafwise move found"

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
                method="leafwise_first_improvement",
                status=status,
                message=message,
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
        """Run deterministic hill climbing for each proposal in a batch.

        Parameters
        ----------
        query : ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]
            Proposal batch and evaluation context to optimize.
        runner : Callable[[ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]], EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]]
            Evaluator runner used to score candidate neighbors.

        Returns
        -------
        EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]
            Locally improved attempts and recorded failed local-search trials.
        """
        runtime: PreparedStructuredLocalSearchRuntime[
            BoundaryT,
            StructuredCandidateT,
        ] = prepare_structured_local_search_runtime(
            query=query,
            runner=runner,
        )
        return EvaluationAttemptBatch[
            StructuredCandidateT,
            ObservationPayload,
        ].concatenate(
            tuple(
                self._optimize_proposal(
                    runtime=runtime,
                    proposal_index=proposal_index,
                    proposal=proposal,
                    reserved_count=len(query.proposals) - proposal_index - 1,
                )
                for proposal_index, proposal in enumerate(query.proposals)
            )
        )
