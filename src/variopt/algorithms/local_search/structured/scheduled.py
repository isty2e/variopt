"""Scheduled neighborhood kernel for structured discrete local search."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

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
from ....spaces import LeafPath
from .neighborhood import BoundaryT, DiscreteLeafSpace, StructuredCandidateT
from .runtime.prepared import (
    PreparedStructuredLocalSearchRuntime,
    prepare_structured_local_search_runtime,
)
from .runtime.search import (
    first_improving_pair_move_outcome,
    first_improving_single_leaf_outcome,
)


@dataclass(frozen=True, slots=True)
class StructuredScheduledLocalSearchKernel(FrozenGenericSlotsCompat,
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
    """Deterministic staged local-search kernel over structured discrete spaces.

    Parameters
    ----------
    max_steps : int, default=8
        Maximum number of improving steps accepted for each proposal.
    pair_move_leaf_limit : int, default=3
        Number of leading scheduled leaves considered for pair moves.

    Notes
    -----
    This kernel realizes a stronger neighborhood contract than pure hill climb:
    it first scans ordered single-leaf moves, then scans bounded pair moves
    over the first ``pair_move_leaf_limit`` scheduled leaf paths.
    """

    max_steps: int = 8
    pair_move_leaf_limit: int = 3

    def __post_init__(self) -> None:
        """Validate scheduled local-search budgets.

        Raises
        ------
        ValueError
            Raised when ``max_steps`` or ``pair_move_leaf_limit`` is invalid.
        """
        if self.max_steps <= 0:
            msg = "max_steps must be positive"
            raise ValueError(msg)

        if self.pair_move_leaf_limit <= 0:
            msg = "pair_move_leaf_limit must be positive"
            raise ValueError(msg)

    def _evaluate_candidate(
        self,
        *,
        runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
        candidate: StructuredCandidateT,
        proposal_evaluation_spec: ProposalEvaluationSpec | None,
    ) -> EvaluationOutcome[StructuredCandidateT]:
        """Evaluate one candidate through the supplied evaluator runner."""
        return runtime.evaluate_candidate(
            candidate=candidate,
            proposal_evaluation_spec=proposal_evaluation_spec,
        )

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
            method="scheduled_single_then_pair",
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
    ) -> EvaluationOutcome[StructuredCandidateT]:
        """Run one scheduled local-search episode for one original proposal."""
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

        current_outcome = self._evaluate_candidate(
            runtime=runtime,
            candidate=proposal.candidate,
            proposal_evaluation_spec=proposal_evaluation_spec,
        )
        current_record = current_outcome.record
        current_candidate = current_record.candidate
        current_value = current_record.value
        current_score = current_record.score
        evaluation_count = current_outcome.evaluation_count
        completed_steps = 0
        converged = False
        episode_max_steps = self._episode_max_steps(runtime=runtime, context=context)
        leaf_schedule = self._ordered_leaf_schedule(
            runtime=runtime,
            context=context,
        )

        while completed_steps < episode_max_steps:
            proposed_outcome, neighbor_evaluation_count = (
                first_improving_single_leaf_outcome(
                    runtime=runtime,
                    candidate=current_candidate,
                    current_score=current_score,
                    leaf_schedule=leaf_schedule,
                    proposal_evaluation_spec=proposal_evaluation_spec,
                )
            )
            evaluation_count += neighbor_evaluation_count
            if proposed_outcome is None:
                proposed_outcome, neighbor_evaluation_count = (
                    first_improving_pair_move_outcome(
                        runtime=runtime,
                        candidate=current_candidate,
                        current_score=current_score,
                        leaf_schedule=leaf_schedule,
                        proposal_evaluation_spec=proposal_evaluation_spec,
                        pair_move_leaf_limit=self.pair_move_leaf_limit,
                    )
                )
                evaluation_count += neighbor_evaluation_count

            if proposed_outcome is None:
                converged = True
                break

            proposed_record = proposed_outcome.record
            current_candidate = proposed_record.candidate
            current_value = proposed_record.value
            current_score = proposed_record.score
            completed_steps += 1

        status = KernelStatus.STOPPED
        message = "max_steps reached before local convergence"
        if converged:
            status = KernelStatus.CONVERGED
            message = "no improving scheduled move found"

        refinement = None
        if completed_steps > 0:
            refinement = runtime.candidate_refinement(
                source_candidate=proposal.candidate,
                refined_candidate=current_candidate,
            )

        return EvaluationOutcome(
            record=Observation.from_objective_value(
                proposal=proposal,
                proposal_evaluation_spec=proposal_evaluation_spec,
                candidate=current_candidate,
                value=current_value,
                direction=runtime.query.problem.direction,
            ),
            evaluation_count=evaluation_count,
            kernel_diagnostics=KernelDiagnostics(
                backend="structured.local_search",
                method="scheduled_single_then_pair",
                status=status,
                message=message,
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
        """Run scheduled local search for each proposal in a batch.

        Parameters
        ----------
        query : ProposalBatchQuery[BoundaryT, StructuredCandidateT]
            Proposal batch and evaluation context to optimize.
        runner : Callable[[ProposalBatchQuery[BoundaryT, StructuredCandidateT]], tuple[EvaluationOutcome[StructuredCandidateT], ...]]
            Evaluator runner used to score single-leaf and pair moves.

        Returns
        -------
        tuple[EvaluationOutcome[StructuredCandidateT], ...]
            Locally improved outcomes aligned to ``query.proposals``.
        """
        runtime: PreparedStructuredLocalSearchRuntime[
            BoundaryT,
            StructuredCandidateT,
        ] = prepare_structured_local_search_runtime(
            query=query,
            runner=runner,
        )
        return tuple(
            self._optimize_proposal(
                runtime=runtime,
                proposal_index=proposal_index,
                proposal=proposal,
            )
            for proposal_index, proposal in enumerate(query.proposals)
        )
