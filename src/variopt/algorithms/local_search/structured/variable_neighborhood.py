"""Variable-neighborhood kernel for structured discrete local search."""

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
    StructuredVariableNeighborhoodStage,
)
from .runtime.prepared import (
    PreparedStructuredLocalSearchRuntime,
    prepare_structured_local_search_runtime,
)
from .runtime.search import run_structured_variable_neighborhood_stage_once


@dataclass(frozen=True, slots=True)
class StructuredVariableNeighborhoodKernel(FrozenGenericSlotsCompat,
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
    """True variable-neighborhood wrapper over structured discrete kernels.

    Parameters
    ----------
    max_steps : int, default=8
        Maximum number of accepted improvements for each proposal.
    stages : tuple[StructuredVariableNeighborhoodStage, ...], optional
        Ordered neighborhood stages attempted during each episode.
    random_state : RandomSeed, optional
        Seed or random-state object used by stochastic stages.

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
    ) -> EvaluationOutcome[StructuredCandidateT]:
        """Run one variable-neighborhood local-search episode for one proposal."""
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
                random_state=random_state,
            )
            evaluation_count += stage_attempt.evaluation_count

            if stage_attempt.improved_outcome is not None:
                proposed_record = stage_attempt.improved_outcome.record
                current_candidate = proposed_record.candidate
                current_value = proposed_record.value
                current_score = proposed_record.score
                completed_steps += 1
                current_stage_index = 0
                continue

            if current_stage_index == len(self.stages) - 1:
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
                        method="variable_neighborhood_search",
                        status=stage_attempt.terminal_status,
                        message=(
                            stage_attempt.terminal_message
                            + " after exhausting the configured variable-neighborhood stages"
                        ),
                    ),
                )

            current_stage_index += 1

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
                method="variable_neighborhood_search",
                status=KernelStatus.STOPPED,
                message="max_steps reached before variable-neighborhood termination",
            ),
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
        """Run variable-neighborhood search for each proposal in a batch.

        Parameters
        ----------
        query : ProposalBatchQuery[BoundaryT, StructuredCandidateT]
            Proposal batch and evaluation context to optimize.
        runner : Callable[[ProposalBatchQuery[BoundaryT, StructuredCandidateT]], tuple[EvaluationOutcome[StructuredCandidateT], ...]]
            Evaluator runner used to score candidate moves.

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
            )
            for proposal_index, proposal in enumerate(query.proposals)
        )
