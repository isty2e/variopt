"""Sampled-neighborhood kernel for structured discrete local search."""

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
from .neighborhood import BoundaryT, DiscreteLeafSpace, StructuredCandidateT
from .runtime.prepared import (
    PreparedStructuredLocalSearchRuntime,
    prepare_structured_local_search_runtime,
)
from .runtime.search import sample_structured_discrete_neighborhood


@dataclass(frozen=True, slots=True)
class StructuredStochasticNeighborhoodKernel(FrozenGenericSlotsCompat,
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
    """Bounded stochastic single-leaf local-search kernel for discrete spaces.

    Parameters
    ----------
    max_steps : int, default=8
        Maximum number of improving steps accepted for each proposal.
    max_neighbors_per_step : int, default=8
        Number of sampled moves considered at each local-search step.
    max_categorical_neighbors_per_leaf : int | None, default=None
        Optional cap on categorical alternatives sampled per leaf.
    random_state : RandomSeed, optional
        Seed or random-state object used to sample neighborhoods.
    """

    max_steps: int = 8
    max_neighbors_per_step: int = 8
    max_categorical_neighbors_per_leaf: int | None = None
    random_state: RandomSeed = 0

    def __post_init__(self) -> None:
        """Validate stochastic local-search budgets.

        Raises
        ------
        ValueError
            Raised when any configured budget or categorical cap is invalid.
        """
        if self.max_steps <= 0:
            msg = "max_steps must be positive"
            raise ValueError(msg)

        if self.max_neighbors_per_step <= 0:
            msg = "max_neighbors_per_step must be positive"
            raise ValueError(msg)

        if (
            self.max_categorical_neighbors_per_leaf is not None
            and self.max_categorical_neighbors_per_leaf <= 0
        ):
            msg = "max_categorical_neighbors_per_leaf must be positive when provided"
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
            method="sampled_leafwise_first_improvement",
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
        """Run one bounded stochastic local-search episode for one proposal."""
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
        found_full_neighborhood_stop = False

        while completed_steps < episode_max_steps:
            sampled_neighborhood = sample_structured_discrete_neighborhood(
                runtime=runtime,
                current_candidate=current_candidate,
                leaf_schedule=leaf_schedule,
                random_state=random_state,
                max_neighbors_per_step=self.max_neighbors_per_step,
                max_categorical_neighbors_per_leaf=(
                    self.max_categorical_neighbors_per_leaf
                ),
            )
            found_full_neighborhood_stop = sampled_neighborhood.covers_full_neighborhood

            improved = False
            for move in sampled_neighborhood.moves:
                proposed_candidate = runtime.neighborhood.space.replace_leaf_values(
                    current_candidate,
                    {move.path: move.replacement},
                )
                proposed_outcome = self._evaluate_candidate(
                    runtime=runtime,
                    candidate=proposed_candidate,
                    proposal_evaluation_spec=proposal_evaluation_spec,
                )
                evaluation_count += proposed_outcome.evaluation_count
                proposed_record = proposed_outcome.record
                proposed_score = proposed_record.score
                if proposed_score < current_score:
                    current_candidate = proposed_record.candidate
                    current_value = proposed_record.value
                    current_score = proposed_score
                    completed_steps += 1
                    improved = True
                    break

            if not improved:
                break

        status = KernelStatus.STOPPED
        message = "max_steps reached before stochastic local-search termination"
        if completed_steps < episode_max_steps:
            if found_full_neighborhood_stop:
                status = KernelStatus.CONVERGED
                message = "no improving move found in the full discrete neighborhood"
            else:
                message = "no improving move found in the sampled discrete neighborhood"

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
                method="sampled_leafwise_first_improvement",
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
        """Run sampled local search for each proposal in a batch.

        Parameters
        ----------
        query : ProposalBatchQuery[BoundaryT, StructuredCandidateT]
            Proposal batch and evaluation context to optimize.
        runner : Callable[[ProposalBatchQuery[BoundaryT, StructuredCandidateT]], tuple[EvaluationOutcome[StructuredCandidateT], ...]]
            Evaluator runner used to score sampled neighbors.

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
