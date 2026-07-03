"""Prepared runtime objects for structured local-search kernels."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....artifacts import CandidateRefinement, Proposal, ProposalEvaluationSpec
from .....kernel import (
    KernelDiagnostics,
    KernelStatus,
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
)
from .....outcomes import EvaluationOutcome
from .....spaces import LeafPath
from ..neighborhood import (
    BoundaryT,
    DiscreteLeafSpace,
    StructuredCandidateT,
    StructuredDiscreteNeighborhood,
)


def _as_local_search_context(
    hint: ProposalKernelHint | None,
) -> ProposalLocalSearchContext | None:
    """Return one local-search context after validating the kernel hint type."""
    if hint is None:
        return None

    if not isinstance(hint, ProposalLocalSearchContext):
        msg = "structured local-search kernels require ProposalLocalSearchContext hints"
        raise TypeError(msg)

    return hint


@dataclass(frozen=True, slots=True)
class PreparedStructuredLocalSearchRuntime(
    FrozenGenericSlotsCompat,
    Generic[BoundaryT, StructuredCandidateT],
):
    """Prepared per-run helpers shared by structured local-search kernels.

    Parameters
    ----------
    query : ProposalBatchQuery[BoundaryT, StructuredCandidateT]
        Original kernel batch query.
    runner : Callable[[ProposalBatchQuery[BoundaryT, StructuredCandidateT]], tuple[EvaluationOutcome[StructuredCandidateT], ...]]
        Evaluation runner bound to the current kernel invocation.
    neighborhood : StructuredDiscreteNeighborhood[BoundaryT, StructuredCandidateT]
        Prepared discrete neighborhood metadata.
    default_schedule : tuple[tuple[LeafPath, DiscreteLeafSpace], ...]
        Default leaf traversal order.
    leaf_space_by_path : dict[LeafPath, DiscreteLeafSpace]
        Lookup table from leaf path to leaf space.
    """

    query: ProposalBatchQuery[BoundaryT, StructuredCandidateT]
    runner: Callable[
        [ProposalBatchQuery[BoundaryT, StructuredCandidateT]],
        tuple[EvaluationOutcome[StructuredCandidateT], ...],
    ]
    neighborhood: StructuredDiscreteNeighborhood[BoundaryT, StructuredCandidateT]
    default_schedule: tuple[tuple[LeafPath, DiscreteLeafSpace], ...]
    leaf_space_by_path: dict[LeafPath, DiscreteLeafSpace]

    def can_evaluate(self, *, reserved_count: int = 0) -> bool:
        """Return whether another evaluator call is allowed by the budget.

        Parameters
        ----------
        reserved_count : int, default=0
            Evaluation units that must remain available for later proposals in
            the same top-level batch.
        """
        budget = self.query.evaluation_budget
        return budget is None or budget.can_consume(1 + reserved_count)

    def evaluate_candidate(
        self,
        *,
        candidate: StructuredCandidateT,
        proposal_evaluation_spec: ProposalEvaluationSpec | None = None,
    ) -> EvaluationOutcome[StructuredCandidateT]:
        """Evaluate one canonical candidate through the prepared runner.

        Parameters
        ----------
        candidate : StructuredCandidateT
            Canonical candidate to evaluate.
        proposal_evaluation_spec : ProposalEvaluationSpec | None, default=None
            Optional proposal evaluation spec to forward to the runner.

        Returns
        -------
        EvaluationOutcome[StructuredCandidateT]
            Single evaluation outcome returned by the runner.

        Raises
        ------
        ValueError
            If the runner returns anything other than exactly one outcome.
        """
        local_outcomes = self.runner(
            ProposalBatchQuery(
                problem=self.query.problem,
                proposals=(Proposal(candidate=candidate),),
                execution_resources=self.query.execution_resources,
                proposal_evaluation_specs=(
                    None
                    if proposal_evaluation_spec is None
                    else (proposal_evaluation_spec,)
                ),
                evaluation_budget=self.query.evaluation_budget,
            ),
        )
        if len(local_outcomes) != 1:
            msg = "kernel runner must return exactly one outcome for one proposal"
            raise ValueError(msg)
        return local_outcomes[0]

    def evaluate_original_proposal(
        self,
        *,
        proposal: Proposal[StructuredCandidateT],
        proposal_evaluation_spec: ProposalEvaluationSpec | None,
        method: str,
    ) -> EvaluationOutcome[StructuredCandidateT]:
        """Evaluate one original proposal once without local search.

        Parameters
        ----------
        proposal : Proposal[StructuredCandidateT]
            Original proposal to evaluate.
        proposal_evaluation_spec : ProposalEvaluationSpec | None
            Optional proposal evaluation spec to forward to the runner.
        method : str
            Kernel method name written into diagnostics.

        Returns
        -------
        EvaluationOutcome[StructuredCandidateT]
            Evaluation outcome marked as stopped because local search was
            disabled.

        Raises
        ------
        ValueError
            If the runner returns anything other than exactly one outcome.
        """
        local_outcomes = self.runner(
            ProposalBatchQuery(
                problem=self.query.problem,
                proposals=(proposal,),
                execution_resources=self.query.execution_resources,
                proposal_evaluation_specs=(
                    None
                    if proposal_evaluation_spec is None
                    else (proposal_evaluation_spec,)
                ),
                evaluation_budget=self.query.evaluation_budget,
            ),
        )
        if len(local_outcomes) != 1:
            msg = "kernel runner must return exactly one outcome for one proposal"
            raise ValueError(msg)

        local_outcome = local_outcomes[0]
        return EvaluationOutcome(
            record=local_outcome.record,
            evaluation_count=local_outcome.evaluation_count,
            kernel_diagnostics=KernelDiagnostics(
                backend="structured.local_search",
                method=method,
                status=KernelStatus.STOPPED,
                message="local search disabled by run-method context",
            ),
        )

    def proposal_context(
        self,
        *,
        proposal_index: int,
    ) -> ProposalLocalSearchContext | None:
        """Return one canonical local-search context for a proposal index.

        Parameters
        ----------
        proposal_index : int
            Proposal index within the current batch query.

        Returns
        -------
        ProposalLocalSearchContext | None
            Proposal-local local-search context, or ``None`` when the query did
            not provide one.
        """
        if self.query.proposal_kernel_hints is None:
            return None
        hint = self.query.proposal_kernel_hints[proposal_index]
        return _as_local_search_context(hint)

    def proposal_evaluation_spec(
        self,
        *,
        proposal_index: int,
    ) -> ProposalEvaluationSpec | None:
        """Return one canonical evaluation spec for a proposal index.

        Parameters
        ----------
        proposal_index : int
            Proposal index within the current batch query.

        Returns
        -------
        ProposalEvaluationSpec | None
            Proposal evaluation spec aligned with ``proposal_index``, or
            ``None`` when the query did not provide one.
        """
        if self.query.proposal_evaluation_specs is None:
            return None
        return self.query.proposal_evaluation_specs[proposal_index]

    def episode_max_steps(
        self,
        *,
        default_max_steps: int,
        context: ProposalLocalSearchContext | None,
    ) -> int:
        """Return one per-episode step budget after context overrides.

        Parameters
        ----------
        default_max_steps : int
            Kernel-level default episode budget.
        context : ProposalLocalSearchContext | None
            Optional proposal-local override context.

        Returns
        -------
        int
            Effective per-episode step budget.
        """
        if context is None or context.local_budget is None:
            return default_max_steps
        return context.local_budget

    def ordered_leaf_schedule(
        self,
        *,
        context: ProposalLocalSearchContext | None,
    ) -> tuple[tuple[LeafPath, DiscreteLeafSpace], ...]:
        """Return the proposal-specific traversal order over prepared leaves.

        Parameters
        ----------
        context : ProposalLocalSearchContext | None
            Optional proposal-local prioritization context.

        Returns
        -------
        tuple[tuple[LeafPath, DiscreteLeafSpace], ...]
            Leaf traversal order for the proposal.

        Raises
        ------
        ValueError
            If ``context`` references a leaf path outside the prepared
            neighborhood.
        """
        if context is None or len(context.prioritized_leaf_paths) == 0:
            return self.default_schedule

        ordered_schedule: list[tuple[LeafPath, DiscreteLeafSpace]] = []
        seen_paths: set[LeafPath] = set()
        for path in context.prioritized_leaf_paths:
            leaf_space = self.leaf_space_by_path.get(path)
            if leaf_space is None:
                msg = (
                    "proposal local-search context referenced a leaf path "
                    "outside the structured neighborhood"
                )
                raise ValueError(msg)
            ordered_schedule.append((path, leaf_space))
            seen_paths.add(path)

        for path, leaf_space in self.default_schedule:
            if path in seen_paths:
                continue
            ordered_schedule.append((path, leaf_space))

        return tuple(ordered_schedule)

    def candidate_refinement(
        self,
        *,
        source_candidate: StructuredCandidateT,
        refined_candidate: StructuredCandidateT,
    ) -> CandidateRefinement[StructuredCandidateT] | None:
        """Return candidate-refinement provenance for a changed candidate.

        Parameters
        ----------
        source_candidate : StructuredCandidateT
            Candidate before structured local-search refinement.
        refined_candidate : StructuredCandidateT
            Candidate returned by structured local-search refinement.

        Returns
        -------
        CandidateRefinement[StructuredCandidateT] | None
            Refinement payload with changed leaf paths, or ``None`` when the
            source and refined candidates have identical structured leaf values.
        """
        space = self.neighborhood.space
        space.validate(source_candidate)
        space.validate(refined_candidate)
        changed_leaf_paths = tuple(
            path
            for path in self.neighborhood.leaf_paths
            if space.leaf_value_at_validated_path(source_candidate, path)
            != space.leaf_value_at_validated_path(refined_candidate, path)
        )
        if len(changed_leaf_paths) == 0:
            return None

        return CandidateRefinement(
            source_candidate=source_candidate,
            refined_candidate=refined_candidate,
            changed_leaf_paths=changed_leaf_paths,
        )


def prepare_structured_local_search_runtime(
    *,
    query: ProposalBatchQuery[BoundaryT, StructuredCandidateT],
    runner: Callable[
        [ProposalBatchQuery[BoundaryT, StructuredCandidateT]],
        tuple[EvaluationOutcome[StructuredCandidateT], ...],
    ],
) -> PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT]:
    """Prepare immutable neighborhood helpers once for one query.

    Parameters
    ----------
    query : ProposalBatchQuery[BoundaryT, StructuredCandidateT]
        Original kernel batch query.
    runner : Callable[[ProposalBatchQuery[BoundaryT, StructuredCandidateT]], tuple[EvaluationOutcome[StructuredCandidateT], ...]]
        Evaluation runner bound to the current kernel invocation.

    Returns
    -------
    PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT]
        Prepared runtime bundle reused by structured local-search kernels for
        this query.
    """
    neighborhood = StructuredDiscreteNeighborhood[
        BoundaryT,
        StructuredCandidateT,
    ].from_space(query.problem.space)
    default_schedule = tuple(
        zip(
            neighborhood.leaf_paths,
            neighborhood.leaf_spaces,
            strict=True,
        )
    )
    return PreparedStructuredLocalSearchRuntime(
        query=query,
        runner=runner,
        neighborhood=neighborhood,
        default_schedule=default_schedule,
        leaf_space_by_path={path: leaf_space for path, leaf_space in default_schedule},
    )
