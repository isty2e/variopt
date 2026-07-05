"""Prepared runtime objects for structured local-search kernels."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....artifacts import (
    CandidateRefinement,
    EvaluationAttemptBatch,
    EvaluationRequest,
    EvaluationSuccess,
    KernelDiagnostics,
    KernelStatus,
    ObservationPayload,
    Proposal,
    ProposalEvaluationSpec,
)
from .....kernel import (
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
)
from .....spaces import LeafPath
from ...diagnostics import (
    diagnostics_with_failed_attempts,
    top_level_failure_from_failed_attempts,
)
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


def structured_episode_attempt_batch(
    *,
    success: EvaluationSuccess[StructuredCandidateT, ObservationPayload] | None,
    failed_attempts: Sequence[
        EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]
    ],
) -> EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]:
    """Build one structured local-search episode attempt batch.

    Parameters
    ----------
    success : EvaluationSuccess[StructuredCandidateT, ObservationPayload] | None
        Terminal successful attempt selected by the local-search episode, or
        ``None`` when no candidate evaluation succeeded.
    failed_attempts : Sequence[EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]]
        One-request failed evaluator attempts in encounter order.

    Returns
    -------
    EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]
        One top-level attempt slot for the original proposal. Inner failed
        local-search attempts are summarized in successful kernel diagnostics;
        if the episode has no success, the representative failed attempt keeps
        the total failed evaluation cost.
    """
    if success is not None:
        diagnostics = diagnostics_with_failed_attempts(
            success.kernel_diagnostics,
            failed_attempts,
        )
        return EvaluationAttemptBatch(
            attempts=(success.with_kernel_diagnostics(diagnostics),),
        )

    return top_level_failure_from_failed_attempts(failed_attempts)


@dataclass(frozen=True, slots=True)
class PreparedStructuredLocalSearchRuntime(
    FrozenGenericSlotsCompat,
    Generic[BoundaryT, StructuredCandidateT],
):
    """Prepared per-run helpers shared by structured local-search kernels.

    Parameters
    ----------
    query : ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]
        Original kernel batch query.
    runner : Callable[[ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]], EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]]
        Evaluation runner bound to the current kernel invocation.
    neighborhood : StructuredDiscreteNeighborhood[BoundaryT, StructuredCandidateT]
        Prepared discrete neighborhood metadata.
    default_schedule : tuple[tuple[LeafPath, DiscreteLeafSpace], ...]
        Default leaf traversal order.
    leaf_space_by_path : dict[LeafPath, DiscreteLeafSpace]
        Lookup table from leaf path to leaf space.
    """

    query: ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]
    runner: Callable[
        [ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]],
        EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload],
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

    def evaluate_candidate_attempt(
        self,
        *,
        candidate: StructuredCandidateT,
        proposal: Proposal[StructuredCandidateT] | None = None,
        proposal_evaluation_spec: ProposalEvaluationSpec | None = None,
    ) -> EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]:
        """Evaluate one canonical candidate through the prepared runner.

        Parameters
        ----------
        candidate : StructuredCandidateT
            Canonical candidate to evaluate.
        proposal : Proposal[StructuredCandidateT] | None, default=None
            Optional proposal to preserve as the request owner. When omitted, a
            synthetic proposal is created for the candidate.
        proposal_evaluation_spec : ProposalEvaluationSpec | None, default=None
            Optional proposal evaluation spec to forward to the runner.

        Returns
        -------
        EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]
            Single-slot evaluation attempt batch returned by the runner.

        Raises
        ------
        ValueError
            If the runner returns anything other than exactly one attempt.
        """
        local_proposal = proposal
        if local_proposal is None:
            local_proposal = Proposal(candidate=candidate)

        local_attempt = self.runner(
            ProposalBatchQuery(
                problem=self.query.problem,
                proposals=(local_proposal,),
                execution_resources=self.query.execution_resources,
                proposal_evaluation_specs=(
                    None
                    if proposal_evaluation_spec is None
                    else (proposal_evaluation_spec,)
                ),
                evaluation_budget=self.query.evaluation_budget,
            ),
        )
        if local_attempt.attempt_count != 1:
            msg = "kernel runner must return exactly one attempt for one proposal"
            raise ValueError(msg)
        return local_attempt

    def evaluate_candidate_success(
        self,
        *,
        candidate: StructuredCandidateT,
        proposal: Proposal[StructuredCandidateT] | None = None,
        proposal_evaluation_spec: ProposalEvaluationSpec | None = None,
    ) -> EvaluationSuccess[StructuredCandidateT, ObservationPayload] | None:
        """Evaluate one candidate and return its successful attempt, if any.

        Parameters
        ----------
        candidate : StructuredCandidateT
            Canonical candidate to evaluate.
        proposal : Proposal[StructuredCandidateT] | None, default=None
            Optional proposal to preserve as the request owner.
        proposal_evaluation_spec : ProposalEvaluationSpec | None, default=None
            Optional proposal evaluation spec to forward to the runner.

        Returns
        -------
        EvaluationSuccess[StructuredCandidateT, ObservationPayload] | None
            Successful attempt returned by the runner, or ``None`` when the
            single attempt slot records an evaluator failure.

        Raises
        ------
        ValueError
            If the runner returns anything other than exactly one attempt.
        RuntimeError
            If the one-slot attempt batch is internally inconsistent.
        """
        return self.evaluate_candidate_attempt(
            candidate=candidate,
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
        ).single_success_or_none()

    def evaluate_original_proposal(
        self,
        *,
        proposal: Proposal[StructuredCandidateT],
        proposal_evaluation_spec: ProposalEvaluationSpec | None,
        method: str,
    ) -> EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]:
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
        EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]
            Single attempt marked as stopped when successful, or the original
            failure attempt when evaluation failed.

        Raises
        ------
        ValueError
            If the runner returns anything other than exactly one attempt.
        """
        local_attempt = self.runner(
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
        if local_attempt.attempt_count != 1:
            msg = "kernel runner must return exactly one attempt for one proposal"
            raise ValueError(msg)

        local_success = local_attempt.single_success_or_none()
        if local_success is None:
            return local_attempt

        success = EvaluationSuccess[
            StructuredCandidateT,
            ObservationPayload,
        ].from_scalar_observation(
            observation=local_success.scalar_observation(),
            request=local_success.request,
            evaluation_count=local_success.evaluation_count,
            kernel_diagnostics=KernelDiagnostics(
                backend="structured.local_search",
                method=method,
                status=KernelStatus.STOPPED,
                message="local search disabled by run-method context",
            ),
            refinement=local_success.refinement,
            candidate_equal=self.query.problem.space.candidates_equal,
        )
        return structured_episode_attempt_batch(
            success=success,
            failed_attempts=(),
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

    def scalar_success(
        self,
        *,
        proposal: Proposal[StructuredCandidateT],
        proposal_evaluation_spec: ProposalEvaluationSpec | None,
        candidate: StructuredCandidateT,
        value: float,
        evaluation_count: int,
        refinement: CandidateRefinement[StructuredCandidateT] | None = None,
        kernel_diagnostics: KernelDiagnostics | None = None,
    ) -> EvaluationSuccess[StructuredCandidateT, ObservationPayload]:
        """Build a canonical scalar success for a structured kernel result.

        Parameters
        ----------
        proposal : Proposal[StructuredCandidateT]
            Original proposal whose id should be preserved for the kernel result.
        proposal_evaluation_spec : ProposalEvaluationSpec | None
            Optional proposal-local evaluation metadata to preserve.
        candidate : StructuredCandidateT
            Candidate whose scalar payload is reported by the result.
        value : float
            Raw scalar objective value.
        evaluation_count : int
            Logical evaluation cost accumulated by the kernel episode.
        refinement : CandidateRefinement[StructuredCandidateT] | None, optional
            Optional provenance from the original proposal candidate to
            ``candidate``.
        kernel_diagnostics : KernelDiagnostics | None, optional
            Optional diagnostics emitted by the kernel episode.

        Returns
        -------
        EvaluationSuccess[StructuredCandidateT, ObservationPayload]
            Request-owned success with request-free scalar payload.
        """
        return EvaluationSuccess(
            request=EvaluationRequest(
                proposal=Proposal(
                    candidate=candidate,
                    proposal_id=proposal.proposal_id,
                ),
                proposal_evaluation_spec=proposal_evaluation_spec,
            ),
            payload=ObservationPayload.from_objective_value(
                value=value,
                direction=self.query.problem.direction,
            ),
            evaluation_count=evaluation_count,
            refinement=refinement,
            kernel_diagnostics=kernel_diagnostics,
            candidate_equal=self.query.problem.space.candidates_equal,
        )


def prepare_structured_local_search_runtime(
    *,
    query: ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload],
    runner: Callable[
        [ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]],
        EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload],
    ],
) -> PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT]:
    """Prepare immutable neighborhood helpers once for one query.

    Parameters
    ----------
    query : ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]
        Original kernel batch query.
    runner : Callable[[ProposalBatchQuery[BoundaryT, StructuredCandidateT, ObservationPayload]], EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]]
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
