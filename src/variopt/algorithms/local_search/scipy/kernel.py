"""SciPy-backed kernel implementation for continuous local search."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....artifacts import (
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
from ....execution import EvaluationBudgetExhausted
from ....kernel import (
    Kernel,
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
)
from ....spaces import SearchSpace
from ....spaces.projections import ContinuousStructuredSpaceCodec
from ....spaces.types import SpaceCandidateValue
from ..diagnostics import (
    diagnostics_with_failed_attempts,
    top_level_failure_from_failed_attempts,
)
from .contracts import ScipyMinimizeMethod
from .results import ScipyMinimizeResult
from .runner import run_scipy_minimize

BoundaryT = TypeVar("BoundaryT")
ContinuousCandidateT = TypeVar("ContinuousCandidateT", bound=SpaceCandidateValue)


def _candidate_refinement_from_codec(
    *,
    codec: ContinuousStructuredSpaceCodec[BoundaryT, ContinuousCandidateT],
    source_candidate: ContinuousCandidateT,
    refined_candidate: ContinuousCandidateT,
) -> CandidateRefinement[ContinuousCandidateT] | None:
    """Return candidate-refinement provenance from continuous codec topology.

    Parameters
    ----------
    codec : ContinuousStructuredSpaceCodec[BoundaryT, ContinuousCandidateT]
        Codec that owns the real-valued structured leaf paths.
    source_candidate : ContinuousCandidateT
        Candidate before SciPy-backed local optimization.
    refined_candidate : ContinuousCandidateT
        Candidate returned by SciPy-backed local optimization.

    Returns
    -------
    CandidateRefinement[ContinuousCandidateT] | None
        Refinement payload with changed leaf paths, or ``None`` when the final
        optimized candidate preserves every structured leaf value.
    """
    codec.space.validate(source_candidate)
    codec.space.validate(refined_candidate)
    changed_leaf_paths = tuple(
        path
        for path in codec.leaf_paths
        if codec.space.leaf_value_at_validated_path(source_candidate, path)
        != codec.space.leaf_value_at_validated_path(refined_candidate, path)
    )
    if len(changed_leaf_paths) == 0:
        return None

    return CandidateRefinement(
        source_candidate=source_candidate,
        refined_candidate=refined_candidate,
        changed_leaf_paths=changed_leaf_paths,
    )


def _as_local_search_context(
    hint: ProposalKernelHint | None,
) -> ProposalLocalSearchContext | None:
    """Return one local-search context after validating the kernel hint type."""
    if hint is None:
        return None

    if not isinstance(hint, ProposalLocalSearchContext):
        msg = "SciPy local-search kernel requires ProposalLocalSearchContext hints"
        raise TypeError(msg)

    return hint


@dataclass(slots=True)
class _ContinuousCodecProvider(Generic[BoundaryT, ContinuousCandidateT]):
    space: SearchSpace[BoundaryT, ContinuousCandidateT]
    _codec: ContinuousStructuredSpaceCodec[BoundaryT, ContinuousCandidateT] | None = (
        field(default=None, init=False, repr=False)
    )

    def codec(
        self,
    ) -> ContinuousStructuredSpaceCodec[BoundaryT, ContinuousCandidateT]:
        codec = self._codec
        if codec is None:
            codec = ContinuousStructuredSpaceCodec[
                BoundaryT,
                ContinuousCandidateT,
            ].from_space(self.space)
            self._codec = codec
        return codec


@dataclass(frozen=True, slots=True)
class ScipyMinimizeKernel(
    FrozenGenericSlotsCompat,
    Kernel[
        ProposalBatchQuery[
            BoundaryT,
            ContinuousCandidateT,
            ObservationPayload,
        ],
        EvaluationAttemptBatch[
            ContinuousCandidateT,
            ObservationPayload,
        ],
    ],
    Generic[BoundaryT, ContinuousCandidateT],
):
    """SciPy ``minimize`` kernel for continuous structured search spaces.

    Parameters
    ----------
    method : ScipyMinimizeMethod, default="L-BFGS-B"
        SciPy minimization method used for each proposal-local optimization
        episode.
    tolerance : float | None, default=None
        Optional termination tolerance forwarded to SciPy.
    max_iterations : int | None, default=None
        Optional global iteration budget forwarded to SciPy, unless a proposal
        context overrides it.

    Notes
    -----
    This kernel currently supports only structured spaces whose leaves are all
    ``RealSpace`` instances. It evaluates the objective in the space's
    coordinate system so that log-scaled real variables are optimized in log
    coordinates rather than raw value space.
    """

    method: ScipyMinimizeMethod = "L-BFGS-B"
    tolerance: float | None = None
    max_iterations: int | None = None

    def __post_init__(self) -> None:
        """Validate SciPy adapter boundary settings.

        Raises
        ------
        ValueError
            Raised when the configured method, tolerance, or iteration limit is
            invalid.
        """
        if self.method not in {"L-BFGS-B", "Powell"}:
            msg = "method must be 'L-BFGS-B' or 'Powell'"
            raise ValueError(msg)

        if self.tolerance is not None and self.tolerance <= 0.0:
            msg = "tolerance must be positive when provided"
            raise ValueError(msg)

        if self.max_iterations is not None and self.max_iterations <= 0:
            msg = "max_iterations must be positive when provided"
            raise ValueError(msg)

    def scipy_options(self) -> dict[str, int]:
        """Return SciPy options derived from kernel settings.

        Returns
        -------
        dict[str, int]
            SciPy options dictionary containing the configured iteration cap
            when one is set.
        """
        if self.max_iterations is None:
            return {}
        return {"maxiter": self.max_iterations}

    def _proposal_context(
        self,
        *,
        query: ProposalBatchQuery[BoundaryT, ContinuousCandidateT, ObservationPayload],
        proposal_index: int,
    ) -> ProposalLocalSearchContext | None:
        """Return the canonical local-search context for one proposal index."""
        if query.proposal_kernel_hints is None:
            return None
        hint = query.proposal_kernel_hints[proposal_index]
        return _as_local_search_context(hint)

    def _attempt_batch_from_success_and_failures(
        self,
        *,
        success: EvaluationSuccess[ContinuousCandidateT, ObservationPayload] | None,
        failed_attempts: Sequence[
            EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload]
        ],
        failure_request: EvaluationRequest[ContinuousCandidateT] | None = None,
    ) -> EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload]:
        """Return one top-level attempt slot for a SciPy local-search episode."""
        if success is not None:
            fallback_diagnostics = None
            if success.kernel_diagnostics is None and len(failed_attempts) > 0:
                fallback_diagnostics = KernelDiagnostics(
                    backend="scipy.optimize.minimize",
                    method=self.method,
                )
            diagnostics = diagnostics_with_failed_attempts(
                success.kernel_diagnostics,
                failed_attempts,
                fallback_diagnostics=fallback_diagnostics,
            )
            return EvaluationAttemptBatch(
                attempts=(success.with_kernel_diagnostics(diagnostics),),
            )

        return top_level_failure_from_failed_attempts(
            failed_attempts,
            failure_request=failure_request,
        )

    def _evaluate_original_proposal(
        self,
        *,
        query: ProposalBatchQuery[BoundaryT, ContinuousCandidateT, ObservationPayload],
        proposal: Proposal[ContinuousCandidateT],
        proposal_evaluation_spec: ProposalEvaluationSpec | None,
        runner: Callable[
            [ProposalBatchQuery[BoundaryT, ContinuousCandidateT, ObservationPayload]],
            EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload]:
        """Evaluate one original proposal once without local search."""
        local_attempt = self._evaluate_proposal_attempt(
            query=query,
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
            runner=runner,
        )
        local_success = local_attempt.single_success_or_none()
        if local_success is None:
            return local_attempt

        success = EvaluationSuccess[
            ContinuousCandidateT,
            ObservationPayload,
        ].from_scalar_observation(
            observation=local_success.scalar_observation(),
            request=local_success.request,
            evaluation_count=local_success.evaluation_count,
            kernel_diagnostics=KernelDiagnostics(
                backend="scipy.optimize.minimize",
                method=self.method,
                status=KernelStatus.STOPPED,
                message="local search disabled by run-method context",
            ),
            refinement=local_success.refinement,
            candidate_equal=query.problem.space.candidates_equal,
        )
        return self._attempt_batch_from_success_and_failures(
            success=success,
            failed_attempts=(),
        )

    def _evaluate_proposal_attempt(
        self,
        *,
        query: ProposalBatchQuery[BoundaryT, ContinuousCandidateT, ObservationPayload],
        proposal: Proposal[ContinuousCandidateT],
        proposal_evaluation_spec: ProposalEvaluationSpec | None,
        runner: Callable[
            [ProposalBatchQuery[BoundaryT, ContinuousCandidateT, ObservationPayload]],
            EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload]:
        """Evaluate one concrete proposal through the supplied evaluator runner."""
        local_attempts = runner(
            ProposalBatchQuery(
                problem=query.problem,
                proposals=(proposal,),
                execution_resources=query.execution_resources,
                proposal_evaluation_specs=(
                    None
                    if proposal_evaluation_spec is None
                    else (proposal_evaluation_spec,)
                ),
                evaluation_budget=query.evaluation_budget,
            ),
        )
        if local_attempts.attempt_count != 1:
            msg = "kernel runner must return exactly one attempt for one proposal"
            raise ValueError(msg)

        return local_attempts

    def _scipy_options(
        self,
        *,
        context: ProposalLocalSearchContext | None,
    ) -> dict[str, int]:
        """Return SciPy options after one per-proposal budget override."""
        max_iterations = self.max_iterations
        if context is not None and context.local_budget is not None:
            max_iterations = context.local_budget

        if max_iterations is None:
            return {}
        return {"maxiter": max_iterations}

    def _evaluate_candidate_attempt(
        self,
        *,
        query: ProposalBatchQuery[BoundaryT, ContinuousCandidateT, ObservationPayload],
        candidate: ContinuousCandidateT,
        proposal_evaluation_spec: ProposalEvaluationSpec | None,
        runner: Callable[
            [ProposalBatchQuery[BoundaryT, ContinuousCandidateT, ObservationPayload]],
            EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload]:
        """Evaluate one candidate through the supplied evaluator runner."""
        local_attempts = runner(
            ProposalBatchQuery(
                problem=query.problem,
                proposals=(Proposal(candidate=candidate),),
                execution_resources=query.execution_resources,
                proposal_evaluation_specs=(
                    None
                    if proposal_evaluation_spec is None
                    else (proposal_evaluation_spec,)
                ),
                evaluation_budget=query.evaluation_budget,
            ),
        )
        if local_attempts.attempt_count != 1:
            msg = "kernel runner must return exactly one attempt for one proposal"
            raise ValueError(msg)
        return local_attempts

    def _optimize_proposal(
        self,
        *,
        query: ProposalBatchQuery[BoundaryT, ContinuousCandidateT, ObservationPayload],
        proposal_index: int,
        proposal: Proposal[ContinuousCandidateT],
        codec_provider: Callable[
            [],
            ContinuousStructuredSpaceCodec[BoundaryT, ContinuousCandidateT],
        ],
        runner: Callable[
            [ProposalBatchQuery[BoundaryT, ContinuousCandidateT, ObservationPayload]],
            EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload],
        ],
        reserved_count: int,
    ) -> EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload]:
        """Run one local descent episode for one original proposal."""
        context = self._proposal_context(query=query, proposal_index=proposal_index)
        proposal_evaluation_spec = (
            None
            if query.proposal_evaluation_specs is None
            else query.proposal_evaluation_specs[proposal_index]
        )
        if context is not None and not context.enabled:
            return self._evaluate_original_proposal(
                query=query,
                proposal=proposal,
                proposal_evaluation_spec=proposal_evaluation_spec,
                runner=runner,
            )

        failed_attempts: list[
            EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload]
        ] = []
        codec = codec_provider()
        initial_coordinates = codec.coordinates_from_candidate(proposal.candidate)
        evaluated_attempts_by_coordinates: dict[
            tuple[float, ...],
            EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload],
        ] = {}
        evaluated_successes_by_coordinates: dict[
            tuple[float, ...],
            EvaluationSuccess[ContinuousCandidateT, ObservationPayload],
        ] = {}

        def record_attempt_success(
            attempt: EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload],
        ) -> EvaluationSuccess[ContinuousCandidateT, ObservationPayload] | None:
            success = attempt.single_success_or_none()
            if success is None:
                failed_attempts.append(attempt)
            return success

        def successful_evaluation_count() -> int:
            return sum(
                success.evaluation_count
                for success in evaluated_successes_by_coordinates.values()
            )

        def failed_evaluation_count() -> int:
            return sum(attempt.evaluation_count for attempt in failed_attempts)

        def total_evaluation_count() -> int:
            return successful_evaluation_count() + failed_evaluation_count()

        def can_evaluate_local_candidate() -> bool:
            budget = query.evaluation_budget
            return budget is None or budget.can_consume(1 + reserved_count)

        def result_request(
            candidate: ContinuousCandidateT,
        ) -> EvaluationRequest[ContinuousCandidateT]:
            return EvaluationRequest(
                proposal=Proposal(
                    candidate=candidate,
                    proposal_id=proposal.proposal_id,
                ),
                proposal_evaluation_spec=proposal_evaluation_spec,
            )

        original_request = EvaluationRequest(
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
        )

        def success_from_success(
            optimized_success: EvaluationSuccess[
                ContinuousCandidateT,
                ObservationPayload,
            ],
            *,
            status: KernelStatus,
            message: str,
        ) -> EvaluationSuccess[ContinuousCandidateT, ObservationPayload]:
            observation = optimized_success.scalar_observation()
            optimized_candidate = optimized_success.candidate
            refinement = _candidate_refinement_from_codec(
                codec=codec,
                source_candidate=proposal.candidate,
                refined_candidate=optimized_candidate,
            )
            return EvaluationSuccess[
                ContinuousCandidateT,
                ObservationPayload,
            ].from_scalar_observation(
                observation=observation,
                request=result_request(optimized_candidate),
                evaluation_count=total_evaluation_count(),
                kernel_diagnostics=KernelDiagnostics(
                    backend="scipy.optimize.minimize",
                    method=self.method,
                    status=status,
                    message=message,
                ),
                refinement=refinement,
                candidate_equal=query.problem.space.candidates_equal,
            )

        def batch_from_success_and_failures(
            optimized_success: EvaluationSuccess[
                ContinuousCandidateT,
                ObservationPayload,
            ],
            *,
            status: KernelStatus,
            message: str,
        ) -> EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload]:
            return self._attempt_batch_from_success_and_failures(
                success=success_from_success(
                    optimized_success,
                    status=status,
                    message=message,
                ),
                failed_attempts=failed_attempts,
            )

        def objective_in_coordinate_space(
            coordinates: Sequence[float],
        ) -> float:
            coordinate_key = tuple(float(coordinate) for coordinate in coordinates)
            cached_attempt = evaluated_attempts_by_coordinates.get(coordinate_key)
            if cached_attempt is not None:
                cached_success = cached_attempt.single_success_or_none()
                if cached_success is None:
                    return float("inf")
                return cached_success.payload.score

            if not can_evaluate_local_candidate():
                msg = "evaluation budget exhausted"
                raise EvaluationBudgetExhausted(msg)

            if coordinate_key == initial_coordinates:
                local_attempt = self._evaluate_proposal_attempt(
                    query=query,
                    proposal=proposal,
                    proposal_evaluation_spec=proposal_evaluation_spec,
                    runner=runner,
                )
            else:
                local_candidate = codec.candidate_from_coordinates(
                    proposal.candidate,
                    coordinate_key,
                )
                local_attempt = self._evaluate_candidate_attempt(
                    query=query,
                    candidate=local_candidate,
                    proposal_evaluation_spec=proposal_evaluation_spec,
                    runner=runner,
                )
            evaluated_attempts_by_coordinates[coordinate_key] = local_attempt
            local_success = record_attempt_success(local_attempt)
            if local_success is None:
                return float("inf")

            evaluated_successes_by_coordinates[coordinate_key] = local_success
            return local_success.payload.score

        try:
            scipy_result = ScipyMinimizeResult.from_optimize_result(
                run_scipy_minimize(
                    objective_in_coordinate_space=objective_in_coordinate_space,
                    initial_coordinates=initial_coordinates,
                    method=self.method,
                    coordinate_bounds=codec.coordinate_bounds,
                    tolerance=self.tolerance,
                    options=self._scipy_options(context=context),
                )
            )
        except EvaluationBudgetExhausted:
            if len(evaluated_successes_by_coordinates) == 0:
                raise
            optimized_success = min(
                evaluated_successes_by_coordinates.values(),
                key=lambda success: success.payload.score,
            )
            return batch_from_success_and_failures(
                optimized_success,
                status=KernelStatus.STOPPED,
                message="evaluation budget exhausted before local convergence",
            )
        if not scipy_result.has_finite_solution:
            original_attempt = evaluated_attempts_by_coordinates.get(
                initial_coordinates,
            )
            original_success = (
                None if original_attempt is None else original_attempt.single_success_or_none()
            )
            if original_success is None and original_attempt is None:
                if (
                    query.evaluation_budget is not None
                    and not can_evaluate_local_candidate()
                    and len(evaluated_successes_by_coordinates) > 0
                ):
                    original_success = min(
                        evaluated_successes_by_coordinates.values(),
                        key=lambda success: success.payload.score,
                    )
                else:
                    original_attempt = self._evaluate_proposal_attempt(
                        query=query,
                        proposal=proposal,
                        proposal_evaluation_spec=proposal_evaluation_spec,
                        runner=runner,
                    )
                    evaluated_attempts_by_coordinates[
                        initial_coordinates
                    ] = original_attempt
                    original_success = record_attempt_success(original_attempt)
                    if original_success is not None:
                        evaluated_successes_by_coordinates[
                            initial_coordinates
                        ] = original_success

            if original_success is None:
                return self._attempt_batch_from_success_and_failures(
                    success=None,
                    failed_attempts=failed_attempts,
                    failure_request=original_request,
                )
            fallback_observation = original_success.scalar_observation()
            fallback_candidate = original_success.candidate
            refinement = None
            if not query.problem.space.candidates_equal(
                proposal.candidate,
                fallback_candidate,
            ):
                refinement = _candidate_refinement_from_codec(
                    codec=codec,
                    source_candidate=proposal.candidate,
                    refined_candidate=fallback_candidate,
                )
            success = EvaluationSuccess[
                ContinuousCandidateT,
                ObservationPayload,
            ].from_scalar_observation(
                observation=fallback_observation,
                request=result_request(fallback_candidate),
                evaluation_count=total_evaluation_count(),
                kernel_diagnostics=scipy_result.diagnostics(method=self.method),
                refinement=refinement,
                candidate_equal=query.problem.space.candidates_equal,
            )
            return self._attempt_batch_from_success_and_failures(
                success=success,
                failed_attempts=failed_attempts,
            )

        optimized_coordinates = scipy_result.coordinates
        cached_optimized_attempt = evaluated_attempts_by_coordinates.get(
            optimized_coordinates,
        )
        cached_optimized_success = (
            None
            if cached_optimized_attempt is None
            else cached_optimized_attempt.single_success_or_none()
        )
        if cached_optimized_success is None:
            if cached_optimized_attempt is not None:
                if len(evaluated_successes_by_coordinates) == 0:
                    return self._attempt_batch_from_success_and_failures(
                        success=None,
                        failed_attempts=failed_attempts,
                        failure_request=original_request,
                    )
                best_seen_success = min(
                    evaluated_successes_by_coordinates.values(),
                    key=lambda success: success.payload.score,
                )
                return batch_from_success_and_failures(
                    best_seen_success,
                    status=KernelStatus.FAILED,
                    message="optimized candidate evaluation failed",
                )

            if (
                query.evaluation_budget is not None
                and not can_evaluate_local_candidate()
                and len(evaluated_successes_by_coordinates) > 0
            ):
                best_seen_success = min(
                    evaluated_successes_by_coordinates.values(),
                    key=lambda success: success.payload.score,
                )
                return batch_from_success_and_failures(
                    best_seen_success,
                    status=KernelStatus.STOPPED,
                    message="evaluation budget exhausted before local convergence",
                )

            optimized_candidate = codec.candidate_from_coordinates(
                proposal.candidate,
                optimized_coordinates,
            )
            optimized_attempt = self._evaluate_candidate_attempt(
                query=query,
                candidate=optimized_candidate,
                proposal_evaluation_spec=proposal_evaluation_spec,
                runner=runner,
            )
            evaluated_attempts_by_coordinates[optimized_coordinates] = optimized_attempt
            cached_optimized_success = record_attempt_success(optimized_attempt)
            if cached_optimized_success is None:
                if len(evaluated_successes_by_coordinates) == 0:
                    return self._attempt_batch_from_success_and_failures(
                        success=None,
                        failed_attempts=failed_attempts,
                        failure_request=original_request,
                    )
                best_seen_success = min(
                    evaluated_successes_by_coordinates.values(),
                    key=lambda success: success.payload.score,
                )
                return batch_from_success_and_failures(
                    best_seen_success,
                    status=KernelStatus.FAILED,
                    message="optimized candidate evaluation failed",
                )
            evaluated_successes_by_coordinates[
                optimized_coordinates
            ] = cached_optimized_success
        optimized_candidate = cached_optimized_success.candidate
        refinement = _candidate_refinement_from_codec(
            codec=codec,
            source_candidate=proposal.candidate,
            refined_candidate=optimized_candidate,
        )
        success = EvaluationSuccess[
            ContinuousCandidateT,
            ObservationPayload,
        ].from_scalar_observation(
            observation=cached_optimized_success.scalar_observation(),
            request=result_request(optimized_candidate),
            evaluation_count=total_evaluation_count(),
            kernel_diagnostics=scipy_result.diagnostics(method=self.method),
            refinement=refinement,
            candidate_equal=query.problem.space.candidates_equal,
        )
        return self._attempt_batch_from_success_and_failures(
            success=success,
            failed_attempts=failed_attempts,
        )

    @override
    def run(
        self,
        query: ProposalBatchQuery[BoundaryT, ContinuousCandidateT, ObservationPayload],
        runner: Callable[
            [ProposalBatchQuery[BoundaryT, ContinuousCandidateT, ObservationPayload]],
            EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload]:
        """Run proposal-local SciPy minimization for each proposal in a batch.

        Parameters
        ----------
        query : ProposalBatchQuery[BoundaryT, ContinuousCandidateT, ObservationPayload]
            Proposal batch and evaluation context to optimize.
        runner : Callable[[ProposalBatchQuery[BoundaryT, ContinuousCandidateT, ObservationPayload]], EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload]]
            Evaluator runner used to score proposals during local search.

        Returns
        -------
        EvaluationAttemptBatch[ContinuousCandidateT, ObservationPayload]
            Locally improved top-level attempts with inner failures summarized in diagnostics.
        """
        codec_provider = _ContinuousCodecProvider[
            BoundaryT,
            ContinuousCandidateT,
        ](space=query.problem.space)

        return EvaluationAttemptBatch[
            ContinuousCandidateT,
            ObservationPayload,
        ].concatenate(
            tuple(
                self._optimize_proposal(
                    query=query,
                    proposal_index=proposal_index,
                    proposal=proposal,
                    codec_provider=codec_provider.codec,
                    runner=runner,
                    reserved_count=len(query.proposals) - proposal_index - 1,
                )
                for proposal_index, proposal in enumerate(query.proposals)
            )
        )
