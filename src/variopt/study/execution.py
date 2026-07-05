"""Generic study step and run orchestration."""

from dataclasses import dataclass, replace
from typing import Generic, NoReturn, Protocol, TypeGuard

from typing_extensions import TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..artifacts import (
    DefaultEvaluationAttemptMaterializer,
    EvaluationAttemptBatch,
    EvaluationAttemptMaterializer,
    EvaluationFailure,
    EvaluationRequest,
    EvaluationSuccess,
    Observation,
    ObservationPayload,
    Proposal,
    RunReport,
    RunResult,
    Trace,
    TraceEvent,
    materialize_success_records,
)
from ..evaluators.sequential import SequentialEvaluator
from ..execution import (
    EXACT_ASYNC_EXECUTION_MODEL,
    SEQUENTIAL_EXECUTION_MODEL,
    STALE_ASYNC_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
    EvaluationBudget,
    EvaluationBudgetExhausted,
    ExecutionAssimilationMode,
    ExecutionModel,
)
from ..kernel import DirectKernel, Kernel, ProposalBatchQuery
from ..methods import RunMethod
from ..objective import Objective, ScalarEvaluationProtocol
from ..problem import Problem
from ..spaces import CandidateEquality
from ..typevars import CandidateT, RunMethodStateT
from .common import (
    StudyEvaluator,
    StudyPayloadT,
    StudyRecordT,
    build_evaluation_requests,
    supports_attempt_batches,
    trace_value_for_records,
    validate_aligned_attempts,
    validate_materialized_attempts,
)
from .exact_async.orchestration import evaluate_batch_exact_async
from .failures import RunExecutionFailed
from .validation import validate_execution_request

BoundaryT = TypeVar("BoundaryT")


@dataclass(frozen=True, slots=True)
class StudyStepResult(
    FrozenGenericSlotsCompat,
    Generic[CandidateT, RunMethodStateT, StudyRecordT],
):
    """Canonical in-process result for one ask/evaluate/tell study step.

    Parameters
    ----------
    attempts : EvaluationAttemptBatch[CandidateT, StudyRecordT]
        Dense materialized feedback attempts assimilated by the run method.
    state : RunMethodStateT
        Run-method state after assimilating ``attempts``.
    evaluation_count : int
        Evaluation units consumed by the step after hard-budget reconciliation.
    """

    attempts: EvaluationAttemptBatch[CandidateT, StudyRecordT]
    state: RunMethodStateT
    evaluation_count: int


@dataclass(frozen=True, slots=True)
class _CheckpointSafeRunSnapshot(
    Generic[CandidateT, RunMethodStateT, StudyRecordT]
):
    """Last known checkpoint-safe run projection."""

    successes: tuple[EvaluationSuccess[CandidateT, StudyRecordT], ...]
    failures: tuple[EvaluationFailure[CandidateT], ...]
    trace: Trace
    evaluation_count: int
    state: RunMethodStateT


class StudyExecutionOwner(
    Protocol[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
):
    """Protocol for the subset of Study state generic execution needs.

    Notes
    -----
    The generic execution helpers operate on this protocol instead of the
    concrete :class:`variopt.study.Study` class so the same orchestration can
    be reused across sync, exact-async, and stale-async study tiers.
    """

    @property
    def problem(
        self,
    ) -> Problem[BoundaryT, CandidateT, StudyPayloadT]:
        """Return the configured problem."""
        ...

    @property
    def run_method(
        self,
    ) -> RunMethod[
        RunMethodStateT,
        Proposal[CandidateT],
        StudyRecordT,
    ]:
        """Return the configured run method."""
        ...

    @property
    def evaluator(
        self,
    ) -> StudyEvaluator[BoundaryT, CandidateT, StudyPayloadT]:
        """Return the configured evaluator."""
        ...

    @property
    def kernel(
        self,
    ) -> Kernel[
        ProposalBatchQuery[BoundaryT, CandidateT, StudyPayloadT],
        EvaluationAttemptBatch[CandidateT, StudyPayloadT],
    ]:
        """Return the configured kernel."""
        ...

    @property
    def attempt_materializer(
        self,
    ) -> EvaluationAttemptMaterializer[CandidateT, StudyPayloadT, StudyRecordT]:
        """Return the payload-to-record feedback materializer."""
        ...


class DirectScalarSequentialStudyOwner(
    Protocol[BoundaryT, CandidateT, RunMethodStateT],
):
    """Protocol for the narrow direct-scalar sequential optimize fast path."""

    @property
    def problem(self) -> Problem[BoundaryT, CandidateT, Observation[CandidateT]]:
        """Return the configured scalar problem."""
        ...

    @property
    def run_method(
        self,
    ) -> RunMethod[RunMethodStateT, Proposal[CandidateT], Observation[CandidateT]]:
        """Return the configured run method."""
        ...

    @property
    def evaluator(
        self,
    ) -> SequentialEvaluator[BoundaryT, CandidateT, Observation[CandidateT]]:
        """Return the configured sequential evaluator."""
        ...

    @property
    def kernel(
        self,
    ) -> DirectKernel[
        ProposalBatchQuery[BoundaryT, CandidateT, Observation[CandidateT]],
        EvaluationAttemptBatch[CandidateT, Observation[CandidateT]],
    ]:
        """Return the configured direct kernel."""
        ...


def _supports_direct_scalar_sequential_path(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    *,
    execution_model: ExecutionModel,
) -> TypeGuard[
    DirectScalarSequentialStudyOwner[BoundaryT, CandidateT, RunMethodStateT]
]:
    if execution_model not in {
        SEQUENTIAL_EXECUTION_MODEL,
        SYNC_BATCH_EXECUTION_MODEL,
    }:
        return False

    kernel = study.kernel
    if not isinstance(kernel, DirectKernel):
        return False
    if kernel.__class__ is not DirectKernel:
        return False

    evaluator = study.evaluator
    if not isinstance(evaluator, SequentialEvaluator):
        return False
    if evaluator.__class__ is not SequentialEvaluator:
        return False

    attempt_materializer = study.attempt_materializer
    if not isinstance(attempt_materializer, DefaultEvaluationAttemptMaterializer):
        return False
    if attempt_materializer.__class__ is not DefaultEvaluationAttemptMaterializer:
        return False

    objective = study.problem.direct_objective
    return objective is not None and _uses_default_scalar_request_evaluation(objective)


def _uses_default_scalar_request_evaluation(
    objective: Objective[CandidateT],
) -> bool:
    for cls in type(objective).__mro__:
        if "evaluate_request" in cls.__dict__:
            return cls is ScalarEvaluationProtocol

    return False


def _query_with_evaluation_budget(
    query: ProposalBatchQuery[BoundaryT, CandidateT, StudyPayloadT],
    evaluation_budget: EvaluationBudget | None,
) -> ProposalBatchQuery[BoundaryT, CandidateT, StudyPayloadT]:
    """Return ``query`` with the active budget ledger attached."""
    if evaluation_budget is None or query.evaluation_budget is not None:
        return query
    return replace(query, evaluation_budget=evaluation_budget)


def _consume_reported_evaluation_cost(
    *,
    evaluation_budget: EvaluationBudget | None,
    remaining_before: int | None,
    reported_evaluation_count: int,
) -> int:
    """Reconcile reported outcome cost against pre-consumed runner cost."""
    if evaluation_budget is None or remaining_before is None:
        return reported_evaluation_count

    consumed_by_runner = remaining_before - evaluation_budget.remaining
    if reported_evaluation_count > consumed_by_runner:
        evaluation_budget.consume(reported_evaluation_count - consumed_by_runner)
        return reported_evaluation_count
    return consumed_by_runner


def _current_remaining_budget(
    *,
    evaluation_budget: EvaluationBudget | None,
    record_budget_remaining: int,
) -> int:
    """Return the active loop budget for evaluation- or record-count mode."""
    if evaluation_budget is None:
        return record_budget_remaining
    return evaluation_budget.remaining


def _current_evaluation_count(
    *,
    max_evaluations: int,
    evaluation_budget: EvaluationBudget | None,
    record_budget_remaining: int,
) -> int:
    """Return the current terminal accounting projection."""
    if evaluation_budget is None:
        return max_evaluations - record_budget_remaining
    return max_evaluations - evaluation_budget.remaining


def _build_run_report(
    *,
    successes: tuple[EvaluationSuccess[CandidateT, StudyRecordT], ...],
    failures: tuple[EvaluationFailure[CandidateT], ...],
    trace_events: tuple[TraceEvent, ...],
    evaluation_count: int,
    candidate_equal: CandidateEquality[CandidateT],
) -> RunReport[CandidateT, StudyRecordT]:
    """Build one report projection from the current run-history state."""
    return RunReport[CandidateT, StudyRecordT].from_successes(
        successes=successes,
        evaluation_count=evaluation_count,
        trace=Trace(events=trace_events),
        failures=failures,
        candidate_equal=candidate_equal,
    )


def _run_report_from_snapshot(
    *,
    snapshot: _CheckpointSafeRunSnapshot[
        CandidateT,
        RunMethodStateT,
        StudyRecordT,
    ],
    candidate_equal: CandidateEquality[CandidateT],
) -> RunReport[CandidateT, StudyRecordT]:
    """Build one report projection from a checkpoint-safe snapshot."""
    return RunReport[CandidateT, StudyRecordT].from_successes(
        successes=snapshot.successes,
        evaluation_count=snapshot.evaluation_count,
        trace=snapshot.trace,
        failures=snapshot.failures,
        candidate_equal=candidate_equal,
    )


def _raise_run_execution_failed(
    *,
    cause: Exception,
    successes: tuple[EvaluationSuccess[CandidateT, StudyRecordT], ...],
    failures: tuple[EvaluationFailure[CandidateT], ...],
    trace_events: tuple[TraceEvent, ...],
    evaluation_count: int,
    state: RunMethodStateT,
    safe_snapshot: _CheckpointSafeRunSnapshot[
        CandidateT,
        RunMethodStateT,
        StudyRecordT,
    ]
    | None,
    candidate_equal: CandidateEquality[CandidateT],
) -> NoReturn:
    """Raise a hard run failure carrying current and checkpoint-safe projections."""
    checkpoint_safe_report: RunReport[CandidateT, StudyRecordT] | None = None
    checkpoint_safe_state: RunMethodStateT | None = None
    if safe_snapshot is not None:
        checkpoint_safe_report = _run_report_from_snapshot(
            snapshot=safe_snapshot,
            candidate_equal=candidate_equal,
        )
        checkpoint_safe_state = safe_snapshot.state

    raise RunExecutionFailed[
        CandidateT,
        RunMethodStateT,
        StudyRecordT,
    ](
        partial_report=_build_run_report(
            successes=successes,
            failures=failures,
            trace_events=trace_events,
            evaluation_count=evaluation_count,
            candidate_equal=candidate_equal,
        ),
        partial_state=state,
        checkpoint_safe_report=checkpoint_safe_report,
        checkpoint_safe_state=checkpoint_safe_state,
        cause=cause,
    ) from cause


def _optimize_direct_scalar_sequential(
    study: DirectScalarSequentialStudyOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
    ],
    max_evaluations: int,
    batch_size: int,
    *,
    execution_model: ExecutionModel,
    count_evaluation_cost: bool,
    initial_state: RunMethodStateT | None,
) -> tuple[RunResult[CandidateT], RunMethodStateT]:
    if max_evaluations < 0:
        msg = "max_evaluations must be non-negative"
        raise ValueError(msg)

    validate_execution_request(
        study,
        batch_size=batch_size,
        execution_model=execution_model,
    )

    objective = study.problem.direct_objective
    if objective is None:
        msg = "direct scalar objective fast path requires a direct Objective"
        raise RuntimeError(msg)

    observations: list[Observation[CandidateT]] = []
    successes: list[EvaluationSuccess[CandidateT, Observation[CandidateT]]] = []
    failures: list[EvaluationFailure[CandidateT]] = []
    trace_events: list[TraceEvent] = []
    evaluation_budget = (
        EvaluationBudget(max_evaluations) if count_evaluation_cost else None
    )
    record_budget_remaining = max_evaluations
    state = (
        study.run_method.create_initial_state()
        if initial_state is None
        else initial_state
    )

    while _current_remaining_budget(
        evaluation_budget=evaluation_budget,
        record_budget_remaining=record_budget_remaining,
    ) > 0 and not study.run_method.is_exhausted(state):
        remaining = _current_remaining_budget(
            evaluation_budget=evaluation_budget,
            record_budget_remaining=record_budget_remaining,
        )
        current_batch_size = min(batch_size, remaining)
        try:
            proposals, next_state = study.run_method.ask(
                state,
                batch_size=current_batch_size,
            )
            if len(proposals) == 0:
                msg = "run_method returned no proposals"
                raise RuntimeError(msg)

            if len(proposals) > current_batch_size:
                msg = "run_method returned more proposals than requested"
                raise ValueError(msg)

            # DirectKernel ignores hint payloads, but the generic path still
            # validates alignment when it builds ProposalBatchQuery.
            proposal_kernel_hints = study.run_method.proposal_kernel_hints(
                next_state,
                proposals,
            )
            if proposal_kernel_hints is not None and (
                len(proposal_kernel_hints) != len(proposals)
            ):
                msg = "proposal_kernel_hints must align one-to-one with proposals"
                raise ValueError(msg)

            proposal_evaluation_specs = study.run_method.proposal_evaluation_specs(
                next_state,
                proposals,
            )
            if proposal_evaluation_specs is not None and (
                len(proposal_evaluation_specs) != len(proposals)
            ):
                msg = "proposal_evaluation_specs must align one-to-one with proposals"
                raise ValueError(msg)

            batch_observations: list[Observation[CandidateT]] = []
            batch_successes: list[
                EvaluationSuccess[CandidateT, Observation[CandidateT]]
            ] = []
            batch_requests: list[EvaluationRequest[CandidateT]] = []
            batch_failures: list[EvaluationFailure[CandidateT]] = []
            batch_attempt_slots: list[
                EvaluationSuccess[CandidateT, Observation[CandidateT]]
                | EvaluationFailure[CandidateT]
            ] = []
            for index, proposal in enumerate(proposals):
                candidate = proposal.candidate
                study.problem.space.validate(candidate)
                proposal_evaluation_spec = (
                    None
                    if proposal_evaluation_specs is None
                    else proposal_evaluation_specs[index]
                )
                request = EvaluationRequest(
                    proposal=proposal,
                    proposal_evaluation_spec=proposal_evaluation_spec,
                )
                batch_requests.append(request)
                if evaluation_budget is not None:
                    evaluation_budget.consume()
                try:
                    value = objective.evaluate(candidate)
                except Exception as exception:
                    batch_failures.append(
                        EvaluationFailure[CandidateT].from_exception(
                            request=request,
                            exception=exception,
                        ),
                    )
                    batch_attempt_slots.append(batch_failures[-1])
                    continue

                observation = Observation.from_objective_value(
                    request=request,
                    candidate=candidate,
                    value=value,
                    direction=study.problem.direction,
                )
                batch_observations.append(observation)
                batch_successes.append(
                    EvaluationSuccess(
                        request=request,
                        payload=observation,
                        evaluation_count=1,
                    ),
                )
                batch_attempt_slots.append(batch_successes[-1])

            batch_observation_tuple = tuple(batch_observations)
            batch_attempts: EvaluationAttemptBatch[
                CandidateT,
                Observation[CandidateT],
            ] = EvaluationAttemptBatch(
                attempts=tuple(batch_attempt_slots),
            )
            next_run_state = study.run_method.tell_attempts(next_state, batch_attempts)
        except EvaluationBudgetExhausted:
            raise
        except Exception as exception:
            _raise_run_execution_failed(
                cause=exception,
                successes=tuple(successes),
                failures=tuple(failures),
                trace_events=tuple(trace_events),
                evaluation_count=_current_evaluation_count(
                    max_evaluations=max_evaluations,
                    evaluation_budget=evaluation_budget,
                    record_budget_remaining=record_budget_remaining,
                ),
                state=state,
                safe_snapshot=None,
                candidate_equal=study.problem.space.candidates_equal,
            )
        state = next_run_state
        observations.extend(batch_observation_tuple)
        successes.extend(batch_successes)
        failures.extend(batch_failures)
        if evaluation_budget is None:
            record_budget_remaining -= batch_attempts.attempt_count

        trace_events.append(
            TraceEvent(
                kind="study.step",
                message=(
                    f"completed {batch_attempts.attempt_count} attempt(s): "
                    f"{len(batch_observation_tuple)} succeeded, "
                    f"{len(batch_failures)} failed"
                ),
                value=trace_value_for_records(batch_observation_tuple),
            ),
        )

    return (
        RunResult[CandidateT].from_observations(
            observations=tuple(observations),
            evaluation_count=(
                max_evaluations - record_budget_remaining
                if evaluation_budget is None
                else max_evaluations - evaluation_budget.remaining
            ),
            trace=Trace(events=tuple(trace_events)),
            failures=tuple(failures),
            candidate_equal=study.problem.space.candidates_equal,
        ),
        state,
    )


def evaluate_attempts_sync(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    query: ProposalBatchQuery[BoundaryT, CandidateT, StudyPayloadT],
    *,
    requests: tuple[EvaluationRequest[CandidateT], ...] | None = None,
) -> EvaluationAttemptBatch[CandidateT, StudyPayloadT]:
    """Execute one synchronous request batch into a dense attempt batch.

    Parameters
    ----------
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Study-like owner exposing the problem, evaluator, and kernel.
    query : ProposalBatchQuery[BoundaryT, CandidateT, StudyPayloadT]
        Proposal batch and evaluation metadata to execute.
    requests : tuple[EvaluationRequest[CandidateT], ...] | None, default=None
        Optional prebuilt request batch aligned with ``query``.

    Returns
    -------
    EvaluationAttemptBatch[CandidateT, StudyPayloadT]
        Dense attempt batch preserving successful and failed request slots.
    """
    resolved_requests = (
        build_evaluation_requests(
            query.proposals,
            proposal_evaluation_specs=query.proposal_evaluation_specs,
        )
        if requests is None
        else requests
    )
    if query.evaluation_budget is not None:
        query.evaluation_budget.consume(len(resolved_requests))

    if not supports_attempt_batches(study.evaluator):
        msg = "sync execution models require evaluator.evaluate_attempts"
        raise TypeError(msg)

    return study.evaluator.evaluate_attempts(
        query.problem,
        resolved_requests,
    )


def evaluate_step(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    state: RunMethodStateT,
    batch_size: int,
    *,
    execution_model: ExecutionModel,
    evaluation_budget: EvaluationBudget | None = None,
) -> StudyStepResult[CandidateT, RunMethodStateT, StudyRecordT]:
    """Run one ask/kernel/evaluate/tell step and return attempts and state.

    Parameters
    ----------
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Study-like owner exposing the problem, run method, evaluator, and
        kernel.
    state : RunMethodStateT
        Run-method state to advance.
    batch_size : int
        Maximum number of proposals to request from the run method.
    execution_model : ExecutionModel
        Execution model controlling whether evaluation is synchronous or
        exact-async.
    evaluation_budget : EvaluationBudget | None, default=None
        Optional hard evaluation-budget ledger shared across kernel subqueries.

    Returns
    -------
    StudyStepResult[CandidateT, RunMethodStateT, StudyRecordT]
        Evaluator attempts and the next run-method state.

    Raises
    ------
    ValueError
        If ``batch_size`` is invalid or the run method returns an invalid
        proposal batch.
    RuntimeError
        If the run method is exhausted or returns no proposals.
    """
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    if study.run_method.is_exhausted(state):
        msg = "run_method is exhausted"
        raise RuntimeError(msg)

    proposals, next_state = study.run_method.ask(state, batch_size=batch_size)
    if len(proposals) == 0:
        msg = "run_method returned no proposals"
        raise RuntimeError(msg)

    if len(proposals) > batch_size:
        msg = "run_method returned more proposals than requested"
        raise ValueError(msg)

    proposal_kernel_hints = study.run_method.proposal_kernel_hints(
        next_state,
        proposals,
    )
    proposal_evaluation_specs = study.run_method.proposal_evaluation_specs(
        next_state,
        proposals,
    )
    top_level_query = ProposalBatchQuery(
        problem=study.problem,
        proposals=proposals,
        execution_resources=study.evaluator.execution_resources(),
        proposal_evaluation_specs=proposal_evaluation_specs,
        proposal_kernel_hints=proposal_kernel_hints,
        evaluation_budget=evaluation_budget,
    )
    top_level_requests = build_evaluation_requests(
        top_level_query.proposals,
        proposal_evaluation_specs=top_level_query.proposal_evaluation_specs,
    )

    def requests_for_query(
        query: ProposalBatchQuery[
            BoundaryT,
            CandidateT,
            StudyPayloadT,
        ],
    ) -> tuple[EvaluationRequest[CandidateT], ...]:
        if query is top_level_query:
            return top_level_requests

        return build_evaluation_requests(
            query.proposals,
            proposal_evaluation_specs=query.proposal_evaluation_specs,
        )

    if execution_model == EXACT_ASYNC_EXECUTION_MODEL:

        def batch_executor(
            query: ProposalBatchQuery[
                BoundaryT,
                CandidateT,
                StudyPayloadT,
            ],
        ) -> EvaluationAttemptBatch[CandidateT, StudyPayloadT]:
            query = _query_with_evaluation_budget(query, evaluation_budget)
            query_requests = requests_for_query(query)
            if query.evaluation_budget is not None:
                query.evaluation_budget.consume(len(query_requests))
            attempts = evaluate_batch_exact_async(
                study.evaluator,
                query.problem,
                query_requests,
            )
            validate_aligned_attempts(
                query_requests,
                attempts,
                candidate_equal=study.problem.space.candidates_equal,
            )
            return attempts
    else:

        def batch_executor(
            query: ProposalBatchQuery[
                BoundaryT,
                CandidateT,
                StudyPayloadT,
            ],
        ) -> EvaluationAttemptBatch[CandidateT, StudyPayloadT]:
            query = _query_with_evaluation_budget(query, evaluation_budget)
            query_requests = requests_for_query(query)
            attempts = evaluate_attempts_sync(
                study,
                query,
                requests=query_requests,
            )
            validate_aligned_attempts(
                query_requests,
                attempts,
                candidate_equal=study.problem.space.candidates_equal,
            )
            return attempts

    remaining_before = (
        None if evaluation_budget is None else evaluation_budget.remaining
    )
    kernel_attempts = study.kernel.run(top_level_query, batch_executor)
    requests = requests_for_query(top_level_query)
    validate_aligned_attempts(
        requests,
        kernel_attempts,
        candidate_equal=study.problem.space.candidates_equal,
    )
    reported_evaluation_count = kernel_attempts.evaluation_count
    step_evaluation_count = _consume_reported_evaluation_cost(
        evaluation_budget=evaluation_budget,
        remaining_before=remaining_before,
        reported_evaluation_count=reported_evaluation_count,
    )
    feedback_attempts = study.attempt_materializer.materialize_attempts(
        kernel_attempts
    )
    validate_materialized_attempts(
        kernel_attempts,
        feedback_attempts,
        candidate_equal=study.problem.space.candidates_equal,
    )
    next_state = study.run_method.tell_attempts(next_state, feedback_attempts)
    return StudyStepResult(
        attempts=feedback_attempts,
        state=next_state,
        evaluation_count=step_evaluation_count,
    )


def step(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    state: RunMethodStateT,
    batch_size: int = 1,
    *,
    execution_model: ExecutionModel = SYNC_BATCH_EXECUTION_MODEL,
) -> tuple[tuple[StudyRecordT, ...], RunMethodStateT]:
    """Run one ask/evaluate/tell step and return records and next state.

    Parameters
    ----------
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Study-like owner exposing the problem, run method, evaluator, and
        kernel.
    state : RunMethodStateT
        Run-method state to advance.
    batch_size : int, default=1
        Maximum number of proposals to request from the run method.
    execution_model : ExecutionModel, default=SYNC_BATCH_EXECUTION_MODEL
        Execution model controlling evaluation behavior.

    Returns
    -------
    tuple[tuple[StudyRecordT, ...], RunMethodStateT]
        Step records and the next run-method state.

    Raises
    ------
    NotImplementedError
        If ``execution_model`` requests stale-async semantics.
    """
    if execution_model.assimilation_mode is ExecutionAssimilationMode.STALE_INCREMENTAL:
        msg = (
            "stale_async execution model is only supported by "
            "Study.run and Study.optimize"
        )
        raise NotImplementedError(msg)

    validate_execution_request(
        study,
        batch_size=batch_size,
        execution_model=execution_model,
    )
    step_result = evaluate_step(
        study,
        state,
        batch_size,
        execution_model=execution_model,
    )
    records = materialize_success_records(step_result.attempts.successes)
    return records, step_result.state


def run(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    max_evaluations: int,
    batch_size: int = 1,
    *,
    execution_model: ExecutionModel = SYNC_BATCH_EXECUTION_MODEL,
    count_evaluation_cost: bool = True,
    initial_state: RunMethodStateT | None = None,
    stop_at_checkpoint_boundary: bool = False,
) -> tuple[RunReport[CandidateT, StudyRecordT], RunMethodStateT]:
    """Run repeated ask/evaluate/tell steps and return one generic run report.

    Parameters
    ----------
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Study-like owner exposing the problem, run method, evaluator, and
        kernel.
    max_evaluations : int
        Evaluation budget to consume.
    batch_size : int, default=1
        Maximum number of proposals requested per step.
    execution_model : ExecutionModel, default=SYNC_BATCH_EXECUTION_MODEL
        Execution model controlling evaluation behavior.
    count_evaluation_cost : bool, default=True
        Whether to debit the budget using evaluator-reported evaluation counts
        instead of completed record count.
    initial_state : RunMethodStateT | None, default=None
        Optional initial run-method state. ``None`` creates a fresh state.
    stop_at_checkpoint_boundary : bool, default=False
        Whether to return a checkpoint-safe state. If the budget ends inside an
        unsafe run-method segment, the report and state are rolled back to the
        most recent checkpoint-safe boundary reached during this call.

    Returns
    -------
    tuple[RunReport[CandidateT, StudyRecordT], RunMethodStateT]
        Run report and the final run-method state.

    Raises
    ------
    ValueError
        If ``max_evaluations`` is negative or ``execution_model`` requests
        stale-async semantics.
    """
    if max_evaluations < 0:
        msg = "max_evaluations must be non-negative"
        raise ValueError(msg)

    if execution_model == STALE_ASYNC_EXECUTION_MODEL:
        msg = "stale_async execution is handled by the stale_async study tier"
        raise ValueError(msg)

    validate_execution_request(
        study,
        batch_size=batch_size,
        execution_model=execution_model,
    )

    successes: list[EvaluationSuccess[CandidateT, StudyRecordT]] = []
    failures: list[EvaluationFailure[CandidateT]] = []
    trace_events: list[TraceEvent] = []
    evaluation_budget = (
        EvaluationBudget(max_evaluations) if count_evaluation_cost else None
    )
    record_budget_remaining = max_evaluations
    reported_evaluation_count_total = 0
    state = (
        study.run_method.create_initial_state()
        if initial_state is None
        else initial_state
    )
    safe_snapshot: (
        _CheckpointSafeRunSnapshot[
            CandidateT,
            RunMethodStateT,
            StudyRecordT,
        ]
        | None
    ) = None
    unsafe_since_safe_snapshot = False
    if stop_at_checkpoint_boundary and study.run_method.is_checkpoint_safe_state(state):
        safe_snapshot = _CheckpointSafeRunSnapshot(
            successes=(),
            failures=(),
            trace=Trace(),
            evaluation_count=0,
            state=state,
        )

    while _current_remaining_budget(
        evaluation_budget=evaluation_budget,
        record_budget_remaining=record_budget_remaining,
    ) > 0 and not study.run_method.is_exhausted(state):
        remaining = _current_remaining_budget(
            evaluation_budget=evaluation_budget,
            record_budget_remaining=record_budget_remaining,
        )
        current_batch_size = min(batch_size, remaining)
        try:
            step_result = evaluate_step(
                study,
                state,
                batch_size=current_batch_size,
                execution_model=execution_model,
                evaluation_budget=evaluation_budget,
            )
        except EvaluationBudgetExhausted:
            if stop_at_checkpoint_boundary and safe_snapshot is not None:
                return (
                    _run_report_from_snapshot(
                        snapshot=safe_snapshot,
                        candidate_equal=study.problem.space.candidates_equal,
                    ),
                    safe_snapshot.state,
                )
            raise
        except Exception as exception:
            _raise_run_execution_failed(
                cause=exception,
                successes=tuple(successes),
                failures=tuple(failures),
                trace_events=tuple(trace_events),
                evaluation_count=_current_evaluation_count(
                    max_evaluations=max_evaluations,
                    evaluation_budget=evaluation_budget,
                    record_budget_remaining=record_budget_remaining,
                ),
                state=state,
                safe_snapshot=safe_snapshot,
                candidate_equal=study.problem.space.candidates_equal,
            )
        batch_successes = step_result.attempts.successes
        batch_records = materialize_success_records(step_result.attempts.successes)
        state = step_result.state
        reported_evaluation_count_total += step_result.evaluation_count
        successes.extend(batch_successes)
        failures.extend(step_result.attempts.failures)
        if evaluation_budget is None:
            remaining -= step_result.attempts.attempt_count
            record_budget_remaining = remaining
        trace_events.append(
            TraceEvent(
                kind="study.step",
                message=(
                    f"completed {step_result.attempts.attempt_count} attempt(s): "
                    f"{len(batch_records)} succeeded, "
                    f"{len(step_result.attempts.failures)} failed"
                ),
                value=trace_value_for_records(batch_records),
            ),
        )
        if stop_at_checkpoint_boundary:
            if study.run_method.is_checkpoint_safe_state(state):
                safe_snapshot = _CheckpointSafeRunSnapshot(
                    successes=tuple(successes),
                    failures=tuple(failures),
                    trace=Trace(events=tuple(trace_events)),
                    evaluation_count=reported_evaluation_count_total,
                    state=state,
                )
                if unsafe_since_safe_snapshot:
                    break
                unsafe_since_safe_snapshot = False
            else:
                unsafe_since_safe_snapshot = True

    if stop_at_checkpoint_boundary and not study.run_method.is_checkpoint_safe_state(
        state
    ):
        if safe_snapshot is None:
            msg = (
                "run did not reach a checkpoint-safe state within the evaluation budget"
            )
            raise RuntimeError(msg)
        return (
            _run_report_from_snapshot(
                snapshot=safe_snapshot,
                candidate_equal=study.problem.space.candidates_equal,
            ),
            safe_snapshot.state,
        )

    return (
        _build_run_report(
            successes=tuple(successes),
            failures=tuple(failures),
            trace_events=tuple(trace_events),
            evaluation_count=reported_evaluation_count_total,
            candidate_equal=study.problem.space.candidates_equal,
        ),
        state,
    )


def materialize_scalar_run_result(
    run_report: RunReport[CandidateT, StudyRecordT],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None = None,
) -> RunResult[CandidateT]:
    """Project a generic run report into a scalar terminal result.

    Parameters
    ----------
    run_report : RunReport[CandidateT, StudyRecordT]
        Generic terminal report to project.
    candidate_equal : CandidateEquality[CandidateT] | None, optional
        Explicit candidate equality predicate used to validate refinement
        alignment. When absent, strict scalar Python equality is used.

    Returns
    -------
    RunResult[CandidateT]
        Scalar terminal result that preserves trace, accounting, and aligned
        refinement provenance from ``run_report``.

    Raises
    ------
    TypeError
        If the report does not contain scalar
        :class:`~variopt.artifacts.ObservationPayload` successes.
    """
    successes: list[EvaluationSuccess[CandidateT, ObservationPayload]] = []
    for success in run_report.successes:
        payload = success.payload
        if isinstance(payload, Observation):
            scalar_payload = ObservationPayload(
                value=payload.value,
                score=payload.score,
                elapsed_seconds=payload.elapsed_seconds,
            )
        else:
            msg = (
                "Study.optimize currently requires scalar ObservationPayload records; "
                "use Study.run for non-scalar evaluation protocols"
            )
            raise TypeError(msg)
        successes.append(
            EvaluationSuccess(
                request=success.request,
                payload=scalar_payload,
                evaluation_count=success.evaluation_count,
                refinement=success.refinement,
                kernel_diagnostics=success.kernel_diagnostics,
                candidate_equal=candidate_equal,
            ),
        )

    return RunResult[CandidateT].from_successes(
        successes=tuple(successes),
        evaluation_count=run_report.evaluation_count,
        trace=run_report.trace,
        failures=run_report.failures,
        candidate_equal=candidate_equal,
    )


def optimize(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    max_evaluations: int,
    batch_size: int = 1,
    *,
    execution_model: ExecutionModel = SYNC_BATCH_EXECUTION_MODEL,
    count_evaluation_cost: bool = True,
    initial_state: RunMethodStateT | None = None,
    stop_at_checkpoint_boundary: bool = False,
) -> tuple[RunResult[CandidateT], RunMethodStateT]:
    """Run repeated ask/evaluate/tell steps until the budget is exhausted.

    Parameters
    ----------
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Study-like owner exposing the problem, run method, evaluator, and
        kernel.
    max_evaluations : int
        Evaluation budget to consume.
    batch_size : int, default=1
        Maximum number of proposals requested per step.
    execution_model : ExecutionModel, default=SYNC_BATCH_EXECUTION_MODEL
        Execution model controlling evaluation behavior.
    count_evaluation_cost : bool, default=True
        Whether to debit the budget using evaluator-reported evaluation counts
        instead of completed record count.
    initial_state : RunMethodStateT | None, default=None
        Optional initial run-method state. ``None`` creates a fresh state.
    stop_at_checkpoint_boundary : bool, default=False
        Whether to return only checkpoint-safe terminal state boundaries.

    Returns
    -------
    tuple[RunResult[CandidateT], RunMethodStateT]
        Scalar optimization result and the final run-method state.

    Raises
    ------
    TypeError
        If the study does not emit scalar :class:`Observation` records.
    """
    if not stop_at_checkpoint_boundary and _supports_direct_scalar_sequential_path(
        study,
        execution_model=execution_model,
    ):
        return _optimize_direct_scalar_sequential(
            study,
            max_evaluations=max_evaluations,
            batch_size=batch_size,
            execution_model=execution_model,
            count_evaluation_cost=count_evaluation_cost,
            initial_state=initial_state,
        )

    run_report, state = run(
        study,
        max_evaluations=max_evaluations,
        batch_size=batch_size,
        execution_model=execution_model,
        count_evaluation_cost=count_evaluation_cost,
        initial_state=initial_state,
        stop_at_checkpoint_boundary=stop_at_checkpoint_boundary,
    )

    return (
        materialize_scalar_run_result(
            run_report,
            candidate_equal=study.problem.space.candidates_equal,
        ),
        state,
    )
