"""Stale-async study helpers."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import sleep
from typing import Generic, Protocol

from typing_extensions import TypeVar

from ..artifacts import (
    CandidateRefinement,
    EvaluationFailure,
    EvaluationRequest,
    Proposal,
    ProposalEvaluationSpec,
    RunReport,
    Trace,
    TraceEvent,
)
from ..evaluators.async_evaluator.contracts import AsyncEvaluator
from ..evaluators.async_evaluator.sessions import EvaluationBatchSession
from ..evaluators.base import Evaluator
from ..execution import (
    STALE_ASYNC_EXECUTION_MODEL,
    EvaluationBudget,
    EvaluationBudgetExhausted,
)
from ..kernel import DirectKernel, Kernel, ProposalBatchQuery
from ..methods import RunMethod
from ..outcomes import EvaluationAttemptBatch, EvaluationOutcome
from ..problem import Problem
from ..spaces import CandidateEquality
from ..typevars import CandidateT, RunMethodStateT
from .common import (
    OutcomeToAttemptBatchSession,
    StudyEvaluationRecordT,
    build_evaluation_requests,
    supports_attempt_batch_sessions,
    trace_value_for_records,
    validate_aligned_attempts,
)
from .failures import RunExecutionFailed
from .validation import require_async_evaluator, validate_execution_request

BoundaryT = TypeVar("BoundaryT")
_STALE_ASYNC_IDLE_SLEEP_SECONDS = 0.001


class _StudyStaleAsyncOwner(
    Protocol[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
):
    """Protocol for the subset of Study state stale-async orchestration needs."""

    @property
    def problem(
        self,
    ) -> Problem[BoundaryT, CandidateT, StudyEvaluationRecordT]:
        """Return the configured problem."""
        ...

    @property
    def run_method(
        self,
    ) -> RunMethod[
        RunMethodStateT,
        Proposal[CandidateT],
        StudyEvaluationRecordT,
    ]:
        """Return the configured run method."""
        ...

    @property
    def evaluator(
        self,
    ) -> Evaluator[
        Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT],
    ]:
        """Return the configured evaluator."""
        ...

    @property
    def kernel(
        self,
    ) -> Kernel[
        ProposalBatchQuery[BoundaryT, CandidateT, StudyEvaluationRecordT],
        EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT],
    ]:
        """Return the configured kernel."""
        ...


@dataclass(slots=True)
class StaleAsyncActiveBatchSession(Generic[CandidateT, StudyEvaluationRecordT]):
    """Run-owned active async batch with incremental stale assimilation tracking.

    Parameters
    ----------
    requests : tuple[EvaluationRequest[CandidateT], ...]
        Logical request batch owned by the session.
    batch_session : EvaluationBatchSession[EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]]
        Evaluator-owned async batch session that executes ``requests``.
    completed_indices : set[int], default=set()
        Logical request indices already emitted through completed groups.
    """

    requests: tuple[EvaluationRequest[CandidateT], ...]
    batch_session: EvaluationBatchSession[
        EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]
    ]
    candidate_equal: CandidateEquality[CandidateT]
    completed_indices: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Reject inconsistent active-session payloads."""
        if len(self.requests) != self.batch_session.handle.request_count:
            msg = "requests must align with batch_session.handle.request_count"
            raise ValueError(msg)

    def poll_completed_groups(
        self,
    ) -> tuple[EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT], ...]:
        """Poll and validate newly completed groups for one stale-async session."""
        completed_groups: list[
            EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]
        ] = []
        for completion_group in self.batch_session.poll():
            end_index = completion_group.start_index + len(completion_group.outcomes)
            if end_index > self.batch_session.handle.request_count:
                msg = "completion group exceeds logical batch bounds"
                raise ValueError(msg)

            group_requests = self.requests[completion_group.start_index : end_index]
            attempts = EvaluationAttemptBatch[
                CandidateT,
                StudyEvaluationRecordT,
            ].from_single_request_attempts(completion_group.outcomes)
            validate_aligned_attempts(
                group_requests,
                attempts,
                candidate_equal=self.candidate_equal,
            )

            for offset, _outcome in enumerate(completion_group.outcomes):
                target_index = completion_group.start_index + offset
                if target_index in self.completed_indices:
                    msg = "completion groups must not overlap"
                    raise ValueError(msg)
                self.completed_indices.add(target_index)

            completed_groups.append(attempts)

        return tuple(completed_groups)

    @property
    def is_completed(self) -> bool:
        """Return whether this active session has completed all requests."""
        return len(self.completed_indices) == self.batch_session.handle.request_count

    def cancel(self) -> None:
        """Cancel the underlying evaluator-owned batch session."""
        self.batch_session.cancel()


def _cancel_active_stale_async_sessions(
    active_sessions: Sequence[
        StaleAsyncActiveBatchSession[CandidateT, StudyEvaluationRecordT]
    ],
) -> None:
    """Cancel all active stale-async sessions while preserving the run failure."""
    for active_session in active_sessions:
        try:
            active_session.cancel()
        except Exception:
            # Cancellation is best-effort; do not let one backend cleanup failure
            # leak later sessions or mask the original run failure.
            continue


def open_stale_async_batch_session(
    *,
    async_evaluator: AsyncEvaluator[
        Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT],
    ],
    problem: Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
    run_method_ask: Callable[
        ...,
        tuple[tuple[Proposal[CandidateT], ...], RunMethodStateT],
    ],
    proposal_evaluation_specs_for: Callable[
        ...,
        tuple[ProposalEvaluationSpec | None, ...] | None,
    ],
    state: RunMethodStateT,
    batch_size: int,
    evaluation_budget: EvaluationBudget | None = None,
) -> tuple[
    StaleAsyncActiveBatchSession[CandidateT, StudyEvaluationRecordT], RunMethodStateT
]:
    """Ask one batch and open its stale-async evaluator session.

    Parameters
    ----------
    async_evaluator : AsyncEvaluator[Problem[BoundaryT, CandidateT, StudyEvaluationRecordT], EvaluationRequest[CandidateT], EvaluationOutcome[CandidateT, StudyEvaluationRecordT]]
        Async evaluator used to open the backend session.
    problem : Problem[BoundaryT, CandidateT, StudyEvaluationRecordT]
        Problem being optimized.
    run_method_ask : Callable[..., tuple[tuple[Proposal[CandidateT], ...], RunMethodStateT]]
        ``RunMethod.ask``-compatible callable.
    proposal_evaluation_specs_for : Callable[..., tuple[ProposalEvaluationSpec | None, ...] | None]
        Callable that resolves proposal evaluation specs for the requested proposals.
    state : RunMethodStateT
        Current run-method state.
    batch_size : int
        Requested logical batch size.
    evaluation_budget : EvaluationBudget | None, default=None
        Optional hard evaluation-budget ledger consumed before submitting work.

    Returns
    -------
    tuple[StaleAsyncActiveBatchSession[CandidateT, StudyEvaluationRecordT], RunMethodStateT]
        Opened active batch session and the post-ask run-method state.

    Raises
    ------
    RuntimeError
        If ``run_method_ask`` returns no proposals.
    ValueError
        If ``batch_size`` is invalid or ``run_method_ask`` overproduces.
    """
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    proposals, next_state = run_method_ask(state, batch_size=batch_size)
    if len(proposals) == 0:
        msg = "run_method returned no proposals"
        raise RuntimeError(msg)

    if len(proposals) > batch_size:
        msg = "run_method returned more proposals than requested"
        raise ValueError(msg)

    proposal_evaluation_specs = proposal_evaluation_specs_for(
        next_state,
        proposals,
    )
    requests = build_evaluation_requests(
        proposals,
        proposal_evaluation_specs=proposal_evaluation_specs,
    )
    if evaluation_budget is not None:
        evaluation_budget.consume(len(requests))
    return (
        StaleAsyncActiveBatchSession(
            requests=requests,
            batch_session=(
                async_evaluator.open_attempt_session(problem, requests)
                if supports_attempt_batch_sessions(async_evaluator)
                else OutcomeToAttemptBatchSession(
                    requests=requests,
                    outcome_session=async_evaluator.open_session(problem, requests),
                    candidate_equal=problem.space.candidates_equal,
                )
            ),
            candidate_equal=problem.space.candidates_equal,
        ),
        next_state,
    )


def _validate_stale_async_run_request(
    study: _StudyStaleAsyncOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ],
    *,
    batch_size: int,
) -> None:
    """Reject invalid stale-async run requests."""
    validate_execution_request(
        study,
        batch_size=batch_size,
        execution_model=STALE_ASYNC_EXECUTION_MODEL,
    )
    if not isinstance(study.kernel, DirectKernel):
        msg = "stale_async execution model currently requires DirectKernel"
        raise ValueError(msg)


def _open_stale_async_batch_session_for_study(
    study: _StudyStaleAsyncOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ],
    state: RunMethodStateT,
    *,
    batch_size: int,
    evaluation_budget: EvaluationBudget | None = None,
) -> tuple[
    StaleAsyncActiveBatchSession[CandidateT, StudyEvaluationRecordT],
    RunMethodStateT,
]:
    """Ask one direct batch and open one active stale-async evaluator session."""
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    if not isinstance(study.kernel, DirectKernel):
        msg = "stale_async execution model currently requires DirectKernel"
        raise ValueError(msg)

    return open_stale_async_batch_session(
        async_evaluator=require_async_evaluator(study),
        problem=study.problem,
        run_method_ask=study.run_method.ask,
        proposal_evaluation_specs_for=study.run_method.proposal_evaluation_specs,
        state=state,
        batch_size=batch_size,
        evaluation_budget=evaluation_budget,
    )


def run_stale_async(
    study: _StudyStaleAsyncOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ],
    *,
    max_evaluations: int,
    batch_size: int,
    count_evaluation_cost: bool,
    initial_state: RunMethodStateT | None,
    stop_at_checkpoint_boundary: bool = False,
) -> tuple[RunReport[CandidateT, StudyEvaluationRecordT], RunMethodStateT]:
    """Run stale-incremental async orchestration with rolling batch refill.

    Parameters
    ----------
    study : _StudyStaleAsyncOwner[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
        Study-like owner providing problem, run method, evaluator, and kernel.
    max_evaluations : int
        Evaluation budget for the run.
    batch_size : int
        Maximum number of requests per active stale batch.
    count_evaluation_cost : bool
        Whether to decrement the budget by evaluator-reported evaluation cost
        instead of completed record count.
    initial_state : RunMethodStateT | None
        Optional initial run-method state.
    stop_at_checkpoint_boundary : bool, default=False
        Whether to roll the returned state/report back to the latest
        checkpoint-safe boundary if the budget ends inside an unsafe segment.

    Returns
    -------
    tuple[RunReport[CandidateT, StudyEvaluationRecordT], RunMethodStateT]
        Run report and terminal run-method state.
    """
    _validate_stale_async_run_request(study, batch_size=batch_size)

    records: list[StudyEvaluationRecordT] = []
    refinements: list[CandidateRefinement[CandidateT] | None] | None = None
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
    safe_records: tuple[StudyEvaluationRecordT, ...] | None = None
    safe_refinements: tuple[CandidateRefinement[CandidateT] | None, ...] | None = None
    safe_failures: tuple[EvaluationFailure[CandidateT], ...] = ()
    safe_trace = Trace()
    safe_evaluation_count = 0
    safe_state = state
    if stop_at_checkpoint_boundary and study.run_method.is_checkpoint_safe_state(state):
        safe_records = ()
    active_sessions: list[
        StaleAsyncActiveBatchSession[CandidateT, StudyEvaluationRecordT]
    ] = []

    try:
        while active_sessions or (
            (
                record_budget_remaining
                if evaluation_budget is None
                else evaluation_budget.remaining
            )
            > 0
            and not study.run_method.is_exhausted(state)
        ):
            remaining = (
                record_budget_remaining
                if evaluation_budget is None
                else evaluation_budget.remaining
            )
            if (
                len(active_sessions) == 0
                and remaining > 0
                and not study.run_method.is_exhausted(state)
            ):
                current_batch_size = min(batch_size, remaining)
                active_session, state = _open_stale_async_batch_session_for_study(
                    study,
                    state,
                    batch_size=current_batch_size,
                    evaluation_budget=evaluation_budget,
                )
                active_sessions.append(active_session)

            if len(active_sessions) == 0:
                break

            completed_any = False
            for active_session in tuple(active_sessions):
                completed_groups = active_session.poll_completed_groups()
                if len(completed_groups) > 0:
                    completed_any = True
                for completed_group in completed_groups:
                    group_records = completed_group.records
                    group_refinements = tuple(
                        outcome.refinement for outcome in completed_group.outcomes
                    )
                    group_evaluation_count = completed_group.evaluation_count
                    if evaluation_budget is not None:
                        unmetered_evaluation_count = (
                            group_evaluation_count - completed_group.attempt_count
                        )
                        if unmetered_evaluation_count > 0:
                            evaluation_budget.consume(unmetered_evaluation_count)
                    else:
                        record_budget_remaining -= completed_group.attempt_count
                    next_state = study.run_method.tell_attempts(
                        state,
                        completed_group,
                    )
                    records_before_group = len(records)
                    records.extend(group_records)
                    failures.extend(completed_group.failures)
                    if refinements is not None:
                        refinements.extend(group_refinements)
                    elif any(
                        refinement is not None for refinement in group_refinements
                    ):
                        refinement_history: list[
                            CandidateRefinement[CandidateT] | None
                        ] = [None for _index in range(records_before_group)]
                        refinement_history.extend(group_refinements)
                        refinements = refinement_history
                    state = next_state
                    trace_events.append(
                        TraceEvent(
                            kind="study.step",
                            message=(
                                f"completed {completed_group.attempt_count} attempt(s): "
                                f"{len(group_records)} succeeded, "
                                f"{len(completed_group.failures)} failed"
                            ),
                            value=trace_value_for_records(group_records),
                        ),
                    )
                    if (
                        stop_at_checkpoint_boundary
                        and study.run_method.is_checkpoint_safe_state(state)
                    ):
                        safe_records = tuple(records)
                        safe_refinements = (
                            None if refinements is None else tuple(refinements)
                        )
                        safe_failures = tuple(failures)
                        safe_trace = Trace(events=tuple(trace_events))
                        safe_evaluation_count = (
                            max_evaluations - record_budget_remaining
                            if evaluation_budget is None
                            else max_evaluations - evaluation_budget.remaining
                        )
                        safe_state = state

                    remaining = (
                        record_budget_remaining
                        if evaluation_budget is None
                        else evaluation_budget.remaining
                    )
                    if remaining > 0 and not study.run_method.is_exhausted(state):
                        refill_batch_size = min(completed_group.attempt_count, remaining)
                        refill_session, state = (
                            _open_stale_async_batch_session_for_study(
                                study,
                                state,
                                batch_size=refill_batch_size,
                                evaluation_budget=evaluation_budget,
                            )
                        )
                        active_sessions.append(refill_session)

            active_sessions = [
                active_session
                for active_session in active_sessions
                if not active_session.is_completed
            ]
            if not completed_any and len(active_sessions) > 0:
                sleep(_STALE_ASYNC_IDLE_SLEEP_SECONDS)
    except EvaluationBudgetExhausted:
        _cancel_active_stale_async_sessions(active_sessions)
        raise
    except Exception as exception:
        _cancel_active_stale_async_sessions(active_sessions)
        partial_report = RunReport[CandidateT, StudyEvaluationRecordT].from_records(
            records=tuple(records),
            evaluation_count=(
                max_evaluations - record_budget_remaining
                if evaluation_budget is None
                else max_evaluations - evaluation_budget.remaining
            ),
            trace=Trace(events=tuple(trace_events)),
            refinements=None if refinements is None else tuple(refinements),
            failures=tuple(failures),
            candidate_equal=study.problem.space.candidates_equal,
        )
        checkpoint_safe_report: RunReport[
            CandidateT,
            StudyEvaluationRecordT,
        ] | None = None
        checkpoint_safe_state: RunMethodStateT | None = None
        if safe_records is not None:
            checkpoint_safe_report = RunReport[
                CandidateT,
                StudyEvaluationRecordT,
            ].from_records(
                records=safe_records,
                evaluation_count=safe_evaluation_count,
                trace=safe_trace,
                refinements=safe_refinements,
                failures=safe_failures,
                candidate_equal=study.problem.space.candidates_equal,
            )
            checkpoint_safe_state = safe_state
        raise RunExecutionFailed[
            CandidateT,
            RunMethodStateT,
            StudyEvaluationRecordT,
        ](
            partial_report=partial_report,
            partial_state=state,
            checkpoint_safe_report=checkpoint_safe_report,
            checkpoint_safe_state=checkpoint_safe_state,
            cause=exception,
        ) from exception
    except BaseException:
        _cancel_active_stale_async_sessions(active_sessions)
        raise

    if stop_at_checkpoint_boundary and not study.run_method.is_checkpoint_safe_state(
        state
    ):
        if safe_records is None:
            msg = (
                "run did not reach a checkpoint-safe state within the evaluation budget"
            )
            raise RuntimeError(msg)
        return (
            RunReport[CandidateT, StudyEvaluationRecordT].from_records(
                records=safe_records,
                evaluation_count=safe_evaluation_count,
                trace=safe_trace,
                refinements=safe_refinements,
                failures=safe_failures,
                candidate_equal=study.problem.space.candidates_equal,
            ),
            safe_state,
        )

    return (
        RunReport[CandidateT, StudyEvaluationRecordT].from_records(
            records=records,
            evaluation_count=(
                max_evaluations - record_budget_remaining
                if evaluation_budget is None
                else max_evaluations - evaluation_budget.remaining
            ),
            trace=Trace(events=tuple(trace_events)),
            refinements=None if refinements is None else tuple(refinements),
            failures=tuple(failures),
            candidate_equal=study.problem.space.candidates_equal,
        ),
        state,
    )
