"""Study execution failure artifacts."""

from collections.abc import Callable, Sequence
from typing import Generic

from typing_extensions import TypeVar, override

from ..artifacts import (
    EvaluationFailure,
    EvaluationSuccess,
    RunReport,
    Trace,
    TraceEvent,
)
from ..artifacts.records import RequestAlignedEvaluationRecord
from ..spaces import CandidateEquality
from ..typevars import CandidateT, RunMethodStateT
from .common import CheckpointSafeRunSnapshot

RunFailureRecordT = TypeVar(
    "RunFailureRecordT",
    bound=RequestAlignedEvaluationRecord[object],
)
# Hard failures are raised after Study has crossed the payload-to-record
# boundary. Their partial reports therefore carry run-method feedback records,
# not raw request-free evaluator payloads.


class RunExecutionFailed(
    RuntimeError,
    Generic[CandidateT, RunMethodStateT, RunFailureRecordT],
):
    """Hard study-run failure with the recoverable partial run projection.

    Parameters
    ----------
    partial_report : RunReport[CandidateT, RunFailureRecordT]
        Report materialized from attempts fully assimilated before the hard
        failure. This report is not necessarily checkpoint-safe.
    partial_state : RunMethodStateT
        Run-method state aligned with ``partial_report``.
    checkpoint_safe_report : RunReport[CandidateT, RunFailureRecordT] | None
        Latest checkpoint-safe report reached before the failure, if any.
    checkpoint_safe_state : RunMethodStateT | None
        Run-method state aligned with ``checkpoint_safe_report``.
    cause : Exception
        Original hard failure. Recordable user-code evaluation exceptions belong
        in ``EvaluationFailure`` instead of this runtime exception.
    """

    partial_report: RunReport[CandidateT, RunFailureRecordT]
    partial_state: RunMethodStateT
    checkpoint_safe_report: RunReport[CandidateT, RunFailureRecordT] | None
    checkpoint_safe_state: RunMethodStateT | None
    cause: Exception

    def __init__(
        self,
        *,
        partial_report: RunReport[CandidateT, RunFailureRecordT],
        partial_state: RunMethodStateT,
        checkpoint_safe_report: RunReport[CandidateT, RunFailureRecordT] | None,
        checkpoint_safe_state: RunMethodStateT | None,
        cause: Exception,
    ) -> None:
        """Create one hard run-failure exception."""
        self.partial_report = partial_report
        self.partial_state = partial_state
        self.checkpoint_safe_report = checkpoint_safe_report
        self.checkpoint_safe_state = checkpoint_safe_state
        self.cause = cause
        super().__init__(f"study execution failed: {cause}")

    @override
    def __reduce__(
        self,
    ) -> tuple[
        Callable[
            [
                RunReport[CandidateT, RunFailureRecordT],
                RunMethodStateT,
                RunReport[CandidateT, RunFailureRecordT] | None,
                RunMethodStateT | None,
                Exception,
                BaseException | None,
                BaseException | None,
                bool,
            ],
            "RunExecutionFailed[CandidateT, RunMethodStateT, RunFailureRecordT]",
        ],
        tuple[
            RunReport[CandidateT, RunFailureRecordT],
            RunMethodStateT,
            RunReport[CandidateT, RunFailureRecordT] | None,
            RunMethodStateT | None,
            Exception,
            BaseException | None,
            BaseException | None,
            bool,
        ],
    ]:
        return (
            _restore_run_execution_failed,
            (
                self.partial_report,
                self.partial_state,
                self.checkpoint_safe_report,
                self.checkpoint_safe_state,
                self.cause,
                self.__cause__,
                self.__context__,
                self.__suppress_context__,
            ),
        )


def _restore_run_execution_failed(
    partial_report: RunReport[CandidateT, RunFailureRecordT],
    partial_state: RunMethodStateT,
    checkpoint_safe_report: RunReport[CandidateT, RunFailureRecordT] | None,
    checkpoint_safe_state: RunMethodStateT | None,
    cause: Exception,
    exception_cause: BaseException | None,
    exception_context: BaseException | None,
    suppress_context: bool,
) -> RunExecutionFailed[CandidateT, RunMethodStateT, RunFailureRecordT]:
    restored = RunExecutionFailed[
        CandidateT,
        RunMethodStateT,
        RunFailureRecordT,
    ](
        partial_report=partial_report,
        partial_state=partial_state,
        checkpoint_safe_report=checkpoint_safe_report,
        checkpoint_safe_state=checkpoint_safe_state,
        cause=cause,
    )
    restored.__cause__ = exception_cause
    restored.__context__ = exception_context
    restored.__suppress_context__ = suppress_context
    return restored


def build_run_report_or_raise_cause(
    *,
    cause: Exception,
    successes: Sequence[EvaluationSuccess[CandidateT, RunFailureRecordT]],
    evaluation_count: int,
    trace: Trace,
    failures: Sequence[EvaluationFailure[CandidateT]],
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> RunReport[CandidateT, RunFailureRecordT]:
    """Build a hard-failure report or re-raise the original run failure.

    Parameters
    ----------
    cause : Exception
        Original hard run failure being wrapped.
    successes : Sequence[EvaluationSuccess[CandidateT, RunFailureRecordT]]
        Successful attempts fully assimilated before ``cause``.
    evaluation_count : int
        Logical evaluation count to report.
    trace : Trace
        Trace accumulated before ``cause``.
    failures : Sequence[EvaluationFailure[CandidateT]]
        Recorded user-code failures accumulated before ``cause``.
    candidate_equal : CandidateEquality[CandidateT] | None
        Candidate equality predicate used to validate refinement alignment.

    Returns
    -------
    RunReport[CandidateT, RunFailureRecordT]
        Recoverable partial run report.

    Raises
    ------
    Exception
        Re-raises ``cause`` if report construction fails. The report
        construction failure is attached as ``cause.__cause__`` so the original
        hard failure remains the top-level exception instead of being masked by
        recovery projection invariants.
    """
    try:
        return RunReport[CandidateT, RunFailureRecordT].from_successes(
            successes=successes,
            evaluation_count=evaluation_count,
            trace=trace,
            failures=failures,
            candidate_equal=candidate_equal,
        )
    except Exception as report_failure:
        raise cause from report_failure


def build_checkpoint_safe_report_or_raise_cause(
    *,
    cause: Exception,
    snapshot: CheckpointSafeRunSnapshot[RunMethodStateT],
    successes: Sequence[EvaluationSuccess[CandidateT, RunFailureRecordT]],
    failures: Sequence[EvaluationFailure[CandidateT]],
    trace_events: Sequence[TraceEvent],
    candidate_equal: CandidateEquality[CandidateT],
) -> RunReport[CandidateT, RunFailureRecordT]:
    """Build a checkpoint-safe report or re-raise the original run failure.

    Parameters
    ----------
    cause : Exception
        Original hard run failure being wrapped.
    snapshot : CheckpointSafeRunSnapshot[RunMethodStateT]
        Checkpoint-safe cut point to project.
    successes : Sequence[EvaluationSuccess[CandidateT, RunFailureRecordT]]
        Append-only success history available at materialization time.
    failures : Sequence[EvaluationFailure[CandidateT]]
        Append-only failure history available at materialization time.
    trace_events : Sequence[TraceEvent]
        Append-only trace history available at materialization time.
    candidate_equal : CandidateEquality[CandidateT]
        Candidate equality predicate used to validate refinement alignment.

    Returns
    -------
    RunReport[CandidateT, RunFailureRecordT]
        Recoverable checkpoint-safe run report.

    Raises
    ------
    Exception
        Re-raises ``cause`` if report construction fails. The report
        construction failure is attached as ``cause.__cause__`` so the original
        hard failure remains the top-level exception.
    """
    try:
        return snapshot.to_report(
            successes=successes,
            failures=failures,
            trace_events=trace_events,
            candidate_equal=candidate_equal,
        )
    except Exception as report_failure:
        raise cause from report_failure
