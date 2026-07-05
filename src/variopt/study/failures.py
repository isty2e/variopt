"""Study execution failure artifacts."""

from typing import Generic

from typing_extensions import TypeVar

from ..artifacts import RunReport
from ..artifacts.records import RequestAlignedEvaluationRecord
from ..typevars import CandidateT, RunMethodStateT

RunFailureRecordT = TypeVar(
    "RunFailureRecordT",
    bound=RequestAlignedEvaluationRecord[object],
)


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
