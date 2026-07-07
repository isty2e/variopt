"""Study-owned exact-async step-session lifecycle object."""

from dataclasses import dataclass, field
from typing import Generic

from typing_extensions import TypeVar

from ...artifacts import (
    EvaluationAttemptBatch,
    EvaluationRequest,
    materialize_success_records,
)
from ...evaluators.async_evaluator.artifacts import (
    BatchExecutionFailed,
    CompletionGroup,
    EvaluationBatchHandle,
    EvaluationBatchSessionState,
)
from ...evaluators.async_evaluator.sessions import (
    EvaluationBatchSession,
    ResumableBatchSession,
)
from ...spaces import CandidateEquality
from ...typevars import CandidateT, RunMethodStateT
from ..assimilation import materialize_feedback_attempts
from ..common import (
    StudyPayloadT,
    StudyRecordT,
    finalize_ordered_attempts,
    store_completion_group,
    validate_aligned_attempts,
)
from .artifacts import (
    StudyExactAsyncSessionLifecycle,
    StudyExactAsyncStepResumeHandle,
)
from .contracts import StudyRunMethodOwner

BoundaryT = TypeVar("BoundaryT")


@dataclass(slots=True)
class StudyExactAsyncStepSession(
    Generic[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ]
):
    """Study-owned exact-async step session over one pre-tell request batch.

    Parameters
    ----------
    study : StudyRunMethodOwner[CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Study façade that owns the run method and evaluator.
    requests : tuple[EvaluationRequest[CandidateT], ...]
        Requests issued for this step.
    post_ask_state : RunMethodStateT
        Run-method state captured immediately after the corresponding ``ask``.
    batch_session : EvaluationBatchSession[EvaluationAttemptBatch[CandidateT, StudyPayloadT]]
        Evaluator-owned async batch session for the issued requests.
    ordered_attempts : list[EvaluationAttemptBatch[CandidateT, StudyPayloadT] | None], optional
        Optional pre-filled attempt slots aligned to ``requests``.
    """

    study: StudyRunMethodOwner[
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ]
    requests: tuple[EvaluationRequest[CandidateT], ...]
    post_ask_state: RunMethodStateT
    batch_session: EvaluationBatchSession[
        EvaluationAttemptBatch[CandidateT, StudyPayloadT]
    ]
    candidate_equal: CandidateEquality[CandidateT]
    ordered_attempts: list[EvaluationAttemptBatch[CandidateT, StudyPayloadT] | None] = (
        field(default_factory=list)
    )
    _lifecycle: StudyExactAsyncSessionLifecycle = "active"
    _final_records: tuple[StudyRecordT, ...] | None = None
    _final_state: RunMethodStateT | None = None

    def __post_init__(self) -> None:
        """Normalize ordered storage for this study-owned exact-async step session.

        Raises
        ------
        ValueError
            Raised when the provided request or outcome counts do not align
            with the evaluator-owned batch handle.
        """
        if len(self.requests) != self.batch_session.handle.request_count:
            msg = "requests must align with batch_session.handle.request_count"
            raise ValueError(msg)

        if len(self.ordered_attempts) == 0:
            self.ordered_attempts = [None] * self.batch_session.handle.request_count
            return

        if len(self.ordered_attempts) != self.batch_session.handle.request_count:
            msg = "ordered_attempts must align with batch_session.handle.request_count"
            raise ValueError(msg)

    @property
    def handle(self) -> EvaluationBatchHandle:
        """Return the evaluator-owned logical batch handle.

        Returns
        -------
        EvaluationBatchHandle
            Logical batch handle associated with this study step session.
        """
        return self.batch_session.handle

    def state(self) -> EvaluationBatchSessionState:
        """Return canonical lifecycle state for this step session.

        Returns
        -------
        EvaluationBatchSessionState
            Canonical lifecycle summary derived from stored outcomes and the
            evaluator-owned handle.
        """
        completed_count = sum(attempt is not None for attempt in self.ordered_attempts)
        return EvaluationBatchSessionState(
            request_count=self.handle.request_count,
            completed_count=completed_count,
            pending_count=self.handle.request_count - completed_count,
            lifecycle=self._lifecycle,
        )

    def poll(
        self,
    ) -> tuple[
        CompletionGroup[EvaluationAttemptBatch[CandidateT, StudyPayloadT]],
        ...,
    ]:
        """Poll newly completed exact-async groups for this step session.

        Returns
        -------
        tuple[CompletionGroup[EvaluationAttemptBatch[CandidateT, StudyPayloadT]], ...]
            Newly completed attempt groups emitted by the evaluator.

        Raises
        ------
        RuntimeError
            Raised when polling a session that is no longer active.
        BatchExecutionFailed
            Raised when the evaluator reports batch failure or cancellation.
        """
        if self._lifecycle != "active":
            msg = "study exact-async step session is no longer active"
            raise RuntimeError(msg)

        try:
            completion_groups = tuple(self.batch_session.poll())
        except BatchExecutionFailed as exception:
            self._lifecycle = "cancelled" if exception.kind == "cancelled" else "failed"
            raise

        try:
            self._record_completion_groups(completion_groups)
        except Exception:
            self._fail_and_cancel_batch_session()
            raise
        return completion_groups

    def wait(
        self,
        *,
        timeout: float | None = None,
    ) -> tuple[
        CompletionGroup[EvaluationAttemptBatch[CandidateT, StudyPayloadT]],
        ...,
    ]:
        """Wait for newly completed exact-async groups for this step session.

        Parameters
        ----------
        timeout : float | None, default=None
            Maximum number of seconds to wait. ``None`` waits indefinitely.

        Returns
        -------
        tuple[CompletionGroup[EvaluationAttemptBatch[CandidateT, StudyPayloadT]], ...]
            Newly completed attempt groups emitted by the evaluator, or an
            empty tuple when ``timeout`` expires first.

        Raises
        ------
        RuntimeError
            Raised when waiting on a session that is no longer active.
        BatchExecutionFailed
            Raised when the evaluator reports batch failure or cancellation.
        """
        if self._lifecycle != "active":
            msg = "study exact-async step session is no longer active"
            raise RuntimeError(msg)

        try:
            completion_groups = tuple(self.batch_session.wait(timeout=timeout))
        except BatchExecutionFailed as exception:
            self._lifecycle = "cancelled" if exception.kind == "cancelled" else "failed"
            raise

        try:
            self._record_completion_groups(completion_groups)
        except Exception:
            self._fail_and_cancel_batch_session()
            raise
        return completion_groups

    def cancel(self) -> None:
        """Cancel the evaluator-owned logical batch for this study step.

        Notes
        -----
        Cancellation is idempotent once the session is already terminal.
        """
        if self._lifecycle in {"cancelled", "failed", "completed"}:
            return
        self.batch_session.cancel()
        self._lifecycle = "cancelled"

    def _record_completion_groups(
        self,
        completion_groups: tuple[
            CompletionGroup[EvaluationAttemptBatch[CandidateT, StudyPayloadT]],
            ...,
        ],
    ) -> None:
        """Store newly observed completion groups and update lifecycle."""
        for completion_group in completion_groups:
            _ = store_completion_group(
                self.ordered_attempts,
                completion_group,
                request_count=self.handle.request_count,
            )

        if all(attempt is not None for attempt in self.ordered_attempts):
            self._lifecycle = "completed"

    def _fail_and_cancel_batch_session(self) -> None:
        """Mark the study session failed and best-effort cancel evaluator work."""
        self._lifecycle = "failed"
        try:
            self.batch_session.cancel()
        except Exception as cancel_exception:
            # Preserve the validation failure as the user-visible cause.
            _ = cancel_exception

    def suspend(
        self,
    ) -> StudyExactAsyncStepResumeHandle[
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
    ]:
        """Suspend an active step session and return a study-owned handle.

        Returns
        -------
        StudyExactAsyncStepResumeHandle[CandidateT, RunMethodStateT, StudyPayloadT]
            Resume handle that captures evaluator and study-side state.

        Raises
        ------
        RuntimeError
            Raised when the session is not active.
        TypeError
            Raised when the underlying evaluator session is not resumable.
        """
        if self._lifecycle != "active":
            msg = "only active study exact-async step sessions can be suspended"
            raise RuntimeError(msg)

        if not isinstance(self.batch_session, ResumableBatchSession):
            msg = "study exact-async step session is not resumable"
            raise TypeError(msg)

        evaluator_handle = self.batch_session.suspend()
        self._lifecycle = "suspended"
        return StudyExactAsyncStepResumeHandle(
            evaluator_handle=evaluator_handle,
            requests=self.requests,
            post_ask_state=self.post_ask_state,
            ordered_attempts=tuple(self.ordered_attempts),
        )

    def finish(self) -> tuple[tuple[StudyRecordT, ...], RunMethodStateT]:
        """Drain completions, assimilate the batch, and return final records.

        Returns
        -------
        tuple[tuple[StudyRecordT, ...], RunMethodStateT]
            Finalized evaluation records together with the advanced run-method
            state after ``tell_attempts``.

        Raises
        ------
        RuntimeError
            Raised when finishing a cancelled, failed, or suspended session.
        """
        if self._final_records is not None and self._final_state is not None:
            return self._final_records, self._final_state

        if self._lifecycle in {"cancelled", "failed", "suspended"}:
            msg = "study exact-async step session must be active or completed to finish"
            raise RuntimeError(msg)

        while self._lifecycle == "active":
            _ = self.wait()

        attempts = finalize_ordered_attempts(self.ordered_attempts)
        validate_aligned_attempts(
            self.requests,
            attempts,
            candidate_equal=self.candidate_equal,
        )
        feedback_attempts = materialize_feedback_attempts(
            attempts,
            self.study.attempt_materializer,
            candidate_equal=self.candidate_equal,
        )
        records = materialize_success_records(feedback_attempts.successes)
        next_state = self.study.run_method.tell_attempts(
            self.post_ask_state,
            feedback_attempts,
        )
        self._final_records = records
        self._final_state = next_state
        return records, next_state
