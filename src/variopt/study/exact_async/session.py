"""Study-owned exact-async step-session lifecycle object."""

from dataclasses import dataclass, field
from typing import Generic

from typing_extensions import TypeVar

from ...artifacts import EvaluationRequest
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
from ...outcomes import EvaluationOutcome
from ...typevars import CandidateT, RunMethodStateT
from ..common import (
    StudyEvaluationRecordT,
    finalize_ordered_outcomes,
    store_completion_group,
    validate_aligned_outcomes,
)
from .artifacts import (
    StudyExactAsyncSessionLifecycle,
    StudyExactAsyncStepResumeHandle,
)
from .contracts import StudyRunMethodOwner

BoundaryT = TypeVar("BoundaryT")


@dataclass(slots=True)
class StudyExactAsyncStepSession(
    Generic[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
):
    """Study-owned exact-async step session over one pre-tell request batch.

    Parameters
    ----------
    study : StudyRunMethodOwner[CandidateT, RunMethodStateT, StudyEvaluationRecordT]
        Study façade that owns the run method and evaluator.
    requests : tuple[EvaluationRequest[CandidateT], ...]
        Requests issued for this step.
    post_ask_state : RunMethodStateT
        Run-method state captured immediately after the corresponding ``ask``.
    batch_session : EvaluationBatchSession[EvaluationOutcome[CandidateT, StudyEvaluationRecordT]]
        Evaluator-owned async batch session for the issued requests.
    ordered_outcomes : list[EvaluationOutcome[CandidateT, StudyEvaluationRecordT] | None], optional
        Optional pre-filled outcome slots aligned to ``requests``.
    """

    study: StudyRunMethodOwner[
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ]
    requests: tuple[EvaluationRequest[CandidateT], ...]
    post_ask_state: RunMethodStateT
    batch_session: EvaluationBatchSession[
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT]
    ]
    ordered_outcomes: list[
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT] | None
    ] = field(default_factory=list)
    _lifecycle: StudyExactAsyncSessionLifecycle = "active"
    _final_records: tuple[StudyEvaluationRecordT, ...] | None = None
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

        if len(self.ordered_outcomes) == 0:
            self.ordered_outcomes = [None] * self.batch_session.handle.request_count
            return

        if len(self.ordered_outcomes) != self.batch_session.handle.request_count:
            msg = "ordered_outcomes must align with batch_session.handle.request_count"
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
        completed_count = sum(
            outcome is not None for outcome in self.ordered_outcomes
        )
        return EvaluationBatchSessionState(
            request_count=self.handle.request_count,
            completed_count=completed_count,
            pending_count=self.handle.request_count - completed_count,
            lifecycle=self._lifecycle,
        )

    def poll(
        self,
    ) -> tuple[
        CompletionGroup[EvaluationOutcome[CandidateT, StudyEvaluationRecordT]],
        ...,
    ]:
        """Poll newly completed exact-async groups for this step session.

        Returns
        -------
        tuple[CompletionGroup[EvaluationOutcome[CandidateT, StudyEvaluationRecordT]], ...]
            Newly completed outcome groups emitted by the evaluator.

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
            self._lifecycle = (
                "cancelled" if exception.kind == "cancelled" else "failed"
            )
            raise

        for completion_group in completion_groups:
            _ = store_completion_group(
                self.ordered_outcomes,
                completion_group,
                request_count=self.handle.request_count,
            )

        if all(outcome is not None for outcome in self.ordered_outcomes):
            self._lifecycle = "completed"

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

    def suspend(
        self,
    ) -> StudyExactAsyncStepResumeHandle[
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ]:
        """Suspend an active step session and return a study-owned handle.

        Returns
        -------
        StudyExactAsyncStepResumeHandle[CandidateT, RunMethodStateT, StudyEvaluationRecordT]
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
            ordered_outcomes=tuple(self.ordered_outcomes),
        )

    def finish(self) -> tuple[tuple[StudyEvaluationRecordT, ...], RunMethodStateT]:
        """Drain completions, assimilate the batch, and return final records.

        Returns
        -------
        tuple[tuple[StudyEvaluationRecordT, ...], RunMethodStateT]
            Finalized evaluation records together with the advanced run-method
            state after ``tell``.

        Raises
        ------
        RuntimeError
            Raised when finishing a cancelled, failed, or suspended session.
        """
        if self._final_records is not None and self._final_state is not None:
            return self._final_records, self._final_state

        if self._lifecycle in {"cancelled", "failed", "suspended"}:
            msg = (
                "study exact-async step session must be active or completed to finish"
            )
            raise RuntimeError(msg)

        while self._lifecycle == "active":
            _ = self.poll()

        outcomes = finalize_ordered_outcomes(self.ordered_outcomes)
        validate_aligned_outcomes(self.requests, outcomes)
        records = tuple(outcome.record for outcome in outcomes)
        next_state = self.study.run_method.tell(self.post_ask_state, records)
        self._final_records = records
        self._final_state = next_state
        return records, next_state
