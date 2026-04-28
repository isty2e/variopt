"""Async batch state for joblib-backed evaluators."""

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeVar

from typing_extensions import override

from ...artifacts import EvaluationRequest, RequestAlignedEvaluationRecord
from ...execution import ExecutionResources
from ...outcomes import EvaluationOutcome
from ...problem import Problem
from ...typevars import CandidateT
from ..async_evaluator.artifacts import (
    BatchExecutionFailed,
    CompletionGroup,
    EvaluationBatchHandle,
    EvaluationBatchResumeHandle,
    EvaluationBatchSessionState,
)
from ..async_evaluator.sessions import PendingAwareBatchSession, ResumableBatchSession
from .contracts import BoundaryT, JoblibEvaluationRecordT

SessionCandidateT = TypeVar("SessionCandidateT")
SessionRecordT = TypeVar(
    "SessionRecordT",
    bound=RequestAlignedEvaluationRecord,
)


class AsyncJoblibBatchSessionEvaluator(
    Protocol[SessionCandidateT, SessionRecordT]
):
    """Minimal evaluator surface required by one resumable joblib batch session.

    Notes
    -----
    The session object depends only on this narrow evaluator surface so the
    resumable joblib session logic stays decoupled from the full evaluator
    implementation.
    """

    def poll(
        self,
        handle: EvaluationBatchHandle,
    ) -> tuple[
        CompletionGroup[EvaluationOutcome[SessionCandidateT, SessionRecordT]],
        ...,
    ]:
        """Poll one submitted batch handle.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Logical batch handle to poll.

        Returns
        -------
        tuple[CompletionGroup[EvaluationOutcome[SessionCandidateT, SessionRecordT]], ...]
            Newly completed outcome groups.
        """
        ...

    def cancel(self, handle: EvaluationBatchHandle) -> None:
        """Cancel one submitted batch handle.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Logical batch handle to cancel.
        """
        ...

    def suspend_batch(
        self,
        handle: EvaluationBatchHandle,
    ) -> EvaluationBatchResumeHandle:
        """Suspend one active batch and return a resume handle.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Logical batch handle to suspend.

        Returns
        -------
        EvaluationBatchResumeHandle
            Resume handle for the suspended batch.
        """
        ...

    def discard_suspended_batch(self, handle: EvaluationBatchHandle) -> None:
        """Discard one suspended batch, if present.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Logical batch handle for the suspended batch.
        """
        ...


@dataclass(slots=True)
class AsyncJoblibRequestInput(Generic[CandidateT]):
    """One indexed async joblib evaluation request.

    Parameters
    ----------
    index : int
        Original request index within the submitted logical batch.
    request : EvaluationRequest[CandidateT]
        Evaluation request associated with the index.
    """

    index: int
    request: EvaluationRequest[CandidateT]


@dataclass(slots=True)
class ActiveAsyncJoblibBatch(Generic[BoundaryT, CandidateT, JoblibEvaluationRecordT]):
    """In-flight async joblib batch state.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT]
        Problem definition used to evaluate the batch.
    request_inputs : tuple[AsyncJoblibRequestInput[CandidateT], ...]
        Indexed requests that belong to the active batch.
    execution_resources : ExecutionResources
        Execution resources reserved for the batch.
    result_generator : Generator[tuple[int, EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]], None, None]
        Joblib-backed generator that yields indexed outcomes.
    completed_indices : set[int], optional
        Request indices already completed inside the active batch.
    infrastructure_retry_count : int, default=0
        Number of infrastructure retries already consumed.
    """

    problem: Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT]
    request_inputs: tuple[AsyncJoblibRequestInput[CandidateT], ...]
    execution_resources: ExecutionResources
    result_generator: Generator[
        tuple[int, EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]],
        None,
        None,
    ]
    completed_indices: set[int] = field(default_factory=set)
    infrastructure_retry_count: int = 0


@dataclass(slots=True)
class SuspendedAsyncJoblibBatch(
    Generic[BoundaryT, CandidateT, JoblibEvaluationRecordT]
):
    """Suspended async joblib batch state kept by one evaluator instance.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT]
        Problem definition used to evaluate the batch.
    request_inputs : tuple[AsyncJoblibRequestInput[CandidateT], ...]
        Indexed requests that belong to the suspended batch.
    execution_resources : ExecutionResources
        Execution resources reserved for the batch.
    completed_indices : set[int], optional
        Request indices already completed before suspension.
    infrastructure_retry_count : int, default=0
        Number of infrastructure retries already consumed.
    """

    problem: Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT]
    request_inputs: tuple[AsyncJoblibRequestInput[CandidateT], ...]
    execution_resources: ExecutionResources
    completed_indices: set[int] = field(default_factory=set)
    infrastructure_retry_count: int = 0


@dataclass(slots=True)
class ResumablePendingAwareAsyncJoblibBatchSession(
    PendingAwareBatchSession[
        EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]
    ],
    ResumableBatchSession[
        EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]
    ],
    Generic[CandidateT, JoblibEvaluationRecordT],
):
    """Pending-aware and resumable session for one async joblib logical batch.

    Parameters
    ----------
    evaluator : AsyncJoblibBatchSessionEvaluator[CandidateT, JoblibEvaluationRecordT]
        Evaluator instance that owns the logical batch.
    _handle : EvaluationBatchHandle
        Logical batch handle associated with the session.
    _completed_count : int, default=0
        Number of requests completed so far.
    _lifecycle : Literal["active", "completed", "failed", "cancelled", "suspended"], default="active"
        Current logical batch lifecycle.
    """

    evaluator: AsyncJoblibBatchSessionEvaluator[
        CandidateT,
        JoblibEvaluationRecordT,
    ]
    _handle: EvaluationBatchHandle
    _completed_count: int = 0
    _lifecycle: Literal[
        "active",
        "completed",
        "failed",
        "cancelled",
        "suspended",
    ] = "active"

    @property
    @override
    def handle(self) -> EvaluationBatchHandle:
        """Return the logical batch handle for this session.

        Returns
        -------
        EvaluationBatchHandle
            Handle associated with the current logical batch.
        """
        return self._handle

    @override
    def poll(
        self,
    ) -> tuple[
        CompletionGroup[EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]],
        ...,
    ]:
        """Poll newly completed outcomes for this logical batch.

        Returns
        -------
        tuple[CompletionGroup[EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]], ...]
            Newly completed outcome groups.

        Raises
        ------
        RuntimeError
            Raised when polling a session that is no longer active.
        BatchExecutionFailed
            Raised when the evaluator reports batch failure or cancellation.
        """
        if self._lifecycle != "active":
            msg = "batch session is no longer active"
            raise RuntimeError(msg)

        try:
            completion_groups = tuple(self.evaluator.poll(self.handle))
        except BatchExecutionFailed as exception:
            self._lifecycle = (
                "cancelled" if exception.kind == "cancelled" else "failed"
            )
            raise

        self._completed_count += sum(
            len(completion_group.outcomes)
            for completion_group in completion_groups
        )
        if self._completed_count >= self.handle.request_count:
            self._completed_count = self.handle.request_count
            self._lifecycle = "completed"
        return completion_groups

    @override
    def cancel(self) -> None:
        """Cancel the logical batch owned by this session.

        Notes
        -----
        Cancellation is idempotent once the session is already terminal.
        """
        if self._lifecycle == "suspended":
            self.evaluator.discard_suspended_batch(self.handle)
            self._lifecycle = "cancelled"
            return

        if self._lifecycle != "active":
            return
        self.evaluator.cancel(self.handle)
        self._lifecycle = "cancelled"

    @override
    def state(self) -> EvaluationBatchSessionState:
        """Return the current logical-batch session state.

        Returns
        -------
        EvaluationBatchSessionState
            Canonical lifecycle summary for the session.
        """
        return EvaluationBatchSessionState(
            request_count=self.handle.request_count,
            completed_count=self._completed_count,
            pending_count=self.handle.request_count - self._completed_count,
            lifecycle=self._lifecycle,
        )

    @override
    def suspend(self) -> EvaluationBatchResumeHandle:
        """Suspend the active logical batch and return a resume handle.

        Returns
        -------
        EvaluationBatchResumeHandle
            Resume handle for the suspended logical batch.

        Raises
        ------
        RuntimeError
            Raised when the session is not active.
        """
        if self._lifecycle != "active":
            msg = "only active batch sessions can be suspended"
            raise RuntimeError(msg)

        resume_handle = self.evaluator.suspend_batch(self.handle)
        self._completed_count = resume_handle.completed_count
        self._lifecycle = "suspended"
        return resume_handle
