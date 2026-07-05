"""Async batch state for joblib-backed evaluators."""

from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from queue import Queue
from threading import Event, Thread
from typing import Generic, Literal, Protocol, TypeVar

from typing_extensions import override

from ...artifacts import EvaluationRequest
from ...execution import ExecutionResources
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
from .contracts import BoundaryT, JoblibEvaluationPayloadT

SessionEvaluationT = TypeVar("SessionEvaluationT", covariant=True)


class AsyncJoblibBatchSessionEvaluator(Protocol[SessionEvaluationT]):
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
    ) -> tuple[CompletionGroup[SessionEvaluationT], ...]:
        """Poll one submitted batch handle without blocking.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Logical batch handle to poll.

        Returns
        -------
        tuple[CompletionGroup[SessionEvaluationT], ...]
            Newly completed groups, or an empty tuple when none are currently
            available.
        """
        ...

    def wait(
        self,
        handle: EvaluationBatchHandle,
        *,
        timeout: float | None = None,
    ) -> tuple[CompletionGroup[SessionEvaluationT], ...]:
        """Wait for at least one submitted-batch completion group.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Logical batch handle to wait on.
        timeout : float | None, default=None
            Maximum number of seconds to wait. ``None`` waits indefinitely.

        Returns
        -------
        tuple[CompletionGroup[SessionEvaluationT], ...]
            Newly completed groups, or an empty tuple when ``timeout`` expires
            before a completion is available.
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


@dataclass(frozen=True, slots=True)
class AsyncJoblibCompletedResult(Generic[SessionEvaluationT]):
    """One completed result emitted by a joblib result-drain worker.

    Parameters
    ----------
    index : int
        Logical request index completed by the joblib attempt.
    outcome : SessionEvaluationT
        Completed evaluation slot for ``index``.
    attempt_generation : int, default=0
        Active-attempt generation that produced this event. Retry replacement
        increments the generation so stale events from aborted attempts are not
        accepted as current completions.
    """

    index: int
    outcome: SessionEvaluationT
    attempt_generation: int = 0


@dataclass(frozen=True, slots=True)
class AsyncJoblibFailedResult:
    """Failure emitted by a joblib result-drain worker.

    Parameters
    ----------
    exception : BaseException
        Exception raised while draining the joblib result stream.
    attempt_generation : int, default=0
        Active-attempt generation that produced this event.
    """

    exception: BaseException
    attempt_generation: int = 0


@dataclass(frozen=True, slots=True)
class AsyncJoblibExhaustedResult:
    """Marker emitted when a joblib result stream is exhausted."""

    attempt_generation: int = 0


@dataclass(slots=True)
class ActiveAsyncJoblibBatch(
    Generic[BoundaryT, CandidateT, SessionEvaluationT, JoblibEvaluationPayloadT]
):
    """In-flight async joblib batch state.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
        Problem definition used to evaluate the batch.
    request_inputs : tuple[AsyncJoblibRequestInput[CandidateT], ...]
        Indexed requests that belong to the active batch.
    execution_resources : ExecutionResources
        Execution resources reserved for the batch.
    result_generator : Generator[tuple[int, SessionEvaluationT], None, None]
        Joblib-backed generator that yields indexed evaluation slots.
    result_queue : Queue[AsyncJoblibCompletedResult[SessionEvaluationT] | AsyncJoblibFailedResult | AsyncJoblibExhaustedResult]
        Non-blocking handoff queue populated by the drain worker.
    result_worker : Thread
        Daemon worker that drains the blocking joblib result stream.
    abort_attempt : Callable[[], None] | None, optional
        Best-effort abort hook for the underlying joblib attempt.
    abort_event : Event, optional
        Cooperative stop signal observed by the drain worker when bounded
        handoff backpressure would otherwise block producer shutdown.
    attempt_generation : int, default=0
        Monotonic active-attempt generation used to discard stale events after
        retry replacement.
    completed_indices : set[int], optional
        Request indices already completed inside the active batch.
    infrastructure_retry_count : int, default=0
        Number of infrastructure retries already consumed.
    """

    problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
    request_inputs: tuple[AsyncJoblibRequestInput[CandidateT], ...]
    execution_resources: ExecutionResources
    result_generator: Generator[tuple[int, SessionEvaluationT], None, None]
    result_queue: Queue[
        AsyncJoblibCompletedResult[SessionEvaluationT]
        | AsyncJoblibFailedResult
        | AsyncJoblibExhaustedResult
    ]
    result_worker: Thread
    abort_attempt: Callable[[], None] | None = None
    abort_event: Event = field(default_factory=Event, repr=False)
    attempt_generation: int = 0
    completed_indices: set[int] = field(default_factory=set)
    infrastructure_retry_count: int = 0


@dataclass(slots=True)
class SuspendedAsyncJoblibBatch(
    Generic[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
):
    """Suspended async joblib batch state kept by one evaluator instance.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
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

    problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
    request_inputs: tuple[AsyncJoblibRequestInput[CandidateT], ...]
    execution_resources: ExecutionResources
    completed_indices: set[int] = field(default_factory=set)
    infrastructure_retry_count: int = 0


@dataclass(slots=True)
class ResumablePendingAwareAsyncJoblibBatchSession(
    PendingAwareBatchSession[SessionEvaluationT],
    ResumableBatchSession[SessionEvaluationT],
    Generic[SessionEvaluationT],
):
    """Pending-aware and resumable session for one async joblib logical batch.

    Parameters
    ----------
    evaluator : AsyncJoblibBatchSessionEvaluator[SessionEvaluationT]
        Evaluator instance that owns the logical batch.
    _handle : EvaluationBatchHandle
        Logical batch handle associated with the session.
    _completed_count : int, default=0
        Number of requests completed so far.
    _lifecycle : Literal["active", "completed", "failed", "cancelled", "suspended"], default="active"
        Current logical batch lifecycle.
    """

    evaluator: AsyncJoblibBatchSessionEvaluator[SessionEvaluationT]
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
    ) -> tuple[CompletionGroup[SessionEvaluationT], ...]:
        """Poll newly completed evaluation slots for this logical batch.

        Returns
        -------
        tuple[CompletionGroup[SessionEvaluationT], ...]
            Newly completed groups.

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
            self._lifecycle = "cancelled" if exception.kind == "cancelled" else "failed"
            raise

        self._record_completion_groups(completion_groups)
        return completion_groups

    @override
    def wait(
        self,
        *,
        timeout: float | None = None,
    ) -> tuple[CompletionGroup[SessionEvaluationT], ...]:
        """Wait for newly completed evaluation slots for this logical batch.

        Parameters
        ----------
        timeout : float | None, default=None
            Maximum number of seconds to wait. ``None`` waits indefinitely.

        Returns
        -------
        tuple[CompletionGroup[SessionEvaluationT], ...]
            Newly completed groups, or an empty tuple when ``timeout`` expires
            before a completion is available.

        Raises
        ------
        RuntimeError
            Raised when waiting on a session that is no longer active.
        BatchExecutionFailed
            Raised when the evaluator reports batch failure or cancellation.
        """
        if self._lifecycle != "active":
            msg = "batch session is no longer active"
            raise RuntimeError(msg)

        try:
            completion_groups = tuple(
                self.evaluator.wait(self.handle, timeout=timeout),
            )
        except BatchExecutionFailed as exception:
            self._lifecycle = "cancelled" if exception.kind == "cancelled" else "failed"
            raise

        self._record_completion_groups(completion_groups)
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

    def _record_completion_groups(
        self,
        completion_groups: tuple[CompletionGroup[SessionEvaluationT], ...],
    ) -> None:
        """Update lifecycle state after newly observed completion groups."""
        self._completed_count += sum(
            len(completion_group.outcomes) for completion_group in completion_groups
        )
        if self._completed_count >= self.handle.request_count:
            self._completed_count = self.handle.request_count
            self._lifecycle = "completed"

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

        try:
            resume_handle = self.evaluator.suspend_batch(self.handle)
        except BatchExecutionFailed as exception:
            self._lifecycle = "cancelled" if exception.kind == "cancelled" else "failed"
            raise
        self._completed_count = resume_handle.completed_count
        self._lifecycle = "suspended"
        return resume_handle
