"""Session capability hierarchy for exact-async evaluator batches."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from time import monotonic, sleep
from typing import Generic

from typing_extensions import TypeVar

from .artifacts import (
    CompletionGroup,
    EvaluationBatchHandle,
    EvaluationBatchResumeHandle,
    EvaluationBatchSessionState,
)

EvaluationT = TypeVar("EvaluationT", covariant=True)
_DEFAULT_WAIT_POLL_INTERVAL_SECONDS = 0.001


class EvaluationBatchSession(ABC, Generic[EvaluationT]):
    """Lifecycle contract for one submitted logical evaluation batch.

    Notes
    -----
    Concrete evaluators provide session objects that satisfy this contract so
    orchestration code can poll, cancel, and optionally suspend batches without
    depending on backend-specific details.
    """

    @property
    @abstractmethod
    def handle(self) -> EvaluationBatchHandle:
        """Return the immutable identity of the submitted batch.

        Returns
        -------
        EvaluationBatchHandle
            Stable handle for this session.
        """

    @abstractmethod
    def poll(self) -> Sequence[CompletionGroup[EvaluationT]]:
        """Return immediately with newly completed ordered groups.

        Returns
        -------
        Sequence[CompletionGroup[EvaluationT]]
            Newly available completion groups since the previous poll. An
            empty sequence means no completion is currently available; callers
            that want blocking behavior should use :meth:`wait`.
        """

    def wait(
        self,
        *,
        timeout: float | None = None,
    ) -> Sequence[CompletionGroup[EvaluationT]]:
        """Block until at least one completion group is available.

        Parameters
        ----------
        timeout : float | None, default=None
            Maximum number of seconds to wait. ``None`` waits indefinitely.

        Returns
        -------
        Sequence[CompletionGroup[EvaluationT]]
            Newly available completion groups, or an empty sequence when
            ``timeout`` expires before any completion is available.

        Raises
        ------
        ValueError
            If ``timeout`` is negative.
        """
        if timeout is not None and timeout < 0.0:
            msg = "timeout must be non-negative"
            raise ValueError(msg)

        deadline = None if timeout is None else monotonic() + timeout
        while True:
            completion_groups = self.poll()
            if len(completion_groups) > 0:
                return completion_groups

            if deadline is None:
                sleep(_DEFAULT_WAIT_POLL_INTERVAL_SECONDS)
                continue

            remaining_seconds = deadline - monotonic()
            if remaining_seconds <= 0.0:
                return ()
            sleep(min(_DEFAULT_WAIT_POLL_INTERVAL_SECONDS, remaining_seconds))

    @abstractmethod
    def cancel(self) -> None:
        """Best-effort cancellation for the session."""


class PendingAwareBatchSession(
    EvaluationBatchSession[EvaluationT],
    ABC,
    Generic[EvaluationT],
):
    """Optional batch-session capability that exposes pending-aware state.

    Notes
    -----
    This capability is used by exact-async orchestration that needs to inspect
    whether work is still pending without cancelling the batch.
    """

    @abstractmethod
    def state(self) -> EvaluationBatchSessionState:
        """Return the current pending-aware lifecycle state.

        Returns
        -------
        EvaluationBatchSessionState
            Pending-aware state summary for the session.
        """


class ResumableBatchSession(
    EvaluationBatchSession[EvaluationT],
    ABC,
    Generic[EvaluationT],
):
    """Optional batch-session capability that can suspend and later resume.

    Notes
    -----
    Resumable sessions allow study-level exact-async orchestration to survive
    process or control-flow boundaries by persisting evaluator-owned resume
    handles.
    """

    @abstractmethod
    def suspend(self) -> EvaluationBatchResumeHandle:
        """Suspend the session and return a resume handle.

        Returns
        -------
        EvaluationBatchResumeHandle
            Evaluator-owned handle that can later be used to resume the
            session.
        """
