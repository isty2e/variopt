"""Session capability hierarchy for exact-async evaluator batches."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import isfinite
from numbers import Real
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


def normalize_wait_timeout(timeout: object | None) -> float | None:
    """Return one canonical async wait timeout or reject invalid values.

    Parameters
    ----------
    timeout : object | None
        Raw timeout value supplied to an async wait boundary.

    Returns
    -------
    float | None
        ``None`` for an unbounded wait or a finite non-negative timeout in
        seconds.

    Raises
    ------
    TypeError
        If ``timeout`` is not a real numeric value.
    ValueError
        If ``timeout`` is non-finite or negative.
    """
    if timeout is None:
        return None

    if type(timeout) is bool or not isinstance(timeout, Real):
        msg = "timeout must be a real number"
        raise TypeError(msg)

    normalized_timeout = float(timeout)
    if not isfinite(normalized_timeout):
        msg = "timeout must be finite"
        raise ValueError(msg)

    if normalized_timeout < 0.0:
        msg = "timeout must be non-negative"
        raise ValueError(msg)

    return normalized_timeout


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
        TypeError
            If ``timeout`` is not a real numeric value.
        ValueError
            If ``timeout`` is non-finite or negative.
        """
        normalized_timeout = normalize_wait_timeout(timeout)

        deadline = (
            None if normalized_timeout is None else monotonic() + normalized_timeout
        )
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
    Resumable sessions allow study-level exact-async orchestration to detach
    and reattach to a logical batch when the evaluator keeps the suspended work
    alive. The returned handle is evaluator-owned runtime state, not a generic
    durable checkpoint; persistence and process-restart guarantees are
    evaluator-specific.
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
