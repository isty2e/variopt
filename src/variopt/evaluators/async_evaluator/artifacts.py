"""Value artifacts for exact-async evaluator batch lifecycle."""

from dataclasses import dataclass
from typing import Generic, Literal

from typing_extensions import TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

EvaluationT = TypeVar("EvaluationT")


@dataclass(frozen=True, slots=True)
class EvaluationBatchHandle:
    """Immutable identity for one submitted logical evaluation batch.

    Parameters
    ----------
    batch_id : str
        Evaluator-owned stable batch identifier.
    request_count : int
        Number of logical requests submitted in the batch.
    """

    batch_id: str
    request_count: int

    def __post_init__(self) -> None:
        """Validate batch-handle payloads.

        Raises
        ------
        ValueError
            If ``batch_id`` is empty or ``request_count`` is not positive.
        """
        if self.batch_id == "":
            msg = "batch_id must not be empty"
            raise ValueError(msg)

        if self.request_count <= 0:
            msg = "request_count must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CompletionGroup(FrozenGenericSlotsCompat, Generic[EvaluationT]):
    """Ordered completion slice for one logical batch.

    Parameters
    ----------
    start_index : int
        Inclusive start index in the original batch order.
    outcomes : tuple[EvaluationT, ...]
        Contiguous completed outcomes beginning at ``start_index``.
    """

    start_index: int
    outcomes: tuple[EvaluationT, ...]

    def __post_init__(self) -> None:
        """Validate completion-group payloads.

        Raises
        ------
        ValueError
            If ``start_index`` is negative or ``outcomes`` is empty.
        """
        if self.start_index < 0:
            msg = "start_index must be non-negative"
            raise ValueError(msg)

        if len(self.outcomes) == 0:
            msg = "completion groups must contain at least one outcome"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class BatchExecutionFailed(RuntimeError):
    """Terminal async-batch failure.

    Parameters
    ----------
    handle : EvaluationBatchHandle
        Batch that failed.
    kind : {"user_code", "infrastructure", "cancelled"}
        High-level failure category.
    cause : BaseException
        Original exception that triggered the failure.
    """

    handle: EvaluationBatchHandle
    kind: Literal["user_code", "infrastructure", "cancelled"]
    cause: BaseException

    def __post_init__(self) -> None:
        """Initialize the runtime error message."""
        RuntimeError.__init__(
            self,
            f"{self.kind} failure while executing batch {self.handle.batch_id}",
        )


@dataclass(frozen=True, slots=True)
class EvaluationBatchSessionState:
    """Pending-aware lifecycle summary for a submitted batch session.

    Parameters
    ----------
    request_count : int
        Total number of requests in the logical batch.
    completed_count : int
        Number of requests whose outcomes are already committed to the session.
    pending_count : int
        Number of requests still pending.
    lifecycle : {"active", "completed", "failed", "cancelled", "suspended"}
        Current lifecycle label for the session.
    """

    request_count: int
    completed_count: int
    pending_count: int
    lifecycle: Literal["active", "completed", "failed", "cancelled", "suspended"]

    def __post_init__(self) -> None:
        """Validate session-state payloads.

        Raises
        ------
        ValueError
            If counts are negative, ``request_count`` is not positive, or the
            completed and pending counts do not partition ``request_count``.
        """
        if self.request_count <= 0:
            msg = "request_count must be positive"
            raise ValueError(msg)

        if self.completed_count < 0:
            msg = "completed_count must be non-negative"
            raise ValueError(msg)

        if self.pending_count < 0:
            msg = "pending_count must be non-negative"
            raise ValueError(msg)

        if self.completed_count + self.pending_count != self.request_count:
            msg = "completed_count and pending_count must partition request_count"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class EvaluationBatchResumeHandle:
    """Immutable identity for resuming a suspended batch session.

    Parameters
    ----------
    batch_id : str
        Evaluator-owned stable batch identifier.
    request_count : int
        Total number of requests in the original batch.
    completed_count : int
        Number of requests already completed before suspension.
    """

    batch_id: str
    request_count: int
    completed_count: int

    def __post_init__(self) -> None:
        """Validate resume-handle payloads.

        Raises
        ------
        ValueError
            If the identifier is empty, counts are invalid, or
            ``completed_count`` exceeds ``request_count``.
        """
        if self.batch_id == "":
            msg = "batch_id must not be empty"
            raise ValueError(msg)

        if self.request_count <= 0:
            msg = "request_count must be positive"
            raise ValueError(msg)

        if self.completed_count < 0:
            msg = "completed_count must be non-negative"
            raise ValueError(msg)

        if self.completed_count > self.request_count:
            msg = "completed_count must not exceed request_count"
            raise ValueError(msg)


def store_completion_group(
    ordered_outcomes: list[EvaluationT | None],
    completion_group: CompletionGroup[EvaluationT],
    *,
    request_count: int,
) -> int:
    """Store a completion group into an ordered outcome buffer.

    Parameters
    ----------
    ordered_outcomes : list[EvaluationT | None]
        Mutable outcome buffer aligned to original request order.
    completion_group : CompletionGroup[EvaluationT]
        Newly completed contiguous slice.
    request_count : int
        Total number of requests in the logical batch.

    Returns
    -------
    int
        Number of newly covered outcomes written into ``ordered_outcomes``.

    Raises
    ------
    ValueError
        If the completion group would exceed batch bounds or overlap an
        existing completed region.
    """
    end_index = completion_group.start_index + len(completion_group.outcomes)
    if end_index > request_count:
        msg = "completion group exceeds logical batch bounds"
        raise ValueError(msg)

    for offset, outcome in enumerate(completion_group.outcomes):
        target_index = completion_group.start_index + offset
        if ordered_outcomes[target_index] is not None:
            msg = "completion groups must not overlap"
            raise ValueError(msg)

        ordered_outcomes[target_index] = outcome

    return len(completion_group.outcomes)
