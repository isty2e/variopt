"""Execution resource and execution-model contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, NoReturn, SupportsIndex


class EvaluationBudgetExhausted(RuntimeError):
    """Raised when an execution path would exceed its evaluation budget."""


@dataclass(slots=True)
class EvaluationBudget:
    """Mutable runtime ledger for a hard evaluation budget.

    Parameters
    ----------
    remaining : int
        Number of evaluation units still available.

    Notes
    -----
    Study orchestration consumes this ledger before submitting evaluator work.
    Kernels that compute outcomes without calling the evaluator may consume it
    directly, and the study layer reconciles any unmetered reported cost before
    assimilating outcomes.
    """

    remaining: int

    def __post_init__(self) -> None:
        """Validate the initial budget."""
        if self.remaining < 0:
            msg = "remaining evaluation budget must be non-negative"
            raise ValueError(msg)

    @property
    def is_exhausted(self) -> bool:
        """Return whether no evaluation units remain."""
        return self.remaining == 0

    def can_consume(self, count: int = 1) -> bool:
        """Return whether ``count`` evaluation units can be consumed.

        Parameters
        ----------
        count : int, default=1
            Requested evaluation-unit count.

        Returns
        -------
        bool
            ``True`` when the ledger has at least ``count`` units remaining.

        Raises
        ------
        ValueError
            If ``count`` is negative.
        """
        if count < 0:
            msg = "evaluation budget count must be non-negative"
            raise ValueError(msg)
        return count <= self.remaining

    def consume(self, count: int = 1) -> None:
        """Consume ``count`` evaluation units or raise.

        Parameters
        ----------
        count : int, default=1
            Evaluation-unit count to consume.

        Raises
        ------
        ValueError
            If ``count`` is negative.
        EvaluationBudgetExhausted
            If consuming ``count`` would exceed the budget.
        """
        if count < 0:
            msg = "evaluation budget count must be non-negative"
            raise ValueError(msg)
        if count > self.remaining:
            msg = "evaluation budget exhausted"
            raise EvaluationBudgetExhausted(msg)
        self.remaining -= count


@dataclass(slots=True, init=False)
class EvaluationReservationBatch:
    """One-shot coordinator reservation for a batch of evaluation limits.

    Parameters
    ----------
    budget : EvaluationBudget
        Coordinator-owned hard-budget ledger.
    limits : Sequence[int]
        Positive per-request evaluation limits reserved atomically from
        ``budget``.

    Notes
    -----
    A reservation is finalized exactly once by :meth:`settle` or
    :meth:`forfeit`. Invalid settlement data fails closed: the reservation
    becomes final and no capacity is refunded.
    """

    budget: EvaluationBudget
    limits: tuple[int, ...]
    _is_finalized: bool

    def __init__(
        self,
        *,
        budget: EvaluationBudget,
        limits: Sequence[int],
    ) -> None:
        """Reserve all requested evaluation limits atomically.

        Parameters
        ----------
        budget : EvaluationBudget
            Coordinator-owned hard-budget ledger.
        limits : Sequence[int]
            Non-empty sequence of positive per-request limits.

        Raises
        ------
        TypeError
            If a limit is not an exact integer.
        ValueError
            If ``limits`` is empty or contains a non-positive value.
        EvaluationBudgetExhausted
            If the total reservation exceeds the remaining budget.
        """
        normalized_limits = tuple(limits)
        if len(normalized_limits) == 0:
            msg = "evaluation reservation limits must not be empty"
            raise ValueError(msg)

        for limit in normalized_limits:
            if type(limit) is not int:
                msg = "evaluation reservation limits must be exact integers"
                raise TypeError(msg)
            if limit <= 0:
                msg = "evaluation reservation limits must be positive"
                raise ValueError(msg)

        budget.consume(sum(normalized_limits))
        self.budget = budget
        self.limits = normalized_limits
        self._is_finalized = False

    @property
    def reserved_evaluation_count(self) -> int:
        """Return the total evaluation capacity reserved by this batch.

        Returns
        -------
        int
            Sum of the per-request limits.
        """
        return sum(self.limits)

    @property
    def is_finalized(self) -> bool:
        """Return whether this reservation has been settled or forfeited.

        Returns
        -------
        bool
            ``True`` after the first settlement or forfeiture attempt.
        """
        return self._is_finalized

    def settle(self, consumed_counts: Sequence[int]) -> int:
        """Settle trusted per-request costs and refund unused capacity.

        Parameters
        ----------
        consumed_counts : Sequence[int]
            Non-negative exact integer costs aligned one-to-one with
            :attr:`limits`.

        Returns
        -------
        int
            Total trusted evaluation cost retained by the budget.

        Raises
        ------
        RuntimeError
            If this reservation is already finalized.
        TypeError
            If a consumed count is not an exact integer.
        ValueError
            If counts are misaligned, negative, or exceed their limits.

        Notes
        -----
        The reservation is finalized before validating ``consumed_counts``.
        Malformed output therefore forfeits all reserved capacity.
        """
        self._require_open()
        self._is_finalized = True

        normalized_counts = tuple(consumed_counts)
        if len(normalized_counts) != len(self.limits):
            msg = "consumed counts must align one-to-one with reservation limits"
            raise ValueError(msg)

        for consumed_count, limit in zip(
            normalized_counts,
            self.limits,
            strict=True,
        ):
            if type(consumed_count) is not int:
                msg = "consumed counts must be exact integers"
                raise TypeError(msg)
            if consumed_count < 0:
                msg = "consumed counts must be non-negative"
                raise ValueError(msg)
            if consumed_count > limit:
                msg = "consumed count must not exceed its reservation limit"
                raise ValueError(msg)

        total_consumed = sum(normalized_counts)
        self.budget.remaining += self.reserved_evaluation_count - total_consumed
        return total_consumed

    def forfeit(self) -> None:
        """Finalize this reservation without refunding unused capacity.

        Raises
        ------
        RuntimeError
            If this reservation is already finalized.
        """
        self._require_open()
        self._is_finalized = True

    def _require_open(self) -> None:
        if self._is_finalized:
            msg = "evaluation reservation is already finalized"
            raise RuntimeError(msg)

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        """Reject serialization of coordinator-owned mutable budget state."""
        del protocol
        msg = "EvaluationReservationBatch is coordinator-local and cannot be pickled"
        raise TypeError(msg)


class ExecutionCompletionMode(Enum):
    """Completion-order axis for an execution model.

    Attributes
    ----------
    IMMEDIATE : str
        Work becomes visible immediately after evaluation.
    ORDERED_BATCH : str
        Work becomes visible only after a full ordered batch completes.
    ORDERED_ASYNC : str
        Work completes asynchronously but still commits in logical order.
    """

    IMMEDIATE = "immediate"
    ORDERED_BATCH = "ordered_batch"
    ORDERED_ASYNC = "ordered_async"


class ExecutionAssimilationMode(Enum):
    """Feedback-assimilation axis for an execution model.

    Attributes
    ----------
    INCREMENTAL : str
        Assimilate each completed item immediately.
    BATCH_COMMIT : str
        Assimilate only after the full ordered batch is complete.
    STALE_INCREMENTAL : str
        Assimilate incrementally against a stale proposal frontier.
    """

    INCREMENTAL = "incremental"
    BATCH_COMMIT = "batch_commit"
    STALE_INCREMENTAL = "stale_incremental"


@dataclass(frozen=True, slots=True)
class ExecutionModel:
    """Execution model for study ask/evaluate/tell transitions.

    Parameters
    ----------
    completion_mode : ExecutionCompletionMode
        Ordering law that determines when evaluated work becomes visible.
    assimilation_mode : ExecutionAssimilationMode
        State-assimilation law used when completed work is committed back into
        the run method.
    """

    completion_mode: ExecutionCompletionMode
    assimilation_mode: ExecutionAssimilationMode

    def __post_init__(self) -> None:
        """Validate execution-model combinations.

        Raises
        ------
        ValueError
            If ``completion_mode`` and ``assimilation_mode`` describe an
            unsupported combination.
        RuntimeError
            If an unknown completion mode reaches this validation path.
        """
        if self.completion_mode is ExecutionCompletionMode.IMMEDIATE:
            if self.assimilation_mode is not ExecutionAssimilationMode.INCREMENTAL:
                msg = "immediate completion requires incremental assimilation"
                raise ValueError(msg)
            return

        if self.completion_mode is ExecutionCompletionMode.ORDERED_BATCH:
            if self.assimilation_mode is not ExecutionAssimilationMode.BATCH_COMMIT:
                msg = "ordered_batch completion requires batch_commit assimilation"
                raise ValueError(msg)
            return

        if self.completion_mode is ExecutionCompletionMode.ORDERED_ASYNC:
            if self.assimilation_mode in {
                ExecutionAssimilationMode.BATCH_COMMIT,
                ExecutionAssimilationMode.STALE_INCREMENTAL,
            }:
                return
            msg = (
                "ordered_async completion requires batch_commit or "
                "stale_incremental assimilation"
            )
            raise ValueError(msg)

        msg = "unknown execution completion mode"
        raise RuntimeError(msg)

    @property
    def name(self) -> str:
        """Return a stable human-readable name for the execution model.

        Returns
        -------
        str
            Stable label used in diagnostics and documentation.
        """
        if self == SEQUENTIAL_EXECUTION_MODEL:
            return "sequential"
        if self == SYNC_BATCH_EXECUTION_MODEL:
            return "sync_batch"
        if self == EXACT_ASYNC_EXECUTION_MODEL:
            return "exact_async"
        if self == STALE_ASYNC_EXECUTION_MODEL:
            return "stale_async"
        return f"{self.completion_mode.value}+{self.assimilation_mode.value}"


SEQUENTIAL_EXECUTION_MODEL = ExecutionModel(
    completion_mode=ExecutionCompletionMode.IMMEDIATE,
    assimilation_mode=ExecutionAssimilationMode.INCREMENTAL,
)
SYNC_BATCH_EXECUTION_MODEL = ExecutionModel(
    completion_mode=ExecutionCompletionMode.ORDERED_BATCH,
    assimilation_mode=ExecutionAssimilationMode.BATCH_COMMIT,
)
EXACT_ASYNC_EXECUTION_MODEL = ExecutionModel(
    completion_mode=ExecutionCompletionMode.ORDERED_ASYNC,
    assimilation_mode=ExecutionAssimilationMode.BATCH_COMMIT,
)
STALE_ASYNC_EXECUTION_MODEL = ExecutionModel(
    completion_mode=ExecutionCompletionMode.ORDERED_ASYNC,
    assimilation_mode=ExecutionAssimilationMode.STALE_INCREMENTAL,
)


class NestedParallelismPolicy(Enum):
    """Policy for nested parallel execution below the current owner.

    Attributes
    ----------
    FORBID : str
        Nested parallel work below the current owner is not allowed.
    ALLOW : str
        Nested parallel work below the current owner is allowed.
    """

    FORBID = "forbid"
    ALLOW = "allow"


@dataclass(frozen=True, slots=True)
class ExecutionResources:
    """Execution-side resource contract for one request-local run.

    Parameters
    ----------
    parallel_owner : Literal["evaluator", "kernel"]
        Component that owns the active worker pool for the current execution
        boundary.
    nested_parallelism_policy : NestedParallelismPolicy
        Policy controlling whether nested parallel work is allowed below the
        current owner.
    owner_worker_count : int | None, optional
        Optional worker count available to the current owner.
    owner_backend : str | None, optional
        Optional backend label for diagnostics and backend-specific branching.
    """

    parallel_owner: Literal["evaluator", "kernel"]
    nested_parallelism_policy: NestedParallelismPolicy
    owner_worker_count: int | None = None
    owner_backend: str | None = None

    def __post_init__(self) -> None:
        """Validate execution-resource metadata.

        Raises
        ------
        ValueError
            If ``owner_worker_count`` is non-positive or ``owner_backend`` is an
            empty string.
        """
        if self.owner_worker_count is not None and self.owner_worker_count <= 0:
            msg = "owner_worker_count must be positive when provided"
            raise ValueError(msg)

        if self.owner_backend == "":
            msg = "owner_backend must not be empty"
            raise ValueError(msg)
