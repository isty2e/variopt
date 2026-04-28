"""Execution resource and execution-model contracts."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


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
