"""Shared validation helpers for study orchestration."""

from typing import Protocol

from ..evaluators.async_evaluator.contracts import (
    AsyncEvaluator,
    ResumableAsyncEvaluator,
)
from ..execution import (
    SEQUENTIAL_EXECUTION_MODEL,
    ExecutionCompletionMode,
    ExecutionModel,
)


class AsyncEvaluatorValidationOwner(Protocol):
    """Study-like owner for async evaluator capability validation.

    Notes
    -----
    Async capability checks only need the evaluator object. Execution-model
    validation has a separate owner protocol for run-method capability checks.
    """

    @property
    def evaluator(self) -> object:
        """Return the configured evaluator."""
        ...


class ExecutionModelRunMethod(Protocol):
    """Run-method capability needed for execution model validation."""

    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        """Return execution models supported by this run method."""
        ...


class ExecutionValidationOwner(Protocol):
    """Study-like owner for execution model validation only."""

    @property
    def run_method(self) -> ExecutionModelRunMethod:
        """Return a run-method capability view."""
        ...

    @property
    def evaluator(self) -> object:
        """Return the configured evaluator for runtime capability checks."""
        ...


def validate_execution_request(
    study: ExecutionValidationOwner,
    *,
    batch_size: int,
    execution_model: ExecutionModel,
) -> None:
    """Reject invalid execution-model requests for one study-like owner.

    Parameters
    ----------
    study : ExecutionValidationOwner
        Study-like owner exposing execution-model and evaluator capabilities.
    batch_size : int
        Requested batch size.
    execution_model : ExecutionModel
        Requested execution model.

    Raises
    ------
    ValueError
        If the batch size or execution model is incompatible with the owner.
    """
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    if execution_model == SEQUENTIAL_EXECUTION_MODEL and batch_size != 1:
        msg = "sequential execution model requires batch_size == 1"
        raise ValueError(msg)

    supported_models = study.run_method.supported_execution_models()
    if execution_model not in supported_models:
        msg = (
            "run_method does not support the requested execution model: "
            f"{execution_model.name}"
        )
        raise ValueError(msg)

    if (
        execution_model.completion_mode is ExecutionCompletionMode.ORDERED_ASYNC
        and not isinstance(study.evaluator, AsyncEvaluator)
    ):
        msg = "ordered_async execution models require an AsyncEvaluator"
        raise ValueError(msg)


def require_async_evaluator(
    study: AsyncEvaluatorValidationOwner,
) -> None:
    """Require the configured evaluator to expose async capability.

    Parameters
    ----------
    study : AsyncEvaluatorValidationOwner
        Study-like owner exposing the evaluator.

    Raises
    ------
    TypeError
        If the evaluator is not async-capable.
    """
    if not isinstance(study.evaluator, AsyncEvaluator):
        msg = "ordered_async execution models require an AsyncEvaluator"
        raise TypeError(msg)


def require_resumable_async_evaluator(
    study: AsyncEvaluatorValidationOwner,
) -> None:
    """Require the configured evaluator to expose resumable async capability.

    Parameters
    ----------
    study : AsyncEvaluatorValidationOwner
        Study-like owner exposing the evaluator.

    Raises
    ------
    TypeError
        If the evaluator does not support resumable async sessions.
    """
    require_async_evaluator(study)
    if not isinstance(study.evaluator, ResumableAsyncEvaluator):
        msg = (
            "study-level resumable exact_async orchestration requires a "
            "ResumableAsyncEvaluator"
        )
        raise TypeError(msg)
