"""Evaluator contracts and concrete backend families."""

from typing import TYPE_CHECKING

from .async_evaluator import (
    AsyncEvaluator,
    BatchExecutionFailed,
    CompletionGroup,
    EvaluationBatchHandle,
    EvaluationBatchResumeHandle,
    EvaluationBatchSession,
    EvaluationBatchSessionState,
    PendingAwareBatchSession,
    ResumableAsyncEvaluator,
    ResumableBatchSession,
)
from .base import Evaluator
from .joblib import AsyncJoblibEvaluator, JoblibEvaluator
from .sequential import SequentialEvaluator

if TYPE_CHECKING:
    from .mpi import MpiEvaluator, MpiExecutorFactory

__all__ = [
    "AsyncEvaluator",
    "AsyncJoblibEvaluator",
    "BatchExecutionFailed",
    "CompletionGroup",
    "EvaluationBatchHandle",
    "EvaluationBatchResumeHandle",
    "EvaluationBatchSession",
    "EvaluationBatchSessionState",
    "Evaluator",
    "JoblibEvaluator",
    "MpiEvaluator",
    "MpiExecutorFactory",
    "PendingAwareBatchSession",
    "ResumableAsyncEvaluator",
    "ResumableBatchSession",
    "SequentialEvaluator",
]


def __getattr__(name: str) -> object:
    """Lazily expose optional evaluator backends from the root facade."""
    if name == "MpiEvaluator":
        from .mpi import MpiEvaluator

        return MpiEvaluator

    if name == "MpiExecutorFactory":
        from .mpi import MpiExecutorFactory

        return MpiExecutorFactory

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
