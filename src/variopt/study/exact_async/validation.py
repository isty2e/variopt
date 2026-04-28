"""Validation helpers for resumable exact-async study orchestration."""

from typing_extensions import TypeVar

from ...execution import EXACT_ASYNC_EXECUTION_MODEL
from ...kernel import DirectKernel
from ...typevars import CandidateT, RunMethodStateT
from ..common import StudyEvaluationRecordT
from ..validation import (
    require_resumable_async_evaluator,
    validate_execution_request,
)
from .contracts import StudyExactAsyncOwner

BoundaryT = TypeVar("BoundaryT")


def validate_resumable_exact_async_request(
    study: StudyExactAsyncOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ],
    *,
    state: RunMethodStateT,
    batch_size: int,
) -> None:
    """Validate one resumable exact-async step-session request.

    Parameters
    ----------
    study : StudyExactAsyncOwner[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
        Study-like owner providing the exact-async execution boundary.
    state : RunMethodStateT
        Current run-method state.
    batch_size : int
        Requested logical batch size.

    Raises
    ------
    RuntimeError
        If the run method is already exhausted.
    ValueError
        If the execution request is incompatible with exact-async resumable
        orchestration.
    """
    validate_execution_request(
        study,
        batch_size=batch_size,
        execution_model=EXACT_ASYNC_EXECUTION_MODEL,
    )
    _ = require_resumable_async_evaluator(study)

    if not isinstance(study.kernel, DirectKernel):
        msg = (
            "study-level resumable exact_async orchestration currently "
            "requires DirectKernel"
        )
        raise ValueError(msg)

    if study.run_method.is_exhausted(state):
        msg = "run_method is exhausted"
        raise RuntimeError(msg)
