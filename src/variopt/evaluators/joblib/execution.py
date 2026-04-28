"""Shared joblib evaluator execution helpers."""

from typing import Literal

from ...execution import ExecutionResources, NestedParallelismPolicy


def build_execution_resources(
    *,
    n_jobs: int,
    backend: Literal["loky", "threading"],
) -> ExecutionResources:
    """Build evaluator-owned execution resources for a joblib backend.

    Parameters
    ----------
    n_jobs : int
        Requested joblib worker count.
    backend : Literal["loky", "threading"]
        Joblib backend label.

    Returns
    -------
    ExecutionResources
        Execution-resource contract for the configured joblib backend.
    """
    return ExecutionResources(
        parallel_owner="evaluator",
        nested_parallelism_policy=NestedParallelismPolicy.FORBID,
        owner_worker_count=None if n_jobs < 0 else n_jobs,
        owner_backend=backend,
    )


def validate_joblib_configuration(
    *,
    n_jobs: int,
    backend: Literal["loky", "threading"],
) -> None:
    """Validate one joblib evaluator configuration.

    Parameters
    ----------
    n_jobs : int
        Requested joblib worker count.
    backend : Literal["loky", "threading"]
        Joblib backend label.

    Raises
    ------
    ValueError
        If ``n_jobs`` is zero or ``backend`` is unsupported.
    """
    if n_jobs == 0:
        msg = "n_jobs must not be zero"
        raise ValueError(msg)

    if backend not in {"loky", "threading"}:
        msg = "backend must be 'loky' or 'threading'"
        raise ValueError(msg)
