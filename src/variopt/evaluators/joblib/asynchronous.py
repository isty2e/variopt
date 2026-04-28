"""Asynchronous joblib-backed evaluator and retry runtime."""

from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import Generic, Literal, cast

import joblib  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import override

from ...artifacts import EvaluationRequest
from ...evaluation_pipeline import evaluate_request_outcome
from ...execution import ExecutionResources
from ...outcomes import EvaluationOutcome
from ...problem import Problem
from ...typevars import CandidateT
from ..async_evaluator.artifacts import (
    BatchExecutionFailed,
    CompletionGroup,
    EvaluationBatchHandle,
    EvaluationBatchResumeHandle,
)
from ..async_evaluator.contracts import ResumableAsyncEvaluator
from ..async_evaluator.sessions import EvaluationBatchSession
from .batches import (
    ActiveAsyncJoblibBatch,
    AsyncJoblibRequestInput,
    ResumablePendingAwareAsyncJoblibBatchSession,
    SuspendedAsyncJoblibBatch,
)
from .contracts import (
    BoundaryT,
    JoblibDelayedFactory,
    JoblibEvaluationRecordT,
    JoblibGeneratorParallelFactory,
)
from .execution import build_execution_resources, validate_joblib_configuration


def _evaluate_indexed_request_outcome(
    *,
    index: int,
    problem: Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT],
    request: EvaluationRequest[CandidateT],
) -> tuple[int, EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]]:
    """Execute one request and carry its original logical index."""
    return (
        index,
        evaluate_request_outcome(
            problem=problem,
            request=request,
        ),
    )


def _classify_joblib_failure(
    exception: BaseException,
) -> Literal["user_code", "infrastructure", "cancelled"]:
    """Classify one joblib failure for async batch reporting."""
    if isinstance(exception, KeyboardInterrupt):
        return "cancelled"

    if isinstance(exception, TimeoutError):
        return "infrastructure"

    if type(exception).__name__ in {
        "BrokenProcessPool",
        "TerminatedWorkerError",
    }:
        return "infrastructure"

    return "user_code"


def _remaining_async_joblib_request_inputs(
    request_inputs: Sequence[AsyncJoblibRequestInput[CandidateT]],
    completed_indices: set[int],
) -> tuple[AsyncJoblibRequestInput[CandidateT], ...]:
    """Return unfinished request inputs for one logical async batch state."""
    return tuple(
        request_input
        for request_input in request_inputs
        if request_input.index not in completed_indices
    )


@dataclass(slots=True)
class AsyncJoblibEvaluator(
    ResumableAsyncEvaluator[
        Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, JoblibEvaluationRecordT],
    ],
    Generic[BoundaryT, CandidateT, JoblibEvaluationRecordT],
):
    """Joblib-backed async evaluator with ordered completion groups.

    Parameters
    ----------
    n_jobs : int, default=-1
        Joblib worker count. ``-1`` delegates to joblib's all-available-worker
        behavior.
    backend : {"loky", "threading"}, default="loky"
        Joblib backend used for request execution.
    infrastructure_retry_limit : int, default=0
        Number of times to retry unfinished work after infrastructure failures.
    """

    n_jobs: int = -1
    backend: Literal["loky", "threading"] = "loky"
    infrastructure_retry_limit: int = 0
    _batch_counter: Iterator[int] = field(init=False, repr=False)
    _active_batches: dict[
        str,
        ActiveAsyncJoblibBatch[BoundaryT, CandidateT, JoblibEvaluationRecordT],
    ] = field(
        init=False,
        repr=False,
    )
    _suspended_batches: dict[
        str,
        SuspendedAsyncJoblibBatch[BoundaryT, CandidateT, JoblibEvaluationRecordT],
    ] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate async joblib evaluator configuration.

        Raises
        ------
        ValueError
            If ``infrastructure_retry_limit`` is negative or the joblib
            configuration is invalid.
        """
        validate_joblib_configuration(
            n_jobs=self.n_jobs,
            backend=self.backend,
        )
        if self.infrastructure_retry_limit < 0:
            msg = "infrastructure_retry_limit must be non-negative"
            raise ValueError(msg)
        self._batch_counter = count()
        self._active_batches = {}
        self._suspended_batches = {}

    @override
    def execution_resources(self) -> ExecutionResources:
        """Return execution resources for async joblib evaluation.

        Returns
        -------
        ExecutionResources
            Resource contract describing evaluator-owned joblib parallelism.
        """
        return build_execution_resources(
            n_jobs=self.n_jobs,
            backend=self.backend,
        )

    @override
    def open_session(
        self,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationBatchSession[EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]]:
        """Open a resumable pending-aware joblib session.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT]
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequest[CandidateT]]
            Request batch to execute asynchronously.

        Returns
        -------
        EvaluationBatchSession[EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]]
            Session backed by joblib's unordered completion generator.
        """
        return ResumablePendingAwareAsyncJoblibBatchSession(
            evaluator=self,
            _handle=self.submit_batch(problem, requests),
        )

    @override
    def resume_session(
        self,
        handle: EvaluationBatchResumeHandle,
    ) -> EvaluationBatchSession[EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]]:
        """Resume a suspended async joblib session.

        Parameters
        ----------
        handle : EvaluationBatchResumeHandle
            Resume handle produced when a batch session was suspended.

        Returns
        -------
        EvaluationBatchSession[EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]]
            Resumed batch session.
        """
        suspended_batch = self._suspended_batches.pop(handle.batch_id, None)
        if suspended_batch is None:
            msg = f"unknown suspended batch handle: {handle.batch_id}"
            raise ValueError(msg)

        if handle.request_count != len(suspended_batch.request_inputs):
            msg = "resume handle request_count does not match suspended batch"
            raise ValueError(msg)

        if handle.completed_count != len(suspended_batch.completed_indices):
            msg = "resume handle completed_count does not match suspended batch"
            raise ValueError(msg)

        remaining_inputs = _remaining_async_joblib_request_inputs(
            suspended_batch.request_inputs,
            suspended_batch.completed_indices,
        )
        result_generator = self._start_attempt(
            problem=suspended_batch.problem,
            request_inputs=remaining_inputs,
            execution_resources=suspended_batch.execution_resources,
        )
        self._active_batches[handle.batch_id] = ActiveAsyncJoblibBatch(
            problem=suspended_batch.problem,
            request_inputs=suspended_batch.request_inputs,
            execution_resources=suspended_batch.execution_resources,
            result_generator=result_generator,
            completed_indices=set(suspended_batch.completed_indices),
            infrastructure_retry_count=suspended_batch.infrastructure_retry_count,
        )
        return ResumablePendingAwareAsyncJoblibBatchSession(
            evaluator=self,
            _handle=EvaluationBatchHandle(
                batch_id=handle.batch_id,
                request_count=handle.request_count,
            ),
            _completed_count=handle.completed_count,
        )

    def _start_attempt(
        self,
        *,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT],
        request_inputs: Sequence[AsyncJoblibRequestInput[CandidateT]],
        execution_resources: ExecutionResources,
    ) -> Generator[
        tuple[int, EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]],
        None,
        None,
    ]:
        """Start one async joblib attempt for the provided logical request slice."""
        _ = execution_resources
        parallel_factory = cast(
            JoblibGeneratorParallelFactory[
                tuple[int, EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]]
            ],
            getattr(joblib, "Parallel"),
        )
        delayed_factory = cast(
            JoblibDelayedFactory,
            getattr(joblib, "delayed"),
        )
        return parallel_factory(
            n_jobs=self.n_jobs,
            backend=self.backend,
            return_as="generator_unordered",
        )(
            delayed_factory(_evaluate_indexed_request_outcome)(
                index=request_input.index,
                problem=problem,
                request=request_input.request,
            )
            for request_input in request_inputs
        )

    def _retry_infrastructure_failure(
        self,
        *,
        handle: EvaluationBatchHandle,
        active_batch: ActiveAsyncJoblibBatch[
            BoundaryT,
            CandidateT,
            JoblibEvaluationRecordT,
        ],
        cause: BaseException,
    ) -> bool:
        """Retry unfinished proposals after one infrastructure failure."""
        if active_batch.infrastructure_retry_count >= self.infrastructure_retry_limit:
            return False

        remaining_inputs = _remaining_async_joblib_request_inputs(
            active_batch.request_inputs,
            active_batch.completed_indices,
        )
        if len(remaining_inputs) == 0:
            return False

        close_method = getattr(active_batch.result_generator, "close", None)
        if callable(close_method):
            _ = close_method()

        active_batch.infrastructure_retry_count += 1
        try:
            active_batch.result_generator = self._start_attempt(
                problem=active_batch.problem,
                request_inputs=remaining_inputs,
                execution_resources=active_batch.execution_resources,
            )
        except BaseException as retry_exception:
            _ = self._active_batches.pop(handle.batch_id, None)
            raise BatchExecutionFailed(
                handle=handle,
                kind=_classify_joblib_failure(retry_exception),
                cause=retry_exception,
            ) from retry_exception

        _ = cause
        return True

    @override
    def submit_batch(
        self,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationBatchHandle:
        """Submit a logical batch for async joblib execution.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT]
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequest[CandidateT]]
            Request batch to execute asynchronously.

        Returns
        -------
        EvaluationBatchHandle
            Immutable handle for the submitted batch.

        Raises
        ------
        ValueError
            If ``requests`` is empty.
        """
        if len(requests) == 0:
            msg = "async batches must contain at least one request"
            raise ValueError(msg)

        execution_resources = self.execution_resources()
        request_inputs = tuple(
            AsyncJoblibRequestInput(
                index=index,
                request=request,
            )
            for index, request in enumerate(requests)
        )
        result_generator = self._start_attempt(
            problem=problem,
            request_inputs=request_inputs,
            execution_resources=execution_resources,
        )

        handle = EvaluationBatchHandle(
            batch_id=f"joblib-{next(self._batch_counter)}",
            request_count=len(requests),
        )
        self._active_batches[handle.batch_id] = ActiveAsyncJoblibBatch(
            problem=problem,
            request_inputs=request_inputs,
            execution_resources=execution_resources,
            result_generator=result_generator,
        )
        return handle

    @override
    def poll(
        self,
        handle: EvaluationBatchHandle,
    ) -> tuple[
        CompletionGroup[EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]],
        ...,
    ]:
        """Poll a submitted async joblib batch.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Handle returned when the batch was submitted.

        Returns
        -------
        tuple[CompletionGroup[EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]], ...]
            Newly completed groups in logical batch order.
        """
        active_batch = self._active_batches.get(handle.batch_id)
        if active_batch is None:
            msg = f"unknown async batch handle: {handle.batch_id}"
            raise ValueError(msg)

        while True:
            try:
                proposal_index, outcome = next(active_batch.result_generator)
            except StopIteration as exception:
                if self._retry_infrastructure_failure(
                    handle=handle,
                    active_batch=active_batch,
                    cause=RuntimeError(
                        "async batch ended before reporting all request outcomes",
                    ),
                ):
                    continue

                _ = self._active_batches.pop(handle.batch_id, None)
                msg = "async batch ended before reporting all request outcomes"
                raise BatchExecutionFailed(
                    handle=handle,
                    kind="infrastructure",
                    cause=RuntimeError(msg),
                ) from exception
            except BaseException as exception:
                failure_kind = _classify_joblib_failure(exception)
                if (
                    failure_kind == "infrastructure"
                    and self._retry_infrastructure_failure(
                        handle=handle,
                        active_batch=active_batch,
                        cause=exception,
                    )
                ):
                    continue

                _ = self._active_batches.pop(handle.batch_id, None)
                raise BatchExecutionFailed(
                    handle=handle,
                    kind=failure_kind,
                    cause=exception,
                ) from exception

            if proposal_index in active_batch.completed_indices:
                _ = self._active_batches.pop(handle.batch_id, None)
                msg = "async batch reported one request outcome more than once"
                raise BatchExecutionFailed(
                    handle=handle,
                    kind="infrastructure",
                    cause=RuntimeError(msg),
                )

            active_batch.completed_indices.add(proposal_index)
            if len(active_batch.completed_indices) >= handle.request_count:
                _ = self._active_batches.pop(handle.batch_id, None)

            return (
                CompletionGroup(
                    start_index=proposal_index,
                    outcomes=(outcome,),
                ),
            )

    @override
    def cancel(self, handle: EvaluationBatchHandle) -> None:
        """Cancel an active async joblib batch.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Handle identifying the active batch.
        """
        active_batch = self._active_batches.pop(handle.batch_id, None)
        if active_batch is None:
            return

        close_method = getattr(active_batch.result_generator, "close", None)
        if callable(close_method):
            _ = close_method()

    def suspend_batch(
        self,
        handle: EvaluationBatchHandle,
    ) -> EvaluationBatchResumeHandle:
        """Suspend an active async batch and return a resume handle.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Handle identifying the active batch.

        Returns
        -------
        EvaluationBatchResumeHandle
            Resume handle describing the suspended batch state.
        """
        active_batch = self._active_batches.pop(handle.batch_id, None)
        if active_batch is None:
            msg = f"unknown active batch handle: {handle.batch_id}"
            raise ValueError(msg)

        close_method = getattr(active_batch.result_generator, "close", None)
        if callable(close_method):
            _ = close_method()

        self._suspended_batches[handle.batch_id] = SuspendedAsyncJoblibBatch(
            problem=active_batch.problem,
            request_inputs=active_batch.request_inputs,
            execution_resources=active_batch.execution_resources,
            completed_indices=set(active_batch.completed_indices),
            infrastructure_retry_count=active_batch.infrastructure_retry_count,
        )
        return EvaluationBatchResumeHandle(
            batch_id=handle.batch_id,
            request_count=handle.request_count,
            completed_count=len(active_batch.completed_indices),
        )

    def discard_suspended_batch(self, handle: EvaluationBatchHandle) -> None:
        """Discard a suspended async batch, if present.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Handle identifying the suspended batch.
        """
        _ = self._suspended_batches.pop(handle.batch_id, None)
