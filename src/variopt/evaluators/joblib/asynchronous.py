"""Asynchronous joblib-backed evaluator and retry runtime."""

from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import count
from queue import Empty, Queue
from threading import Thread, current_thread
from time import monotonic
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
    AsyncJoblibCompletedResult,
    AsyncJoblibExhaustedResult,
    AsyncJoblibFailedResult,
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

_ABORT_JOIN_TIMEOUT_SECONDS = 0.01


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


@dataclass(frozen=True, slots=True)
class _AsyncJoblibAttempt(Generic[CandidateT, JoblibEvaluationRecordT]):
    """Result stream and abort hook for one joblib attempt."""

    result_generator: Generator[
        tuple[int, EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]],
        None,
        None,
    ]
    abort: Callable[[], None] | None


def _drain_async_joblib_results(
    *,
    result_generator: Generator[
        tuple[int, EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]],
        None,
        None,
    ],
    result_queue: Queue[
        AsyncJoblibCompletedResult[CandidateT, JoblibEvaluationRecordT]
        | AsyncJoblibFailedResult
        | AsyncJoblibExhaustedResult
    ],
) -> None:
    """Drain a blocking joblib result stream into a non-blocking queue."""
    try:
        for proposal_index, outcome in result_generator:
            result_queue.put(
                AsyncJoblibCompletedResult(
                    index=proposal_index,
                    outcome=outcome,
                ),
            )
    except GeneratorExit:
        result_queue.put(AsyncJoblibExhaustedResult())
    except BaseException as exception:
        result_queue.put(AsyncJoblibFailedResult(exception=exception))
    else:
        result_queue.put(AsyncJoblibExhaustedResult())


def _start_async_joblib_result_worker(
    *,
    result_generator: Generator[
        tuple[int, EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]],
        None,
        None,
    ],
    result_queue: Queue[
        AsyncJoblibCompletedResult[CandidateT, JoblibEvaluationRecordT]
        | AsyncJoblibFailedResult
        | AsyncJoblibExhaustedResult
    ],
) -> Thread:
    """Start a daemon worker that drains one blocking joblib result stream."""
    result_worker = Thread(
        target=_drain_async_joblib_results,
        kwargs={
            "result_generator": result_generator,
            "result_queue": result_queue,
        },
        name="variopt-async-joblib-drain",
        daemon=True,
    )
    result_worker.start()
    return result_worker


def _validate_wait_timeout(timeout: float | None) -> None:
    """Reject invalid timeout values for async joblib waits."""
    if timeout is not None and timeout < 0.0:
        msg = "timeout must be non-negative"
        raise ValueError(msg)


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
        self._active_batches[handle.batch_id] = self._start_active_batch(
            problem=suspended_batch.problem,
            request_inputs=suspended_batch.request_inputs,
            execution_resources=suspended_batch.execution_resources,
            completed_indices=set(suspended_batch.completed_indices),
            infrastructure_retry_count=suspended_batch.infrastructure_retry_count,
            attempt_inputs=remaining_inputs,
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
        return self._start_controlled_joblib_attempt(
            problem=problem,
            request_inputs=request_inputs,
            execution_resources=execution_resources,
        ).result_generator

    def _start_controlled_joblib_attempt(
        self,
        *,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT],
        request_inputs: Sequence[AsyncJoblibRequestInput[CandidateT]],
        execution_resources: ExecutionResources,
    ) -> _AsyncJoblibAttempt[CandidateT, JoblibEvaluationRecordT]:
        """Start one default joblib attempt with an explicit abort hook."""
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
        parallel_runner = parallel_factory(
            n_jobs=self.n_jobs,
            backend=self.backend,
            return_as="generator_unordered",
        )
        abort_method = getattr(parallel_runner, "_abort", None)
        abort: Callable[[], None] | None = None
        if callable(abort_method):
            # joblib has no public abort hook for generator_unordered; use the
            # private hook when available and fall back to generator.close().
            def abort_parallel_runner() -> None:
                _ = abort_method()

            abort = abort_parallel_runner

        return _AsyncJoblibAttempt(
            result_generator=parallel_runner(
                delayed_factory(_evaluate_indexed_request_outcome)(
                    index=request_input.index,
                    problem=problem,
                    request=request_input.request,
                )
                for request_input in request_inputs
            ),
            abort=abort,
        )

    def _start_controlled_attempt(
        self,
        *,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT],
        request_inputs: Sequence[AsyncJoblibRequestInput[CandidateT]],
        execution_resources: ExecutionResources,
    ) -> _AsyncJoblibAttempt[CandidateT, JoblibEvaluationRecordT]:
        """Start one attempt while preserving subclass ``_start_attempt`` hooks."""
        if type(self) is AsyncJoblibEvaluator:
            return self._start_controlled_joblib_attempt(
                problem=problem,
                request_inputs=request_inputs,
                execution_resources=execution_resources,
            )

        return _AsyncJoblibAttempt(
            result_generator=self._start_attempt(
                problem=problem,
                request_inputs=request_inputs,
                execution_resources=execution_resources,
            ),
            abort=None,
        )

    def _start_active_batch(
        self,
        *,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationRecordT],
        request_inputs: tuple[AsyncJoblibRequestInput[CandidateT], ...],
        execution_resources: ExecutionResources,
        completed_indices: set[int] | None = None,
        infrastructure_retry_count: int = 0,
        attempt_inputs: Sequence[AsyncJoblibRequestInput[CandidateT]] | None = None,
    ) -> ActiveAsyncJoblibBatch[BoundaryT, CandidateT, JoblibEvaluationRecordT]:
        """Start one active batch and its non-blocking result buffer."""
        active_attempt_inputs = (
            request_inputs if attempt_inputs is None else attempt_inputs
        )
        attempt = self._start_controlled_attempt(
            problem=problem,
            request_inputs=active_attempt_inputs,
            execution_resources=execution_resources,
        )
        result_queue: Queue[
            AsyncJoblibCompletedResult[CandidateT, JoblibEvaluationRecordT]
            | AsyncJoblibFailedResult
            | AsyncJoblibExhaustedResult
        ] = Queue()
        result_worker = _start_async_joblib_result_worker(
            result_generator=attempt.result_generator,
            result_queue=result_queue,
        )
        return ActiveAsyncJoblibBatch(
            problem=problem,
            request_inputs=request_inputs,
            execution_resources=execution_resources,
            result_generator=attempt.result_generator,
            result_queue=result_queue,
            result_worker=result_worker,
            abort_attempt=attempt.abort,
            completed_indices=set() if completed_indices is None else completed_indices,
            infrastructure_retry_count=infrastructure_retry_count,
        )

    def _replace_active_batch_attempt(
        self,
        *,
        active_batch: ActiveAsyncJoblibBatch[
            BoundaryT,
            CandidateT,
            JoblibEvaluationRecordT,
        ],
        request_inputs: Sequence[AsyncJoblibRequestInput[CandidateT]],
    ) -> None:
        """Replace the running attempt for an already active logical batch."""
        attempt = self._start_controlled_attempt(
            problem=active_batch.problem,
            request_inputs=request_inputs,
            execution_resources=active_batch.execution_resources,
        )
        result_queue: Queue[
            AsyncJoblibCompletedResult[CandidateT, JoblibEvaluationRecordT]
            | AsyncJoblibFailedResult
            | AsyncJoblibExhaustedResult
        ] = Queue()
        active_batch.result_generator = attempt.result_generator
        active_batch.result_queue = result_queue
        active_batch.result_worker = _start_async_joblib_result_worker(
            result_generator=attempt.result_generator,
            result_queue=result_queue,
        )
        active_batch.abort_attempt = attempt.abort

    def _abort_active_batch_attempt(
        self,
        active_batch: ActiveAsyncJoblibBatch[
            BoundaryT,
            CandidateT,
            JoblibEvaluationRecordT,
        ],
    ) -> None:
        """Best-effort abort for one active joblib attempt."""
        try:
            if active_batch.abort_attempt is not None:
                active_batch.abort_attempt()
            else:
                close_method = getattr(active_batch.result_generator, "close", None)
                if callable(close_method):
                    _ = close_method()
        except Exception:
            pass

        if active_batch.result_worker is not current_thread():
            active_batch.result_worker.join(timeout=_ABORT_JOIN_TIMEOUT_SECONDS)

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

        self._abort_active_batch_attempt(active_batch)

        active_batch.infrastructure_retry_count += 1
        try:
            self._replace_active_batch_attempt(
                active_batch=active_batch,
                request_inputs=remaining_inputs,
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
        handle = EvaluationBatchHandle(
            batch_id=f"joblib-{next(self._batch_counter)}",
            request_count=len(requests),
        )
        self._active_batches[handle.batch_id] = self._start_active_batch(
            problem=problem,
            request_inputs=request_inputs,
            execution_resources=execution_resources,
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
        """Poll a submitted async joblib batch without blocking.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Handle returned when the batch was submitted.

        Returns
        -------
        tuple[CompletionGroup[EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]], ...]
            Newly completed groups in logical batch order, or an empty tuple
            when none are currently available.
        """
        return self._collect_next_completion_group(
            handle,
            block=False,
            timeout=None,
        )

    def wait(
        self,
        handle: EvaluationBatchHandle,
        *,
        timeout: float | None = None,
    ) -> tuple[
        CompletionGroup[EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]],
        ...,
    ]:
        """Wait for at least one async joblib completion group.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Handle returned when the batch was submitted.
        timeout : float | None, default=None
            Maximum number of seconds to wait. ``None`` waits indefinitely.

        Returns
        -------
        tuple[CompletionGroup[EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]], ...]
            Newly completed groups in logical batch order, or an empty tuple
            when ``timeout`` expires before any completion is available.
        """
        _validate_wait_timeout(timeout)
        return self._collect_next_completion_group(
            handle,
            block=True,
            timeout=timeout,
        )

    def _collect_next_completion_group(
        self,
        handle: EvaluationBatchHandle,
        *,
        block: bool,
        timeout: float | None,
    ) -> tuple[
        CompletionGroup[EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]],
        ...,
    ]:
        """Collect one queued completion group, retrying infrastructure failures."""
        active_batch = self._active_batches.get(handle.batch_id)
        if active_batch is None:
            msg = f"unknown async batch handle: {handle.batch_id}"
            raise ValueError(msg)

        deadline = None if timeout is None else monotonic() + timeout
        while True:
            try:
                if block:
                    queue_timeout = (
                        None
                        if deadline is None
                        else max(0.0, deadline - monotonic())
                    )
                    result_event = active_batch.result_queue.get(
                        timeout=queue_timeout,
                    )
                else:
                    result_event = active_batch.result_queue.get_nowait()
            except Empty:
                return ()

            if isinstance(result_event, AsyncJoblibCompletedResult):
                return self._completion_group_for_result_event(
                    handle=handle,
                    active_batch=active_batch,
                    result_event=result_event,
                )

            if isinstance(result_event, AsyncJoblibExhaustedResult):
                retry_cause: BaseException = RuntimeError(
                    "async batch ended before reporting all request outcomes",
                )
                if self._retry_infrastructure_failure(
                    handle=handle,
                    active_batch=active_batch,
                    cause=retry_cause,
                ):
                    if not block:
                        return ()
                    continue

                _ = self._active_batches.pop(handle.batch_id, None)
                msg = "async batch ended before reporting all request outcomes"
                raise BatchExecutionFailed(
                    handle=handle,
                    kind="infrastructure",
                    cause=RuntimeError(msg),
                )

            failure_kind = _classify_joblib_failure(result_event.exception)
            if (
                failure_kind == "infrastructure"
                and self._retry_infrastructure_failure(
                    handle=handle,
                    active_batch=active_batch,
                    cause=result_event.exception,
                )
            ):
                if not block:
                    return ()
                continue

            _ = self._active_batches.pop(handle.batch_id, None)
            raise BatchExecutionFailed(
                handle=handle,
                kind=failure_kind,
                cause=result_event.exception,
            ) from result_event.exception

    def _completion_group_for_result_event(
        self,
        *,
        handle: EvaluationBatchHandle,
        active_batch: ActiveAsyncJoblibBatch[
            BoundaryT,
            CandidateT,
            JoblibEvaluationRecordT,
        ],
        result_event: AsyncJoblibCompletedResult[
            CandidateT,
            JoblibEvaluationRecordT,
        ],
    ) -> tuple[
        CompletionGroup[EvaluationOutcome[CandidateT, JoblibEvaluationRecordT]],
        ...,
    ]:
        """Convert one queued result event into a completion group."""
        proposal_index = result_event.index
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
                outcomes=(result_event.outcome,),
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

        self._abort_active_batch_attempt(active_batch)

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

        self._abort_active_batch_attempt(active_batch)

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
