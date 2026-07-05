"""Asynchronous joblib-backed evaluator and retry runtime."""

from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures.process import BrokenProcessPool as FuturesBrokenProcessPool
from dataclasses import dataclass, field
from importlib import import_module
from itertools import count
from os import cpu_count
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread, current_thread
from time import monotonic
from types import ModuleType
from typing import Generic, Literal, TypeVar, cast
from warnings import warn

import joblib  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import override

from ...artifacts import EvaluationAttemptBatch, EvaluationRequest
from ...artifacts.records import RequestAlignedEvaluationRecord
from ...evaluation_pipeline import evaluate_request_attempt, evaluate_request_outcome
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
from ..async_evaluator.sessions import EvaluationBatchSession, normalize_wait_timeout
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
    JoblibEvaluationPayloadT,
    JoblibGeneratorParallelFactory,
    JoblibListParallelFactory,
)
from .execution import build_execution_resources, validate_joblib_configuration

_ABORT_JOIN_TIMEOUT_SECONDS = 0.01
_RESULT_QUEUE_PUT_TIMEOUT_SECONDS = 0.01
_RESULT_QUEUE_WAIT_POLL_SECONDS = 0.05
_RESULT_QUEUE_COMPLETIONS_PER_WORKER = 2
_RESULT_QUEUE_TERMINAL_EVENT_CAPACITY = 1
_JOBLIB_PROCESS_EXECUTOR_MODULE = "joblib.externals.loky.process_executor"
JoblibEvaluationResultT = TypeVar("JoblibEvaluationResultT")


def _exception_type_from_module(
    module: ModuleType,
    name: str,
) -> type[BaseException] | None:
    """Return one exception type from a dynamically imported module."""
    candidate: object = getattr(module, name, None)
    if not isinstance(candidate, type):
        return None

    if not issubclass(candidate, BaseException):
        return None

    return candidate


def _joblib_process_failure_types() -> tuple[type[BaseException], ...]:
    """Return joblib/loky process-pool exception types when available."""
    try:
        process_executor_module = import_module(_JOBLIB_PROCESS_EXECUTOR_MODULE)
    except ModuleNotFoundError as exception:
        if exception.name is None or not _JOBLIB_PROCESS_EXECUTOR_MODULE.startswith(
            exception.name
        ):
            raise
        return ()

    failure_types: list[type[BaseException]] = []
    for name in ("BrokenProcessPool", "TerminatedWorkerError"):
        failure_type = _exception_type_from_module(process_executor_module, name)
        if failure_type is not None:
            failure_types.append(failure_type)
    return tuple(failure_types)


_PROCESS_POOL_INFRASTRUCTURE_FAILURE_TYPES: tuple[type[BaseException], ...] = (
    FuturesBrokenProcessPool,
    *_joblib_process_failure_types(),
)


def _evaluate_indexed_request_outcome(
    *,
    index: int,
    problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
    request: EvaluationRequest[CandidateT],
) -> tuple[int, EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]]:
    """Execute one request and carry its original logical index."""
    return (
        index,
        evaluate_request_outcome(
            problem=problem,
            request=request,
        ),
    )


def _evaluate_indexed_request_attempt(
    *,
    index: int,
    problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
    request: EvaluationRequest[CandidateT],
) -> tuple[
    int,
    EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT],
]:
    """Execute one request attempt and carry its original logical index."""
    return (
        index,
        evaluate_request_attempt(
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

    if _is_retryable_joblib_infrastructure_failure(exception):
        return "infrastructure"

    return "user_code"


def _classify_joblib_attempt_failure(
    exception: BaseException,
) -> Literal["infrastructure", "cancelled"]:
    """Classify hard failures that escape native attempt-slot evaluation.

    User-code ``Exception`` is already recorded inside ``EvaluationFailure`` by
    the attempt evaluator. Anything that still escapes the worker is a hard
    boundary/backend failure rather than a recordable user failure slot.
    """
    if isinstance(exception, KeyboardInterrupt):
        return "cancelled"

    return "infrastructure"


def _is_retryable_joblib_infrastructure_failure(exception: BaseException) -> bool:
    """Return whether a joblib failure may be retried for unfinished requests."""
    return isinstance(exception, _PROCESS_POOL_INFRASTRUCTURE_FAILURE_TYPES)


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


def _effective_joblib_worker_count(n_jobs: int) -> int:
    """Return a conservative positive worker count for joblib queue sizing."""
    if n_jobs > 0:
        return n_jobs

    detected_cpu_count = cpu_count()
    available_cpu_count = (
        1 if detected_cpu_count is None else max(1, detected_cpu_count)
    )
    if n_jobs == -1:
        return available_cpu_count

    return max(1, available_cpu_count + 1 + n_jobs)


def _async_joblib_result_queue_max_size(
    *,
    n_jobs: int,
    request_count: int,
) -> int:
    """Return bounded result-buffer capacity for one active joblib attempt."""
    request_limited_capacity = max(
        1,
        request_count + _RESULT_QUEUE_TERMINAL_EVENT_CAPACITY,
    )
    worker_limited_capacity = (
        _effective_joblib_worker_count(n_jobs) * _RESULT_QUEUE_COMPLETIONS_PER_WORKER
        + _RESULT_QUEUE_TERMINAL_EVENT_CAPACITY
    )
    return min(request_limited_capacity, worker_limited_capacity)


def _new_async_joblib_result_queue(
    *,
    n_jobs: int,
    request_count: int,
) -> Queue[
    AsyncJoblibCompletedResult[JoblibEvaluationResultT]
    | AsyncJoblibFailedResult
    | AsyncJoblibExhaustedResult
]:
    """Return a bounded queue for one drain-worker to waiter handoff."""
    return Queue(
        maxsize=_async_joblib_result_queue_max_size(
            n_jobs=n_jobs,
            request_count=request_count,
        ),
    )


def _result_queue_get_timeout(deadline: float | None) -> float:
    """Return one bounded queue wait interval for cancellable blocking waits."""
    if deadline is None:
        return _RESULT_QUEUE_WAIT_POLL_SECONDS

    return min(_RESULT_QUEUE_WAIT_POLL_SECONDS, max(0.0, deadline - monotonic()))


def _inactive_batch_failure(handle: EvaluationBatchHandle) -> BatchExecutionFailed:
    """Return the standard failure for a waiter that lost handle ownership."""
    return BatchExecutionFailed(
        handle=handle,
        kind="cancelled",
        cause=RuntimeError(
            f"async batch handle is no longer active: {handle.batch_id}",
        ),
    )


@dataclass(frozen=True, slots=True)
class _AsyncJoblibAttempt(Generic[JoblibEvaluationResultT]):
    """Result stream and abort hook for one joblib attempt."""

    result_generator: Generator[tuple[int, JoblibEvaluationResultT], None, None]
    abort: Callable[[], None] | None


def _drain_async_joblib_results(
    *,
    result_generator: Generator[tuple[int, JoblibEvaluationResultT], None, None],
    result_queue: Queue[
        AsyncJoblibCompletedResult[JoblibEvaluationResultT]
        | AsyncJoblibFailedResult
        | AsyncJoblibExhaustedResult
    ],
    abort_event: Event,
    attempt_generation: int,
) -> None:
    """Drain a blocking joblib result stream into a non-blocking queue."""

    def put_result_event(
        result_event: (
            AsyncJoblibCompletedResult[JoblibEvaluationResultT]
            | AsyncJoblibFailedResult
            | AsyncJoblibExhaustedResult
        ),
    ) -> bool:
        while not abort_event.is_set():
            try:
                result_queue.put(
                    result_event,
                    timeout=_RESULT_QUEUE_PUT_TIMEOUT_SECONDS,
                )
                return True
            except Full:
                continue
        return False

    try:
        for proposal_index, outcome in result_generator:
            if not put_result_event(
                AsyncJoblibCompletedResult(
                    index=proposal_index,
                    outcome=outcome,
                    attempt_generation=attempt_generation,
                ),
            ):
                return
    except GeneratorExit:
        _ = put_result_event(
            AsyncJoblibExhaustedResult(attempt_generation=attempt_generation),
        )
    except BaseException as exception:
        _ = put_result_event(
            AsyncJoblibFailedResult(
                exception=exception,
                attempt_generation=attempt_generation,
            ),
        )
    else:
        _ = put_result_event(
            AsyncJoblibExhaustedResult(attempt_generation=attempt_generation),
        )


def _start_async_joblib_result_worker(
    *,
    result_generator: Generator[tuple[int, JoblibEvaluationResultT], None, None],
    result_queue: Queue[
        AsyncJoblibCompletedResult[JoblibEvaluationResultT]
        | AsyncJoblibFailedResult
        | AsyncJoblibExhaustedResult
    ],
    abort_event: Event,
    attempt_generation: int,
) -> Thread:
    """Start a daemon worker that drains one blocking joblib result stream."""
    result_worker = Thread(
        target=_drain_async_joblib_results,
        kwargs={
            "result_generator": result_generator,
            "result_queue": result_queue,
            "abort_event": abort_event,
            "attempt_generation": attempt_generation,
        },
        name="variopt-async-joblib-drain",
        daemon=True,
    )
    result_worker.start()
    return result_worker


@dataclass(slots=True)
class AsyncJoblibEvaluator(
    ResumableAsyncEvaluator[
        Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord],
    ],
    Generic[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
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
        Retries apply only to backend boundary failures such as process-pool
        termination, not to ordinary objective exceptions.

    Notes
    -----
    Suspended joblib batches are kept in this evaluator instance's in-memory
    runtime state. Resume handles must be used with the same live evaluator
    instance; they are not durable process-restart checkpoints.
    """

    n_jobs: int = -1
    backend: Literal["loky", "threading"] = "loky"
    infrastructure_retry_limit: int = 0
    _batch_counter: Iterator[int] = field(init=False, repr=False)
    _active_batches: dict[
        str,
        ActiveAsyncJoblibBatch[
            BoundaryT,
            CandidateT,
            EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord],
            JoblibEvaluationPayloadT,
        ],
    ] = field(
        init=False,
        repr=False,
    )
    _active_attempt_batches: dict[
        str,
        ActiveAsyncJoblibBatch[
            BoundaryT,
            CandidateT,
            EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT],
            JoblibEvaluationPayloadT,
        ],
    ] = field(
        init=False,
        repr=False,
    )
    _suspended_batches: dict[
        str,
        SuspendedAsyncJoblibBatch[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
    ] = field(
        init=False,
        repr=False,
    )
    _suspended_attempt_batches: dict[
        str,
        SuspendedAsyncJoblibBatch[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
    ] = field(
        init=False,
        repr=False,
    )
    _resuming_batches: set[str] = field(init=False, repr=False)
    _resuming_attempt_batches: set[str] = field(init=False, repr=False)
    _discarded_resuming_batches: set[str] = field(init=False, repr=False)
    _discarded_resuming_attempt_batches: set[str] = field(init=False, repr=False)
    _lifecycle_lock: RLock = field(init=False, repr=False)

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
        self._active_attempt_batches = {}
        self._suspended_batches = {}
        self._suspended_attempt_batches = {}
        self._resuming_batches = set()
        self._resuming_attempt_batches = set()
        self._discarded_resuming_batches = set()
        self._discarded_resuming_attempt_batches = set()
        self._lifecycle_lock = RLock()

    def __getstate__(self) -> tuple[int, Literal["loky", "threading"], int]:
        """Return pickle state for evaluator configuration.

        Active, suspended, and resuming batch state is process-local runtime
        state and is intentionally not part of the pickle payload.
        """
        return (
            self.n_jobs,
            self.backend,
            self.infrastructure_retry_limit,
        )

    def __setstate__(
        self,
        state: tuple[int, Literal["loky", "threading"], int],
    ) -> None:
        """Restore evaluator configuration and fresh runtime lifecycle state."""
        n_jobs, backend, infrastructure_retry_limit = state
        self.n_jobs = n_jobs
        self.backend = backend
        self.infrastructure_retry_limit = infrastructure_retry_limit
        self.__post_init__()

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
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationBatchSession[
        EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]
    ]:
        """Open a resumable pending-aware joblib session.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequest[CandidateT]]
            Request batch to execute asynchronously.

        Returns
        -------
        EvaluationBatchSession[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]]
            Session backed by joblib's unordered completion generator.
        """
        return ResumablePendingAwareAsyncJoblibBatchSession(
            evaluator=self,
            _handle=self.submit_batch(problem, requests),
        )

    def open_attempt_session(
        self,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationBatchSession[
        EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]
    ]:
        """Open a native request-owned attempt-batch session.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequest[CandidateT]]
            Request batch to execute asynchronously.

        Returns
        -------
        EvaluationBatchSession[EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]]
            Session that streams one-slot attempt batches in completion order.
        """
        return ResumablePendingAwareAsyncJoblibBatchSession(
            evaluator=_AsyncJoblibAttemptSessionEvaluator(self),
            _handle=self.submit_attempt_batch(problem, requests),
        )

    def evaluate_attempts(
        self,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]:
        """Execute a request batch through joblib into a dense attempt batch.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequest[CandidateT]]
            Request batch to execute.

        Returns
        -------
        EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]
            Dense attempt batch aligned to ``requests``.

        Notes
        -----
        This batch-level attempt hook is separate from exact-async session
        orchestration. Session-level attempt wiring is handled by the study
        orchestration migration.
        """
        parallel_factory = cast(
            JoblibListParallelFactory[
                EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]
            ],
            getattr(joblib, "Parallel"),
        )
        delayed_factory = cast(
            JoblibDelayedFactory,
            getattr(joblib, "delayed"),
        )
        attempts = parallel_factory(
            n_jobs=self.n_jobs,
            backend=self.backend,
        )(
            delayed_factory(evaluate_request_attempt)(
                problem=problem,
                request=request,
            )
            for request in requests
        )
        return EvaluationAttemptBatch[
            CandidateT,
            JoblibEvaluationPayloadT,
        ].from_single_request_attempts(attempts)

    @override
    def resume_session(
        self,
        handle: EvaluationBatchResumeHandle,
    ) -> EvaluationBatchSession[
        EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]
    ]:
        """Resume a suspended async joblib session.

        Parameters
        ----------
        handle : EvaluationBatchResumeHandle
            Resume handle produced when a batch session was suspended.

        Returns
        -------
        EvaluationBatchSession[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]]
            Resumed batch session.
        """
        with self._lifecycle_lock:
            suspended_batch = self._suspended_batches.get(handle.batch_id)
            if suspended_batch is None:
                msg = f"unknown suspended batch handle: {handle.batch_id}"
                raise ValueError(msg)

            if handle.request_count != len(suspended_batch.request_inputs):
                msg = "resume handle request_count does not match suspended batch"
                raise ValueError(msg)

            if handle.completed_count != len(suspended_batch.completed_indices):
                msg = "resume handle completed_count does not match suspended batch"
                raise ValueError(msg)
            _ = self._suspended_batches.pop(handle.batch_id)
            self._resuming_batches.add(handle.batch_id)

        remaining_inputs = _remaining_async_joblib_request_inputs(
            suspended_batch.request_inputs,
            suspended_batch.completed_indices,
        )
        try:
            active_batch = self._start_active_batch(
                problem=suspended_batch.problem,
                request_inputs=suspended_batch.request_inputs,
                execution_resources=suspended_batch.execution_resources,
                completed_indices=set(suspended_batch.completed_indices),
                infrastructure_retry_count=suspended_batch.infrastructure_retry_count,
                attempt_inputs=remaining_inputs,
            )
        except BaseException:
            with self._lifecycle_lock:
                self._resuming_batches.discard(handle.batch_id)
                if handle.batch_id in self._discarded_resuming_batches:
                    self._discarded_resuming_batches.remove(handle.batch_id)
                else:
                    self._suspended_batches[handle.batch_id] = suspended_batch
            raise

        resumed_handle = EvaluationBatchHandle(
            batch_id=handle.batch_id,
            request_count=handle.request_count,
        )
        with self._lifecycle_lock:
            self._resuming_batches.discard(handle.batch_id)
            was_discarded = handle.batch_id in self._discarded_resuming_batches
            if was_discarded:
                self._discarded_resuming_batches.remove(handle.batch_id)
            else:
                self._active_batches[handle.batch_id] = active_batch
        if was_discarded:
            self._abort_active_batch_attempt(active_batch)
            msg = "suspended batch was cancelled while resuming"
            raise BatchExecutionFailed(
                handle=resumed_handle,
                kind="cancelled",
                cause=RuntimeError(msg),
            )

        return ResumablePendingAwareAsyncJoblibBatchSession(
            evaluator=self,
            _handle=resumed_handle,
            _completed_count=handle.completed_count,
        )

    def resume_attempt_session(
        self,
        handle: EvaluationBatchResumeHandle,
    ) -> EvaluationBatchSession[
        EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]
    ]:
        """Resume a suspended native attempt-batch session.

        Parameters
        ----------
        handle : EvaluationBatchResumeHandle
            Resume handle produced when an attempt batch session was suspended.

        Returns
        -------
        EvaluationBatchSession[EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]]
            Resumed attempt batch session.
        """
        with self._lifecycle_lock:
            suspended_batch = self._suspended_attempt_batches.get(handle.batch_id)
            if suspended_batch is None:
                msg = f"unknown suspended attempt batch handle: {handle.batch_id}"
                raise ValueError(msg)

            if handle.request_count != len(suspended_batch.request_inputs):
                msg = (
                    "resume handle request_count does not match suspended attempt batch"
                )
                raise ValueError(msg)

            if handle.completed_count != len(suspended_batch.completed_indices):
                msg = "resume handle completed_count does not match suspended attempt batch"
                raise ValueError(msg)
            _ = self._suspended_attempt_batches.pop(handle.batch_id)
            self._resuming_attempt_batches.add(handle.batch_id)

        remaining_inputs = _remaining_async_joblib_request_inputs(
            suspended_batch.request_inputs,
            suspended_batch.completed_indices,
        )
        try:
            active_batch = self._start_active_attempt_batch(
                problem=suspended_batch.problem,
                request_inputs=suspended_batch.request_inputs,
                execution_resources=suspended_batch.execution_resources,
                completed_indices=set(suspended_batch.completed_indices),
                infrastructure_retry_count=suspended_batch.infrastructure_retry_count,
                attempt_inputs=remaining_inputs,
            )
        except BaseException:
            with self._lifecycle_lock:
                self._resuming_attempt_batches.discard(handle.batch_id)
                if handle.batch_id in self._discarded_resuming_attempt_batches:
                    self._discarded_resuming_attempt_batches.remove(handle.batch_id)
                else:
                    self._suspended_attempt_batches[handle.batch_id] = suspended_batch
            raise

        resumed_handle = EvaluationBatchHandle(
            batch_id=handle.batch_id,
            request_count=handle.request_count,
        )
        with self._lifecycle_lock:
            self._resuming_attempt_batches.discard(handle.batch_id)
            was_discarded = handle.batch_id in self._discarded_resuming_attempt_batches
            if was_discarded:
                self._discarded_resuming_attempt_batches.remove(handle.batch_id)
            else:
                self._active_attempt_batches[handle.batch_id] = active_batch
        if was_discarded:
            self._abort_active_batch_attempt(active_batch)
            msg = "suspended attempt batch was cancelled while resuming"
            raise BatchExecutionFailed(
                handle=resumed_handle,
                kind="cancelled",
                cause=RuntimeError(msg),
            )

        return ResumablePendingAwareAsyncJoblibBatchSession(
            evaluator=_AsyncJoblibAttemptSessionEvaluator(self),
            _handle=resumed_handle,
            _completed_count=handle.completed_count,
        )

    def _start_attempt(
        self,
        *,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        request_inputs: Sequence[AsyncJoblibRequestInput[CandidateT]],
        execution_resources: ExecutionResources,
    ) -> Generator[
        tuple[int, EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]],
        None,
        None,
    ]:
        """Start one async joblib attempt for the provided logical request slice."""
        return self._start_controlled_joblib_attempt(
            problem=problem,
            request_inputs=request_inputs,
            execution_resources=execution_resources,
            evaluation_function=_evaluate_indexed_request_outcome,
        ).result_generator

    def _start_controlled_joblib_attempt(
        self,
        *,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        request_inputs: Sequence[AsyncJoblibRequestInput[CandidateT]],
        execution_resources: ExecutionResources,
        evaluation_function: Callable[..., tuple[int, JoblibEvaluationResultT]],
    ) -> _AsyncJoblibAttempt[JoblibEvaluationResultT]:
        """Start one default joblib attempt with an explicit abort hook."""
        _ = execution_resources
        parallel_factory = cast(
            JoblibGeneratorParallelFactory[tuple[int, JoblibEvaluationResultT]],
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
                delayed_factory(evaluation_function)(
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
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        request_inputs: Sequence[AsyncJoblibRequestInput[CandidateT]],
        execution_resources: ExecutionResources,
    ) -> _AsyncJoblibAttempt[
        EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]
    ]:
        """Start one attempt while preserving subclass ``_start_attempt`` hooks."""
        if type(self) is AsyncJoblibEvaluator:
            return self._start_controlled_joblib_attempt(
                problem=problem,
                request_inputs=request_inputs,
                execution_resources=execution_resources,
                evaluation_function=_evaluate_indexed_request_outcome,
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
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        request_inputs: tuple[AsyncJoblibRequestInput[CandidateT], ...],
        execution_resources: ExecutionResources,
        completed_indices: set[int] | None = None,
        infrastructure_retry_count: int = 0,
        attempt_inputs: Sequence[AsyncJoblibRequestInput[CandidateT]] | None = None,
    ) -> ActiveAsyncJoblibBatch[
        BoundaryT,
        CandidateT,
        EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord],
        JoblibEvaluationPayloadT,
    ]:
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
            AsyncJoblibCompletedResult[
                EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]
            ]
            | AsyncJoblibFailedResult
            | AsyncJoblibExhaustedResult
        ] = _new_async_joblib_result_queue(
            n_jobs=self.n_jobs,
            request_count=len(active_attempt_inputs),
        )
        abort_event = Event()
        result_worker = _start_async_joblib_result_worker(
            result_generator=attempt.result_generator,
            result_queue=result_queue,
            abort_event=abort_event,
            attempt_generation=0,
        )
        return ActiveAsyncJoblibBatch(
            problem=problem,
            request_inputs=request_inputs,
            execution_resources=execution_resources,
            result_generator=attempt.result_generator,
            result_queue=result_queue,
            result_worker=result_worker,
            abort_attempt=attempt.abort,
            abort_event=abort_event,
            completed_indices=set() if completed_indices is None else completed_indices,
            infrastructure_retry_count=infrastructure_retry_count,
        )

    def _replace_active_batch_attempt(
        self,
        *,
        active_batch: ActiveAsyncJoblibBatch[
            BoundaryT,
            CandidateT,
            EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord],
            JoblibEvaluationPayloadT,
        ],
        request_inputs: Sequence[AsyncJoblibRequestInput[CandidateT]],
    ) -> None:
        """Replace the running attempt for an already active logical batch."""
        self._abort_active_batch_attempt(active_batch)
        attempt = self._start_controlled_attempt(
            problem=active_batch.problem,
            request_inputs=request_inputs,
            execution_resources=active_batch.execution_resources,
        )
        result_queue: Queue[
            AsyncJoblibCompletedResult[
                EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]
            ]
            | AsyncJoblibFailedResult
            | AsyncJoblibExhaustedResult
        ] = _new_async_joblib_result_queue(
            n_jobs=self.n_jobs,
            request_count=len(request_inputs),
        )
        abort_event = Event()
        attempt_generation = active_batch.attempt_generation + 1
        active_batch.result_generator = attempt.result_generator
        active_batch.result_queue = result_queue
        active_batch.abort_event = abort_event
        active_batch.attempt_generation = attempt_generation
        active_batch.result_worker = _start_async_joblib_result_worker(
            result_generator=attempt.result_generator,
            result_queue=result_queue,
            abort_event=abort_event,
            attempt_generation=attempt_generation,
        )
        active_batch.abort_attempt = attempt.abort

    def _start_active_attempt_batch(
        self,
        *,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        request_inputs: tuple[AsyncJoblibRequestInput[CandidateT], ...],
        execution_resources: ExecutionResources,
        completed_indices: set[int] | None = None,
        infrastructure_retry_count: int = 0,
        attempt_inputs: Sequence[AsyncJoblibRequestInput[CandidateT]] | None = None,
    ) -> ActiveAsyncJoblibBatch[
        BoundaryT,
        CandidateT,
        EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT],
        JoblibEvaluationPayloadT,
    ]:
        """Start one active native attempt-batch stream."""
        active_attempt_inputs = (
            request_inputs if attempt_inputs is None else attempt_inputs
        )
        attempt = self._start_controlled_joblib_attempt(
            problem=problem,
            request_inputs=active_attempt_inputs,
            execution_resources=execution_resources,
            evaluation_function=_evaluate_indexed_request_attempt,
        )
        result_queue: Queue[
            AsyncJoblibCompletedResult[
                EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]
            ]
            | AsyncJoblibFailedResult
            | AsyncJoblibExhaustedResult
        ] = _new_async_joblib_result_queue(
            n_jobs=self.n_jobs,
            request_count=len(active_attempt_inputs),
        )
        abort_event = Event()
        result_worker = _start_async_joblib_result_worker(
            result_generator=attempt.result_generator,
            result_queue=result_queue,
            abort_event=abort_event,
            attempt_generation=0,
        )
        return ActiveAsyncJoblibBatch(
            problem=problem,
            request_inputs=request_inputs,
            execution_resources=execution_resources,
            result_generator=attempt.result_generator,
            result_queue=result_queue,
            result_worker=result_worker,
            abort_attempt=attempt.abort,
            abort_event=abort_event,
            completed_indices=set() if completed_indices is None else completed_indices,
            infrastructure_retry_count=infrastructure_retry_count,
        )

    def _replace_active_attempt_batch_attempt(
        self,
        *,
        active_batch: ActiveAsyncJoblibBatch[
            BoundaryT,
            CandidateT,
            EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT],
            JoblibEvaluationPayloadT,
        ],
        request_inputs: Sequence[AsyncJoblibRequestInput[CandidateT]],
    ) -> None:
        """Replace the running native attempt-batch stream."""
        self._abort_active_batch_attempt(active_batch)
        attempt = self._start_controlled_joblib_attempt(
            problem=active_batch.problem,
            request_inputs=request_inputs,
            execution_resources=active_batch.execution_resources,
            evaluation_function=_evaluate_indexed_request_attempt,
        )
        result_queue: Queue[
            AsyncJoblibCompletedResult[
                EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]
            ]
            | AsyncJoblibFailedResult
            | AsyncJoblibExhaustedResult
        ] = _new_async_joblib_result_queue(
            n_jobs=self.n_jobs,
            request_count=len(request_inputs),
        )
        abort_event = Event()
        attempt_generation = active_batch.attempt_generation + 1
        active_batch.result_generator = attempt.result_generator
        active_batch.result_queue = result_queue
        active_batch.abort_event = abort_event
        active_batch.attempt_generation = attempt_generation
        active_batch.result_worker = _start_async_joblib_result_worker(
            result_generator=attempt.result_generator,
            result_queue=result_queue,
            abort_event=abort_event,
            attempt_generation=attempt_generation,
        )
        active_batch.abort_attempt = attempt.abort

    def _abort_active_batch_attempt(
        self,
        active_batch: ActiveAsyncJoblibBatch[
            BoundaryT,
            CandidateT,
            JoblibEvaluationResultT,
            JoblibEvaluationPayloadT,
        ],
    ) -> None:
        """Best-effort abort for one active joblib attempt."""
        active_batch.abort_event.set()
        abort_failure: Exception | None = None
        if active_batch.abort_attempt is not None:
            try:
                active_batch.abort_attempt()
            except Exception as exception:
                abort_failure = exception

        close_failure: Exception | None = None
        close_method_missing = False
        if active_batch.abort_attempt is None or abort_failure is not None:
            close_method = getattr(active_batch.result_generator, "close", None)
            if callable(close_method):
                try:
                    _ = close_method()
                except Exception as exception:
                    close_failure = exception
            else:
                close_method_missing = True

        if (
            abort_failure is not None
            and close_failure is None
            and not close_method_missing
        ):
            warn(
                (
                    "async joblib abort hook failed; fallback generator close "
                    "was used "
                    f"({type(abort_failure).__name__}: {abort_failure})"
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        if close_failure is not None or close_method_missing:
            close_detail = (
                "close method unavailable"
                if close_method_missing
                else f"{type(close_failure).__name__}: {close_failure}"
            )
            abort_detail = (
                ""
                if abort_failure is None
                else f"; abort hook failed first ({type(abort_failure).__name__}: {abort_failure})"
            )
            warn(
                (
                    "failed to abort async joblib attempt; detached backend work "
                    "may continue after evaluator state is cleared "
                    f"({close_detail}{abort_detail})"
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        if active_batch.result_worker is not current_thread():
            active_batch.result_worker.join(timeout=_ABORT_JOIN_TIMEOUT_SECONDS)

    def _retry_infrastructure_failure(
        self,
        *,
        handle: EvaluationBatchHandle,
        active_batch: ActiveAsyncJoblibBatch[
            BoundaryT,
            CandidateT,
            EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord],
            JoblibEvaluationPayloadT,
        ],
        cause: BaseException,
    ) -> bool:
        """Retry unfinished proposals after one infrastructure failure."""
        with self._lifecycle_lock:
            if self._active_batches.get(handle.batch_id) is not active_batch:
                raise _inactive_batch_failure(handle)

            if (
                active_batch.infrastructure_retry_count
                >= self.infrastructure_retry_limit
            ):
                return False

            remaining_inputs = _remaining_async_joblib_request_inputs(
                active_batch.request_inputs,
                active_batch.completed_indices,
            )
            if len(remaining_inputs) == 0:
                return False

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

    def _retry_attempt_infrastructure_failure(
        self,
        *,
        handle: EvaluationBatchHandle,
        active_batch: ActiveAsyncJoblibBatch[
            BoundaryT,
            CandidateT,
            EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT],
            JoblibEvaluationPayloadT,
        ],
        cause: BaseException,
    ) -> bool:
        """Retry unfinished request attempts after one infrastructure failure."""
        with self._lifecycle_lock:
            if self._active_attempt_batches.get(handle.batch_id) is not active_batch:
                raise _inactive_batch_failure(handle)

            if (
                active_batch.infrastructure_retry_count
                >= self.infrastructure_retry_limit
            ):
                return False

            remaining_inputs = _remaining_async_joblib_request_inputs(
                active_batch.request_inputs,
                active_batch.completed_indices,
            )
            if len(remaining_inputs) == 0:
                return False

            active_batch.infrastructure_retry_count += 1
            try:
                self._replace_active_attempt_batch_attempt(
                    active_batch=active_batch,
                    request_inputs=remaining_inputs,
                )
            except BaseException as retry_exception:
                _ = self._active_attempt_batches.pop(handle.batch_id, None)
                raise BatchExecutionFailed(
                    handle=handle,
                    kind=_classify_joblib_attempt_failure(retry_exception),
                    cause=retry_exception,
                ) from retry_exception

        _ = cause
        return True

    @override
    def submit_batch(
        self,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationBatchHandle:
        """Submit a logical batch for async joblib execution.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
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
        active_batch = self._start_active_batch(
            problem=problem,
            request_inputs=request_inputs,
            execution_resources=execution_resources,
        )
        with self._lifecycle_lock:
            self._active_batches[handle.batch_id] = active_batch
        return handle

    def submit_attempt_batch(
        self,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationBatchHandle:
        """Submit a logical batch for native attempt-slot execution.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequest[CandidateT]]
            Request batch to execute asynchronously.

        Returns
        -------
        EvaluationBatchHandle
            Immutable handle for the submitted attempt batch.

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
            batch_id=f"joblib-attempt-{next(self._batch_counter)}",
            request_count=len(requests),
        )
        active_batch = self._start_active_attempt_batch(
            problem=problem,
            request_inputs=request_inputs,
            execution_resources=execution_resources,
        )
        with self._lifecycle_lock:
            self._active_attempt_batches[handle.batch_id] = active_batch
        return handle

    @override
    def poll(
        self,
        handle: EvaluationBatchHandle,
    ) -> tuple[
        CompletionGroup[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]],
        ...,
    ]:
        """Poll a submitted async joblib batch without blocking.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Handle returned when the batch was submitted.

        Returns
        -------
        tuple[CompletionGroup[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]], ...]
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
        CompletionGroup[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]],
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
        tuple[CompletionGroup[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]], ...]
            Newly completed groups in logical batch order, or an empty tuple
            when ``timeout`` expires before any completion is available.
        """
        normalized_timeout = normalize_wait_timeout(timeout)
        return self._collect_next_completion_group(
            handle,
            block=True,
            timeout=normalized_timeout,
        )

    def poll_attempts(
        self,
        handle: EvaluationBatchHandle,
    ) -> tuple[
        CompletionGroup[EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]],
        ...,
    ]:
        """Poll a native attempt batch without blocking."""
        return self._collect_next_attempt_completion_group(
            handle,
            block=False,
            timeout=None,
        )

    def wait_attempts(
        self,
        handle: EvaluationBatchHandle,
        *,
        timeout: float | None = None,
    ) -> tuple[
        CompletionGroup[EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]],
        ...,
    ]:
        """Wait for at least one native attempt-batch completion group."""
        normalized_timeout = normalize_wait_timeout(timeout)
        return self._collect_next_attempt_completion_group(
            handle,
            block=True,
            timeout=normalized_timeout,
        )

    def _collect_next_completion_group(
        self,
        handle: EvaluationBatchHandle,
        *,
        block: bool,
        timeout: float | None,
    ) -> tuple[
        CompletionGroup[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]],
        ...,
    ]:
        """Collect one queued completion group, retrying infrastructure failures."""
        with self._lifecycle_lock:
            active_batch = self._active_batches.get(handle.batch_id)
        if active_batch is None:
            msg = f"unknown async batch handle: {handle.batch_id}"
            raise ValueError(msg)

        deadline = None if timeout is None else monotonic() + timeout
        while True:
            try:
                if block:
                    queue_timeout = _result_queue_get_timeout(deadline)
                    result_event = active_batch.result_queue.get(
                        timeout=queue_timeout,
                    )
                else:
                    result_event = active_batch.result_queue.get_nowait()
            except Empty:
                if block:
                    with self._lifecycle_lock:
                        if (
                            self._active_batches.get(handle.batch_id)
                            is not active_batch
                        ):
                            raise _inactive_batch_failure(handle)
                    if deadline is None or monotonic() < deadline:
                        continue
                return ()

            with self._lifecycle_lock:
                if self._active_batches.get(handle.batch_id) is not active_batch:
                    raise _inactive_batch_failure(handle)
                is_stale_event = (
                    result_event.attempt_generation != active_batch.attempt_generation
                )
            if is_stale_event:
                if not block:
                    return ()
                continue

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

                with self._lifecycle_lock:
                    if self._active_batches.get(handle.batch_id) is not active_batch:
                        raise _inactive_batch_failure(handle)
                    _ = self._active_batches.pop(handle.batch_id, None)
                msg = "async batch ended before reporting all request outcomes"
                raise BatchExecutionFailed(
                    handle=handle,
                    kind="infrastructure",
                    cause=RuntimeError(msg),
                )

            failure_kind = _classify_joblib_failure(result_event.exception)
            if failure_kind == "infrastructure" and self._retry_infrastructure_failure(
                handle=handle,
                active_batch=active_batch,
                cause=result_event.exception,
            ):
                if not block:
                    return ()
                continue

            with self._lifecycle_lock:
                if self._active_batches.get(handle.batch_id) is not active_batch:
                    raise _inactive_batch_failure(handle)
                _ = self._active_batches.pop(handle.batch_id, None)
            raise BatchExecutionFailed(
                handle=handle,
                kind=failure_kind,
                cause=result_event.exception,
            ) from result_event.exception

    def _collect_next_attempt_completion_group(
        self,
        handle: EvaluationBatchHandle,
        *,
        block: bool,
        timeout: float | None,
    ) -> tuple[
        CompletionGroup[EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]],
        ...,
    ]:
        """Collect one queued attempt completion group."""
        with self._lifecycle_lock:
            active_batch = self._active_attempt_batches.get(handle.batch_id)
        if active_batch is None:
            msg = f"unknown async attempt batch handle: {handle.batch_id}"
            raise ValueError(msg)

        deadline = None if timeout is None else monotonic() + timeout
        while True:
            try:
                if block:
                    queue_timeout = _result_queue_get_timeout(deadline)
                    result_event = active_batch.result_queue.get(
                        timeout=queue_timeout,
                    )
                else:
                    result_event = active_batch.result_queue.get_nowait()
            except Empty:
                if block:
                    with self._lifecycle_lock:
                        if (
                            self._active_attempt_batches.get(handle.batch_id)
                            is not active_batch
                        ):
                            raise _inactive_batch_failure(handle)
                    if deadline is None or monotonic() < deadline:
                        continue
                return ()

            with self._lifecycle_lock:
                if (
                    self._active_attempt_batches.get(handle.batch_id)
                    is not active_batch
                ):
                    raise _inactive_batch_failure(handle)
                is_stale_event = (
                    result_event.attempt_generation != active_batch.attempt_generation
                )
            if is_stale_event:
                if not block:
                    return ()
                continue

            if isinstance(result_event, AsyncJoblibCompletedResult):
                return self._attempt_completion_group_for_result_event(
                    handle=handle,
                    active_batch=active_batch,
                    result_event=result_event,
                )

            if isinstance(result_event, AsyncJoblibExhaustedResult):
                retry_cause: BaseException = RuntimeError(
                    "async attempt batch ended before reporting all request attempts",
                )
                if self._retry_attempt_infrastructure_failure(
                    handle=handle,
                    active_batch=active_batch,
                    cause=retry_cause,
                ):
                    if not block:
                        return ()
                    continue

                with self._lifecycle_lock:
                    if (
                        self._active_attempt_batches.get(handle.batch_id)
                        is not active_batch
                    ):
                        raise _inactive_batch_failure(handle)
                    _ = self._active_attempt_batches.pop(handle.batch_id, None)
                msg = "async attempt batch ended before reporting all request attempts"
                raise BatchExecutionFailed(
                    handle=handle,
                    kind="infrastructure",
                    cause=RuntimeError(msg),
                )

            failure_kind = _classify_joblib_attempt_failure(result_event.exception)
            if _is_retryable_joblib_infrastructure_failure(
                result_event.exception
            ) and self._retry_attempt_infrastructure_failure(
                handle=handle,
                active_batch=active_batch,
                cause=result_event.exception,
            ):
                if not block:
                    return ()
                continue

            with self._lifecycle_lock:
                if (
                    self._active_attempt_batches.get(handle.batch_id)
                    is not active_batch
                ):
                    raise _inactive_batch_failure(handle)
                _ = self._active_attempt_batches.pop(handle.batch_id, None)
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
            EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord],
            JoblibEvaluationPayloadT,
        ],
        result_event: AsyncJoblibCompletedResult[
            EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]
        ],
    ) -> tuple[
        CompletionGroup[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]],
        ...,
    ]:
        """Convert one queued result event into a completion group."""
        proposal_index = result_event.index
        duplicate_batch: (
            ActiveAsyncJoblibBatch[
                BoundaryT,
                CandidateT,
                EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord],
                JoblibEvaluationPayloadT,
            ]
            | None
        ) = None
        with self._lifecycle_lock:
            if self._active_batches.get(handle.batch_id) is not active_batch:
                raise _inactive_batch_failure(handle)

            if proposal_index in active_batch.completed_indices:
                duplicate_batch = self._active_batches.pop(handle.batch_id, None)
            else:
                active_batch.completed_indices.add(proposal_index)
                if len(active_batch.completed_indices) >= handle.request_count:
                    _ = self._active_batches.pop(handle.batch_id, None)

        if duplicate_batch is not None:
            self._abort_active_batch_attempt(duplicate_batch)
            msg = "async batch reported one request outcome more than once"
            raise BatchExecutionFailed(
                handle=handle,
                kind="infrastructure",
                cause=RuntimeError(msg),
            )

        return (
            CompletionGroup(
                start_index=proposal_index,
                outcomes=(result_event.outcome,),
            ),
        )

    def _attempt_completion_group_for_result_event(
        self,
        *,
        handle: EvaluationBatchHandle,
        active_batch: ActiveAsyncJoblibBatch[
            BoundaryT,
            CandidateT,
            EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT],
            JoblibEvaluationPayloadT,
        ],
        result_event: AsyncJoblibCompletedResult[
            EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]
        ],
    ) -> tuple[
        CompletionGroup[EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]],
        ...,
    ]:
        """Convert one queued native attempt event into a completion group."""
        proposal_index = result_event.index
        duplicate_batch: (
            ActiveAsyncJoblibBatch[
                BoundaryT,
                CandidateT,
                EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT],
                JoblibEvaluationPayloadT,
            ]
            | None
        ) = None
        with self._lifecycle_lock:
            if self._active_attempt_batches.get(handle.batch_id) is not active_batch:
                raise _inactive_batch_failure(handle)

            if proposal_index in active_batch.completed_indices:
                duplicate_batch = self._active_attempt_batches.pop(
                    handle.batch_id, None
                )
            else:
                active_batch.completed_indices.add(proposal_index)
                if len(active_batch.completed_indices) >= handle.request_count:
                    _ = self._active_attempt_batches.pop(handle.batch_id, None)

        if duplicate_batch is not None:
            self._abort_active_batch_attempt(duplicate_batch)
            msg = "async batch reported one request attempt more than once"
            raise BatchExecutionFailed(
                handle=handle,
                kind="infrastructure",
                cause=RuntimeError(msg),
            )

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
        with self._lifecycle_lock:
            active_batch = self._active_batches.pop(handle.batch_id, None)
            _ = self._suspended_batches.pop(handle.batch_id, None)
        if active_batch is None:
            return

        self._abort_active_batch_attempt(active_batch)

    def cancel_attempt_batch(self, handle: EvaluationBatchHandle) -> None:
        """Cancel an active native attempt batch."""
        with self._lifecycle_lock:
            active_batch = self._active_attempt_batches.pop(handle.batch_id, None)
            _ = self._suspended_attempt_batches.pop(handle.batch_id, None)
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
        with self._lifecycle_lock:
            active_batch = self._active_batches.pop(handle.batch_id, None)
            if active_batch is None:
                msg = f"unknown active batch handle: {handle.batch_id}"
                raise ValueError(msg)
            suspended_batch = SuspendedAsyncJoblibBatch(
                problem=active_batch.problem,
                request_inputs=active_batch.request_inputs,
                execution_resources=active_batch.execution_resources,
                completed_indices=set(active_batch.completed_indices),
                infrastructure_retry_count=active_batch.infrastructure_retry_count,
            )
            self._suspended_batches[handle.batch_id] = suspended_batch

        self._abort_active_batch_attempt(active_batch)

        with self._lifecycle_lock:
            if self._suspended_batches.get(handle.batch_id) is not suspended_batch:
                msg = "async batch was cancelled while suspending"
                raise BatchExecutionFailed(
                    handle=handle,
                    kind="cancelled",
                    cause=RuntimeError(msg),
                )
        return EvaluationBatchResumeHandle(
            batch_id=handle.batch_id,
            request_count=handle.request_count,
            completed_count=len(active_batch.completed_indices),
        )

    def suspend_attempt_batch(
        self,
        handle: EvaluationBatchHandle,
    ) -> EvaluationBatchResumeHandle:
        """Suspend an active native attempt batch and return a resume handle."""
        with self._lifecycle_lock:
            active_batch = self._active_attempt_batches.pop(handle.batch_id, None)
            if active_batch is None:
                msg = f"unknown active attempt batch handle: {handle.batch_id}"
                raise ValueError(msg)
            suspended_batch = SuspendedAsyncJoblibBatch(
                problem=active_batch.problem,
                request_inputs=active_batch.request_inputs,
                execution_resources=active_batch.execution_resources,
                completed_indices=set(active_batch.completed_indices),
                infrastructure_retry_count=active_batch.infrastructure_retry_count,
            )
            self._suspended_attempt_batches[handle.batch_id] = suspended_batch

        self._abort_active_batch_attempt(active_batch)

        with self._lifecycle_lock:
            if (
                self._suspended_attempt_batches.get(handle.batch_id)
                is not suspended_batch
            ):
                msg = "async attempt batch was cancelled while suspending"
                raise BatchExecutionFailed(
                    handle=handle,
                    kind="cancelled",
                    cause=RuntimeError(msg),
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
        with self._lifecycle_lock:
            removed_batch = self._suspended_batches.pop(handle.batch_id, None)
            if removed_batch is None and handle.batch_id in self._resuming_batches:
                self._discarded_resuming_batches.add(handle.batch_id)

    def discard_suspended_attempt_batch(self, handle: EvaluationBatchHandle) -> None:
        """Discard a suspended native attempt batch, if present."""
        with self._lifecycle_lock:
            removed_batch = self._suspended_attempt_batches.pop(
                handle.batch_id,
                None,
            )
            if (
                removed_batch is None
                and handle.batch_id in self._resuming_attempt_batches
            ):
                self._discarded_resuming_attempt_batches.add(handle.batch_id)


@dataclass(frozen=True, slots=True)
class _AsyncJoblibAttemptSessionEvaluator(
    Generic[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
):
    """Adapt native attempt-batch methods to the generic session protocol."""

    evaluator: AsyncJoblibEvaluator[BoundaryT, CandidateT, JoblibEvaluationPayloadT]

    def poll(
        self,
        handle: EvaluationBatchHandle,
    ) -> tuple[
        CompletionGroup[EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]],
        ...,
    ]:
        """Poll native attempt completions without blocking."""
        return self.evaluator.poll_attempts(handle)

    def wait(
        self,
        handle: EvaluationBatchHandle,
        *,
        timeout: float | None = None,
    ) -> tuple[
        CompletionGroup[EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]],
        ...,
    ]:
        """Wait for native attempt completions."""
        return self.evaluator.wait_attempts(handle, timeout=timeout)

    def cancel(self, handle: EvaluationBatchHandle) -> None:
        """Cancel the native attempt batch."""
        self.evaluator.cancel_attempt_batch(handle)

    def suspend_batch(
        self,
        handle: EvaluationBatchHandle,
    ) -> EvaluationBatchResumeHandle:
        """Suspend the native attempt batch."""
        return self.evaluator.suspend_attempt_batch(handle)

    def discard_suspended_batch(self, handle: EvaluationBatchHandle) -> None:
        """Discard the suspended native attempt batch."""
        self.evaluator.discard_suspended_attempt_batch(handle)
