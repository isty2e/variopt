"""Tests for execution-only joblib evaluators."""

import pickle
import time
import warnings
from collections.abc import Callable, Generator, Sequence
from concurrent.futures.process import BrokenProcessPool as FuturesBrokenProcessPool
from dataclasses import dataclass
from importlib import import_module
from queue import Queue
from threading import Event, Thread, current_thread
from types import ModuleType
from typing import Generic, Literal, TypeGuard, TypeVar, cast, final

import numpy as np
import pytest
from typing_extensions import override

from variopt import (
    EvaluationAttemptBatch,
    EvaluationOutcome,
    EvaluationProtocol,
    EvaluationRequest,
    IntegerSpace,
    Objective,
    Observation,
    OptimizationDirection,
    Problem,
    Proposal,
    RunMethod,
    Study,
)
from variopt.artifacts import EvaluationSuccess
from variopt.artifacts.records import RequestAlignedEvaluationRecord
from variopt.evaluation_pipeline import evaluate_request_outcome
from variopt.evaluators import (
    AsyncJoblibEvaluator,
    BatchExecutionFailed,
    CompletionGroup,
    EvaluationBatchHandle,
    EvaluationBatchResumeHandle,
    EvaluationBatchSessionState,
    JoblibEvaluator,
    PendingAwareBatchSession,
    ResumableAsyncEvaluator,
    ResumableBatchSession,
)
from variopt.evaluators.async_evaluator.sessions import EvaluationBatchSession
from variopt.evaluators.joblib.batches import (
    ActiveAsyncJoblibBatch,
    AsyncJoblibCompletedResult,
    AsyncJoblibExhaustedResult,
    AsyncJoblibFailedResult,
    AsyncJoblibRequestInput,
    SuspendedAsyncJoblibBatch,
)
from variopt.execution import (
    EXACT_ASYNC_EXECUTION_MODEL,
    SEQUENTIAL_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
    ExecutionModel,
    ExecutionResources,
    NestedParallelismPolicy,
)

AttemptCandidateT = TypeVar("AttemptCandidateT")
SessionEvaluationT = TypeVar("SessionEvaluationT")
QueuedEventT = TypeVar("QueuedEventT")
OutcomeJoblibQueueEvent = (
    AsyncJoblibCompletedResult[
        EvaluationOutcome[int, RequestAlignedEvaluationRecord]
    ]
    | AsyncJoblibFailedResult
    | AsyncJoblibExhaustedResult
)
AttemptJoblibQueueEvent = (
    AsyncJoblibCompletedResult[EvaluationAttemptBatch[int, Observation[int]]]
    | AsyncJoblibFailedResult
    | AsyncJoblibExhaustedResult
)
InjectedAsyncJoblibFailureMode = Literal[
    "infrastructure_once",
    "infrastructure_always",
    "user_code_once",
    "user_timeout_error_once",
    "keyboard_interrupt_once",
    "loky_broken_process_pool_once",
    "loky_terminated_worker_once",
    "futures_broken_process_pool_once",
    "user_broken_process_pool_once",
    "user_terminated_worker_once",
]


def _joblib_process_executor_module() -> ModuleType:
    return import_module("joblib.externals.loky.process_executor")


def _joblib_process_executor_exception_type(
    name: Literal["BrokenProcessPool", "TerminatedWorkerError"],
) -> type[BaseException]:
    candidate: object = getattr(_joblib_process_executor_module(), name, None)
    if not isinstance(candidate, type) or not issubclass(candidate, BaseException):
        pytest.fail(f"joblib process executor exception is unavailable: {name}")
    return candidate


def _make_loky_process_executor_exception(
    name: Literal["BrokenProcessPool", "TerminatedWorkerError"],
) -> BaseException:
    exception_type = _joblib_process_executor_exception_type(name)
    return exception_type(f"synthetic loky {name}")


def _is_int_observation_record(
    record: RequestAlignedEvaluationRecord,
) -> TypeGuard[Observation[int]]:
    return isinstance(record, Observation)


def _make_observation_outcome(
    *,
    problem: Problem[int, int, Observation[int]],
    request: EvaluationRequest[int],
) -> EvaluationOutcome[int, Observation[int]]:
    outcome = evaluate_request_outcome(problem=problem, request=request)
    record = outcome.record
    if not _is_int_observation_record(record):
        msg = "test fixture expected a scalar Observation compatibility record"
        raise TypeError(msg)
    return EvaluationOutcome(
        record=record,
        evaluation_count=outcome.evaluation_count,
        refinement=outcome.refinement,
    )


def _make_request_aligned_observation_outcome(
    *,
    problem: Problem[int, int, Observation[int]],
    request: EvaluationRequest[int],
) -> EvaluationOutcome[int, RequestAlignedEvaluationRecord]:
    outcome = _make_observation_outcome(problem=problem, request=request)
    return EvaluationOutcome(
        record=outcome.record,
        evaluation_count=outcome.evaluation_count,
        refinement=outcome.refinement,
    )


def _require_observation_records(
    records: Sequence[RequestAlignedEvaluationRecord],
) -> tuple[Observation[int], ...]:
    """Return scalar observation records from a broad record sequence."""
    observations: list[Observation[int]] = []
    for record in records:
        if not _is_int_observation_record(record):
            msg = "test fixture expected scalar Observation records"
            raise TypeError(msg)
        observations.append(record)
    return tuple(observations)


def _successful_observation_records(
    attempts: EvaluationAttemptBatch[
        AttemptCandidateT,
        RequestAlignedEvaluationRecord,
    ],
) -> tuple[Observation[int], ...]:
    """Project successful scalar attempts into observation compatibility records."""
    records: list[RequestAlignedEvaluationRecord] = []
    for success in attempts.successes:
        records.append(success.scalar_observation())
    return _require_observation_records(records)


@dataclass(frozen=True, slots=True)
class _LegacyObjectiveProtocol(EvaluationProtocol[int, Observation[int]]):
    """Record-returning protocol for tests that still target legacy evaluators."""

    objective: Objective[int]

    @override
    def evaluate_request(self, request: EvaluationRequest[int]) -> Observation[int]:
        value = self.objective.evaluate(request.candidate)
        return Observation.from_objective_value(
            request=request,
            candidate=request.candidate,
            value=value,
            direction=OptimizationDirection.MINIMIZE,
        )


@dataclass(frozen=True, slots=True)
class _LegacyRequestRecordProtocol(
    EvaluationProtocol[int, RequestAlignedEvaluationRecord]
):
    """Record-returning protocol exposed through the broad record contract."""

    objective: Objective[int]

    @override
    def evaluate_request(
        self,
        request: EvaluationRequest[int],
    ) -> RequestAlignedEvaluationRecord:
        value = self.objective.evaluate(request.candidate)
        return Observation.from_objective_value(
            request=request,
            candidate=request.candidate,
            value=value,
            direction=OptimizationDirection.MINIMIZE,
        )


def _legacy_observation_problem(
    objective: Objective[int],
) -> Problem[int, int, Observation[int]]:
    """Build a record-bound problem for pre-migration evaluator contracts."""
    return Problem(
        space=IntegerSpace(low=0, high=10),
        evaluation_protocol=_LegacyObjectiveProtocol(objective=objective),
    )


def _legacy_request_record_problem(
    objective: Objective[int],
) -> Problem[int, int, RequestAlignedEvaluationRecord]:
    """Build a record-bound problem with the broad request-aligned contract."""
    return Problem(
        space=IntegerSpace(low=0, high=10),
        evaluation_protocol=_LegacyRequestRecordProtocol(objective=objective),
    )


def _require_pending_aware_session(
    session: EvaluationBatchSession[SessionEvaluationT],
) -> PendingAwareBatchSession[SessionEvaluationT]:
    if not isinstance(session, PendingAwareBatchSession):
        pytest.fail("async joblib session should be pending-aware")
    return session


def _require_resumable_session(
    session: EvaluationBatchSession[SessionEvaluationT],
) -> ResumableBatchSession[SessionEvaluationT]:
    if not isinstance(session, ResumableBatchSession):
        pytest.fail("async joblib session should be resumable")
    return session


class ImmediateBaseSession(EvaluationBatchSession[int]):
    """Concrete base-session fixture for default wait contract tests."""

    _handle: EvaluationBatchHandle
    _completion_groups: tuple[CompletionGroup[int], ...]
    poll_count: int
    cancel_count: int

    def __init__(
        self,
        *,
        completion_groups: tuple[CompletionGroup[int], ...],
    ) -> None:
        self._handle = EvaluationBatchHandle(batch_id="base-session", request_count=1)
        self._completion_groups = completion_groups
        self.poll_count = 0
        self.cancel_count = 0

    @property
    @override
    def handle(self) -> EvaluationBatchHandle:
        return self._handle

    @override
    def poll(self) -> tuple[CompletionGroup[int], ...]:
        self.poll_count += 1
        return self._completion_groups

    @override
    def cancel(self) -> None:
        self.cancel_count += 1


class SquareObjective(Objective[int]):
    """Toy objective used to test evaluator behavior."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


class DelayedSquareObjective(Objective[int]):
    """Objective whose runtime varies by candidate."""

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 4:
            time.sleep(0.05)
        return float(candidate * candidate)


class SlowSquareObjective(Objective[int]):
    """Objective that makes every candidate slow enough to test polling."""

    @override
    def evaluate(self, candidate: int) -> float:
        time.sleep(0.05)
        return float(candidate * candidate)


class ExplodingObjective(Objective[int]):
    """Objective that raises for one designated candidate."""

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 4:
            msg = "boom"
            raise ValueError(msg)
        return float(candidate * candidate)


class DelayedExplodingObjective(Objective[int]):
    """Objective that delays before raising for one designated candidate."""

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 4:
            time.sleep(0.05)
            msg = "boom"
            raise ValueError(msg)
        return float(candidate * candidate)


@final
class FlakyAsyncJoblibEvaluator(AsyncJoblibEvaluator[int, int, Observation[int]]):
    """Async evaluator that injects deterministic failures between completions."""

    def __init__(
        self,
        *,
        failure_mode: InjectedAsyncJoblibFailureMode,
        infrastructure_retry_limit: int,
    ) -> None:
        super().__init__(
            backend="threading",
            n_jobs=2,
            infrastructure_retry_limit=infrastructure_retry_limit,
        )
        self.failure_mode = failure_mode
        self.attempt_count = 0

    def _build_outcome(
        self,
        problem: Problem[int, int, Observation[int]],
        request: EvaluationRequest[int],
    ) -> EvaluationOutcome[int, RequestAlignedEvaluationRecord]:
        return _make_request_aligned_observation_outcome(
            problem=problem,
            request=request,
        )

    @override
    def _start_attempt(
        self,
        *,
        problem: Problem[int, int, Observation[int]],
        request_inputs: Sequence[AsyncJoblibRequestInput[int]],
        execution_resources: ExecutionResources,
    ) -> Generator[
        tuple[int, EvaluationOutcome[int, RequestAlignedEvaluationRecord]],
        None,
        None,
    ]:
        _ = execution_resources
        self.attempt_count += 1
        attempt_count = self.attempt_count
        immutable_request_inputs = tuple(request_inputs)

        def first_completion() -> tuple[
            int,
            EvaluationOutcome[int, RequestAlignedEvaluationRecord],
        ]:
            first_input = immutable_request_inputs[0]
            return first_input.index, self._build_outcome(
                problem,
                first_input.request,
            )

        def iterator():
            if self.failure_mode == "infrastructure_once" and attempt_count == 1:
                yield first_completion()
                raise FuturesBrokenProcessPool("transient infrastructure failure")

            if self.failure_mode == "infrastructure_always":
                if attempt_count == 1 and len(immutable_request_inputs) > 1:
                    yield first_completion()
                raise FuturesBrokenProcessPool("persistent infrastructure failure")

            if self.failure_mode == "keyboard_interrupt_once" and attempt_count == 1:
                yield first_completion()
                raise KeyboardInterrupt

            if self.failure_mode == "user_code_once" and attempt_count == 1:
                yield first_completion()
                raise ValueError("boom")

            if self.failure_mode == "user_timeout_error_once" and attempt_count == 1:
                yield first_completion()
                raise TimeoutError("user timeout")

            if self.failure_mode == "loky_broken_process_pool_once" and attempt_count == 1:
                yield first_completion()
                raise _make_loky_process_executor_exception("BrokenProcessPool")

            if self.failure_mode == "loky_terminated_worker_once" and attempt_count == 1:
                yield first_completion()
                raise _make_loky_process_executor_exception("TerminatedWorkerError")

            if self.failure_mode == "futures_broken_process_pool_once" and attempt_count == 1:
                yield first_completion()
                raise FuturesBrokenProcessPool("synthetic futures BrokenProcessPool")

            if self.failure_mode == "user_broken_process_pool_once" and attempt_count == 1:
                class BrokenProcessPool(RuntimeError):
                    """User exception with a backend-like class name."""

                yield first_completion()
                raise BrokenProcessPool("user boom")

            if self.failure_mode == "user_terminated_worker_once" and attempt_count == 1:
                class TerminatedWorkerError(RuntimeError):
                    """User exception with a backend-like class name."""

                yield first_completion()
                raise TerminatedWorkerError("user boom")

            for request_input in immutable_request_inputs:
                yield request_input.index, self._build_outcome(
                    problem,
                    request_input.request,
                )

        return iterator()


@final
class ResumeStartFailureAsyncJoblibEvaluator(
    AsyncJoblibEvaluator[int, int, Observation[int]]
):
    """Async evaluator that can fail exactly one resume stream startup."""

    def __init__(self) -> None:
        super().__init__(backend="threading", n_jobs=2)
        self._fail_next_batch_start = False
        self._fail_next_attempt_start = False
        self._batch_start_barrier: tuple[Event, Event] | None = None
        self._attempt_start_barrier: tuple[Event, Event] | None = None
        self.batch_start_count = 0
        self.attempt_start_count = 0

    def fail_next_batch_start(self) -> None:
        """Make the next standard active-batch startup fail."""
        self._fail_next_batch_start = True

    def fail_next_attempt_start(self) -> None:
        """Make the next attempt-batch startup fail."""
        self._fail_next_attempt_start = True

    def block_next_batch_start(self, *, entered: Event, release: Event) -> None:
        """Block the next standard batch startup at the resume claim boundary."""
        self._batch_start_barrier = (entered, release)

    def block_next_attempt_start(self, *, entered: Event, release: Event) -> None:
        """Block the next attempt batch startup at the resume claim boundary."""
        self._attempt_start_barrier = (entered, release)

    def install_suspended_batch(
        self,
        *,
        problem: Problem[int, int, Observation[int]],
        requests: tuple[EvaluationRequest[int], ...],
        completed_indices: set[int],
    ) -> EvaluationBatchResumeHandle:
        """Install a suspended standard batch without starting joblib work."""
        handle = EvaluationBatchResumeHandle(
            batch_id=f"joblib-{next(self._batch_counter)}",
            request_count=len(requests),
            completed_count=len(completed_indices),
        )
        self._suspended_batches[handle.batch_id] = SuspendedAsyncJoblibBatch(
            problem=problem,
            request_inputs=_request_inputs(requests),
            execution_resources=self.execution_resources(),
            completed_indices=set(completed_indices),
        )
        return handle

    def install_suspended_attempt_batch(
        self,
        *,
        problem: Problem[int, int, Observation[int]],
        requests: tuple[EvaluationRequest[int], ...],
        completed_indices: set[int],
    ) -> EvaluationBatchResumeHandle:
        """Install a suspended attempt batch without starting joblib work."""
        handle = EvaluationBatchResumeHandle(
            batch_id=f"joblib-attempt-{next(self._batch_counter)}",
            request_count=len(requests),
            completed_count=len(completed_indices),
        )
        self._suspended_attempt_batches[handle.batch_id] = SuspendedAsyncJoblibBatch(
            problem=problem,
            request_inputs=_request_inputs(requests),
            execution_resources=self.execution_resources(),
            completed_indices=set(completed_indices),
        )
        return handle

    @override
    def _start_active_batch(
        self,
        *,
        problem: Problem[int, int, Observation[int]],
        request_inputs: tuple[AsyncJoblibRequestInput[int], ...],
        execution_resources: ExecutionResources,
        completed_indices: set[int] | None = None,
        infrastructure_retry_count: int = 0,
        attempt_inputs: Sequence[AsyncJoblibRequestInput[int]] | None = None,
    ) -> ActiveAsyncJoblibBatch[
        int,
        int,
        EvaluationOutcome[int, RequestAlignedEvaluationRecord],
        Observation[int],
    ]:
        self.batch_start_count += 1
        if self.batch_start_count > 1 and self._batch_start_barrier is not None:
            msg = "duplicate batch resume startup"
            raise AssertionError(msg)

        if self._batch_start_barrier is not None:
            entered, release = self._batch_start_barrier
            entered.set()
            if not release.wait(timeout=5.0):
                msg = "timed out waiting to release batch startup"
                raise RuntimeError(msg)
            self._batch_start_barrier = None

        if self._fail_next_batch_start:
            self._fail_next_batch_start = False
            msg = "forced resume startup failure"
            raise RuntimeError(msg)

        return super()._start_active_batch(
            problem=problem,
            request_inputs=request_inputs,
            execution_resources=execution_resources,
            completed_indices=completed_indices,
            infrastructure_retry_count=infrastructure_retry_count,
            attempt_inputs=attempt_inputs,
        )

    @override
    def _start_active_attempt_batch(
        self,
        *,
        problem: Problem[int, int, Observation[int]],
        request_inputs: tuple[AsyncJoblibRequestInput[int], ...],
        execution_resources: ExecutionResources,
        completed_indices: set[int] | None = None,
        infrastructure_retry_count: int = 0,
        attempt_inputs: Sequence[AsyncJoblibRequestInput[int]] | None = None,
    ) -> ActiveAsyncJoblibBatch[
        int,
        int,
        EvaluationAttemptBatch[int, Observation[int]],
        Observation[int],
    ]:
        self.attempt_start_count += 1
        if self.attempt_start_count > 1 and self._attempt_start_barrier is not None:
            msg = "duplicate attempt resume startup"
            raise AssertionError(msg)

        if self._attempt_start_barrier is not None:
            entered, release = self._attempt_start_barrier
            entered.set()
            if not release.wait(timeout=5.0):
                msg = "timed out waiting to release attempt startup"
                raise RuntimeError(msg)
            self._attempt_start_barrier = None

        if self._fail_next_attempt_start:
            self._fail_next_attempt_start = False
            msg = "forced attempt resume startup failure"
            raise RuntimeError(msg)

        return super()._start_active_attempt_batch(
            problem=problem,
            request_inputs=request_inputs,
            execution_resources=execution_resources,
            completed_indices=completed_indices,
            infrastructure_retry_count=infrastructure_retry_count,
            attempt_inputs=attempt_inputs,
        )


class AbortInspectionAsyncJoblibEvaluator(
    AsyncJoblibEvaluator[int, int, Observation[int]]
):
    """Test evaluator exposing active-batch inspection without private test access."""

    def install_active_batch(
        self,
        handle: EvaluationBatchHandle,
        active_batch: ActiveAsyncJoblibBatch[
            int,
            int,
            EvaluationOutcome[int, RequestAlignedEvaluationRecord],
            Observation[int],
        ],
    ) -> None:
        self._active_batches[handle.batch_id] = active_batch

    def has_active_batch(self, handle: EvaluationBatchHandle) -> bool:
        return handle.batch_id in self._active_batches

    def active_batch_for(
        self,
        handle: EvaluationBatchHandle,
    ) -> ActiveAsyncJoblibBatch[
        int,
        int,
        EvaluationOutcome[int, RequestAlignedEvaluationRecord],
        Observation[int],
    ]:
        return self._active_batches[handle.batch_id]

    def install_active_attempt_batch(
        self,
        handle: EvaluationBatchHandle,
        active_batch: ActiveAsyncJoblibBatch[
            int,
            int,
            EvaluationAttemptBatch[int, Observation[int]],
            Observation[int],
        ],
    ) -> None:
        self._active_attempt_batches[handle.batch_id] = active_batch

    def has_active_attempt_batch(self, handle: EvaluationBatchHandle) -> bool:
        return handle.batch_id in self._active_attempt_batches

    def active_attempt_batch_for(
        self,
        handle: EvaluationBatchHandle,
    ) -> ActiveAsyncJoblibBatch[
        int,
        int,
        EvaluationAttemptBatch[int, Observation[int]],
        Observation[int],
    ]:
        return self._active_attempt_batches[handle.batch_id]

    def replace_active_batch_attempt(
        self,
        *,
        active_batch: ActiveAsyncJoblibBatch[
            int,
            int,
            EvaluationOutcome[int, RequestAlignedEvaluationRecord],
            Observation[int],
        ],
        request_inputs: Sequence[AsyncJoblibRequestInput[int]],
    ) -> None:
        self._replace_active_batch_attempt(
            active_batch=active_batch,
            request_inputs=request_inputs,
        )

    def replace_active_attempt_batch_attempt(
        self,
        *,
        active_batch: ActiveAsyncJoblibBatch[
            int,
            int,
            EvaluationAttemptBatch[int, Observation[int]],
            Observation[int],
        ],
        request_inputs: Sequence[AsyncJoblibRequestInput[int]],
    ) -> None:
        self._replace_active_attempt_batch_attempt(
            active_batch=active_batch,
            request_inputs=request_inputs,
        )


@final
class CancelAfterRetryDeclinedAsyncJoblibEvaluator(
    AbortInspectionAsyncJoblibEvaluator
):
    """Evaluator that cancels a batch after retry policy declines retry."""

    @override
    def _retry_infrastructure_failure(
        self,
        *,
        handle: EvaluationBatchHandle,
        active_batch: ActiveAsyncJoblibBatch[
            int,
            int,
            EvaluationOutcome[int, RequestAlignedEvaluationRecord],
            Observation[int],
        ],
        cause: BaseException,
    ) -> bool:
        should_retry = super()._retry_infrastructure_failure(
            handle=handle,
            active_batch=active_batch,
            cause=cause,
        )
        if not should_retry:
            self.cancel(handle)
        return should_retry

    @override
    def _retry_attempt_infrastructure_failure(
        self,
        *,
        handle: EvaluationBatchHandle,
        active_batch: ActiveAsyncJoblibBatch[
            int,
            int,
            EvaluationAttemptBatch[int, Observation[int]],
            Observation[int],
        ],
        cause: BaseException,
    ) -> bool:
        should_retry = super()._retry_attempt_infrastructure_failure(
            handle=handle,
            active_batch=active_batch,
            cause=cause,
        )
        if not should_retry:
            self.cancel_attempt_batch(handle)
        return should_retry


@dataclass(slots=True)
class AbortEventLog:
    """Mutable event log for abort-order assertions."""

    events: list[str]


@dataclass(slots=True)
class WaitThreadCapture(Generic[SessionEvaluationT]):
    """Captured result or exception from one waiter thread."""

    completion_groups: tuple[CompletionGroup[SessionEvaluationT], ...] | None = None
    exception: BaseException | None = None


@final
class SignalingQueue(Queue[QueuedEventT], Generic[QueuedEventT]):
    """Queue fixture that signals when a waiter starts a blocking read."""

    def __init__(self) -> None:
        super().__init__()
        self.entered_get = Event()

    @override
    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> QueuedEventT:
        """Signal the blocking read before delegating to ``Queue``."""
        self.entered_get.set()
        return super().get(block=block, timeout=timeout)


def _requests(
    proposals: Sequence[Proposal[int]],
) -> tuple[EvaluationRequest[int], ...]:
    """Lower proposal fixtures into canonical evaluation requests."""
    return tuple(
        EvaluationRequest(proposal=proposal)
        for proposal in proposals
    )


def _request_inputs(
    requests: Sequence[EvaluationRequest[int]],
) -> tuple[AsyncJoblibRequestInput[int], ...]:
    return tuple(
        AsyncJoblibRequestInput(index=index, request=request)
        for index, request in enumerate(requests)
    )


def _active_observation_joblib_batch(
    *,
    problem: Problem[int, int, Observation[int]],
    request_inputs: tuple[AsyncJoblibRequestInput[int], ...],
    result_generator: Generator[
        tuple[int, EvaluationOutcome[int, RequestAlignedEvaluationRecord]],
        None,
        None,
    ],
    abort_attempt: Callable[[], None] | None = None,
) -> ActiveAsyncJoblibBatch[
    int,
    int,
    EvaluationOutcome[int, RequestAlignedEvaluationRecord],
    Observation[int],
]:
    result_queue: Queue[
        AsyncJoblibCompletedResult[
            EvaluationOutcome[int, RequestAlignedEvaluationRecord]
        ]
        | AsyncJoblibFailedResult
        | AsyncJoblibExhaustedResult
    ] = Queue()
    return ActiveAsyncJoblibBatch(
        problem=problem,
        request_inputs=request_inputs,
        execution_resources=ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
        ),
        result_generator=result_generator,
        result_queue=result_queue,
        result_worker=current_thread(),
        abort_attempt=abort_attempt,
    )


def _active_attempt_joblib_batch(
    *,
    problem: Problem[int, int, Observation[int]],
    request_inputs: tuple[AsyncJoblibRequestInput[int], ...],
    result_generator: Generator[
        tuple[int, EvaluationAttemptBatch[int, Observation[int]]],
        None,
        None,
    ],
    abort_attempt: Callable[[], None] | None = None,
) -> ActiveAsyncJoblibBatch[
    int,
    int,
    EvaluationAttemptBatch[int, Observation[int]],
    Observation[int],
]:
    result_queue: Queue[
        AsyncJoblibCompletedResult[EvaluationAttemptBatch[int, Observation[int]]]
        | AsyncJoblibFailedResult
        | AsyncJoblibExhaustedResult
    ] = Queue()
    return ActiveAsyncJoblibBatch(
        problem=problem,
        request_inputs=request_inputs,
        execution_resources=ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
        ),
        result_generator=result_generator,
        result_queue=result_queue,
        result_worker=current_thread(),
        abort_attempt=abort_attempt,
    )


class JoblibEvaluatorTests:
    """Tests for the synchronous joblib evaluator."""

    def test_rejects_zero_n_jobs(self) -> None:
        with pytest.raises(ValueError):
            _ = JoblibEvaluator[int, int, Observation[int]](n_jobs=0)

    def test_preserves_input_proposal_order(self) -> None:
        problem = _legacy_observation_problem(DelayedSquareObjective())
        evaluator = JoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)

        outcomes = evaluator.evaluate(
            problem,
            _requests(
                (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
                )
            ),
        )

        assert tuple(outcome.observation.proposal.proposal_id for outcome in outcomes) == ("p-1", "p-2")
        assert tuple(outcome.observation.value for outcome in outcomes) == (16.0, 1.0)

    def test_loky_backend_evaluates_picklable_problem(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        evaluator = JoblibEvaluator[int, int, Observation[int]](backend="loky", n_jobs=2)

        outcomes = evaluator.evaluate(
            problem,
            _requests((Proposal(candidate=4, proposal_id="p-1"),)),
        )

        assert tuple(outcome.observation.value for outcome in outcomes) == (16.0,)

    def test_loky_backend_evaluate_attempts_records_pickled_failure(self) -> None:
        problem = _legacy_observation_problem(ExplodingObjective())
        evaluator = JoblibEvaluator[int, int, Observation[int]](backend="loky", n_jobs=2)

        attempts = evaluator.evaluate_attempts(
            problem,
            _requests(
                (
                    Proposal(candidate=1, proposal_id="p-1"),
                    Proposal(candidate=4, proposal_id="p-2"),
                )
            ),
        )

        assert attempts.success_indices == (0,)
        assert attempts.failure_indices == (1,)
        assert tuple(
            observation.value for observation in _successful_observation_records(attempts)
        ) == (
            1.0,
        )
        assert attempts.failures[0].proposal_id == "p-2"
        assert attempts.failures[0].exception.exception_type == "builtins.ValueError"
        assert attempts.failures[0].exception.message == "boom"

    def test_evaluate_attempts_records_user_failure_and_preserves_successes(
        self,
    ) -> None:
        problem = _legacy_observation_problem(ExplodingObjective())
        evaluator = JoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)

        attempts = evaluator.evaluate_attempts(
            problem,
            _requests(
                (
                    Proposal(candidate=1, proposal_id="p-1"),
                    Proposal(candidate=4, proposal_id="p-2"),
                    Proposal(candidate=2, proposal_id="p-3"),
                )
            ),
        )

        assert attempts.success_indices == (0, 2)
        assert attempts.failure_indices == (1,)
        assert tuple(
            observation.value for observation in _successful_observation_records(attempts)
        ) == (
            1.0,
            4.0,
        )
        failure = attempts.failures[0]
        assert failure.proposal_id == "p-2"
        assert failure.exception.exception_type == "builtins.ValueError"
        assert failure.exception.message == "boom"

    def test_evaluate_attempts_support_all_failure_batch(self) -> None:
        problem = _legacy_observation_problem(ExplodingObjective())
        evaluator = JoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)

        attempts = evaluator.evaluate_attempts(
            problem,
            _requests(
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=4, proposal_id="p-2"),
                )
            ),
        )

        assert attempts.successes == ()
        assert attempts.success_indices == ()
        assert attempts.failure_indices == (0, 1)
        assert tuple(failure.proposal_id for failure in attempts.failures) == (
            "p-1",
            "p-2",
        )
        assert attempts.evaluation_count == 2

    def test_evaluate_attempts_support_empty_batch(self) -> None:
        problem = _legacy_observation_problem(ExplodingObjective())
        evaluator = JoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)

        attempts = evaluator.evaluate_attempts(problem, ())

        assert attempts.requests == ()
        assert attempts.successes == ()
        assert attempts.failures == ()
        assert attempts.evaluation_count == 0

    def test_evaluate_attempts_does_not_record_validation_failure(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        evaluator = JoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)

        with pytest.raises(ValueError):
            _ = evaluator.evaluate_attempts(
                problem,
                _requests((Proposal(candidate=11, proposal_id="p-1"),)),
            )

    def test_execution_resources_report_evaluator_ownership(self) -> None:
        evaluator = JoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)

        assert evaluator.execution_resources() == ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=2,
                owner_backend="threading",
            )


class AsyncJoblibEvaluatorTests:
    """Tests for the async joblib evaluator."""

    @pytest.mark.parametrize(
        ("timeout", "expected_error"),
        (
            (cast(float, True), TypeError),
            (cast(float, cast(object, np.bool_(False))), TypeError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (-0.001, ValueError),
        ),
    )
    def test_base_session_wait_rejects_non_canonical_timeout(
        self,
        timeout: float,
        expected_error: type[Exception],
    ) -> None:
        session = ImmediateBaseSession(
            completion_groups=(CompletionGroup(start_index=0, outcomes=(1,)),),
        )

        with pytest.raises(expected_error, match="timeout must"):
            _ = session.wait(timeout=timeout)

        assert session.poll_count == 0

    def test_base_session_wait_preserves_none_timeout(self) -> None:
        session = ImmediateBaseSession(
            completion_groups=(CompletionGroup(start_index=0, outcomes=(1,)),),
        )

        completion_groups = tuple(session.wait(timeout=None))

        assert completion_groups == (CompletionGroup(start_index=0, outcomes=(1,)),)
        assert session.poll_count == 1

    def test_base_session_wait_accepts_canonical_numeric_timeout(self) -> None:
        session = ImmediateBaseSession(
            completion_groups=(CompletionGroup(start_index=0, outcomes=(1,)),),
        )

        completion_groups = tuple(
            session.wait(timeout=cast(float, cast(object, np.float64(0.0)))),
        )

        assert completion_groups == (CompletionGroup(start_index=0, outcomes=(1,)),)
        assert session.poll_count == 1

    def test_rejects_negative_infrastructure_retry_limit(self) -> None:
        with pytest.raises(ValueError):
            _ = AsyncJoblibEvaluator[int, int, Observation[int]](infrastructure_retry_limit=-1)

    def test_pickle_round_trip_preserves_config_and_rebuilds_runtime_state(
        self,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](
            backend="threading",
            n_jobs=1,
            infrastructure_retry_limit=2,
        )

        restored = cast(
            AsyncJoblibEvaluator[int, int, Observation[int]],
            pickle.loads(pickle.dumps(evaluator)),
        )

        assert restored.backend == "threading"
        assert restored.n_jobs == 1
        assert restored.infrastructure_retry_limit == 2
        session = restored.open_session(
            problem,
            _requests((Proposal(candidate=3, proposal_id="p-1"),)),
        )
        pending_session = _require_pending_aware_session(session)

        completion_groups = tuple(pending_session.wait(timeout=5.0))

        assert len(completion_groups) == 1
        assert completion_groups[0].outcomes[0].observation.value == 9.0

    def test_evaluate_attempts_records_user_failure_without_retrying(
        self,
    ) -> None:
        problem = _legacy_observation_problem(ExplodingObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](
            backend="threading",
            n_jobs=2,
            infrastructure_retry_limit=3,
        )

        attempts = evaluator.evaluate_attempts(
            problem,
            _requests(
                (
                    Proposal(candidate=1, proposal_id="p-1"),
                    Proposal(candidate=4, proposal_id="p-2"),
                    Proposal(candidate=2, proposal_id="p-3"),
                )
            ),
        )

        assert attempts.success_indices == (0, 2)
        assert attempts.failure_indices == (1,)
        assert tuple(
            observation.value for observation in _successful_observation_records(attempts)
        ) == (
            1.0,
            4.0,
        )
        assert attempts.failures[0].proposal_id == "p-2"
        assert attempts.failures[0].exception.exception_type == "builtins.ValueError"

    def test_evaluate_attempts_support_empty_batch(self) -> None:
        problem = _legacy_observation_problem(ExplodingObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)

        attempts = evaluator.evaluate_attempts(problem, ())

        assert attempts.requests == ()
        assert attempts.successes == ()
        assert attempts.failures == ()
        assert attempts.evaluation_count == 0

    def test_evaluate_preserves_input_proposal_order(self) -> None:
        problem = _legacy_request_record_problem(DelayedSquareObjective())
        evaluator = AsyncJoblibEvaluator[
            int,
            int,
            RequestAlignedEvaluationRecord,
        ](backend="threading", n_jobs=2)

        outcomes = evaluator.evaluate(
            problem,
            _requests(
                (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
                )
            ),
        )

        assert tuple(outcome.observation.proposal.proposal_id for outcome in outcomes) == ("p-1", "p-2")
        assert tuple(outcome.observation.value for outcome in outcomes) == (16.0, 1.0)

    def test_loky_backend_evaluates_picklable_problem(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend="loky", n_jobs=2)

        outcomes = evaluator.evaluate(
            problem,
            _requests((Proposal(candidate=4, proposal_id="p-1"),)),
        )

        assert tuple(outcome.observation.value for outcome in outcomes) == (16.0,)

    def test_execution_resources_report_evaluator_ownership(self) -> None:
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)

        assert evaluator.execution_resources() == ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=2,
                owner_backend="threading",
            )

    @pytest.mark.parametrize("backend", ("threading", "loky"))
    def test_poll_returns_empty_without_blocking_when_no_result_is_ready(
        self,
        backend: Literal["threading", "loky"],
    ) -> None:
        problem = _legacy_observation_problem(SlowSquareObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend=backend, n_jobs=2)
        session = evaluator.open_session(
            problem,
            _requests((Proposal(candidate=4, proposal_id="p-1"),)),
        )

        start_time = time.monotonic()
        completion_groups = tuple(session.poll())

        assert completion_groups == ()
        assert time.monotonic() - start_time < 0.03
        assert len(tuple(session.wait(timeout=5.0))) == 1

    def test_wait_returns_empty_when_timeout_expires_before_result_is_ready(
        self,
    ) -> None:
        problem = _legacy_observation_problem(SlowSquareObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)
        session = evaluator.open_session(
            problem,
            _requests((Proposal(candidate=4, proposal_id="p-1"),)),
        )

        assert tuple(session.wait(timeout=0.001)) == ()

        session.cancel()

    def test_wait_rejects_negative_timeout(self) -> None:
        problem = _legacy_observation_problem(SlowSquareObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)
        session = evaluator.open_session(
            problem,
            _requests((Proposal(candidate=4, proposal_id="p-1"),)),
        )

        with pytest.raises(ValueError, match="timeout must be non-negative"):
            _ = session.wait(timeout=-0.001)

        session.cancel()

    @pytest.mark.parametrize(
        "timeout",
        (
            cast(float, True),
            cast(float, cast(object, float("nan"))),
            cast(float, cast(object, float("inf"))),
        ),
    )
    def test_wait_rejects_non_canonical_timeout(self, timeout: float) -> None:
        problem = _legacy_observation_problem(SlowSquareObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)
        session = evaluator.open_session(
            problem,
            _requests((Proposal(candidate=4, proposal_id="p-1"),)),
        )

        expected_error: type[Exception] = TypeError if type(timeout) is bool else ValueError
        with pytest.raises(expected_error, match="timeout must"):
            _ = session.wait(timeout=timeout)

        session.cancel()

    @pytest.mark.parametrize(
        "timeout",
        (
            cast(float, False),
            cast(float, cast(object, float("-inf"))),
        ),
    )
    def test_attempt_session_wait_rejects_non_canonical_timeout(
        self,
        timeout: float,
    ) -> None:
        problem = _legacy_observation_problem(SlowSquareObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)
        session = evaluator.open_attempt_session(
            problem,
            _requests((Proposal(candidate=4, proposal_id="p-1"),)),
        )

        expected_error: type[Exception] = TypeError if type(timeout) is bool else ValueError
        with pytest.raises(expected_error, match="timeout must"):
            _ = session.wait(timeout=timeout)

        session.cancel()

    def test_poll_wraps_terminal_failure(self) -> None:
        problem = _legacy_observation_problem(ExplodingObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)
        handle = evaluator.submit_batch(
            problem,
            _requests(
                (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
                )
            ),
        )

        with pytest.raises(BatchExecutionFailed) as caught:
            while True:
                _ = evaluator.wait(handle)

        assert caught.value.kind == "user_code"
        assert isinstance(caught.value.cause, ValueError)

    def test_open_session_exposes_exact_async_batch_lifecycle(self) -> None:
        problem = _legacy_request_record_problem(DelayedSquareObjective())
        evaluator = AsyncJoblibEvaluator[
            int,
            int,
            RequestAlignedEvaluationRecord,
        ](backend="threading", n_jobs=2)
        session = evaluator.open_session(
            problem,
            _requests(
                (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
                )
            ),
        )

        ordered_outcomes: list[
            EvaluationOutcome[int, RequestAlignedEvaluationRecord] | None
        ] = [None] * session.handle.request_count
        completed_count = 0
        while completed_count < session.handle.request_count:
            completion_groups = tuple(session.wait())
            for completion_group in completion_groups:
                for offset, outcome in enumerate(completion_group.outcomes):
                    ordered_outcomes[completion_group.start_index + offset] = outcome
                    completed_count += 1

        ordered_values = tuple(
            outcome.observation.value
            for outcome in ordered_outcomes
            if outcome is not None
        )
        assert ordered_values == (16.0, 1.0)

    def test_open_session_exposes_pending_aware_state(self) -> None:
        problem = _legacy_request_record_problem(DelayedSquareObjective())
        evaluator = AsyncJoblibEvaluator[
            int,
            int,
            RequestAlignedEvaluationRecord,
        ](backend="threading", n_jobs=2)
        session = evaluator.open_session(
            problem,
            _requests(
                (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
                )
            ),
        )

        pending_aware_session = _require_pending_aware_session(session)

        assert pending_aware_session.state() == EvaluationBatchSessionState(
                request_count=2,
                completed_count=0,
                pending_count=2,
                lifecycle="active",
        )

        _ = tuple(pending_aware_session.wait())
        after_first_poll = pending_aware_session.state()
        assert after_first_poll.completed_count == 1
        assert after_first_poll.pending_count == 1
        assert after_first_poll.lifecycle == "active"

        _ = tuple(pending_aware_session.wait())
        assert pending_aware_session.state() == EvaluationBatchSessionState(
                request_count=2,
                completed_count=2,
                pending_count=0,
                lifecycle="completed",
            )

    def test_pending_aware_state_tracks_cancelled_session(self) -> None:
        problem = _legacy_observation_problem(DelayedSquareObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)
        session = evaluator.open_session(
            problem,
            _requests(
                (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
                )
            ),
        )
        pending_aware_session = _require_pending_aware_session(session)

        pending_aware_session.cancel()

        assert pending_aware_session.state() == EvaluationBatchSessionState(
                request_count=2,
                completed_count=0,
                pending_count=2,
                lifecycle="cancelled",
            )

    def test_cancel_uses_private_abort_hook_when_available(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        broad_outcome = _make_request_aligned_observation_outcome(
            problem=problem,
            request=requests[0],
        )
        generator_closed = False
        abort_called = False

        def result_generator() -> Generator[
            tuple[int, EvaluationOutcome[int, RequestAlignedEvaluationRecord]],
            None,
            None,
        ]:
            nonlocal generator_closed
            try:
                yield 0, broad_outcome
            finally:
                generator_closed = True

        def abort_attempt() -> None:
            nonlocal abort_called
            abort_called = True

        generator = result_generator()
        _ = next(generator)
        evaluator = AbortInspectionAsyncJoblibEvaluator()
        handle = EvaluationBatchHandle(batch_id="joblib-test", request_count=1)
        evaluator.install_active_batch(
            handle,
            _active_observation_joblib_batch(
                problem=problem,
                request_inputs=_request_inputs(requests),
                result_generator=generator,
                abort_attempt=abort_attempt,
            ),
        )

        evaluator.cancel(handle)

        assert abort_called
        assert not generator_closed
        assert not evaluator.has_active_batch(handle)

    def test_cancel_falls_back_to_generator_close_when_abort_hook_fails(
        self,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        broad_outcome = _make_request_aligned_observation_outcome(
            problem=problem,
            request=requests[0],
        )
        generator_closed = False

        def result_generator() -> Generator[
            tuple[int, EvaluationOutcome[int, RequestAlignedEvaluationRecord]],
            None,
            None,
        ]:
            nonlocal generator_closed
            try:
                yield 0, broad_outcome
            finally:
                generator_closed = True

        def abort_attempt() -> None:
            msg = "abort hook failed"
            raise RuntimeError(msg)

        generator = result_generator()
        _ = next(generator)
        evaluator = AbortInspectionAsyncJoblibEvaluator()
        handle = EvaluationBatchHandle(batch_id="joblib-test", request_count=1)
        evaluator.install_active_batch(
            handle,
            _active_observation_joblib_batch(
                problem=problem,
                request_inputs=_request_inputs(requests),
                result_generator=generator,
                abort_attempt=abort_attempt,
            ),
        )

        with pytest.warns(
            RuntimeWarning,
            match="fallback generator close was used",
        ):
            evaluator.cancel(handle)

        assert generator_closed
        assert not evaluator.has_active_batch(handle)

    def test_cancel_falls_back_to_generator_close_without_private_abort_hook(
        self,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        broad_outcome = _make_request_aligned_observation_outcome(
            problem=problem,
            request=requests[0],
        )
        generator_closed = False

        def result_generator() -> Generator[
            tuple[int, EvaluationOutcome[int, RequestAlignedEvaluationRecord]],
            None,
            None,
        ]:
            nonlocal generator_closed
            try:
                yield 0, broad_outcome
            finally:
                generator_closed = True

        generator = result_generator()
        _ = next(generator)
        evaluator = AbortInspectionAsyncJoblibEvaluator()
        handle = EvaluationBatchHandle(batch_id="joblib-test", request_count=1)
        evaluator.install_active_batch(
            handle,
            _active_observation_joblib_batch(
                problem=problem,
                request_inputs=_request_inputs(requests),
                result_generator=generator,
            ),
        )

        evaluator.cancel(handle)

        assert generator_closed
        assert not evaluator.has_active_batch(handle)

    def test_cancel_warns_when_abort_fallback_close_fails(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        broad_outcome = _make_request_aligned_observation_outcome(
            problem=problem,
            request=requests[0],
        )

        def result_generator() -> Generator[
            tuple[int, EvaluationOutcome[int, RequestAlignedEvaluationRecord]],
            None,
            None,
        ]:
            try:
                yield 0, broad_outcome
            except GeneratorExit as exception:
                msg = "close failed"
                raise RuntimeError(msg) from exception

        generator = result_generator()
        _ = next(generator)
        evaluator = AbortInspectionAsyncJoblibEvaluator()
        handle = EvaluationBatchHandle(batch_id="joblib-test", request_count=1)
        evaluator.install_active_batch(
            handle,
            _active_observation_joblib_batch(
                problem=problem,
                request_inputs=_request_inputs(requests),
                result_generator=generator,
            ),
        )

        with pytest.warns(
            RuntimeWarning,
            match="failed to abort async joblib attempt",
        ):
            evaluator.cancel(handle)

        assert not evaluator.has_active_batch(handle)

    def test_replacement_aborts_old_stream_before_starting_new_attempt(
        self,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        request_inputs = _request_inputs(requests)
        old_outcome = _make_request_aligned_observation_outcome(
            problem=problem,
            request=requests[0],
        )
        event_log = AbortEventLog(events=[])

        def result_generator() -> Generator[
            tuple[int, EvaluationOutcome[int, RequestAlignedEvaluationRecord]],
            None,
            None,
        ]:
            yield 0, old_outcome

        old_generator = result_generator()
        _ = next(old_generator)
        active_batch = _active_observation_joblib_batch(
            problem=problem,
            request_inputs=request_inputs,
            result_generator=old_generator,
        )

        def abort_attempt() -> None:
            event_log.events.append("abort")
            assert active_batch.result_generator is old_generator

        active_batch.abort_attempt = abort_attempt
        evaluator = AbortInspectionAsyncJoblibEvaluator(backend="threading", n_jobs=1)

        evaluator.replace_active_batch_attempt(
            active_batch=active_batch,
            request_inputs=request_inputs,
        )

        assert event_log.events == ["abort"]
        assert active_batch.result_generator is not old_generator

    def test_replacement_falls_back_to_close_when_abort_hook_fails(
        self,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        request_inputs = _request_inputs(requests)
        old_outcome = _make_request_aligned_observation_outcome(
            problem=problem,
            request=requests[0],
        )
        event_log = AbortEventLog(events=[])

        def result_generator() -> Generator[
            tuple[int, EvaluationOutcome[int, RequestAlignedEvaluationRecord]],
            None,
            None,
        ]:
            try:
                yield 0, old_outcome
            finally:
                event_log.events.append("close")

        old_generator = result_generator()
        _ = next(old_generator)
        active_batch = _active_observation_joblib_batch(
            problem=problem,
            request_inputs=request_inputs,
            result_generator=old_generator,
        )

        def abort_attempt() -> None:
            event_log.events.append("abort")
            assert active_batch.result_generator is old_generator
            msg = "abort hook failed"
            raise RuntimeError(msg)

        active_batch.abort_attempt = abort_attempt
        evaluator = AbortInspectionAsyncJoblibEvaluator(backend="threading", n_jobs=1)

        with pytest.warns(
            RuntimeWarning,
            match="fallback generator close was used",
        ):
            evaluator.replace_active_batch_attempt(
                active_batch=active_batch,
                request_inputs=request_inputs,
            )

        assert event_log.events == ["abort", "close"]
        assert active_batch.result_generator is not old_generator

    def test_duplicate_completion_aborts_active_stream(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        request_inputs = _request_inputs(requests)
        outcome = _make_request_aligned_observation_outcome(
            problem=problem,
            request=requests[0],
        )
        event_log = AbortEventLog(events=[])

        def result_generator() -> Generator[
            tuple[int, EvaluationOutcome[int, RequestAlignedEvaluationRecord]],
            None,
            None,
        ]:
            yield 0, outcome

        active_batch = _active_observation_joblib_batch(
            problem=problem,
            request_inputs=request_inputs,
            result_generator=result_generator(),
            abort_attempt=lambda: event_log.events.append("abort"),
        )
        active_batch.completed_indices.add(0)
        active_batch.result_queue.put(
            AsyncJoblibCompletedResult(index=0, outcome=outcome),
        )
        handle = EvaluationBatchHandle(batch_id="joblib-duplicate", request_count=1)
        evaluator = AbortInspectionAsyncJoblibEvaluator(backend="threading", n_jobs=1)
        evaluator.install_active_batch(handle, active_batch)

        with pytest.raises(BatchExecutionFailed, match="infrastructure"):
            _ = evaluator.poll(handle)

        assert event_log.events == ["abort"]
        assert not evaluator.has_active_batch(handle)

    def test_attempt_replacement_aborts_old_stream_before_starting_new_attempt(
        self,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        request_inputs = _request_inputs(requests)
        observation = _make_observation_outcome(
            problem=problem,
            request=requests[0],
        ).record
        old_attempts: EvaluationAttemptBatch[int, Observation[int]] = (
            EvaluationAttemptBatch(
                attempts=(
                    EvaluationSuccess(
                        request=requests[0],
                        payload=observation,
                    ),
                ),
            )
        )
        event_log = AbortEventLog(events=[])

        def result_generator() -> Generator[
            tuple[int, EvaluationAttemptBatch[int, Observation[int]]],
            None,
            None,
        ]:
            yield 0, old_attempts

        old_generator = result_generator()
        _ = next(old_generator)
        active_batch = _active_attempt_joblib_batch(
            problem=problem,
            request_inputs=request_inputs,
            result_generator=old_generator,
        )

        def abort_attempt() -> None:
            event_log.events.append("abort")
            assert active_batch.result_generator is old_generator

        active_batch.abort_attempt = abort_attempt
        evaluator = AbortInspectionAsyncJoblibEvaluator(backend="threading", n_jobs=1)

        evaluator.replace_active_attempt_batch_attempt(
            active_batch=active_batch,
            request_inputs=request_inputs,
        )

        assert event_log.events == ["abort"]
        assert active_batch.result_generator is not old_generator

    def test_duplicate_attempt_completion_aborts_active_stream(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        request_inputs = _request_inputs(requests)
        observation = _make_observation_outcome(
            problem=problem,
            request=requests[0],
        ).record
        attempts: EvaluationAttemptBatch[int, Observation[int]] = (
            EvaluationAttemptBatch(
                attempts=(
                    EvaluationSuccess(
                        request=requests[0],
                        payload=observation,
                    ),
                ),
            )
        )
        event_log = AbortEventLog(events=[])

        def result_generator() -> Generator[
            tuple[int, EvaluationAttemptBatch[int, Observation[int]]],
            None,
            None,
        ]:
            yield 0, attempts

        active_batch = _active_attempt_joblib_batch(
            problem=problem,
            request_inputs=request_inputs,
            result_generator=result_generator(),
            abort_attempt=lambda: event_log.events.append("abort"),
        )
        active_batch.completed_indices.add(0)
        active_batch.result_queue.put(
            AsyncJoblibCompletedResult(index=0, outcome=attempts),
        )
        handle = EvaluationBatchHandle(
            batch_id="joblib-attempt-duplicate",
            request_count=1,
        )
        evaluator = AbortInspectionAsyncJoblibEvaluator(backend="threading", n_jobs=1)
        evaluator.install_active_attempt_batch(handle, active_batch)

        with pytest.raises(BatchExecutionFailed, match="infrastructure"):
            _ = evaluator.poll_attempts(handle)

        assert event_log.events == ["abort"]
        assert not evaluator.has_active_attempt_batch(handle)

    def test_wait_after_concurrent_cancel_does_not_retry_stale_batch(
        self,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        request_inputs = _request_inputs(requests)
        handle = EvaluationBatchHandle(batch_id="joblib-stale-cancel", request_count=1)
        result_queue = SignalingQueue[OutcomeJoblibQueueEvent]()

        def result_generator() -> Generator[
            tuple[int, EvaluationOutcome[int, RequestAlignedEvaluationRecord]],
            None,
            None,
        ]:
            yield from ()

        active_batch = _active_observation_joblib_batch(
            problem=problem,
            request_inputs=request_inputs,
            result_generator=result_generator(),
        )
        active_batch.result_queue = result_queue
        evaluator = AbortInspectionAsyncJoblibEvaluator(
            backend="threading",
            n_jobs=1,
            infrastructure_retry_limit=1,
        )
        evaluator.install_active_batch(handle, active_batch)
        capture: WaitThreadCapture[
            EvaluationOutcome[int, RequestAlignedEvaluationRecord]
        ] = WaitThreadCapture()

        def wait_for_batch() -> None:
            try:
                capture.completion_groups = evaluator.wait(handle, timeout=5.0)
            except BaseException as exception:
                capture.exception = exception

        waiter = Thread(target=wait_for_batch, name="stale-cancel-waiter")
        waiter.start()
        assert result_queue.entered_get.wait(timeout=5.0)

        evaluator.cancel(handle)
        result_queue.put(AsyncJoblibExhaustedResult())
        waiter.join(timeout=5.0)

        assert not waiter.is_alive()
        assert isinstance(capture.exception, BatchExecutionFailed)
        assert capture.exception.kind == "cancelled"
        assert capture.completion_groups is None
        assert active_batch.infrastructure_retry_count == 0
        assert not evaluator.has_active_batch(handle)

    def test_wait_after_concurrent_cancel_rejects_stale_completion(
        self,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        request_inputs = _request_inputs(requests)
        handle = EvaluationBatchHandle(
            batch_id="joblib-stale-cancel-completion",
            request_count=1,
        )
        result_queue = SignalingQueue[OutcomeJoblibQueueEvent]()
        stale_outcome = _make_request_aligned_observation_outcome(
            problem=problem,
            request=requests[0],
        )

        def result_generator() -> Generator[
            tuple[int, EvaluationOutcome[int, RequestAlignedEvaluationRecord]],
            None,
            None,
        ]:
            yield from ()

        active_batch = _active_observation_joblib_batch(
            problem=problem,
            request_inputs=request_inputs,
            result_generator=result_generator(),
        )
        active_batch.result_queue = result_queue
        evaluator = AbortInspectionAsyncJoblibEvaluator(
            backend="threading",
            n_jobs=1,
        )
        evaluator.install_active_batch(handle, active_batch)
        capture: WaitThreadCapture[
            EvaluationOutcome[int, RequestAlignedEvaluationRecord]
        ] = WaitThreadCapture()

        def wait_for_batch() -> None:
            try:
                capture.completion_groups = evaluator.wait(handle, timeout=5.0)
            except BaseException as exception:
                capture.exception = exception

        waiter = Thread(target=wait_for_batch, name="stale-completion-waiter")
        waiter.start()
        assert result_queue.entered_get.wait(timeout=5.0)

        evaluator.cancel(handle)
        result_queue.put(AsyncJoblibCompletedResult(index=0, outcome=stale_outcome))
        waiter.join(timeout=5.0)

        assert not waiter.is_alive()
        assert isinstance(capture.exception, BatchExecutionFailed)
        assert capture.exception.kind == "cancelled"
        assert capture.completion_groups is None
        assert active_batch.completed_indices == set()
        assert not evaluator.has_active_batch(handle)

    def test_attempt_wait_after_concurrent_suspend_does_not_retry_stale_batch(
        self,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        request_inputs = _request_inputs(requests)
        handle = EvaluationBatchHandle(
            batch_id="joblib-attempt-stale-suspend",
            request_count=1,
        )
        result_queue = SignalingQueue[AttemptJoblibQueueEvent]()

        def result_generator() -> Generator[
            tuple[int, EvaluationAttemptBatch[int, Observation[int]]],
            None,
            None,
        ]:
            yield from ()

        active_batch = _active_attempt_joblib_batch(
            problem=problem,
            request_inputs=request_inputs,
            result_generator=result_generator(),
        )
        active_batch.result_queue = result_queue
        evaluator = AbortInspectionAsyncJoblibEvaluator(
            backend="threading",
            n_jobs=1,
            infrastructure_retry_limit=1,
        )
        evaluator.install_active_attempt_batch(handle, active_batch)
        capture: WaitThreadCapture[
            EvaluationAttemptBatch[int, Observation[int]]
        ] = WaitThreadCapture()

        def wait_for_attempt_batch() -> None:
            try:
                capture.completion_groups = evaluator.wait_attempts(
                    handle,
                    timeout=5.0,
                )
            except BaseException as exception:
                capture.exception = exception

        waiter = Thread(target=wait_for_attempt_batch, name="stale-suspend-waiter")
        waiter.start()
        assert result_queue.entered_get.wait(timeout=5.0)

        resume_handle = evaluator.suspend_attempt_batch(handle)
        result_queue.put(AsyncJoblibExhaustedResult())
        waiter.join(timeout=5.0)

        assert resume_handle.batch_id == handle.batch_id
        assert resume_handle.completed_count == 0
        assert not waiter.is_alive()
        assert isinstance(capture.exception, BatchExecutionFailed)
        assert capture.exception.kind == "cancelled"
        assert capture.completion_groups is None
        assert active_batch.infrastructure_retry_count == 0
        assert not evaluator.has_active_attempt_batch(handle)

    def test_attempt_wait_after_concurrent_cancel_does_not_retry_stale_failure(
        self,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        request_inputs = _request_inputs(requests)
        handle = EvaluationBatchHandle(
            batch_id="joblib-attempt-stale-cancel-failure",
            request_count=1,
        )
        result_queue = SignalingQueue[AttemptJoblibQueueEvent]()

        def result_generator() -> Generator[
            tuple[int, EvaluationAttemptBatch[int, Observation[int]]],
            None,
            None,
        ]:
            yield from ()

        active_batch = _active_attempt_joblib_batch(
            problem=problem,
            request_inputs=request_inputs,
            result_generator=result_generator(),
        )
        active_batch.result_queue = result_queue
        evaluator = AbortInspectionAsyncJoblibEvaluator(
            backend="threading",
            n_jobs=1,
            infrastructure_retry_limit=1,
        )
        evaluator.install_active_attempt_batch(handle, active_batch)
        capture: WaitThreadCapture[
            EvaluationAttemptBatch[int, Observation[int]]
        ] = WaitThreadCapture()

        def wait_for_attempt_batch() -> None:
            try:
                capture.completion_groups = evaluator.wait_attempts(
                    handle,
                    timeout=5.0,
                )
            except BaseException as exception:
                capture.exception = exception

        waiter = Thread(target=wait_for_attempt_batch, name="stale-failure-waiter")
        waiter.start()
        assert result_queue.entered_get.wait(timeout=5.0)

        evaluator.cancel_attempt_batch(handle)
        result_queue.put(
            AsyncJoblibFailedResult(
                exception=FuturesBrokenProcessPool("stale backend failure"),
            ),
        )
        waiter.join(timeout=5.0)

        assert not waiter.is_alive()
        assert isinstance(capture.exception, BatchExecutionFailed)
        assert capture.exception.kind == "cancelled"
        assert capture.completion_groups is None
        assert active_batch.infrastructure_retry_count == 0
        assert not evaluator.has_active_attempt_batch(handle)

    def test_cancel_after_retry_declines_prevents_infrastructure_failure(
        self,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        request_inputs = _request_inputs(requests)
        handle = EvaluationBatchHandle(
            batch_id="joblib-cancel-after-retry-declined",
            request_count=1,
        )

        def result_generator() -> Generator[
            tuple[int, EvaluationOutcome[int, RequestAlignedEvaluationRecord]],
            None,
            None,
        ]:
            yield from ()

        active_batch = _active_observation_joblib_batch(
            problem=problem,
            request_inputs=request_inputs,
            result_generator=result_generator(),
        )
        active_batch.result_queue.put(AsyncJoblibExhaustedResult())
        evaluator = CancelAfterRetryDeclinedAsyncJoblibEvaluator(
            backend="threading",
            n_jobs=1,
        )
        evaluator.install_active_batch(handle, active_batch)

        with pytest.raises(BatchExecutionFailed) as caught:
            _ = evaluator.wait(handle, timeout=5.0)

        assert caught.value.kind == "cancelled"
        assert active_batch.infrastructure_retry_count == 0
        assert not evaluator.has_active_batch(handle)

    def test_attempt_cancel_after_retry_declines_prevents_infrastructure_failure(
        self,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        request_inputs = _request_inputs(requests)
        handle = EvaluationBatchHandle(
            batch_id="joblib-attempt-cancel-after-retry-declined",
            request_count=1,
        )

        def result_generator() -> Generator[
            tuple[int, EvaluationAttemptBatch[int, Observation[int]]],
            None,
            None,
        ]:
            yield from ()

        active_batch = _active_attempt_joblib_batch(
            problem=problem,
            request_inputs=request_inputs,
            result_generator=result_generator(),
        )
        active_batch.result_queue.put(AsyncJoblibExhaustedResult())
        evaluator = CancelAfterRetryDeclinedAsyncJoblibEvaluator(
            backend="threading",
            n_jobs=1,
        )
        evaluator.install_active_attempt_batch(handle, active_batch)

        with pytest.raises(BatchExecutionFailed) as caught:
            _ = evaluator.wait_attempts(handle, timeout=5.0)

        assert caught.value.kind == "cancelled"
        assert active_batch.infrastructure_retry_count == 0
        assert not evaluator.has_active_attempt_batch(handle)

    def test_async_result_queue_is_bounded_by_worker_count(self) -> None:
        problem = _legacy_observation_problem(SlowSquareObjective())
        requests = _requests(
            tuple(
                Proposal(candidate=candidate, proposal_id=f"p-{candidate}")
                for candidate in range(20)
            ),
        )
        evaluator = AbortInspectionAsyncJoblibEvaluator(
            backend="threading",
            n_jobs=2,
        )
        handle = evaluator.submit_batch(problem, requests)

        try:
            active_batch = evaluator.active_batch_for(handle)
            assert active_batch.result_queue.maxsize == 5
        finally:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                evaluator.cancel(handle)

    def test_attempt_result_queue_is_bounded_by_worker_count(self) -> None:
        problem = _legacy_observation_problem(SlowSquareObjective())
        requests = _requests(
            tuple(
                Proposal(candidate=candidate, proposal_id=f"p-{candidate}")
                for candidate in range(20)
            ),
        )
        evaluator = AbortInspectionAsyncJoblibEvaluator(
            backend="threading",
            n_jobs=2,
        )
        handle = evaluator.submit_attempt_batch(problem, requests)

        try:
            active_batch = evaluator.active_attempt_batch_for(handle)
            assert active_batch.result_queue.maxsize == 5
        finally:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                evaluator.cancel_attempt_batch(handle)

    def test_cancel_returns_when_result_queue_is_full(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        requests = _requests((Proposal(candidate=4, proposal_id="p-1"),))
        request_inputs = _request_inputs(requests)
        stale_outcome = _make_request_aligned_observation_outcome(
            problem=problem,
            request=requests[0],
        )
        full_queue: Queue[OutcomeJoblibQueueEvent] = Queue(maxsize=1)
        full_queue.put(AsyncJoblibCompletedResult(index=0, outcome=stale_outcome))
        worker_started = Event()

        def blocked_drain_worker() -> None:
            worker_started.set()
            full_queue.put(AsyncJoblibExhaustedResult())

        result_worker = Thread(
            target=blocked_drain_worker,
            name="full-result-queue-worker",
            daemon=True,
        )
        result_worker.start()
        assert worker_started.wait(timeout=5.0)

        def result_generator() -> Generator[
            tuple[int, EvaluationOutcome[int, RequestAlignedEvaluationRecord]],
            None,
            None,
        ]:
            yield from ()

        event_log = AbortEventLog(events=[])
        active_batch = _active_observation_joblib_batch(
            problem=problem,
            request_inputs=request_inputs,
            result_generator=result_generator(),
            abort_attempt=lambda: event_log.events.append("abort"),
        )
        active_batch.result_queue = full_queue
        active_batch.result_worker = result_worker
        handle = EvaluationBatchHandle(
            batch_id="joblib-full-result-queue-cancel",
            request_count=1,
        )
        evaluator = AbortInspectionAsyncJoblibEvaluator(backend="threading", n_jobs=1)
        evaluator.install_active_batch(handle, active_batch)

        evaluator.cancel(handle)

        assert event_log.events == ["abort"]
        assert not evaluator.has_active_batch(handle)
        assert result_worker.is_alive()
        _ = full_queue.get_nowait()
        result_worker.join(timeout=5.0)
        assert not result_worker.is_alive()

    def test_open_session_exposes_resumable_capability(self) -> None:
        problem = _legacy_observation_problem(DelayedSquareObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)

        assert isinstance(evaluator, ResumableAsyncEvaluator)

        session = evaluator.open_session(
            problem,
            _requests(
                (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
                )
            ),
        )

        assert isinstance(session, ResumableBatchSession)
        session.cancel()

    def test_open_attempt_session_streams_user_failures_without_terminal_failure(
        self,
    ) -> None:
        problem = _legacy_observation_problem(ExplodingObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)
        requests = _requests(
            (
                Proposal(candidate=1, proposal_id="p-1"),
                Proposal(candidate=4, proposal_id="p-2"),
                Proposal(candidate=2, proposal_id="p-3"),
            )
        )
        session = evaluator.open_attempt_session(problem, requests)

        ordered_attempts: list[
            EvaluationAttemptBatch[int, RequestAlignedEvaluationRecord] | None
        ] = [None] * session.handle.request_count
        completed_count = 0
        while completed_count < session.handle.request_count:
            for completion_group in session.wait(timeout=5.0):
                for offset, attempt in enumerate(completion_group.outcomes):
                    ordered_attempts[completion_group.start_index + offset] = attempt
                    completed_count += 1

        attempts = EvaluationAttemptBatch[
            int,
            RequestAlignedEvaluationRecord,
        ].from_single_request_attempts(
            tuple(attempt for attempt in ordered_attempts if attempt is not None),
        )
        assert attempts.success_indices == (0, 2)
        assert attempts.failure_indices == (1,)
        assert tuple(failure.proposal_id for failure in attempts.failures) == ("p-2",)
        assert attempts.failures[0].exception.exception_type == "builtins.ValueError"

    def test_open_attempt_session_keeps_validation_failure_hard(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](
            backend="threading",
            n_jobs=2,
            infrastructure_retry_limit=3,
        )
        session = evaluator.open_attempt_session(
            problem,
            _requests((Proposal(candidate=11, proposal_id="p-1"),)),
        )

        with pytest.raises(BatchExecutionFailed) as caught:
            _ = tuple(session.wait(timeout=5.0))

        assert caught.value.kind == "infrastructure"
        assert isinstance(caught.value.cause, ValueError)

    def test_open_attempt_session_supports_all_failure_batch(self) -> None:
        problem = _legacy_observation_problem(ExplodingObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](
            backend="threading",
            n_jobs=2,
        )
        requests = _requests(
            (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=4, proposal_id="p-2"),
            )
        )
        session = evaluator.open_attempt_session(problem, requests)
        pending_aware_session = _require_pending_aware_session(session)

        ordered_attempts: list[
            EvaluationAttemptBatch[int, RequestAlignedEvaluationRecord] | None
        ] = [None] * session.handle.request_count
        while pending_aware_session.state().lifecycle != "completed":
            for completion_group in pending_aware_session.wait(timeout=5.0):
                for offset, attempt in enumerate(completion_group.outcomes):
                    ordered_attempts[completion_group.start_index + offset] = attempt

        attempts = EvaluationAttemptBatch[
            int,
            RequestAlignedEvaluationRecord,
        ].from_single_request_attempts(
            tuple(attempt for attempt in ordered_attempts if attempt is not None),
        )

        assert attempts.successes == ()
        assert attempts.failure_indices == (0, 1)
        assert tuple(failure.proposal_id for failure in attempts.failures) == (
            "p-1",
            "p-2",
        )
        assert pending_aware_session.state() == EvaluationBatchSessionState(
            request_count=2,
            completed_count=2,
            pending_count=0,
            lifecycle="completed",
        )

    def test_suspend_and_resume_attempt_session_preserves_failure_slots(self) -> None:
        problem = _legacy_observation_problem(DelayedExplodingObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)
        requests = _requests(
            (
                Proposal(candidate=1, proposal_id="p-1"),
                Proposal(candidate=4, proposal_id="p-2"),
            )
        )
        session = evaluator.open_attempt_session(problem, requests)
        pending_aware_session = _require_pending_aware_session(session)
        resumable_session = _require_resumable_session(session)

        first_completion_groups = tuple(pending_aware_session.wait(timeout=5.0))
        resume_handle = resumable_session.suspend()

        assert pending_aware_session.state() == EvaluationBatchSessionState(
            request_count=2,
            completed_count=1,
            pending_count=1,
            lifecycle="suspended",
        )

        resumed_session = evaluator.resume_attempt_session(resume_handle)
        pending_resumed_session = _require_pending_aware_session(resumed_session)

        second_completion_groups = tuple(pending_resumed_session.wait(timeout=5.0))

        ordered_attempts: list[
            EvaluationAttemptBatch[int, RequestAlignedEvaluationRecord] | None
        ] = [None, None]
        for completion_group in first_completion_groups + second_completion_groups:
            for offset, attempt in enumerate(completion_group.outcomes):
                ordered_attempts[completion_group.start_index + offset] = attempt

        attempts = EvaluationAttemptBatch[
            int,
            RequestAlignedEvaluationRecord,
        ].from_single_request_attempts(
            tuple(attempt for attempt in ordered_attempts if attempt is not None),
        )
        assert attempts.success_indices == (0,)
        assert attempts.failure_indices == (1,)
        assert attempts.failures[0].proposal_id == "p-2"
        assert attempts.failures[0].exception.exception_type == "builtins.ValueError"

    def test_invalid_attempt_resume_handle_does_not_consume_suspended_state(
        self,
    ) -> None:
        problem = _legacy_observation_problem(DelayedExplodingObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](
            backend="threading",
            n_jobs=2,
        )
        requests = _requests(
            (
                Proposal(candidate=1, proposal_id="p-1"),
                Proposal(candidate=4, proposal_id="p-2"),
            )
        )
        session = evaluator.open_attempt_session(problem, requests)
        pending_aware_session = _require_pending_aware_session(session)
        resumable_session = _require_resumable_session(session)

        _ = tuple(pending_aware_session.wait(timeout=5.0))
        resume_handle = resumable_session.suspend()
        bad_handle = EvaluationBatchResumeHandle(
            batch_id=resume_handle.batch_id,
            request_count=resume_handle.request_count + 1,
            completed_count=resume_handle.completed_count,
        )

        with pytest.raises(ValueError, match="request_count"):
            _ = evaluator.resume_attempt_session(bad_handle)

        resumed_session = evaluator.resume_attempt_session(resume_handle)
        pending_resumed_session = _require_pending_aware_session(resumed_session)

        assert pending_resumed_session.state().lifecycle == "active"
        assert tuple(pending_resumed_session.wait(timeout=5.0)) != ()

    def test_failed_attempt_resume_start_does_not_consume_suspended_state(
        self,
    ) -> None:
        problem = _legacy_observation_problem(DelayedExplodingObjective())
        evaluator = ResumeStartFailureAsyncJoblibEvaluator()
        requests = _requests(
            (
                Proposal(candidate=1, proposal_id="p-1"),
                Proposal(candidate=4, proposal_id="p-2"),
            )
        )
        resume_handle = evaluator.install_suspended_attempt_batch(
            problem=problem,
            requests=requests,
            completed_indices={0},
        )

        evaluator.fail_next_attempt_start()
        with pytest.raises(RuntimeError, match="forced attempt resume startup failure"):
            _ = evaluator.resume_attempt_session(resume_handle)

        resumed_session = evaluator.resume_attempt_session(resume_handle)
        pending_resumed_session = _require_pending_aware_session(resumed_session)

        assert pending_resumed_session.state().lifecycle == "active"
        assert tuple(pending_resumed_session.wait(timeout=5.0)) != ()

    def test_cancelled_attempt_resume_start_does_not_restore_suspended_state(
        self,
    ) -> None:
        problem = _legacy_observation_problem(DelayedExplodingObjective())
        evaluator = ResumeStartFailureAsyncJoblibEvaluator()
        requests = _requests(
            (
                Proposal(candidate=1, proposal_id="p-1"),
                Proposal(candidate=4, proposal_id="p-2"),
            )
        )
        resume_handle = evaluator.install_suspended_attempt_batch(
            problem=problem,
            requests=requests,
            completed_indices={0},
        )
        entered_start = Event()
        release_start = Event()
        evaluator.block_next_attempt_start(
            entered=entered_start,
            release=release_start,
        )
        evaluator.fail_next_attempt_start()
        resume_results: Queue[object] = Queue()

        def resume_once() -> None:
            try:
                resume_results.put(evaluator.resume_attempt_session(resume_handle))
            except BaseException as exception:
                resume_results.put(exception)

        worker = Thread(target=resume_once, name="attempt-resume-cancel-test")
        worker.start()
        assert entered_start.wait(timeout=5.0)
        evaluator.discard_suspended_attempt_batch(
            EvaluationBatchHandle(
                batch_id=resume_handle.batch_id,
                request_count=resume_handle.request_count,
            )
        )

        release_start.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        result = resume_results.get(timeout=5.0)
        assert isinstance(result, RuntimeError)
        assert str(result) == "forced attempt resume startup failure"
        with pytest.raises(ValueError, match="unknown suspended attempt batch handle"):
            _ = evaluator.resume_attempt_session(resume_handle)

    def test_concurrent_attempt_resume_claim_rejects_duplicate_start(self) -> None:
        problem = _legacy_observation_problem(DelayedExplodingObjective())
        evaluator = ResumeStartFailureAsyncJoblibEvaluator()
        requests = _requests(
            (
                Proposal(candidate=1, proposal_id="p-1"),
                Proposal(candidate=4, proposal_id="p-2"),
            )
        )
        resume_handle = evaluator.install_suspended_attempt_batch(
            problem=problem,
            requests=requests,
            completed_indices={0},
        )
        entered_start = Event()
        release_start = Event()
        evaluator.block_next_attempt_start(
            entered=entered_start,
            release=release_start,
        )
        resume_results: Queue[object] = Queue()

        def resume_once() -> None:
            try:
                resume_results.put(evaluator.resume_attempt_session(resume_handle))
            except BaseException as exception:
                resume_results.put(exception)

        worker = Thread(target=resume_once, name="attempt-resume-claim-test")
        worker.start()
        assert entered_start.wait(timeout=5.0)

        with pytest.raises(ValueError, match="unknown suspended attempt batch handle"):
            _ = evaluator.resume_attempt_session(resume_handle)

        release_start.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        result = resume_results.get(timeout=5.0)
        if isinstance(result, BaseException):
            raise AssertionError("first resume should complete") from result

        resumed_session = cast(
            EvaluationBatchSession[EvaluationAttemptBatch[int, Observation[int]]],
            result,
        )
        pending_resumed_session = _require_pending_aware_session(resumed_session)
        assert evaluator.attempt_start_count == 1
        assert tuple(pending_resumed_session.wait(timeout=5.0)) != ()

    def test_suspend_and_resume_session_preserves_remaining_work(self) -> None:
        problem = _legacy_observation_problem(DelayedSquareObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](backend="threading", n_jobs=2)
        session = evaluator.open_session(
            problem,
            _requests(
                (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
                )
            ),
        )
        pending_aware_session = _require_pending_aware_session(session)
        resumable_session = _require_resumable_session(session)

        first_completion_groups = tuple(pending_aware_session.wait())
        resume_handle = resumable_session.suspend()

        assert pending_aware_session.state() == EvaluationBatchSessionState(
                request_count=2,
                completed_count=1,
                pending_count=1,
                lifecycle="suspended",
            )
        assert isinstance(resume_handle, EvaluationBatchResumeHandle)

        resumed_session = evaluator.resume_session(resume_handle)
        pending_resumed_session = _require_pending_aware_session(resumed_session)

        second_completion_groups = tuple(pending_resumed_session.wait())
        assert pending_resumed_session.state() == EvaluationBatchSessionState(
                request_count=2,
                completed_count=2,
                pending_count=0,
                lifecycle="completed",
            )

        ordered_outcomes: list[
            EvaluationOutcome[int, RequestAlignedEvaluationRecord] | None
        ] = [None, None]
        for completion_group in first_completion_groups + second_completion_groups:
            for offset, outcome in enumerate(completion_group.outcomes):
                ordered_outcomes[completion_group.start_index + offset] = outcome

        assert tuple(
                outcome.observation.value
                for outcome in ordered_outcomes
                if outcome is not None
            ) == (16.0, 1.0)

    def test_invalid_resume_handle_does_not_consume_suspended_state(self) -> None:
        problem = _legacy_observation_problem(DelayedSquareObjective())
        evaluator = AsyncJoblibEvaluator[int, int, Observation[int]](
            backend="threading",
            n_jobs=2,
        )
        session = evaluator.open_session(
            problem,
            _requests(
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=1, proposal_id="p-2"),
                )
            ),
        )
        pending_aware_session = _require_pending_aware_session(session)
        resumable_session = _require_resumable_session(session)

        _ = tuple(pending_aware_session.wait(timeout=5.0))
        resume_handle = resumable_session.suspend()
        bad_handle = EvaluationBatchResumeHandle(
            batch_id=resume_handle.batch_id,
            request_count=resume_handle.request_count + 1,
            completed_count=resume_handle.completed_count,
        )

        with pytest.raises(ValueError, match="request_count"):
            _ = evaluator.resume_session(bad_handle)

        resumed_session = evaluator.resume_session(resume_handle)
        pending_resumed_session = _require_pending_aware_session(resumed_session)

        assert pending_resumed_session.state().lifecycle == "active"
        assert tuple(pending_resumed_session.wait(timeout=5.0)) != ()

    def test_failed_resume_start_does_not_consume_suspended_state(
        self,
    ) -> None:
        problem = _legacy_observation_problem(DelayedSquareObjective())
        evaluator = ResumeStartFailureAsyncJoblibEvaluator()
        requests = _requests(
            (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
            )
        )
        resume_handle = evaluator.install_suspended_batch(
            problem=problem,
            requests=requests,
            completed_indices={0},
        )

        evaluator.fail_next_batch_start()
        with pytest.raises(RuntimeError, match="forced resume startup failure"):
            _ = evaluator.resume_session(resume_handle)

        resumed_session = evaluator.resume_session(resume_handle)
        pending_resumed_session = _require_pending_aware_session(resumed_session)

        assert pending_resumed_session.state().lifecycle == "active"
        assert tuple(pending_resumed_session.wait(timeout=5.0)) != ()

    def test_cancelled_resume_start_does_not_restore_suspended_state(self) -> None:
        problem = _legacy_observation_problem(DelayedSquareObjective())
        evaluator = ResumeStartFailureAsyncJoblibEvaluator()
        requests = _requests(
            (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
            )
        )
        resume_handle = evaluator.install_suspended_batch(
            problem=problem,
            requests=requests,
            completed_indices={0},
        )
        entered_start = Event()
        release_start = Event()
        evaluator.block_next_batch_start(entered=entered_start, release=release_start)
        evaluator.fail_next_batch_start()
        resume_results: Queue[object] = Queue()

        def resume_once() -> None:
            try:
                resume_results.put(evaluator.resume_session(resume_handle))
            except BaseException as exception:
                resume_results.put(exception)

        worker = Thread(target=resume_once, name="resume-cancel-test")
        worker.start()
        assert entered_start.wait(timeout=5.0)
        evaluator.discard_suspended_batch(
            EvaluationBatchHandle(
                batch_id=resume_handle.batch_id,
                request_count=resume_handle.request_count,
            )
        )

        release_start.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        result = resume_results.get(timeout=5.0)
        assert isinstance(result, RuntimeError)
        assert str(result) == "forced resume startup failure"
        with pytest.raises(ValueError, match="unknown suspended batch handle"):
            _ = evaluator.resume_session(resume_handle)

    def test_concurrent_resume_claim_rejects_duplicate_start(self) -> None:
        problem = _legacy_observation_problem(DelayedSquareObjective())
        evaluator = ResumeStartFailureAsyncJoblibEvaluator()
        requests = _requests(
            (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
            )
        )
        resume_handle = evaluator.install_suspended_batch(
            problem=problem,
            requests=requests,
            completed_indices={0},
        )
        entered_start = Event()
        release_start = Event()
        evaluator.block_next_batch_start(entered=entered_start, release=release_start)
        resume_results: Queue[object] = Queue()

        def resume_once() -> None:
            try:
                resume_results.put(evaluator.resume_session(resume_handle))
            except BaseException as exception:
                resume_results.put(exception)

        worker = Thread(target=resume_once, name="resume-claim-test")
        worker.start()
        assert entered_start.wait(timeout=5.0)

        with pytest.raises(ValueError, match="unknown suspended batch handle"):
            _ = evaluator.resume_session(resume_handle)

        release_start.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        result = resume_results.get(timeout=5.0)
        if isinstance(result, BaseException):
            raise AssertionError("first resume should complete") from result

        resumed_session = cast(
            EvaluationBatchSession[
                EvaluationOutcome[int, RequestAlignedEvaluationRecord]
            ],
            result,
        )
        pending_resumed_session = _require_pending_aware_session(resumed_session)
        assert evaluator.batch_start_count == 1
        assert tuple(pending_resumed_session.wait(timeout=5.0)) != ()

    def test_retries_infrastructure_failure_for_remaining_proposals(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        evaluator = FlakyAsyncJoblibEvaluator(
            failure_mode="infrastructure_once",
            infrastructure_retry_limit=1,
        )

        outcomes = evaluator.evaluate(
            problem,
            _requests(
                (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
                )
            ),
        )

        assert evaluator.attempt_count == 2
        assert tuple(outcome.observation.proposal.proposal_id for outcome in outcomes) == ("p-1", "p-2")
        assert tuple(outcome.observation.value for outcome in outcomes) == (16.0, 1.0)

    def test_raises_after_infrastructure_retry_limit_exhausted(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        evaluator = FlakyAsyncJoblibEvaluator(
            failure_mode="infrastructure_always",
            infrastructure_retry_limit=1,
        )

        with pytest.raises(BatchExecutionFailed) as caught:
            _ = evaluator.evaluate(
                problem,
                _requests(
                    (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=1, proposal_id="p-2"),
                    )
                ),
        )

        assert evaluator.attempt_count == 2
        assert caught.value.kind == "infrastructure"
        assert isinstance(caught.value.cause, FuturesBrokenProcessPool)

    def test_does_not_retry_user_code_failure(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        evaluator = FlakyAsyncJoblibEvaluator(
            failure_mode="user_code_once",
            infrastructure_retry_limit=3,
        )

        with pytest.raises(BatchExecutionFailed) as caught:
            _ = evaluator.evaluate(
                problem,
                _requests(
                    (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=1, proposal_id="p-2"),
                    )
                ),
            )

        assert evaluator.attempt_count == 1
        assert caught.value.kind == "user_code"
        assert isinstance(caught.value.cause, ValueError)

    def test_does_not_retry_user_timeout_error(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        evaluator = FlakyAsyncJoblibEvaluator(
            failure_mode="user_timeout_error_once",
            infrastructure_retry_limit=3,
        )

        with pytest.raises(BatchExecutionFailed) as caught:
            _ = evaluator.evaluate(
                problem,
                _requests(
                    (
                        Proposal(candidate=4, proposal_id="p-1"),
                        Proposal(candidate=1, proposal_id="p-2"),
                    )
                ),
            )

        assert evaluator.attempt_count == 1
        assert caught.value.kind == "user_code"
        assert isinstance(caught.value.cause, TimeoutError)

    def test_does_not_retry_keyboard_interrupt(self) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        evaluator = FlakyAsyncJoblibEvaluator(
            failure_mode="keyboard_interrupt_once",
            infrastructure_retry_limit=3,
        )

        with pytest.raises(BatchExecutionFailed) as caught:
            _ = evaluator.evaluate(
                problem,
                _requests(
                    (
                        Proposal(candidate=4, proposal_id="p-1"),
                        Proposal(candidate=1, proposal_id="p-2"),
                    )
                ),
            )

        assert evaluator.attempt_count == 1
        assert caught.value.kind == "cancelled"
        assert isinstance(caught.value.cause, KeyboardInterrupt)

    @pytest.mark.parametrize(
        "failure_mode",
        (
            "loky_broken_process_pool_once",
            "loky_terminated_worker_once",
            "futures_broken_process_pool_once",
        ),
    )
    def test_retries_structural_process_pool_infrastructure_failures(
        self,
        failure_mode: InjectedAsyncJoblibFailureMode,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        evaluator = FlakyAsyncJoblibEvaluator(
            failure_mode=failure_mode,
            infrastructure_retry_limit=1,
        )

        outcomes = evaluator.evaluate(
            problem,
            _requests(
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=1, proposal_id="p-2"),
                )
            ),
        )

        assert evaluator.attempt_count == 2
        assert tuple(outcome.observation.proposal.proposal_id for outcome in outcomes) == ("p-1", "p-2")
        assert tuple(outcome.observation.value for outcome in outcomes) == (16.0, 1.0)

    @pytest.mark.parametrize(
        "failure_mode",
        (
            "user_broken_process_pool_once",
            "user_terminated_worker_once",
        ),
    )
    def test_does_not_retry_user_exception_with_backend_like_name(
        self,
        failure_mode: InjectedAsyncJoblibFailureMode,
    ) -> None:
        problem = _legacy_observation_problem(SquareObjective())
        evaluator = FlakyAsyncJoblibEvaluator(
            failure_mode=failure_mode,
            infrastructure_retry_limit=3,
        )

        with pytest.raises(BatchExecutionFailed) as caught:
            _ = evaluator.evaluate(
                problem,
                _requests(
                    (
                        Proposal(candidate=4, proposal_id="p-1"),
                        Proposal(candidate=1, proposal_id="p-2"),
                    )
                ),
            )

        assert evaluator.attempt_count == 1
        assert caught.value.kind == "user_code"
        assert type(caught.value.cause).__name__ in {
            "BrokenProcessPool",
            "TerminatedWorkerError",
        }

    def test_study_step_exact_async_uses_async_joblib_backend(self) -> None:
        @dataclass(frozen=True, slots=True)
        class _State:
            remaining_batches: tuple[tuple[Proposal[int], ...], ...]
            tell_history: tuple[tuple[Observation[int], ...], ...] = ()

        class _Optimizer(
            RunMethod[_State, Proposal[int], RequestAlignedEvaluationRecord]
        ):
            @override
            def create_initial_state(self) -> _State:
                return _State(
                    remaining_batches=(
                        (
                            Proposal(candidate=4, proposal_id="p-1"),
                            Proposal(candidate=1, proposal_id="p-2"),
                        ),
                    ),
                )

            @override
            def is_exhausted(self, state: _State) -> bool:
                return len(state.remaining_batches) == 0

            @override
            def supported_execution_models(self) -> frozenset[ExecutionModel]:
                return frozenset(
                    {
                        SEQUENTIAL_EXECUTION_MODEL,
                        SYNC_BATCH_EXECUTION_MODEL,
                        EXACT_ASYNC_EXECUTION_MODEL,
                    },
                )

            @override
            def ask(
                self,
                state: _State,
                batch_size: int = 1,
            ) -> tuple[tuple[Proposal[int], ...], _State]:
                _ = batch_size
                return state.remaining_batches[0], _State(
                    remaining_batches=state.remaining_batches[1:],
                    tell_history=state.tell_history,
                )

            @override
            def tell(
                self,
                state: _State,
                observations: Sequence[RequestAlignedEvaluationRecord],
            ) -> _State:
                scalar_observations = _require_observation_records(observations)
                return _State(
                    remaining_batches=state.remaining_batches,
                    tell_history=state.tell_history + (scalar_observations,),
                )

        problem = _legacy_request_record_problem(DelayedSquareObjective())
        evaluator = AsyncJoblibEvaluator[
            int,
            int,
            RequestAlignedEvaluationRecord,
        ](backend="threading", n_jobs=2)
        study = Study(
            problem=problem,
            run_method=_Optimizer(),
            evaluator=evaluator,
        )

        records, next_state = study.step(
            study.create_run_method_state(),
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )
        observations = _require_observation_records(records)

        assert tuple(observation.proposal.proposal_id for observation in observations) == ("p-1", "p-2")
        assert tuple(
                observation.proposal.proposal_id
                for observation in next_state.tell_history[0]
            ) == ("p-1", "p-2")

    def test_study_exact_async_with_async_joblib_records_attempt_failures(
        self,
    ) -> None:
        @dataclass(frozen=True, slots=True)
        class _State:
            remaining_batches: tuple[tuple[Proposal[int], ...], ...]
            tell_history: tuple[tuple[Observation[int], ...], ...] = ()
            failure_history: tuple[tuple[str | None, ...], ...] = ()

        class _Optimizer(
            RunMethod[_State, Proposal[int], RequestAlignedEvaluationRecord]
        ):
            @override
            def create_initial_state(self) -> _State:
                return _State(
                    remaining_batches=(
                        (
                            Proposal(candidate=1, proposal_id="p-1"),
                            Proposal(candidate=4, proposal_id="p-2"),
                            Proposal(candidate=2, proposal_id="p-3"),
                        ),
                    ),
                )

            @override
            def is_exhausted(self, state: _State) -> bool:
                return len(state.remaining_batches) == 0

            @override
            def supported_execution_models(self) -> frozenset[ExecutionModel]:
                return frozenset(
                    {
                        SEQUENTIAL_EXECUTION_MODEL,
                        SYNC_BATCH_EXECUTION_MODEL,
                        EXACT_ASYNC_EXECUTION_MODEL,
                    },
                )

            @override
            def ask(
                self,
                state: _State,
                batch_size: int = 1,
            ) -> tuple[tuple[Proposal[int], ...], _State]:
                _ = batch_size
                return state.remaining_batches[0], _State(
                    remaining_batches=state.remaining_batches[1:],
                    tell_history=state.tell_history,
                    failure_history=state.failure_history,
                )

            @override
            def tell(
                self,
                state: _State,
                observations: Sequence[RequestAlignedEvaluationRecord],
            ) -> _State:
                scalar_observations = _require_observation_records(observations)
                return _State(
                    remaining_batches=state.remaining_batches,
                    tell_history=state.tell_history + (scalar_observations,),
                    failure_history=state.failure_history + ((),),
                )

            @override
            def tell_attempts(
                self,
                state: _State,
                attempts: EvaluationAttemptBatch[
                    AttemptCandidateT,
                    RequestAlignedEvaluationRecord,
                ],
            ) -> _State:
                return _State(
                    remaining_batches=state.remaining_batches,
                    tell_history=state.tell_history
                    + (_successful_observation_records(attempts),),
                    failure_history=state.failure_history
                    + (tuple(failure.proposal_id for failure in attempts.failures),),
                )

        problem = _legacy_request_record_problem(ExplodingObjective())
        evaluator = AsyncJoblibEvaluator[
            int,
            int,
            RequestAlignedEvaluationRecord,
        ](
            backend="threading",
            n_jobs=2,
        )
        study = Study(problem=problem, run_method=_Optimizer(), evaluator=evaluator)

        report, final_state = study.run(
            max_evaluations=3,
            batch_size=3,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )
        observations = _require_observation_records(report.records)

        assert tuple(observation.proposal.proposal_id for observation in observations) == (
            "p-1",
            "p-3",
        )
        assert tuple(failure.proposal_id for failure in report.failures) == ("p-2",)
        assert report.evaluation_count == 3
        assert tuple(
            observation.proposal.proposal_id
            for observation in final_state.tell_history[0]
        ) == (
            "p-1",
            "p-3",
        )
        assert tuple(
            failure_proposal_id
            for failure_batch in final_state.failure_history
            for failure_proposal_id in failure_batch
        ) == ("p-2",)
