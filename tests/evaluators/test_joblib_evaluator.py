"""Tests for execution-only joblib evaluators."""

import time
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import Literal, TypeGuard, TypeVar, final

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
from variopt.artifacts.records import RequestAlignedEvaluationRecord
from variopt.evaluation_pipeline import evaluate_request_outcome
from variopt.evaluators import (
    AsyncJoblibEvaluator,
    BatchExecutionFailed,
    EvaluationBatchResumeHandle,
    EvaluationBatchSessionState,
    JoblibEvaluator,
    PendingAwareBatchSession,
    ResumableAsyncEvaluator,
    ResumableBatchSession,
)
from variopt.evaluators.async_evaluator.sessions import EvaluationBatchSession
from variopt.evaluators.joblib.batches import AsyncJoblibRequestInput
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
        failure_mode: Literal[
            "infrastructure_once",
            "infrastructure_always",
            "user_code_once",
        ],
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
        outcome = _make_observation_outcome(problem=problem, request=request)
        return EvaluationOutcome(
            record=outcome.record,
            evaluation_count=outcome.evaluation_count,
            refinement=outcome.refinement,
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

        def iterator():
            if self.failure_mode == "infrastructure_once" and attempt_count == 1:
                first_input = immutable_request_inputs[0]
                yield first_input.index, self._build_outcome(
                    problem,
                    first_input.request,
                )
                raise TimeoutError("transient infrastructure failure")

            if self.failure_mode == "infrastructure_always":
                if attempt_count == 1 and len(immutable_request_inputs) > 1:
                    first_input = immutable_request_inputs[0]
                    yield first_input.index, self._build_outcome(
                        problem,
                        first_input.request,
                    )
                raise TimeoutError("persistent infrastructure failure")

            if self.failure_mode == "user_code_once" and attempt_count == 1:
                first_input = immutable_request_inputs[0]
                yield first_input.index, self._build_outcome(
                    problem,
                    first_input.request,
                )
                raise ValueError("boom")

            for request_input in immutable_request_inputs:
                yield request_input.index, self._build_outcome(
                    problem,
                    request_input.request,
                )

        return iterator()


def _requests(
    proposals: Sequence[Proposal[int]],
) -> tuple[EvaluationRequest[int], ...]:
    """Lower proposal fixtures into canonical evaluation requests."""
    return tuple(
        EvaluationRequest(proposal=proposal)
        for proposal in proposals
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

    def test_rejects_negative_infrastructure_retry_limit(self) -> None:
        with pytest.raises(ValueError):
            _ = AsyncJoblibEvaluator[int, int, Observation[int]](infrastructure_retry_limit=-1)

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
        assert isinstance(caught.value.cause, TimeoutError)

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
