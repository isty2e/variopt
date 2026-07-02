"""Tests for execution-only joblib evaluators."""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, final

import pytest
from typing_extensions import override

from variopt import (
    EvaluationOutcome,
    EvaluationRequest,
    IntegerSpace,
    Objective,
    Observation,
    Problem,
    Proposal,
    RunMethod,
    Study,
)
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

EvaluationRecordT = EvaluationOutcome[int]
JOBLIB_CANCELLED_TASKS_WARNING = (
    r"\d+ tasks? which were still being processed by the workers "
    r"have been cancelled"
)


def _require_pending_aware_session(
    session: EvaluationBatchSession[EvaluationRecordT],
) -> PendingAwareBatchSession[EvaluationRecordT]:
    if not isinstance(session, PendingAwareBatchSession):
        pytest.fail("async joblib session should be pending-aware")
    return session


def _require_resumable_session(
    session: EvaluationBatchSession[EvaluationRecordT],
) -> ResumableBatchSession[EvaluationRecordT]:
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


class ExplodingObjective(Objective[int]):
    """Objective that raises for one designated candidate."""

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 4:
            msg = "boom"
            raise ValueError(msg)
        return float(candidate * candidate)


@final
class FlakyAsyncJoblibEvaluator(AsyncJoblibEvaluator[int, int]):
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
        problem: Problem[int, int],
        request: EvaluationRequest[int],
    ) -> EvaluationOutcome[int]:
        return EvaluationOutcome(
            record=problem.evaluation_protocol.evaluate_request(request),
            evaluation_count=1,
        )

    @override
    def _start_attempt(
        self,
        *,
        problem: Problem[int, int],
        request_inputs: Sequence[AsyncJoblibRequestInput[int]],
        execution_resources: ExecutionResources,
    ):
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
            _ = JoblibEvaluator[int, int](n_jobs=0)

    def test_preserves_input_proposal_order(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=DelayedSquareObjective(),
        )
        evaluator = JoblibEvaluator[int, int](backend="threading", n_jobs=2)

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
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        evaluator = JoblibEvaluator[int, int](backend="loky", n_jobs=2)

        outcomes = evaluator.evaluate(
            problem,
            _requests((Proposal(candidate=4, proposal_id="p-1"),)),
        )

        assert tuple(outcome.observation.value for outcome in outcomes) == (16.0,)

    def test_execution_resources_report_evaluator_ownership(self) -> None:
        evaluator = JoblibEvaluator[int, int](backend="threading", n_jobs=2)

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
            _ = AsyncJoblibEvaluator[int, int](infrastructure_retry_limit=-1)

    def test_evaluate_preserves_input_proposal_order(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=DelayedSquareObjective(),
        )
        evaluator = AsyncJoblibEvaluator[int, int](backend="threading", n_jobs=2)

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
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        evaluator = AsyncJoblibEvaluator[int, int](backend="loky", n_jobs=2)

        outcomes = evaluator.evaluate(
            problem,
            _requests((Proposal(candidate=4, proposal_id="p-1"),)),
        )

        assert tuple(outcome.observation.value for outcome in outcomes) == (16.0,)

    def test_execution_resources_report_evaluator_ownership(self) -> None:
        evaluator = AsyncJoblibEvaluator[int, int](backend="threading", n_jobs=2)

        assert evaluator.execution_resources() == ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=2,
                owner_backend="threading",
            )

    def test_poll_wraps_terminal_failure(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        evaluator = AsyncJoblibEvaluator[int, int](backend="threading", n_jobs=2)
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
                _ = evaluator.poll(handle)

        assert caught.value.kind == "user_code"
        assert isinstance(caught.value.cause, ValueError)

    def test_open_session_exposes_exact_async_batch_lifecycle(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=DelayedSquareObjective(),
        )
        evaluator = AsyncJoblibEvaluator[int, int](backend="threading", n_jobs=2)
        session = evaluator.open_session(
            problem,
            _requests(
                (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
                )
            ),
        )

        ordered_outcomes: list[EvaluationOutcome[int] | None] = [
            None
        ] * session.handle.request_count
        completed_count = 0
        while completed_count < session.handle.request_count:
            completion_groups = tuple(session.poll())
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
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=DelayedSquareObjective(),
        )
        evaluator = AsyncJoblibEvaluator[int, int](backend="threading", n_jobs=2)
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

        _ = tuple(pending_aware_session.poll())
        after_first_poll = pending_aware_session.state()
        assert after_first_poll.completed_count == 1
        assert after_first_poll.pending_count == 1
        assert after_first_poll.lifecycle == "active"

        _ = tuple(pending_aware_session.poll())
        assert pending_aware_session.state() == EvaluationBatchSessionState(
                request_count=2,
                completed_count=2,
                pending_count=0,
                lifecycle="completed",
            )

    def test_pending_aware_state_tracks_cancelled_session(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=DelayedSquareObjective(),
        )
        evaluator = AsyncJoblibEvaluator[int, int](backend="threading", n_jobs=2)
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

        with pytest.warns(UserWarning, match=JOBLIB_CANCELLED_TASKS_WARNING):
            pending_aware_session.cancel()

        assert pending_aware_session.state() == EvaluationBatchSessionState(
                request_count=2,
                completed_count=0,
                pending_count=2,
                lifecycle="cancelled",
            )

    def test_open_session_exposes_resumable_capability(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=DelayedSquareObjective(),
        )
        evaluator = AsyncJoblibEvaluator[int, int](backend="threading", n_jobs=2)

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
        with pytest.warns(UserWarning, match=JOBLIB_CANCELLED_TASKS_WARNING):
            session.cancel()

    def test_suspend_and_resume_session_preserves_remaining_work(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=DelayedSquareObjective(),
        )
        evaluator = AsyncJoblibEvaluator[int, int](backend="threading", n_jobs=2)
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

        first_completion_groups = tuple(pending_aware_session.poll())
        with pytest.warns(UserWarning, match=JOBLIB_CANCELLED_TASKS_WARNING):
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

        second_completion_groups = tuple(pending_resumed_session.poll())
        assert pending_resumed_session.state() == EvaluationBatchSessionState(
                request_count=2,
                completed_count=2,
                pending_count=0,
                lifecycle="completed",
            )

        ordered_outcomes: list[EvaluationOutcome[int] | None] = [None, None]
        for completion_group in first_completion_groups + second_completion_groups:
            for offset, outcome in enumerate(completion_group.outcomes):
                ordered_outcomes[completion_group.start_index + offset] = outcome

        assert tuple(
                outcome.observation.value
                for outcome in ordered_outcomes
                if outcome is not None
            ) == (16.0, 1.0)

    def test_retries_infrastructure_failure_for_remaining_proposals(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
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
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
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
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
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

        class _Optimizer(RunMethod[_State, Proposal[int], Observation[int]]):
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
                observations: Sequence[Observation[int]],
            ) -> _State:
                return _State(
                    remaining_batches=state.remaining_batches,
                    tell_history=state.tell_history + (tuple(observations),),
                )

        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=DelayedSquareObjective(),
        )
        evaluator = AsyncJoblibEvaluator[int, int](backend="threading", n_jobs=2)
        study = Study(
            problem=problem,
            run_method=_Optimizer(),
            evaluator=evaluator,
        )

        observations, next_state = study.step(
            study.create_run_method_state(),
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(observation.proposal.proposal_id for observation in observations) == ("p-1", "p-2")
        assert tuple(
                observation.proposal.proposal_id
                for observation in next_state.tell_history[0]
            ) == ("p-1", "p-2")
