"""Tests for stale-async Study execution."""

from collections.abc import Sequence
from typing import Protocol, TypeAlias, final, runtime_checkable

import pytest
from typing_extensions import override

import variopt.study.stale_async as stale_async_execution
from tests.study_support import (
    AttemptOutOfOrderAsyncEvaluator,
    FailingCandidateObjective,
    FailureRecordingBatchQueueOptimizer,
    FailureRecordingBatchQueueOptimizerState,
    OutOfOrderAsyncEvaluator,
    PayloadResumableOutOfOrderAsyncEvaluator,
    RecordingKernel,
    RollingStaleAsyncOptimizer,
    RollingStaleAsyncOptimizerState,
    SessionRecordingAsyncEvaluator,
    ShiftedObservationProtocol,
    SpaceOwnedEqualityAsyncEvaluator,
    SpaceOwnedEqualityCandidate,
    SpaceOwnedEqualityObjective,
    SpaceOwnedEqualityOptimizer,
    SpaceOwnedEqualityOptimizerState,
    SpaceOwnedEqualitySpace,
    SquareObjective,
    make_observation_payload_attempt,
)
from variopt import (
    EvaluationBudgetExhausted,
    EvaluationOutcome,
    EvaluationRequest,
    IntegerSpace,
    Observation,
    Problem,
    Proposal,
    RunExecutionFailed,
    RunMethod,
    RunReport,
    Study,
    UnsupportedEvaluationFailureError,
)
from variopt.artifacts import (
    EvaluationAttemptBatch,
    EvaluationSuccess,
    ObservationPayload,
    Trace,
    TraceEvent,
)
from variopt.evaluators import (
    AsyncEvaluator,
    BatchExecutionFailed,
    CompletionGroup,
    EvaluationBatchHandle,
    EvaluationBatchSession,
)
from variopt.execution import (
    STALE_ASYNC_EXECUTION_MODEL,
    EvaluationBudget,
    ExecutionModel,
    ExecutionResources,
    NestedParallelismPolicy,
)
from variopt.study.stale_async import open_stale_async_batch_session

StaleAsyncOutcome: TypeAlias = EvaluationOutcome[int, Observation[int]]
StaleAsyncCompletionGroup: TypeAlias = CompletionGroup[StaleAsyncOutcome]
StaleAsyncAttemptBatch: TypeAlias = EvaluationAttemptBatch[int, ObservationPayload]
StaleAsyncAttemptCompletionGroup: TypeAlias = CompletionGroup[StaleAsyncAttemptBatch]
StaleAsyncScalarStudy: TypeAlias = Study[
    int,
    int,
    RollingStaleAsyncOptimizerState,
    ObservationPayload,
    Observation[int],
]
StaleAsyncFailureRecordingStudy: TypeAlias = Study[
    int,
    int,
    FailureRecordingBatchQueueOptimizerState,
    ObservationPayload,
    Observation[int],
]
StaleAsyncSpaceOwnedEqualityStudy: TypeAlias = Study[
    int | SpaceOwnedEqualityCandidate,
    SpaceOwnedEqualityCandidate,
    SpaceOwnedEqualityOptimizerState,
    ObservationPayload,
    Observation[SpaceOwnedEqualityCandidate],
]


@runtime_checkable
class StaleAsyncRunFailure(Protocol):
    """Typed shape for hard-failure assertions over stale-async runs."""

    partial_report: RunReport[int, Observation[int]]
    partial_state: RollingStaleAsyncOptimizerState
    cause: Exception


class StaleAsyncTraceFactoryCounter:
    """Trace factory fixture that records stale-async materialization calls."""

    calls: list[tuple[TraceEvent, ...]]

    def __init__(self) -> None:
        self.calls = []

    def __call__(
        self,
        *,
        events: tuple[TraceEvent, ...] = (),
    ) -> Trace:
        self.calls.append(tuple(events))
        return Trace(events=events)


class FailingSecondBatchAsyncEvaluator(OutOfOrderAsyncEvaluator):
    """Async evaluator that fails a later active batch after a refill opens."""

    cancelled_batch_ids: tuple[str, ...]

    def __init__(self) -> None:
        super().__init__()
        self.cancelled_batch_ids = ()

    @override
    def poll(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[StaleAsyncCompletionGroup]:
        if handle.batch_id == "batch-1":
            raise BatchExecutionFailed(
                handle=handle,
                kind="infrastructure",
                cause=RuntimeError("forced mid-sweep failure"),
            )

        return super().poll(handle)

    @override
    def cancel(self, handle: EvaluationBatchHandle) -> None:
        self.cancelled_batch_ids += (handle.batch_id,)
        super().cancel(handle)

    @override
    def poll_attempts(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[StaleAsyncAttemptCompletionGroup]:
        if handle.batch_id == "attempt-batch-1":
            raise BatchExecutionFailed(
                handle=handle,
                kind="infrastructure",
                cause=RuntimeError("forced mid-sweep failure"),
            )

        return super().poll_attempts(handle)

    @override
    def cancel_attempts(self, handle: EvaluationBatchHandle) -> None:
        self.cancelled_batch_ids += (handle.batch_id,)
        super().cancel_attempts(handle)


@final
class CancelFailingSecondBatchAsyncEvaluator(FailingSecondBatchAsyncEvaluator):
    """Async evaluator whose first cancellation fails during cleanup."""

    @override
    def cancel(self, handle: EvaluationBatchHandle) -> None:
        self.cancelled_batch_ids += (handle.batch_id,)
        if handle.batch_id == "batch-0":
            raise RuntimeError("forced cancellation failure")

        OutOfOrderAsyncEvaluator.cancel(self, handle)

    @override
    def cancel_attempts(self, handle: EvaluationBatchHandle) -> None:
        self.cancelled_batch_ids += (handle.batch_id,)
        if handle.batch_id == "attempt-batch-0":
            raise RuntimeError("forced cancellation failure")

        OutOfOrderAsyncEvaluator.cancel_attempts(self, handle)


@final
class CancellationRecordingSessionAsyncEvaluator(SessionRecordingAsyncEvaluator):
    """Session-recording evaluator that also records attempt cancellations."""

    cancelled_batch_ids: tuple[str, ...]

    def __init__(self) -> None:
        super().__init__()
        self.cancelled_batch_ids = ()

    @override
    def cancel_attempts(self, handle: EvaluationBatchHandle) -> None:
        self.cancelled_batch_ids += (handle.batch_id,)
        super().cancel_attempts(handle)


@final
class AskRecordingStaleAsyncOptimizer(
    RunMethod[RollingStaleAsyncOptimizerState, Proposal[int], Observation[int]],
):
    """Stale-async optimizer that records whether ask was called."""

    _initial_proposals: tuple[Proposal[int], ...]
    ask_count: int

    def __init__(self, proposals: Sequence[Proposal[int]]) -> None:
        self._initial_proposals = tuple(proposals)
        self.ask_count = 0

    @override
    def create_initial_state(self) -> RollingStaleAsyncOptimizerState:
        return RollingStaleAsyncOptimizerState(
            queued_proposals=self._initial_proposals,
        )

    @override
    def is_exhausted(self, state: RollingStaleAsyncOptimizerState) -> bool:
        return len(state.queued_proposals) == 0

    @override
    def ask(
        self,
        state: RollingStaleAsyncOptimizerState,
        batch_size: int = 1,
    ) -> tuple[tuple[Proposal[int], ...], RollingStaleAsyncOptimizerState]:
        self.ask_count += 1
        proposal_batch = state.queued_proposals[:batch_size]
        return (
            proposal_batch,
            RollingStaleAsyncOptimizerState(
                queued_proposals=state.queued_proposals[len(proposal_batch) :],
                ask_history=state.ask_history + (batch_size,),
                tell_history=state.tell_history,
            ),
        )

    @override
    def tell(
        self,
        state: RollingStaleAsyncOptimizerState,
        observations: Sequence[Observation[int]],
    ) -> RollingStaleAsyncOptimizerState:
        return RollingStaleAsyncOptimizerState(
            queued_proposals=state.queued_proposals,
            ask_history=state.ask_history,
            tell_history=state.tell_history + (tuple(observations),),
        )

    @override
    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        return frozenset({STALE_ASYNC_EXECUTION_MODEL})


@final
class AttemptOnlyEvaluator:
    """Evaluator that can evaluate attempts but cannot open async sessions."""

    evaluate_attempt_count: int

    def __init__(self) -> None:
        self.evaluate_attempt_count = 0

    def execution_resources(self) -> ExecutionResources:
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
        )

    def evaluate_attempts(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        _ = problem, requests
        self.evaluate_attempt_count += 1
        return EvaluationAttemptBatch(attempts=())


@final
class PendingAskUnsafeRollingStaleAsyncOptimizer(
    RunMethod[RollingStaleAsyncOptimizerState, Proposal[int], Observation[int]],
):
    """Rolling optimizer whose post-ask state is not checkpoint-safe."""

    _initial_proposals: tuple[Proposal[int], ...]

    def __init__(self, proposals: Sequence[Proposal[int]]) -> None:
        self._initial_proposals = tuple(proposals)

    @override
    def create_initial_state(self) -> RollingStaleAsyncOptimizerState:
        return RollingStaleAsyncOptimizerState(
            queued_proposals=self._initial_proposals,
        )

    @override
    def is_exhausted(self, state: RollingStaleAsyncOptimizerState) -> bool:
        return len(state.queued_proposals) == 0

    @override
    def ask(
        self,
        state: RollingStaleAsyncOptimizerState,
        batch_size: int = 1,
    ) -> tuple[tuple[Proposal[int], ...], RollingStaleAsyncOptimizerState]:
        proposal_batch = state.queued_proposals[:batch_size]
        return (
            proposal_batch,
            RollingStaleAsyncOptimizerState(
                queued_proposals=state.queued_proposals[len(proposal_batch) :],
                ask_history=state.ask_history + (batch_size,),
                tell_history=state.tell_history,
            ),
        )

    @override
    def tell(
        self,
        state: RollingStaleAsyncOptimizerState,
        observations: Sequence[Observation[int]],
    ) -> RollingStaleAsyncOptimizerState:
        spawned_proposals = tuple(
            Proposal(
                candidate=observation.candidate + 10,
                proposal_id=f"spawn-{observation.proposal.proposal_id}",
            )
            for observation in observations
            if observation.candidate < 10
        )
        return RollingStaleAsyncOptimizerState(
            queued_proposals=state.queued_proposals + spawned_proposals,
            ask_history=state.ask_history,
            tell_history=state.tell_history + (tuple(observations),),
        )

    @override
    def is_checkpoint_safe_state(
        self,
        state: RollingStaleAsyncOptimizerState,
    ) -> bool:
        return len(state.ask_history) == len(state.tell_history)

    @override
    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        return frozenset({STALE_ASYNC_EXECUTION_MODEL})


@final
class TellFailingStaleAsyncOptimizer(
    RunMethod[RollingStaleAsyncOptimizerState, Proposal[int], Observation[int]],
):
    """Stale-async optimizer that fails when assimilating completed records."""

    _initial_proposals: tuple[Proposal[int], ...]

    def __init__(self, proposals: Sequence[Proposal[int]]) -> None:
        self._initial_proposals = tuple(proposals)

    @override
    def create_initial_state(self) -> RollingStaleAsyncOptimizerState:
        return RollingStaleAsyncOptimizerState(
            queued_proposals=self._initial_proposals,
        )

    @override
    def is_exhausted(self, state: RollingStaleAsyncOptimizerState) -> bool:
        return len(state.queued_proposals) == 0

    @override
    def ask(
        self,
        state: RollingStaleAsyncOptimizerState,
        batch_size: int = 1,
    ) -> tuple[tuple[Proposal[int], ...], RollingStaleAsyncOptimizerState]:
        proposal_batch = state.queued_proposals[:batch_size]
        return (
            proposal_batch,
            RollingStaleAsyncOptimizerState(
                queued_proposals=state.queued_proposals[len(proposal_batch) :],
                ask_history=state.ask_history + (batch_size,),
                tell_history=state.tell_history,
            ),
        )

    @override
    def tell(
        self,
        state: RollingStaleAsyncOptimizerState,
        observations: Sequence[Observation[int]],
    ) -> RollingStaleAsyncOptimizerState:
        _ = state, observations
        msg = "forced stale tell failure"
        raise RuntimeError(msg)

    @override
    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        return frozenset({STALE_ASYNC_EXECUTION_MODEL})


@final
class ScriptedNonBlockingAttemptBatchSession(
    EvaluationBatchSession[StaleAsyncAttemptBatch],
):
    """Batch session whose poll results follow a deterministic script."""

    _handle: EvaluationBatchHandle
    _poll_results: list[tuple[StaleAsyncAttemptCompletionGroup, ...]]
    _cancelled_handles: list[EvaluationBatchHandle] | None

    def __init__(
        self,
        *,
        handle: EvaluationBatchHandle,
        poll_results: Sequence[tuple[StaleAsyncAttemptCompletionGroup, ...]],
        cancelled_handles: list[EvaluationBatchHandle] | None = None,
    ) -> None:
        self._handle = handle
        self._poll_results = list(poll_results)
        self._cancelled_handles = cancelled_handles

    @property
    @override
    def handle(self) -> EvaluationBatchHandle:
        return self._handle

    @override
    def poll(
        self,
    ) -> tuple[StaleAsyncAttemptCompletionGroup, ...]:
        if len(self._poll_results) == 0:
            return ()
        return self._poll_results.pop(0)

    @override
    def cancel(self) -> None:
        if self._cancelled_handles is not None:
            self._cancelled_handles.append(self.handle)
        self._poll_results.clear()


@final
class BudgetExhaustingAttemptAsyncEvaluator(
    AsyncEvaluator[
        Problem[int, int],
        EvaluationRequest[int],
        StaleAsyncOutcome,
    ],
):
    """Attempt-session evaluator whose completions overrun logical budget."""

    _next_batch_id: int
    _evaluation_count: int
    _cancelled_handles: list[EvaluationBatchHandle]

    def __init__(self, *, evaluation_count: int) -> None:
        self._next_batch_id = 0
        self._evaluation_count = evaluation_count
        self._cancelled_handles = []

    @property
    def cancelled_batch_ids(self) -> tuple[str, ...]:
        """Return batch ids cancelled through active-session cleanup."""
        return tuple(handle.batch_id for handle in self._cancelled_handles)

    @override
    def open_session(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchSession[StaleAsyncOutcome]:
        _ = problem, requests
        msg = "budget-exhaustion test double only supports attempt sessions"
        raise NotImplementedError(msg)

    def open_attempt_session(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchSession[StaleAsyncAttemptBatch]:
        handle = EvaluationBatchHandle(
            batch_id=f"budget-attempt-batch-{self._next_batch_id}",
            request_count=len(requests),
        )
        self._next_batch_id += 1
        completion_groups = tuple(
            CompletionGroup(
                start_index=index,
                outcomes=(
                    self._with_reported_evaluation_count(
                        make_observation_payload_attempt(
                            problem=problem,
                            request=request,
                        ),
                    ),
                ),
            )
            for index, request in reversed(tuple(enumerate(requests)))
        )
        return ScriptedNonBlockingAttemptBatchSession(
            handle=handle,
            poll_results=tuple((group,) for group in completion_groups),
            cancelled_handles=self._cancelled_handles,
        )

    def _with_reported_evaluation_count(
        self,
        attempt: StaleAsyncAttemptBatch,
    ) -> StaleAsyncAttemptBatch:
        success = attempt.single_success_or_none()
        if success is None:
            return attempt

        return EvaluationAttemptBatch(
            attempts=(
                EvaluationSuccess(
                    request=success.request,
                    payload=success.payload,
                    evaluation_count=self._evaluation_count,
                    refinement=success.refinement,
                    kernel_diagnostics=success.kernel_diagnostics,
                ),
            ),
        )

    @override
    def submit_batch(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchHandle:
        _ = problem, requests
        msg = "budget-exhaustion test double only supports attempt sessions"
        raise NotImplementedError(msg)

    @override
    def poll(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[StaleAsyncCompletionGroup]:
        _ = handle
        msg = "budget-exhaustion test double only supports attempt polling"
        raise NotImplementedError(msg)

    @override
    def cancel(self, handle: EvaluationBatchHandle) -> None:
        _ = handle
        msg = "budget-exhaustion test double only supports attempt cancellation"
        raise NotImplementedError(msg)


@final
class HolAvoidanceAsyncEvaluator(
    AsyncEvaluator[
        Problem[int, int],
        EvaluationRequest[int],
        StaleAsyncOutcome,
    ],
):
    """Async evaluator with one empty earlier poll before a ready later batch."""

    _next_batch_id: int
    _sessions: dict[str, ScriptedNonBlockingAttemptBatchSession]

    def __init__(self) -> None:
        self._next_batch_id = 0
        self._sessions = {}

    @override
    def open_session(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchSession[StaleAsyncOutcome]:
        _ = problem, requests
        msg = "HOL avoidance test double only supports attempt-batch sessions"
        raise NotImplementedError(msg)

    def open_attempt_session(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchSession[StaleAsyncAttemptBatch]:
        handle = EvaluationBatchHandle(
            batch_id=f"attempt-batch-{self._next_batch_id}",
            request_count=len(requests),
        )
        self._next_batch_id += 1
        attempts: tuple[StaleAsyncAttemptBatch, ...] = tuple(
            make_observation_payload_attempt(problem=problem, request=request)
            for request in requests
        )
        poll_results: tuple[tuple[StaleAsyncAttemptCompletionGroup, ...], ...]
        if handle.batch_id == "attempt-batch-0":
            poll_results = (
                (CompletionGroup(start_index=1, outcomes=(attempts[1],)),),
                (),
                (CompletionGroup(start_index=0, outcomes=(attempts[0],)),),
            )
        else:
            poll_results = (
                (CompletionGroup(start_index=0, outcomes=(attempts[0],)),),
            )
        session = ScriptedNonBlockingAttemptBatchSession(
            handle=handle,
            poll_results=poll_results,
        )
        self._sessions[handle.batch_id] = session
        return session

    @override
    def submit_batch(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchHandle:
        return self.open_attempt_session(problem, requests).handle

    @override
    def poll(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[StaleAsyncCompletionGroup]:
        _ = handle
        msg = "HOL avoidance test double only supports attempt-batch polling"
        raise NotImplementedError(msg)

    @override
    def cancel(self, handle: EvaluationBatchHandle) -> None:
        _ = handle
        msg = "HOL avoidance test double only supports attempt-batch cancellation"
        raise NotImplementedError(msg)

    def poll_attempts(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[StaleAsyncAttemptCompletionGroup]:
        return self._sessions[handle.batch_id].poll()

    def cancel_attempts(self, handle: EvaluationBatchHandle) -> None:
        self._sessions[handle.batch_id].cancel()


class StudyStaleAsyncTests:
    """Coverage for stale-async Study execution behavior."""

    def test_step_rejects_stale_async_model_outside_run_or_optimize(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(Proposal(candidate=3, proposal_id="p-1"),),
        )
        evaluator = OutOfOrderAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )
        state = optimizer.create_initial_state()

        with pytest.raises(
            NotImplementedError,
            match="stale_async execution model is only supported by Study.run and Study.optimize",
        ):
            _ = study.step(
                state,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
            )

    def test_run_preflights_attempt_session_capability_before_ask_or_budget(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = AskRecordingStaleAsyncOptimizer(
            proposals=(Proposal(candidate=3, proposal_id="p-1"),),
        )
        evaluator = AttemptOnlyEvaluator()
        state = optimizer.create_initial_state()
        budget = EvaluationBudget(1)

        with pytest.raises(
            TypeError,
            match="stale_async evaluator must expose attempt-batch sessions",
        ):
            _ = open_stale_async_batch_session(
                async_evaluator=evaluator,
                problem=problem,
                run_method_ask=optimizer.ask,
                proposal_evaluation_specs_for=optimizer.proposal_evaluation_specs,
                state=state,
                batch_size=1,
                evaluation_budget=budget,
            )

        assert optimizer.ask_count == 0
        assert evaluator.evaluate_attempt_count == 0
        assert budget.remaining == 1

    def test_run_stale_async_assimilates_incrementally_and_refills_frontier(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=2, proposal_id="p-2"),
            ),
        )
        evaluator = OutOfOrderAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        report, final_state = study.run(
            max_evaluations=4,
            batch_size=2,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
        )

        assert final_state.ask_history == (2, 1, 1)
        assert tuple(
            observation.proposal.proposal_id
            for observation_batch in final_state.tell_history
            for observation in observation_batch
        ) == ("p-2", "p-1", "spawn-p-2", "spawn-p-1")
        assert tuple(
            observation.proposal.proposal_id for observation in report.records
        ) == ("p-2", "p-1", "spawn-p-2", "spawn-p-1")
        assert report.refinements == ()

    def test_run_stale_async_refill_respects_default_evaluation_budget(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=2, proposal_id="p-2"),
            ),
        )
        evaluator = SessionRecordingAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        report, final_state = study.run(
            max_evaluations=3,
            batch_size=2,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
        )

        assert evaluator.opened_batch_sizes == (2, 1)
        assert final_state.ask_history == (2, 1)
        assert report.evaluation_count == 3
        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-2",
            "p-1",
            "spawn-p-2",
        )

    @pytest.mark.parametrize("count_evaluation_cost", [True, False])
    def test_run_stale_async_rejects_negative_budget_before_opening_session(
        self,
        *,
        count_evaluation_cost: bool,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(Proposal(candidate=4, proposal_id="p-1"),),
        )
        evaluator = SessionRecordingAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        with pytest.raises(ValueError, match="max_evaluations must be non-negative"):
            _ = study.run(
                max_evaluations=-1,
                batch_size=1,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
                count_evaluation_cost=count_evaluation_cost,
            )

        assert evaluator.opened_batch_sizes == ()

    def test_optimize_stale_async_rejects_negative_budget_before_opening_session(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(Proposal(candidate=4, proposal_id="p-1"),),
        )
        evaluator = SessionRecordingAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        with pytest.raises(ValueError, match="max_evaluations must be non-negative"):
            _ = study.optimize(
                max_evaluations=-1,
                batch_size=1,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
            )

        assert evaluator.opened_batch_sizes == ()

    def test_run_stale_async_stops_refill_after_checkpoint_safe_boundary(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = PendingAskUnsafeRollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=2, proposal_id="p-2"),
            ),
        )
        evaluator = CancellationRecordingSessionAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        report, final_state = study.run(
            max_evaluations=4,
            batch_size=2,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
            stop_at_checkpoint_boundary=True,
        )

        assert evaluator.opened_batch_sizes == (2,)
        assert evaluator.cancelled_batch_ids == ("attempt-batch-0",)
        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-2",
        )
        assert final_state.tell_history == ((report.records[0],),)

    def test_run_stale_async_keeps_refilling_without_unsafe_checkpoint_segment(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=2, proposal_id="p-2"),
            ),
        )
        evaluator = SessionRecordingAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        report, final_state = study.run(
            max_evaluations=3,
            batch_size=2,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
            stop_at_checkpoint_boundary=True,
        )

        assert evaluator.opened_batch_sizes == (2, 1)
        assert final_state.ask_history == (2, 1)
        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-2",
            "p-1",
            "spawn-p-2",
        )

    def test_run_stale_async_checkpoint_snapshots_do_not_materialize_trace_each_step(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=2, proposal_id="p-2"),
            ),
        )
        evaluator = SessionRecordingAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )
        trace_counter = StaleAsyncTraceFactoryCounter()
        monkeypatch.setattr(stale_async_execution, "Trace", trace_counter)

        report, final_state = study.run(
            max_evaluations=3,
            batch_size=2,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
            stop_at_checkpoint_boundary=True,
        )

        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-2",
            "p-1",
            "spawn-p-2",
        )
        assert final_state.ask_history == (2, 1)
        assert len(trace_counter.calls) == 1
        assert len(trace_counter.calls[0]) == 3

    def test_run_stale_async_safe_boundary_return_survives_cancel_failure(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = PendingAskUnsafeRollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=2, proposal_id="p-2"),
            ),
        )
        evaluator = CancelFailingSecondBatchAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        report, final_state = study.run(
            max_evaluations=4,
            batch_size=2,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
            stop_at_checkpoint_boundary=True,
        )

        assert evaluator.cancelled_batch_ids == ("attempt-batch-0",)
        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-2",
        )
        assert final_state.tell_history == ((report.records[0],),)

    def test_run_stale_async_returns_checkpoint_snapshot_after_budget_exhaustion(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = PendingAskUnsafeRollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=2, proposal_id="p-2"),
            ),
        )
        evaluator = BudgetExhaustingAttemptAsyncEvaluator(evaluation_count=7)
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        report, final_state = study.run(
            max_evaluations=2,
            batch_size=2,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
            stop_at_checkpoint_boundary=True,
        )

        assert report.records == ()
        assert report.failures == ()
        assert report.evaluation_count == 0
        assert final_state == optimizer.create_initial_state()
        assert evaluator.cancelled_batch_ids == ("budget-attempt-batch-0",)

    def test_run_stale_async_keeps_budget_exhaustion_without_checkpoint_boundary(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = PendingAskUnsafeRollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=2, proposal_id="p-2"),
            ),
        )
        evaluator = BudgetExhaustingAttemptAsyncEvaluator(evaluation_count=7)
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        with pytest.raises(EvaluationBudgetExhausted):
            _ = study.run(
                max_evaluations=2,
                batch_size=2,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
            )

        assert evaluator.cancelled_batch_ids == ("budget-attempt-batch-0",)

    def test_run_stale_async_materializes_payload_groups_before_feedback(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=2, proposal_id="p-2"),
            ),
        )
        evaluator = PayloadResumableOutOfOrderAsyncEvaluator()
        study: Study[
            int,
            int,
            RollingStaleAsyncOptimizerState,
            ObservationPayload,
            Observation[int],
        ] = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        report, final_state = study.run(
            max_evaluations=3,
            batch_size=2,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
        )

        assert final_state.ask_history == (2, 1)
        assert all(isinstance(record, Observation) for record in report.records)
        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-2",
            "p-1",
            "spawn-p-2",
        )
        assert tuple(
            record.proposal.proposal_id
            for record_batch in final_state.tell_history
            for record in record_batch
        ) == ("p-2", "p-1", "spawn-p-2")

    def test_run_stale_async_polls_ready_later_session_after_empty_earlier_session(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=2, proposal_id="p-2"),
            ),
        )
        evaluator = HolAvoidanceAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        report, final_state = study.run(
            max_evaluations=3,
            batch_size=2,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-2",
            "spawn-p-2",
            "p-1",
        )
        assert tuple(
            observation.proposal.proposal_id
            for observation_batch in final_state.tell_history
            for observation in observation_batch
        ) == ("p-2", "spawn-p-2", "p-1")

    def test_run_stale_async_records_out_of_order_attempt_failures(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=FailingCandidateObjective(failed_candidates=(5,)),
        )
        optimizer = FailureRecordingBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=2, proposal_id="p-1"),
                    Proposal(candidate=5, proposal_id="p-2"),
                    Proposal(candidate=1, proposal_id="p-3"),
                ),
            ],
        )
        evaluator = AttemptOutOfOrderAsyncEvaluator()
        study: StaleAsyncFailureRecordingStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        report, final_state = study.run(
            max_evaluations=3,
            batch_size=3,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-3",
            "p-1",
        )
        assert all(type(success.payload) is Observation for success in report.successes)
        assert tuple(failure.proposal_id for failure in report.failures) == ("p-2",)
        assert report.evaluation_count == 3
        assert tuple(
            failure_proposal_id
            for failure_batch in final_state.failure_history
            for failure_proposal_id in failure_batch
        ) == ("p-2",)
        assert tuple(
            record.proposal.proposal_id
            for record_batch in final_state.tell_history
            for record in record_batch
        ) == ("p-3", "p-1")

    def test_run_stale_async_tell_failure_preserves_completed_attempts(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=FailingCandidateObjective(failed_candidates=(5,)),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=2, proposal_id="p-1"),
                Proposal(candidate=5, proposal_id="p-2"),
            ),
        )
        evaluator = AttemptOutOfOrderAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        try:
            _ = study.run(
                max_evaluations=2,
                batch_size=2,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
            )
        except RuntimeError as raw_exception:
            assert isinstance(raw_exception, StaleAsyncRunFailure)
            exception: StaleAsyncRunFailure = raw_exception
            assert type(raw_exception) is RunExecutionFailed
        else:
            pytest.fail("expected stale-async tell failure")

        assert isinstance(exception.cause, UnsupportedEvaluationFailureError)
        assert exception.partial_report.records == ()
        assert tuple(
            failure.proposal_id for failure in exception.partial_report.failures
        ) == ("p-2",)
        assert exception.partial_report.evaluation_count == 2
        assert exception.partial_state.tell_history == ()

    def test_run_stale_async_tell_failure_preserves_successful_completed_group(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = TellFailingStaleAsyncOptimizer(
            proposals=(Proposal(candidate=2, proposal_id="p-1"),),
        )
        evaluator = OutOfOrderAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        try:
            _ = study.run(
                max_evaluations=1,
                batch_size=1,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
            )
        except RuntimeError as raw_exception:
            assert isinstance(raw_exception, StaleAsyncRunFailure)
            exception: StaleAsyncRunFailure = raw_exception
            assert type(raw_exception) is RunExecutionFailed
        else:
            pytest.fail("expected stale-async tell failure")

        assert isinstance(exception.cause, RuntimeError)
        assert tuple(
            record.proposal.proposal_id for record in exception.partial_report.records
        ) == ("p-1",)
        assert exception.partial_report.failures == ()
        assert exception.partial_report.evaluation_count == 1
        assert exception.partial_state.tell_history == ()

    def test_run_stale_async_cancels_refill_opened_before_mid_sweep_failure(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=2, proposal_id="p-2"),
            ),
        )
        evaluator = FailingSecondBatchAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        try:
            _ = study.run(
                max_evaluations=4,
                batch_size=2,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
            )
        except RuntimeError as exception:
            exception_message = str(exception)
            assert type(exception) is RunExecutionFailed
            assert "attempt-batch-1" in exception_message
            assert isinstance(exception.__cause__, BatchExecutionFailed)
        else:
            pytest.fail("expected stale-async run failure")

        assert "attempt-batch-2" in evaluator.cancelled_batch_ids

    def test_run_stale_async_cancels_remaining_sessions_after_cancel_failure(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=2, proposal_id="p-2"),
            ),
        )
        evaluator = CancelFailingSecondBatchAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        try:
            _ = study.run(
                max_evaluations=4,
                batch_size=2,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
            )
        except RuntimeError as exception:
            exception_message = str(exception)
            assert type(exception) is RunExecutionFailed
            assert "attempt-batch-1" in exception_message
            assert isinstance(exception.__cause__, BatchExecutionFailed)
        else:
            pytest.fail("expected stale-async run failure")

        assert evaluator.cancelled_batch_ids == (
            "attempt-batch-0",
            "attempt-batch-1",
            "attempt-batch-2",
        )

    def test_run_stale_async_buffers_trace_events_without_trace_append(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def reject_trace_append(_trace: Trace, _event: TraceEvent) -> Trace:
            raise AssertionError(
                "stale-async run should buffer trace events before materialization",
            )

        monkeypatch.setattr(Trace, "append", reject_trace_append)
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=2, proposal_id="p-2"),
            ),
        )
        evaluator = OutOfOrderAsyncEvaluator()
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        report, _ = study.run(
            max_evaluations=2,
            batch_size=2,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-2",
            "p-1",
        )
        assert len(report.trace.events) == 2

    def test_run_stale_async_preserves_refinement_completion_order(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=14, proposal_id="p-1"),
                Proposal(candidate=12, proposal_id="p-2"),
            ),
        )
        evaluator = OutOfOrderAsyncEvaluator(attach_refinement=True)
        study: StaleAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        report, final_state = study.run(
            max_evaluations=2,
            batch_size=2,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-2",
            "p-1",
        )
        assert tuple(record.candidate for record in report.records) == (12, 14)
        assert len(report.refinements) == 2
        first_refinement = report.refinements[0]
        second_refinement = report.refinements[1]
        assert first_refinement is not None
        assert second_refinement is not None
        assert first_refinement.source_candidate == 12
        assert first_refinement.refined_candidate == 12
        assert second_refinement.source_candidate == 14
        assert second_refinement.refined_candidate == 14
        assert tuple(
            observation.proposal.proposal_id
            for observation_batch in final_state.tell_history
            for observation in observation_batch
        ) == ("p-2", "p-1")

    def test_run_stale_async_uses_space_candidate_equality_for_refinement(
        self,
    ) -> None:
        problem = Problem(
            space=SpaceOwnedEqualitySpace(),
            objective=SpaceOwnedEqualityObjective(),
        )
        optimizer = SpaceOwnedEqualityOptimizer()
        evaluator = SpaceOwnedEqualityAsyncEvaluator()
        study: StaleAsyncSpaceOwnedEqualityStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        report, _ = study.run(
            max_evaluations=1,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
        )

        assert len(report.refinements) == 1
        refinement = report.refinements[0]
        assert refinement is not None
        assert (
            refinement.refined_candidate.stable_id
            == report.records[0].candidate.stable_id
        )

    def test_optimize_stale_async_uses_space_candidate_equality_for_refinement(
        self,
    ) -> None:
        problem = Problem(
            space=SpaceOwnedEqualitySpace(),
            objective=SpaceOwnedEqualityObjective(),
        )
        optimizer = SpaceOwnedEqualityOptimizer()
        evaluator = SpaceOwnedEqualityAsyncEvaluator()
        study: StaleAsyncSpaceOwnedEqualityStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        result, _ = study.optimize(
            max_evaluations=1,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
        )

        assert len(result.refinements) == 1
        refinement = result.refinements[0]
        assert refinement is not None
        assert (
            refinement.refined_candidate.stable_id
            == result.observations[0].candidate.stable_id
        )

    def test_optimize_stale_async_projects_completion_order_refinements(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(
                Proposal(candidate=14, proposal_id="p-1"),
                Proposal(candidate=12, proposal_id="p-2"),
            ),
        )
        evaluator = OutOfOrderAsyncEvaluator(attach_refinement=True)
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, final_state = study.optimize(
            max_evaluations=2,
            batch_size=2,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(
            observation.proposal.proposal_id for observation in result.observations
        ) == ("p-2", "p-1")
        assert tuple(observation.candidate for observation in result.observations) == (
            12,
            14,
        )
        assert len(result.refinements) == 2
        first_refinement = result.refinements[0]
        second_refinement = result.refinements[1]
        assert first_refinement is not None
        assert second_refinement is not None
        assert first_refinement.source_candidate == 12
        assert first_refinement.refined_candidate == 12
        assert second_refinement.source_candidate == 14
        assert second_refinement.refined_candidate == 14
        assert tuple(
            observation.proposal.proposal_id
            for observation_batch in final_state.tell_history
            for observation in observation_batch
        ) == ("p-2", "p-1")

    def test_run_stale_async_rejects_non_direct_kernel(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=20),
            objective=SquareObjective(),
        )
        optimizer = RollingStaleAsyncOptimizer(
            proposals=(Proposal(candidate=4, proposal_id="p-1"),),
        )
        evaluator = OutOfOrderAsyncEvaluator()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=RecordingKernel(),
        )

        with pytest.raises(
            ValueError,
            match="stale_async execution model currently requires DirectKernel",
        ):
            _ = study.run(
                max_evaluations=1,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
            )
