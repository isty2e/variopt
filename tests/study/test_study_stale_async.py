"""Tests for stale-async Study execution."""

from collections.abc import Sequence
from typing import Protocol, TypeAlias, final, runtime_checkable

import pytest
from typing_extensions import override

from tests.study_support import (
    AttemptOutOfOrderAsyncEvaluator,
    FailingCandidateObjective,
    FailureRecordingBatchQueueOptimizer,
    OutOfOrderAsyncEvaluator,
    RecordingKernel,
    RollingStaleAsyncOptimizer,
    RollingStaleAsyncOptimizerState,
    SessionRecordingAsyncEvaluator,
    ShiftedObservationProtocol,
    SpaceOwnedEqualityAsyncEvaluator,
    SpaceOwnedEqualityObjective,
    SpaceOwnedEqualityOptimizer,
    SpaceOwnedEqualitySpace,
    SquareObjective,
    make_observation_attempt,
)
from variopt import (
    EvaluationOutcome,
    EvaluationRequest,
    IntegerSpace,
    Observation,
    Problem,
    Proposal,
    RunReport,
    Study,
)
from variopt.artifacts import EvaluationAttemptBatch, Trace, TraceEvent
from variopt.evaluators import (
    AsyncEvaluator,
    BatchExecutionFailed,
    CompletionGroup,
    EvaluationBatchHandle,
    EvaluationBatchSession,
)
from variopt.execution import STALE_ASYNC_EXECUTION_MODEL

StaleAsyncOutcome: TypeAlias = EvaluationOutcome[int, Observation[int]]
StaleAsyncCompletionGroup: TypeAlias = CompletionGroup[StaleAsyncOutcome]
StaleAsyncAttemptBatch: TypeAlias = EvaluationAttemptBatch[int, Observation[int]]
StaleAsyncAttemptCompletionGroup: TypeAlias = CompletionGroup[StaleAsyncAttemptBatch]


@runtime_checkable
class StaleAsyncRunFailure(Protocol):
    """Typed shape for hard-failure assertions over stale-async runs."""

    partial_report: RunReport[int, Observation[int]]
    partial_state: RollingStaleAsyncOptimizerState
    cause: Exception


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
class ScriptedNonBlockingAttemptBatchSession(
    EvaluationBatchSession[StaleAsyncAttemptBatch],
):
    """Batch session whose poll results follow a deterministic script."""

    _handle: EvaluationBatchHandle
    _poll_results: list[tuple[StaleAsyncAttemptCompletionGroup, ...]]

    def __init__(
        self,
        *,
        handle: EvaluationBatchHandle,
        poll_results: Sequence[tuple[StaleAsyncAttemptCompletionGroup, ...]],
    ) -> None:
        self._handle = handle
        self._poll_results = list(poll_results)

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
        self._poll_results.clear()


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
            make_observation_attempt(problem=problem, request=request)
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
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)
        state = optimizer.create_initial_state()

        with pytest.raises(
            NotImplementedError,
            match="stale_async execution model is only supported by Study.run and Study.optimize",
        ):
            _ = study.step(
                state,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
            )

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
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

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
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

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
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

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
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        report, final_state = study.run(
            max_evaluations=3,
            batch_size=3,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-3",
            "p-1",
        )
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

    def test_run_stale_async_tell_failure_excludes_unassimilated_attempts(
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
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        try:
            _ = study.run(
                max_evaluations=2,
                batch_size=2,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
            )
        except RuntimeError as raw_exception:
            assert raw_exception.__class__.__name__ == "RunExecutionFailed"
            assert isinstance(raw_exception, StaleAsyncRunFailure)
            exception = raw_exception
        else:
            pytest.fail("expected stale-async tell failure")

        assert exception.cause.__class__.__name__ == "UnsupportedEvaluationFailureError"
        assert exception.partial_report.records == ()
        assert exception.partial_report.failures == ()
        assert exception.partial_report.evaluation_count == 2
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
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        try:
            _ = study.run(
                max_evaluations=4,
                batch_size=2,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
            )
        except RuntimeError as exception:
            assert exception.__class__.__name__ == "RunExecutionFailed"
            assert "attempt-batch-1" in str(exception)
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
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        try:
            _ = study.run(
                max_evaluations=4,
                batch_size=2,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
            )
        except RuntimeError as exception:
            assert exception.__class__.__name__ == "RunExecutionFailed"
            assert "attempt-batch-1" in str(exception)
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
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

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
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

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
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

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
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

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
