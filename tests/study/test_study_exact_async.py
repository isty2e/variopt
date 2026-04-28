"""Tests for exact-async Study execution."""

import pytest

from tests.study_support import (
    ExactAsyncCapableBatchQueueOptimizer,
    OutOfOrderAsyncEvaluator,
    RecordingKernel,
    ResumableOutOfOrderAsyncEvaluator,
    SessionRecordingAsyncEvaluator,
    SquareObjective,
)
from variopt import IntegerSpace, Problem, Proposal, Study
from variopt.evaluators import EvaluationBatchSessionState, SequentialEvaluator
from variopt.execution import EXACT_ASYNC_EXECUTION_MODEL


class StudyExactAsyncTests:
    """Coverage for exact-async Study execution and session lifecycle."""

    def test_step_rejects_exact_async_model_with_non_async_evaluator(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)
        state = optimizer.create_initial_state()

        with pytest.raises(ValueError, match="ordered_async execution models require an AsyncEvaluator"):
            _ = study.step(
                state,
                execution_model=EXACT_ASYNC_EXECUTION_MODEL,
            )

    def test_step_exact_async_reorders_out_of_order_completions(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = OutOfOrderAsyncEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)
        state = optimizer.create_initial_state()

        observations, next_state = study.step(
            state,
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(observation.proposal.proposal_id for observation in observations) == ("p-1", "p-2")
        assert tuple(observation.value for observation in observations) == (16.0, 4.0)
        assert tuple(
                observation.proposal.proposal_id
                for observation in next_state.tell_history[0]
            ) == ("p-1", "p-2")

    def test_step_exact_async_opens_batch_session(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = SessionRecordingAsyncEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        _records, _next_state = study.step(
            optimizer.create_initial_state(),
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert evaluator.opened_batch_sizes == (2,)

    def test_step_exact_async_runs_through_custom_kernel(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = OutOfOrderAsyncEvaluator()
        kernel = RecordingKernel()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=kernel,
        )
        state = optimizer.create_initial_state()

        observations, _ = study.step(
            state,
            batch_size=1,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert len(kernel.queries) == 1
        assert kernel.queries[0].proposals[0].proposal_id == "p-1"
        assert observations[0].proposal.proposal_id == "p-1"

    def test_open_exact_async_step_session_rejects_non_resumable_async_evaluator(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = OutOfOrderAsyncEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        with pytest.raises(TypeError, match="study-level resumable exact_async orchestration requires a ResumableAsyncEvaluator"):
            _ = study.open_exact_async_step_session(
                optimizer.create_initial_state(),
                batch_size=1,
            )

    def test_open_exact_async_step_session_rejects_non_direct_kernel(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = ResumableOutOfOrderAsyncEvaluator()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=RecordingKernel(),
        )

        with pytest.raises(ValueError, match="study-level resumable exact_async orchestration currently requires DirectKernel"):
            _ = study.open_exact_async_step_session(
                optimizer.create_initial_state(),
                batch_size=1,
            )

    def test_suspend_and_resume_exact_async_step_session_preserves_tell_order(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = ResumableOutOfOrderAsyncEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        session = study.open_exact_async_step_session(
            optimizer.create_initial_state(),
            batch_size=2,
        )
        first_completion_groups = tuple(session.poll())
        resume_handle = session.suspend()

        assert session.state() == EvaluationBatchSessionState(
                request_count=2,
                completed_count=1,
                pending_count=1,
                lifecycle="suspended",
            )
        assert len(first_completion_groups) == 1
        assert first_completion_groups[0].start_index == 1

        resumed_session = study.resume_exact_async_step_session(resume_handle)
        observations, next_state = resumed_session.finish()

        assert tuple(observation.proposal.proposal_id for observation in observations) == ("p-1", "p-2")
        assert tuple(observation.value for observation in observations) == (16.0, 4.0)
        assert tuple(
                observation.proposal.proposal_id
                for observation in next_state.tell_history[0]
            ) == ("p-1", "p-2")
