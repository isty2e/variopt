"""Tests for exact-async Study execution."""

import pytest
from typing_extensions import override

from tests.study_support import (
    ExactAsyncCapableBatchQueueOptimizer,
    OutcomeAwareBatchQueueOptimizer,
    OutOfOrderAsyncEvaluator,
    RecordingKernel,
    RefinementKernel,
    ResumableOutOfOrderAsyncEvaluator,
    SessionRecordingAsyncEvaluator,
    ShiftedObservationProtocol,
    SquareObjective,
)
from variopt import EvaluationRequest, IntegerSpace, Problem, Proposal, Study
from variopt.artifacts import ProposalEvaluationSpec
from variopt.evaluators import EvaluationBatchSessionState, SequentialEvaluator
from variopt.execution import (
    EXACT_ASYNC_EXECUTION_MODEL,
    SEQUENTIAL_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
    ExecutionModel,
)
from variopt.study.common import build_evaluation_requests


class OutcomeAwareExactAsyncBatchQueueOptimizer(OutcomeAwareBatchQueueOptimizer):
    """Outcome-aware batch optimizer that advertises exact-async compatibility."""

    @override
    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        return frozenset(
            {
                SEQUENTIAL_EXECUTION_MODEL,
                SYNC_BATCH_EXECUTION_MODEL,
                EXACT_ASYNC_EXECUTION_MODEL,
            },
        )


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

    def test_direct_step_exact_async_reuses_request_batch_for_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        build_call_count = 0
        original_builder = build_evaluation_requests

        def counting_builder(
            proposals: tuple[Proposal[int], ...],
            *,
            proposal_evaluation_specs: (
                tuple[ProposalEvaluationSpec | None, ...] | None
            ),
        ) -> tuple[EvaluationRequest[int], ...]:
            nonlocal build_call_count
            build_call_count += 1
            return original_builder(
                proposals,
                proposal_evaluation_specs=proposal_evaluation_specs,
            )

        monkeypatch.setattr(
            "variopt.study.execution.build_evaluation_requests",
            counting_builder,
        )
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = OutOfOrderAsyncEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        _ = study.step(
            optimizer.create_initial_state(),
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert build_call_count == 1

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

    def test_run_exact_async_preserves_report_refinement_order(self) -> None:
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
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=RefinementKernel(),
        )

        report, next_state = study.run(
            max_evaluations=2,
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(observation.proposal.proposal_id for observation in report.records) == (
            "p-1",
            "p-2",
        )
        assert tuple(observation.candidate for observation in report.records) == (3, 1)
        assert len(report.refinements) == 2
        first_refinement = report.refinements[0]
        second_refinement = report.refinements[1]
        assert first_refinement is not None
        assert second_refinement is not None
        assert first_refinement.source_candidate == 4
        assert second_refinement.source_candidate == 2
        assert tuple(
                observation.proposal.proposal_id
                for observation in next_state.tell_history[0]
            ) == ("p-1", "p-2")

    def test_run_exact_async_preserves_evaluator_refinement_order(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = OutOfOrderAsyncEvaluator(attach_refinement=True)
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        report, next_state = study.run(
            max_evaluations=2,
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(observation.proposal.proposal_id for observation in report.records) == (
            "p-1",
            "p-2",
        )
        assert tuple(observation.candidate for observation in report.records) == (3, 1)
        assert len(report.refinements) == 2
        first_refinement = report.refinements[0]
        second_refinement = report.refinements[1]
        assert first_refinement is not None
        assert second_refinement is not None
        assert first_refinement.source_candidate == 4
        assert first_refinement.refined_candidate == 3
        assert second_refinement.source_candidate == 2
        assert second_refinement.refined_candidate == 1
        assert tuple(
                observation.proposal.proposal_id
                for observation in next_state.tell_history[0]
            ) == ("p-1", "p-2")

    def test_optimize_exact_async_projects_evaluator_refinements(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = OutOfOrderAsyncEvaluator(attach_refinement=True)
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, next_state = study.optimize(
            max_evaluations=2,
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(observation.proposal.proposal_id for observation in result.observations) == (
            "p-1",
            "p-2",
        )
        assert tuple(observation.candidate for observation in result.observations) == (3, 1)
        assert len(result.refinements) == 2
        first_refinement = result.refinements[0]
        second_refinement = result.refinements[1]
        assert first_refinement is not None
        assert second_refinement is not None
        assert first_refinement.source_candidate == 4
        assert first_refinement.refined_candidate == 3
        assert second_refinement.source_candidate == 2
        assert second_refinement.refined_candidate == 1
        assert tuple(
                observation.proposal.proposal_id
                for observation in next_state.tell_history[0]
            ) == ("p-1", "p-2")

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

    def test_suspend_and_resume_exact_async_step_session_preserves_refinement_payload(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = ResumableOutOfOrderAsyncEvaluator(attach_refinement=True)
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        session = study.open_exact_async_step_session(
            optimizer.create_initial_state(),
            batch_size=2,
        )
        first_completion_groups = tuple(session.poll())
        resume_handle = session.suspend()
        stored_outcome = resume_handle.ordered_outcomes[1]

        assert len(first_completion_groups) == 1
        assert stored_outcome is not None
        assert stored_outcome.refinement is not None
        assert stored_outcome.refinement.source_candidate == 2
        assert stored_outcome.refinement.refined_candidate == 1

        resumed_session = study.resume_exact_async_step_session(resume_handle)
        observations, next_state = resumed_session.finish()
        first_resumed_outcome = resumed_session.ordered_outcomes[0]
        second_resumed_outcome = resumed_session.ordered_outcomes[1]

        assert tuple(observation.candidate for observation in observations) == (3, 1)
        assert first_resumed_outcome is not None
        assert second_resumed_outcome is not None
        assert first_resumed_outcome.refinement is not None
        assert second_resumed_outcome.refinement is not None
        assert first_resumed_outcome.refinement.source_candidate == 4
        assert second_resumed_outcome.refinement.source_candidate == 2
        assert tuple(
                observation.proposal.proposal_id
                for observation in next_state.tell_history[0]
            ) == ("p-1", "p-2")

    def test_suspend_and_resume_exact_async_step_session_preserves_outcome_feedback_order(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        optimizer = OutcomeAwareExactAsyncBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = ResumableOutOfOrderAsyncEvaluator(attach_refinement=True)
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        session = study.open_exact_async_step_session(
            optimizer.create_initial_state(),
            batch_size=2,
        )
        _ = tuple(session.poll())
        resume_handle = session.suspend()

        resumed_session = study.resume_exact_async_step_session(resume_handle)
        observations, _ = resumed_session.finish()

        assert tuple(observation.proposal.proposal_id for observation in observations) == (
            "p-1",
            "p-2",
        )
        assert optimizer.seen_changed_leaf_paths == (((),), ((),))
