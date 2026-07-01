"""Tests for stale-async Study execution."""

import pytest

from tests.study_support import (
    OutOfOrderAsyncEvaluator,
    RecordingKernel,
    RollingStaleAsyncOptimizer,
    ShiftedObservationProtocol,
    SquareObjective,
)
from variopt import IntegerSpace, Problem, Proposal, Study
from variopt.execution import STALE_ASYNC_EXECUTION_MODEL


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

        with pytest.raises(NotImplementedError, match="stale_async execution model is only supported by Study.run and Study.optimize"):
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
                observation.proposal.proposal_id
                for observation in report.records
            ) == ("p-2", "p-1", "spawn-p-2", "spawn-p-1")
        assert report.refinements == ()

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
        assert tuple(record.candidate for record in report.records) == (11, 13)
        assert len(report.refinements) == 2
        first_refinement = report.refinements[0]
        second_refinement = report.refinements[1]
        assert first_refinement is not None
        assert second_refinement is not None
        assert first_refinement.source_candidate == 12
        assert first_refinement.refined_candidate == 11
        assert second_refinement.source_candidate == 14
        assert second_refinement.refined_candidate == 13
        assert tuple(
                observation.proposal.proposal_id
                for observation_batch in final_state.tell_history
                for observation in observation_batch
            ) == ("p-2", "p-1")

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
                observation.proposal.proposal_id
                for observation in result.observations
            ) == ("p-2", "p-1")
        assert tuple(observation.candidate for observation in result.observations) == (
            11,
            13,
        )
        assert len(result.refinements) == 2
        first_refinement = result.refinements[0]
        second_refinement = result.refinements[1]
        assert first_refinement is not None
        assert second_refinement is not None
        assert first_refinement.source_candidate == 12
        assert first_refinement.refined_candidate == 11
        assert second_refinement.source_candidate == 14
        assert second_refinement.refined_candidate == 13
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

        with pytest.raises(ValueError, match="stale_async execution model currently requires DirectKernel"):
            _ = study.run(
                max_evaluations=1,
                execution_model=STALE_ASYNC_EXECUTION_MODEL,
            )
