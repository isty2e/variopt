"""Tests for stale-async Study execution."""

import pytest

from tests.study_support import (
    OutOfOrderAsyncEvaluator,
    RecordingKernel,
    RollingStaleAsyncOptimizer,
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
