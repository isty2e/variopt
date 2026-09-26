"""Process transport of built-in local-search refinement and its accounting."""

from typing import Literal

import pytest

from tests.study_support import SquareObjective
from variopt import IntegerSpace, Problem, Study
from variopt.algorithms.local_search import StructuredHillClimbKernel
from variopt.algorithms.population import CSAOptimizer
from variopt.artifacts import ObservationPayload
from variopt.evaluators import JoblibEvaluator, SequentialEvaluator


@pytest.mark.parametrize("problem_transport", ["per_request", "worker_session"])
def test_loky_refined_csa_matches_sequential_for_large_integer_candidates(
    problem_transport: Literal["per_request", "worker_session"],
) -> None:
    space = IntegerSpace(low=1000, high=1100)
    problem = Problem(space=space, objective=SquareObjective())
    optimizer = CSAOptimizer.from_space_defaults(
        space=space, bank_capacity=4, random_state=17
    )
    sequential = Study(
        problem=problem,
        run_method=optimizer,
        evaluator=SequentialEvaluator[int, int, ObservationPayload](),
        kernel=StructuredHillClimbKernel[int, int](max_steps=2),
    )
    process = Study(
        problem=problem,
        run_method=optimizer,
        evaluator=JoblibEvaluator[int, int, ObservationPayload](
            n_jobs=2, backend="loky", problem_transport=problem_transport
        ),
        kernel=StructuredHillClimbKernel[int, int](max_steps=2),
    )

    expected, expected_state = sequential.run(max_evaluations=32, batch_size=2)
    actual, actual_state = process.run(max_evaluations=32, batch_size=2)

    assert any(refinement is not None for refinement in expected.refinements)
    assert actual.evaluation_count == expected.evaluation_count == 32
    assert actual.failures == expected.failures == ()
    assert actual.refinements == expected.refinements
    assert tuple(
        (
            success.proposal_id,
            success.candidate,
            success.payload.score,
            success.evaluation_count,
        )
        for success in actual.successes
    ) == tuple(
        (
            success.proposal_id,
            success.candidate,
            success.payload.score,
            success.evaluation_count,
        )
        for success in expected.successes
    )
    assert actual_state == expected_state
