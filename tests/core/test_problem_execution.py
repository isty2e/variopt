"""Tests for evaluator-backed problem execution paths."""

import pytest

from tests.problem_artifact_support import (
    LabelProtocol,
    LabelRecord,
    NaNObjective,
    ShiftedObservationProtocol,
)
from variopt import EvaluationRequest, IntegerSpace, Problem, Proposal
from variopt.evaluators import SequentialEvaluator


class ProblemExecutionTests:
    """Coverage for evaluator-backed scalar and non-scalar problem execution."""

    def test_sequential_evaluator_rejects_non_finite_objective_values(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=NaNObjective(),
        )
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(ValueError):
            _ = SequentialEvaluator[int, int]().evaluate(
                problem,
                (EvaluationRequest(proposal=proposal),),
            )

    def test_sequential_evaluator_uses_evaluation_protocol_canonically(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        proposal = Proposal(candidate=4, proposal_id="p-1")

        outcomes = SequentialEvaluator[int, int]().evaluate(
            problem,
            (EvaluationRequest(proposal=proposal),),
        )

        assert len(outcomes) == 1
        assert outcomes[0].observation.proposal == proposal
        assert outcomes[0].observation.candidate == 3
        assert outcomes[0].observation.value == 9.0

    def test_sequential_evaluator_supports_non_scalar_evaluation_records(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=LabelProtocol(),
        )
        proposal = Proposal(candidate=4, proposal_id="p-1")

        outcomes = SequentialEvaluator[int, int, LabelRecord]().evaluate(
            problem,
            (EvaluationRequest(proposal=proposal),),
        )

        assert len(outcomes) == 1
        assert outcomes[0].record.proposal == proposal
        assert outcomes[0].record.candidate == 4
        assert outcomes[0].record.label == "parity:0"
        with pytest.raises(TypeError):
            _ = outcomes[0].observation
