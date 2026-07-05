"""Tests for evaluator-backed problem execution paths."""

import pytest
from typing_extensions import override

from tests.problem_artifact_support import (
    LabelProtocol,
    LabelRecord,
    NaNObjective,
    ShiftedObservationProtocol,
)
from variopt import (
    EvaluationProtocol,
    EvaluationRequest,
    IntegerSpace,
    Objective,
    Problem,
    Proposal,
)
from variopt.evaluators import SequentialEvaluator


class ExplodingObjective(Objective[int]):
    """Objective that raises for one candidate."""

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 4:
            msg = "boom"
            raise ValueError(msg)
        return float(candidate)


class InterruptingObjective(Objective[int]):
    """Objective that raises a non-recordable interruption."""

    @override
    def evaluate(self, candidate: int) -> float:
        _ = candidate
        raise KeyboardInterrupt


class ExitingObjective(Objective[int]):
    """Objective that raises a non-recordable system exit."""

    @override
    def evaluate(self, candidate: int) -> float:
        _ = candidate
        raise SystemExit(2)


class MisalignedLabelProtocol(EvaluationProtocol[int, LabelRecord]):
    """Protocol that returns a record for the wrong request."""

    @override
    def evaluate_request(self, request: EvaluationRequest[int]) -> LabelRecord:
        return LabelRecord(
            request=EvaluationRequest(proposal=Proposal(candidate=0)),
            candidate=request.candidate,
            label="misaligned",
        )


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

    def test_sequential_evaluator_attempts_record_user_exception(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        request_one = EvaluationRequest(
            proposal=Proposal(candidate=1, proposal_id="p-1"),
        )
        request_two = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-2"),
        )
        request_three = EvaluationRequest(
            proposal=Proposal(candidate=2, proposal_id="p-3"),
        )

        attempts = SequentialEvaluator[int, int]().evaluate_attempts(
            problem,
            (request_one, request_two, request_three),
        )

        assert attempts.requests == (request_one, request_two, request_three)
        assert attempts.outcome_indices == (0, 2)
        assert attempts.failure_indices == (1,)
        assert tuple(outcome.observation.value for outcome in attempts.outcomes) == (
            1.0,
            2.0,
        )
        failure = attempts.failures[0]
        assert failure.request is request_two
        assert failure.exception.exception_type == "builtins.ValueError"
        assert failure.exception.message == "boom"
        assert attempts.evaluation_count == 3

    def test_sequential_evaluator_attempts_preserve_first_and_last_failures(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        request_one = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
        )
        request_two = EvaluationRequest(
            proposal=Proposal(candidate=1, proposal_id="p-2"),
        )
        request_three = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-3"),
        )

        attempts = SequentialEvaluator[int, int]().evaluate_attempts(
            problem,
            (request_one, request_two, request_three),
        )

        assert attempts.requests == (request_one, request_two, request_three)
        assert attempts.outcome_indices == (1,)
        assert attempts.failure_indices == (0, 2)
        assert tuple(outcome.observation.value for outcome in attempts.outcomes) == (
            1.0,
        )
        assert tuple(failure.proposal_id for failure in attempts.failures) == (
            "p-1",
            "p-3",
        )
        assert attempts.evaluation_count == 3

    def test_sequential_evaluator_attempts_support_all_failure_batch(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        request_one = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
        )
        request_two = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-2"),
        )

        attempts = SequentialEvaluator[int, int]().evaluate_attempts(
            problem,
            (request_one, request_two),
        )

        assert attempts.outcomes == ()
        assert attempts.outcome_indices == ()
        assert attempts.failure_indices == (0, 1)
        assert tuple(failure.proposal_id for failure in attempts.failures) == (
            "p-1",
            "p-2",
        )
        assert attempts.evaluation_count == 2

    def test_sequential_evaluator_attempts_support_empty_batch(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )

        attempts = SequentialEvaluator[int, int]().evaluate_attempts(problem, ())

        assert attempts.requests == ()
        assert attempts.outcomes == ()
        assert attempts.failures == ()
        assert attempts.evaluation_count == 0

    def test_sequential_evaluator_attempts_record_non_finite_objective(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=NaNObjective(),
        )
        request = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
        )

        attempts = SequentialEvaluator[int, int]().evaluate_attempts(
            problem,
            (request,),
        )

        assert attempts.outcomes == ()
        assert attempts.failure_indices == (0,)
        assert attempts.failures[0].exception.exception_type == "builtins.ValueError"

    def test_sequential_evaluator_attempts_do_not_catch_invalid_candidate(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        request = EvaluationRequest(
            proposal=Proposal(candidate=11, proposal_id="p-1"),
        )

        with pytest.raises(ValueError):
            _ = SequentialEvaluator[int, int]().evaluate_attempts(
                problem,
                (request,),
            )

    def test_sequential_evaluator_attempts_do_not_catch_keyboard_interrupt(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=InterruptingObjective(),
        )
        request = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
        )

        with pytest.raises(KeyboardInterrupt):
            _ = SequentialEvaluator[int, int]().evaluate_attempts(
                problem,
                (request,),
            )

    def test_sequential_evaluator_attempts_do_not_catch_system_exit(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExitingObjective(),
        )
        request = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
        )

        with pytest.raises(SystemExit):
            _ = SequentialEvaluator[int, int]().evaluate_attempts(
                problem,
                (request,),
            )

    def test_sequential_evaluator_attempts_do_not_record_alignment_errors(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=MisalignedLabelProtocol(),
        )
        request = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
        )

        with pytest.raises(ValueError, match="outcome record request"):
            _ = SequentialEvaluator[int, int, LabelRecord]().evaluate_attempts(
                problem,
                (request,),
            )
