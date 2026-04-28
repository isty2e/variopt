"""Tests for problem and interaction contract surfaces."""

import dataclasses

import pytest

from tests.problem_artifact_support import (
    LabelProtocol,
    MatchupProtocol,
    MatchupSpec,
    ShiftedObservationProtocol,
    SquareObjective,
)
from variopt import (
    EvaluationRequest,
    IntegerSpace,
    InteractionProblem,
    Observation,
    OptimizationDirection,
    Problem,
    Proposal,
)
from variopt.artifacts import InteractionEvaluationUnit


class ProblemContractsTests:
    """Coverage for problem construction and interaction contract validation."""

    def test_problem_is_frozen(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
            name="square",
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(problem, "name", "other")

    def test_problem_accepts_evaluation_protocol_basis(self) -> None:
        protocol = ShiftedObservationProtocol()
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=protocol,
            direction=OptimizationDirection.MAXIMIZE,
            name="shifted",
        )

        evaluation_record = problem.evaluation_protocol.evaluate_proposal(
            Proposal(candidate=4),
        )
        assert isinstance(evaluation_record, Observation)
        assert evaluation_record.value == 9.0
        assert evaluation_record.score == -9.0
        assert problem.objective.evaluate(4) == 9.0

    def test_problem_non_scalar_protocol_has_no_objective_compatibility_view(self) -> None:
        protocol = LabelProtocol()
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=protocol,
        )

        assert problem.evaluation_protocol is protocol
        with pytest.raises(TypeError):
            _ = problem.objective

    def test_interaction_evaluation_unit_requires_non_empty_requests(self) -> None:
        with pytest.raises(ValueError):
            _ = InteractionEvaluationUnit[int](requests=())

    def test_interaction_evaluation_unit_exposes_request_compatibility_views(
        self,
    ) -> None:
        first_request = EvaluationRequest(proposal=Proposal(candidate=3, proposal_id="a"))
        second_request = EvaluationRequest(proposal=Proposal(candidate=5, proposal_id="b"))
        interaction_unit = InteractionEvaluationUnit(
            requests=(first_request, second_request),
            interaction_evaluation_spec=MatchupSpec(arena="ladder"),
        )

        assert interaction_unit.request_count == 2
        assert interaction_unit.candidates == (3, 5)
        assert tuple(proposal.proposal_id for proposal in interaction_unit.proposals) == ("a", "b")
        assert isinstance(interaction_unit.interaction_evaluation_spec, MatchupSpec)

    def test_interaction_protocol_evaluates_grouped_requests(self) -> None:
        protocol = MatchupProtocol()
        interaction_record = protocol.evaluate_requests(
            (
                EvaluationRequest(proposal=Proposal(candidate=2, proposal_id="left")),
                EvaluationRequest(proposal=Proposal(candidate=7, proposal_id="right")),
            ),
            interaction_evaluation_spec=MatchupSpec(arena="tournament"),
        )

        assert interaction_record.winner == 7
        assert interaction_record.arena == "tournament"
        assert interaction_record.candidates == (2, 7)

    def test_interaction_problem_binds_space_and_interaction_protocol(self) -> None:
        protocol = MatchupProtocol()
        problem = InteractionProblem(
            space=IntegerSpace(low=0, high=10),
            interaction_evaluation_protocol=protocol,
            name="matchup",
        )

        assert problem.interaction_evaluation_protocol is protocol
        assert problem.name == "matchup"

    def test_interaction_problem_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            _ = InteractionProblem(
                space=IntegerSpace(low=0, high=10),
                interaction_evaluation_protocol=MatchupProtocol(),
                name="",
            )

    def test_problem_rejects_missing_objective_and_protocol(self) -> None:
        with pytest.raises(ValueError):
            _ = Problem(
                space=IntegerSpace(low=0, high=10),
            )

    def test_problem_rejects_both_objective_and_protocol(self) -> None:
        with pytest.raises(ValueError):
            _ = Problem(
                space=IntegerSpace(low=0, high=10),
                objective=SquareObjective(),
                evaluation_protocol=ShiftedObservationProtocol(),
            )

    def test_problem_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            _ = Problem(
                space=IntegerSpace(low=0, high=10),
                objective=SquareObjective(),
                name="",
            )

    def test_problem_rejects_non_direction_value(self) -> None:
        with pytest.raises(TypeError):
            _ = Problem(
                space=IntegerSpace(low=0, high=10),
                objective=SquareObjective(),
                direction="maximize",  # pyright: ignore[reportArgumentType]
            )
