import dataclasses
import pickle
from collections.abc import Sequence
from typing import TypeVar, cast

import pytest
from typing_extensions import override

from tests.problem_artifact_support import (
    LabelProtocol,
    LabelRecord,
    MatchupProtocol,
    MatchupSpec,
    ShiftedObservationProtocol,
    SquareObjective,
)
from variopt import (
    EvaluationOutcome,
    EvaluationProtocol,
    EvaluationRequest,
    Evaluator,
    IntegerSpace,
    InteractionProblem,
    Objective,
    Observation,
    OptimizationDirection,
    Problem,
    Proposal,
    RealSpace,
    Study,
    TupleSpace,
)
from variopt.algorithms.population import CSAOptimizer
from variopt.artifacts import (
    EvaluationAttemptBatch,
    EvaluationSuccess,
    InteractionEvaluationUnit,
    ObservationPayload,
)
from variopt.kernel import DirectKernel
from variopt.spaces import SpaceCandidateValue

PickleRoundTripT = TypeVar("PickleRoundTripT")


def pickle_round_trip(value: PickleRoundTripT) -> PickleRoundTripT:
    """Return one pickle round-trip result with the input type restored."""
    return cast(PickleRoundTripT, pickle.loads(pickle.dumps(value)))


class LegacySquareObservationProtocol(EvaluationProtocol[int, Observation[int]]):
    """Legacy scalar protocol for record-bound study compatibility tests."""

    @override
    def evaluate_request(self, request: EvaluationRequest[int]) -> Observation[int]:
        value = float(request.candidate * request.candidate)
        return Observation(
            request=request,
            candidate=request.candidate,
            value=value,
            score=value,
        )


class LegacyObservationEvaluator(
    Evaluator[
        Problem[int, int, Observation[int]],
        EvaluationRequest[int],
        EvaluationOutcome[int, Observation[int]],
    ]
):
    """Record-bound evaluator for legacy study pickle compatibility tests."""

    @override
    def evaluate(
        self,
        problem: Problem[int, int, Observation[int]],
        requests: Sequence[EvaluationRequest[int]],
    ) -> Sequence[EvaluationOutcome[int, Observation[int]]]:
        return tuple(
            EvaluationOutcome(
                record=problem.evaluation_protocol.evaluate_request(request),
                evaluation_count=1,
            )
            for request in requests
        )

    def evaluate_attempts(
        self,
        problem: Problem[int, int, Observation[int]],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationAttemptBatch[int, Observation[int]]:
        """Return request-owned attempts for the current Study evaluator contract."""
        return EvaluationAttemptBatch(
            attempts=tuple(
                EvaluationSuccess(
                    request=request,
                    payload=problem.evaluation_protocol.evaluate_request(request),
                )
                for request in requests
            ),
        )


class TupleLengthObjective(Objective[tuple[SpaceCandidateValue, ...]]):
    """Objective whose identity lets Problem equality isolate space equality."""

    @override
    def evaluate(self, candidate: tuple[SpaceCandidateValue, ...]) -> float:
        return float(len(candidate))


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

    def test_problem_equality_uses_composite_space_value_equality(self) -> None:
        objective = TupleLengthObjective()
        first = Problem(
            space=TupleSpace(IntegerSpace(0, 5), RealSpace(0.0, 1.0)),
            objective=objective,
            name="tuple",
        )
        second = Problem(
            space=TupleSpace(IntegerSpace(0, 5), RealSpace(0.0, 1.0)),
            objective=objective,
            name="tuple",
        )
        different_space = Problem(
            space=TupleSpace(IntegerSpace(0, 6), RealSpace(0.0, 1.0)),
            objective=objective,
            name="tuple",
        )

        assert first == second
        assert hash(first) == hash(second)
        assert first != different_space

    def test_problem_pickle_round_trips_without_runtime_generic_metadata(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
            name="square",
        )

        restored = pickle_round_trip(problem)

        assert restored.name == "square"
        assert restored.objective.evaluate(4) == 16.0

    def test_observation_protocol_problem_pickle_round_trips(self) -> None:
        problem: Problem[int, int, ObservationPayload] = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=ShiftedObservationProtocol(),
            direction=OptimizationDirection.MAXIMIZE,
            name="shifted",
        )

        restored = pickle_round_trip(problem)
        evaluation_record = restored.evaluation_protocol.evaluate_proposal(
            Proposal(candidate=4),
        )

        assert evaluation_record.value == 9.0
        assert evaluation_record.score == -9.0
        assert restored.objective.evaluate(4) == 9.0

    def test_non_scalar_protocol_problem_pickle_round_trips(self) -> None:
        problem: Problem[int, int, LabelRecord] = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=LabelProtocol(),
        )

        restored = pickle_round_trip(problem)
        evaluation_record = restored.evaluation_protocol.evaluate_proposal(
            Proposal(candidate=5),
        )

        assert evaluation_record.label == "parity:1"
        with pytest.raises(TypeError):
            _ = restored.objective

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
        assert isinstance(evaluation_record, ObservationPayload)
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

    def test_interaction_evaluation_unit_pickle_round_trips(self) -> None:
        first_request = EvaluationRequest(proposal=Proposal(candidate=3, proposal_id="a"))
        second_request = EvaluationRequest(proposal=Proposal(candidate=5, proposal_id="b"))
        interaction_unit = InteractionEvaluationUnit(
            requests=(first_request, second_request),
            interaction_evaluation_spec=MatchupSpec(arena="ladder"),
        )
        plain_interaction_unit = InteractionEvaluationUnit(
            requests=(first_request, second_request),
        )

        restored = pickle_round_trip(interaction_unit)
        plain_restored = pickle_round_trip(plain_interaction_unit)

        assert restored.request_count == 2
        assert tuple(proposal.proposal_id for proposal in restored.proposals) == ("a", "b")
        assert isinstance(restored.interaction_evaluation_spec, MatchupSpec)
        assert plain_restored.request_count == 2
        assert plain_restored.interaction_evaluation_spec is None

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

    def test_interaction_problem_pickle_round_trips(self) -> None:
        problem = InteractionProblem(
            space=IntegerSpace(low=0, high=10),
            interaction_evaluation_protocol=MatchupProtocol(),
            name="matchup",
        )

        restored = pickle_round_trip(problem)
        interaction_record = restored.interaction_evaluation_protocol.evaluate_requests(
            (
                EvaluationRequest(proposal=Proposal(candidate=2, proposal_id="left")),
                EvaluationRequest(proposal=Proposal(candidate=7, proposal_id="right")),
            ),
            interaction_evaluation_spec=MatchupSpec(arena="restored"),
        )

        assert restored.name == "matchup"
        assert interaction_record.winner == 7
        assert interaction_record.arena == "restored"

    def test_study_pickle_round_trips_without_runtime_generic_metadata(self) -> None:
        space = IntegerSpace(low=0, high=10)
        problem: Problem[int, int, Observation[int]] = Problem(
            space=space,
            evaluation_protocol=LegacySquareObservationProtocol(),
        )
        optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=4,
            random_state=0,
        )
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=LegacyObservationEvaluator(),
        )

        restored = pickle_round_trip(study)

        record = restored.problem.evaluation_protocol.evaluate_proposal(
            Proposal(candidate=4),
        )
        assert record.value == 16.0

    def test_study_with_explicit_kernel_pickle_round_trips(self) -> None:
        space = IntegerSpace(low=0, high=10)
        problem: Problem[int, int, Observation[int]] = Problem(
            space=space,
            evaluation_protocol=LegacySquareObservationProtocol(),
        )
        optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=4,
            random_state=0,
        )
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=LegacyObservationEvaluator(),
            kernel=DirectKernel(),
        )

        restored = pickle_round_trip(study)

        assert isinstance(restored.kernel, DirectKernel)
        record = restored.problem.evaluation_protocol.evaluate_proposal(
            Proposal(candidate=4),
        )
        assert record.value == 16.0

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
                direction=cast(
                    OptimizationDirection,
                    cast(object, "maximize"),
                ),
            )
