"""Tests for the restricted-tournament genetic algorithm optimizer."""

from collections.abc import Sequence
from dataclasses import replace
from typing import TypeVar

import numpy as np
import pytest
from typing_extensions import override

from variopt import (
    EvaluationRequest,
    IntegerSpace,
    Objective,
    OptimizationDirection,
    PermutationSpace,
    Problem,
)
from variopt.algorithms import (
    RestrictedTournamentGAProfile,
    RestrictedTournamentGeneticAlgorithmOptimizer,
)
from variopt.algorithms.population.restricted_tournament_ga.state import (
    RestrictedTournamentGAPopulationMember,
)
from variopt.artifacts import Observation, Proposal
from variopt.diversity import DiversityMetric
from variopt.evaluators import SequentialEvaluator
from variopt.operators import VariationOperator
from variopt.sampling import CandidateSampler

CandidateT = TypeVar("CandidateT")


def _requests(
    proposals: Sequence[Proposal[CandidateT]],
) -> tuple[EvaluationRequest[CandidateT], ...]:
    """Lower proposal fixtures into canonical evaluation requests."""
    return tuple(EvaluationRequest(proposal=proposal) for proposal in proposals)


class SquareObjective(Objective[int]):
    """Quadratic integer objective with a minimum at zero."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


class PermutationMismatchObjective(Objective[tuple[int, ...]]):
    """Objective that counts positional mismatches against identity."""

    @override
    def evaluate(self, candidate: tuple[int, ...]) -> float:
        mismatch_count = 0
        for index, value in enumerate(candidate):
            if index != value:
                mismatch_count += 1
        return float(mismatch_count)


class CyclingIntegerSampler(CandidateSampler[int]):
    """Test-only sampler that yields a fixed integer sequence."""

    values: tuple[int, ...]
    index: int

    def __init__(self, values: Sequence[int]) -> None:
        self.values = tuple(values)
        self.index = 0

    @override
    def sample(self, random_state: np.random.RandomState) -> int:
        _ = random_state
        value = self.values[self.index % len(self.values)]
        self.index += 1
        return value


class StepTowardZeroMutation(VariationOperator[int]):
    """Unary mutation that deterministically moves one step toward zero."""

    @property
    @override
    def arity(self) -> int:
        return 1

    @override
    def apply(
        self,
        parents: Sequence[int],
        random_state: np.random.RandomState,
    ) -> int:
        _ = random_state
        parent = parents[0]
        if parent <= 0:
            return 0
        return parent - 1


class IntegerDistance(DiversityMetric[int]):
    """Absolute-difference distance for scalar integer candidates."""

    @override
    def distance(self, left: int, right: int) -> float:
        return float(abs(left - right))


class RestrictedTournamentGeneticAlgorithmOptimizerTests:
    """Regression tests for the restricted-tournament GA optimizer."""

    def test_profile_rejects_invalid_restricted_tournament_fields(self) -> None:
        with pytest.raises(ValueError, match="restricted_tournament_window_size must be positive"):
            _ = RestrictedTournamentGAProfile(restricted_tournament_window_size=0)

    def test_optimizer_rejects_invalid_population_and_operator_shapes(self) -> None:
        with pytest.raises(ValueError, match="population_size must be positive"):
            _ = RestrictedTournamentGeneticAlgorithmOptimizer(
                space=IntegerSpace(0, 10),
                population_size=0,
                diversity_metric=IntegerDistance(),
                mutation_operator=StepTowardZeroMutation(),
            )

        with pytest.raises(ValueError, match="restricted_tournament_window_size must not exceed population_size"):
            _ = RestrictedTournamentGeneticAlgorithmOptimizer(
                space=IntegerSpace(0, 10),
                population_size=4,
                diversity_metric=IntegerDistance(),
                mutation_operator=StepTowardZeroMutation(),
                profile=RestrictedTournamentGAProfile(restricted_tournament_window_size=5),
            )

    def test_restricted_tournament_replacement_is_local(self) -> None:
        optimizer = RestrictedTournamentGeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 30),
            population_size=4,
            diversity_metric=IntegerDistance(),
            mutation_operator=StepTowardZeroMutation(),
            profile=RestrictedTournamentGAProfile(
                crossover_probability=0.0,
                mutation_probability=1.0,
                restricted_tournament_window_size=4,
            ),
            sampler=CyclingIntegerSampler((0, 10, 20, 30)),
            random_state=0,
        )

        initial_state = optimizer.create_initial_state()
        state = replace(
            initial_state,
            population=(
                RestrictedTournamentGAPopulationMember(candidate=0, value=0.0, score=0.0),
                RestrictedTournamentGAPopulationMember(candidate=10, value=100.0, score=100.0),
                RestrictedTournamentGAPopulationMember(candidate=20, value=400.0, score=400.0),
                RestrictedTournamentGAPopulationMember(candidate=30, value=900.0, score=900.0),
            ),
            pending_proposals=(
                Proposal(candidate=9, proposal_id="rt-ga-0"),
                Proposal(candidate=19, proposal_id="rt-ga-1"),
                Proposal(candidate=29, proposal_id="rt-ga-2"),
                Proposal(candidate=15, proposal_id="rt-ga-3"),
            ),
        )
        observations = tuple(
            Observation.from_objective_value(
                proposal=proposal,
                candidate=proposal.candidate,
                value=float(proposal.candidate * proposal.candidate),
                direction=OptimizationDirection.MINIMIZE,
            )
            for proposal in state.pending_proposals
        )

        next_state = optimizer.tell(state, observations)

        assert next_state.generation_index == 1
        assert tuple(member.candidate for member in next_state.population) == (0, 9, 15, 29)

    def test_from_permutation_space_defaults_can_optimize(self) -> None:
        space = PermutationSpace(size=6)
        optimizer = RestrictedTournamentGeneticAlgorithmOptimizer.from_permutation_space_defaults(
            space=space,
            population_size=6,
            profile=RestrictedTournamentGAProfile(
                crossover_probability=1.0,
                mutation_probability=1.0,
                restricted_tournament_window_size=3,
            ),
            random_state=0,
        )
        problem = Problem(
            space=space,
            objective=PermutationMismatchObjective(),
        )
        evaluator = SequentialEvaluator[Sequence[int], tuple[int, ...]]()

        state = optimizer.create_initial_state()
        proposals, state = optimizer.ask(state, batch_size=6)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))

        assert len(state.population) == 6
        assert all(
                tuple(sorted(member.candidate)) == (0, 1, 2, 3, 4, 5)
                for member in state.population
            )
