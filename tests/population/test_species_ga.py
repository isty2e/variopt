"""Tests for the species-conserving genetic algorithm optimizer."""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import pytest
from typing_extensions import override

from variopt import (
    EvaluationRequest,
    IntegerSpace,
    Objective,
    PermutationSpace,
    Problem,
    Proposal,
)
from variopt.algorithms import (
    SpeciesConservingGeneticAlgorithmOptimizer,
    SpeciesGAProfile,
)
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


class SpeciesConservingGeneticAlgorithmOptimizerTests:
    """Regression tests for the species-conserving GA optimizer."""

    def test_profile_rejects_invalid_species_fields(self) -> None:
        with pytest.raises(ValueError, match="species_radius must be non-negative"):
            _ = SpeciesGAProfile(species_radius=-0.1)

        with pytest.raises(ValueError, match="species_capacity must be positive"):
            _ = SpeciesGAProfile(species_capacity=0)

    def test_optimizer_rejects_invalid_population_and_operator_shapes(self) -> None:
        with pytest.raises(ValueError, match="population_size must be positive"):
            _ = SpeciesConservingGeneticAlgorithmOptimizer(
                space=IntegerSpace(0, 10),
                population_size=0,
                diversity_metric=IntegerDistance(),
                mutation_operator=StepTowardZeroMutation(),
            )

        class BinaryMutation(VariationOperator[int]):
            @property
            @override
            def arity(self) -> int:
                return 2

            @override
            def apply(
                self,
                parents: Sequence[int],
                random_state: np.random.RandomState,
            ) -> int:
                _ = random_state
                return parents[0]

        with pytest.raises(ValueError, match="mutation_operator arity must be exactly 1"):
            _ = SpeciesConservingGeneticAlgorithmOptimizer(
                space=IntegerSpace(0, 10),
                population_size=4,
                diversity_metric=IntegerDistance(),
                mutation_operator=BinaryMutation(),
            )

    def test_species_survival_preserves_multiple_seeds(self) -> None:
        optimizer = SpeciesConservingGeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            diversity_metric=IntegerDistance(),
            mutation_operator=StepTowardZeroMutation(),
            profile=SpeciesGAProfile(
                mutation_probability=1.0,
                crossover_probability=0.0,
                species_radius=3.0,
                species_capacity=1,
            ),
            sampler=CyclingIntegerSampler((9, 8, 1, 0)),
            random_state=0,
        )
        problem = Problem(space=IntegerSpace(0, 10), objective=SquareObjective())
        evaluator = SequentialEvaluator[int, int]()

        state = optimizer.create_initial_state()
        proposals, state = optimizer.ask(state, batch_size=4)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))
        assert tuple(member.candidate for member in state.population) == (0, 1, 8, 9)

        proposals, state = optimizer.ask(state, batch_size=4)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))

        assert state.generation_index == 1
        assert tuple(member.candidate for member in state.population) == (0, 1, 7, 8)
        assert len(state.population) == 4

    def test_from_permutation_space_defaults_can_optimize(self) -> None:
        space = PermutationSpace(size=6)
        optimizer = SpeciesConservingGeneticAlgorithmOptimizer.from_permutation_space_defaults(
            space=space,
            population_size=6,
            profile=SpeciesGAProfile(
                crossover_probability=1.0,
                mutation_probability=1.0,
                species_radius=0.25,
                species_capacity=1,
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
