"""Tests for the clearing genetic algorithm optimizer."""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import pytest
from typing_extensions import override

from variopt import (
    EvaluationAttemptBatch,
    EvaluationFailure,
    EvaluationRequest,
    IntegerSpace,
    Objective,
    Observation,
    PermutationSpace,
    Problem,
    Proposal,
    UnsupportedEvaluationFailureError,
)
from variopt.algorithms import (
    ClearingGAProfile,
    ClearingGeneticAlgorithmOptimizer,
    SpeciesConservingGeneticAlgorithmOptimizer,
    SpeciesGAProfile,
)
from variopt.algorithms.population.clearing_ga.state import (
    ClearingGAPopulationMember,
)
from variopt.algorithms.population.species_ga.state import (
    SpeciesGAPopulationMember,
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


class TestableClearingGA(ClearingGeneticAlgorithmOptimizer[int, int]):
    """Test-only clearing GA that exposes one survival seam."""

    def build_next_population_for_test(
        self,
        *,
        parents: tuple[ClearingGAPopulationMember[int], ...],
        offspring: tuple[ClearingGAPopulationMember[int], ...],
    ) -> tuple[ClearingGAPopulationMember[int], ...]:
        return self._build_next_population(parents=parents, offspring=offspring)


class TestableSpeciesGA(SpeciesConservingGeneticAlgorithmOptimizer[int, int]):
    """Test-only species GA that exposes one survival seam."""

    def build_next_population_for_test(
        self,
        *,
        parents: tuple[SpeciesGAPopulationMember[int], ...],
        offspring: tuple[SpeciesGAPopulationMember[int], ...],
    ) -> tuple[SpeciesGAPopulationMember[int], ...]:
        return self._build_next_population(parents=parents, offspring=offspring)


class ClearingGeneticAlgorithmOptimizerTests:
    """Regression tests for the clearing genetic algorithm optimizer."""

    def test_profile_rejects_invalid_clearing_fields(self) -> None:
        with pytest.raises(ValueError, match="clearing_radius must be non-negative"):
            _ = ClearingGAProfile(clearing_radius=-0.1)

        with pytest.raises(ValueError, match="clearing_capacity must be positive"):
            _ = ClearingGAProfile(clearing_capacity=0)

    def test_profile_defaults_use_capacity_two(self) -> None:
        assert ClearingGAProfile().resolve().clearing_capacity == 2

    def test_optimizer_rejects_invalid_population_and_operator_shapes(self) -> None:
        with pytest.raises(ValueError, match="population_size must be positive"):
            _ = ClearingGeneticAlgorithmOptimizer(
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
            _ = ClearingGeneticAlgorithmOptimizer(
                space=IntegerSpace(0, 10),
                population_size=4,
                diversity_metric=IntegerDistance(),
                mutation_operator=BinaryMutation(),
            )

    def test_clearing_survival_limits_local_occupancy(self) -> None:
        optimizer = ClearingGeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            diversity_metric=IntegerDistance(),
            mutation_operator=StepTowardZeroMutation(),
            profile=ClearingGAProfile(
                mutation_probability=1.0,
                crossover_probability=0.0,
                clearing_radius=3.0,
                clearing_capacity=1,
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
        assert tuple(member.candidate for member in state.population) == (0, 1, 7, 9)
        assert len(state.population) == 4

    def test_clearing_optimizer_rejects_failure_attempts_without_consuming_pending(self) -> None:
        optimizer = ClearingGeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            diversity_metric=IntegerDistance(),
            mutation_operator=StepTowardZeroMutation(),
            sampler=CyclingIntegerSampler((9, 8, 1, 0)),
            random_state=0,
        )
        state = optimizer.create_initial_state()
        proposals, state = optimizer.ask(state, batch_size=1)
        request = EvaluationRequest(proposal=proposals[0])
        attempts: EvaluationAttemptBatch[int, Observation[int]] = EvaluationAttemptBatch(
            requests=(request,),
            failures=(
                EvaluationFailure[int].from_exception(
                    request=request,
                    exception=ValueError("failed"),
                ),
            ),
        )

        with pytest.raises(UnsupportedEvaluationFailureError):
            _ = optimizer.tell_attempts(state, attempts)

        assert state.pending_proposals == proposals

    def test_clearing_backfill_differs_from_species_seed_backfill(self) -> None:
        clearing_optimizer = TestableClearingGA(
            space=IntegerSpace(0, 10),
            population_size=4,
            diversity_metric=IntegerDistance(),
            mutation_operator=StepTowardZeroMutation(),
            profile=ClearingGAProfile(
                mutation_probability=0.0,
                crossover_probability=0.0,
                clearing_radius=3.0,
                clearing_capacity=1,
            ),
        )
        species_optimizer = TestableSpeciesGA(
            space=IntegerSpace(0, 10),
            population_size=4,
            diversity_metric=IntegerDistance(),
            mutation_operator=StepTowardZeroMutation(),
            profile=SpeciesGAProfile(
                mutation_probability=0.0,
                crossover_probability=0.0,
                species_radius=3.0,
                species_capacity=1,
            ),
        )

        clearing_pool = tuple(
            ClearingGAPopulationMember(candidate=value, value=float(value), score=float(value))
            for value in (0, 1, 2, 6, 7, 10)
        )
        species_pool = tuple(
            SpeciesGAPopulationMember(candidate=value, value=float(value), score=float(value))
            for value in (0, 1, 2, 6, 7, 10)
        )

        clearing_population = clearing_optimizer.build_next_population_for_test(
            parents=clearing_pool,
            offspring=(),
        )
        species_population = species_optimizer.build_next_population_for_test(
            parents=species_pool,
            offspring=(),
        )

        assert tuple(member.candidate for member in clearing_population) == (0, 2, 6, 10)
        assert tuple(member.candidate for member in species_population) == (0, 1, 6, 10)

    def test_from_permutation_space_defaults_can_optimize(self) -> None:
        space = PermutationSpace(size=6)
        optimizer = ClearingGeneticAlgorithmOptimizer.from_permutation_space_defaults(
            space=space,
            population_size=6,
            profile=ClearingGAProfile(
                crossover_probability=1.0,
                mutation_probability=1.0,
                clearing_radius=0.25,
                clearing_capacity=1,
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
