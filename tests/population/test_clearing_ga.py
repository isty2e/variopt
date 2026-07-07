"""Tests for the clearing genetic algorithm optimizer."""

from collections.abc import Sequence
from dataclasses import dataclass
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
    SearchSpace,
    UnsupportedEvaluationFailureError,
)
from variopt.algorithms import (
    ClearingGAProfile,
    ClearingGeneticAlgorithmOptimizer,
    SpeciesConservingGeneticAlgorithmOptimizer,
    SpeciesGAProfile,
)
from variopt.algorithms.population.generational_ga.state import (
    GenerationalGAPopulationMember,
)
from variopt.diversity import DiversityMetric
from variopt.evaluators import SequentialEvaluator
from variopt.operators import VariationOperator
from variopt.randomness import random_state_randint
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


class CountingIntegerDistance(DiversityMetric[int]):
    """Integer distance metric that records call count."""

    count: int

    def __init__(self) -> None:
        self.count = 0

    @override
    def distance(self, left: int, right: int) -> float:
        self.count += 1
        return float(abs(left - right))


@dataclass(frozen=True, slots=True, eq=False)
class EqualityHostileCandidate:
    """Candidate whose value equality must never be used by backfill."""

    coordinate: int
    stable_id: int

    @override
    def __eq__(self, other: object) -> bool:
        _ = other
        msg = "candidate equality must not be used"
        raise RuntimeError(msg)


class EqualityHostileSpace(
    SearchSpace[EqualityHostileCandidate, EqualityHostileCandidate],
):
    """Minimal search space for equality-hostile candidates."""

    @override
    def normalize(
        self,
        raw_candidate: EqualityHostileCandidate,
    ) -> EqualityHostileCandidate:
        self.validate(raw_candidate)
        return raw_candidate

    @override
    def validate(self, candidate: EqualityHostileCandidate) -> None:
        if type(candidate) is not EqualityHostileCandidate:
            msg = "candidate must be an EqualityHostileCandidate"
            raise TypeError(msg)

    @override
    def sample(
        self,
        random_state: np.random.RandomState,
    ) -> EqualityHostileCandidate:
        return EqualityHostileCandidate(
            coordinate=random_state_randint(random_state, 0, 100),
            stable_id=0,
        )


class EqualityHostileDistance(DiversityMetric[EqualityHostileCandidate]):
    """Coordinate distance for equality-hostile candidates."""

    @override
    def distance(
        self,
        left: EqualityHostileCandidate,
        right: EqualityHostileCandidate,
    ) -> float:
        return float(abs(left.coordinate - right.coordinate))


class EqualityHostileMutation(VariationOperator[EqualityHostileCandidate]):
    """No-op mutation for constructing equality-hostile optimizers."""

    @property
    @override
    def arity(self) -> int:
        return 1

    @override
    def apply(
        self,
        parents: Sequence[EqualityHostileCandidate],
        random_state: np.random.RandomState,
    ) -> EqualityHostileCandidate:
        _ = random_state
        return parents[0]


class TestableClearingGA(ClearingGeneticAlgorithmOptimizer[int, int]):
    """Test-only clearing GA that exposes one survival seam."""

    def build_next_population_for_test(
        self,
        *,
        parents: tuple[GenerationalGAPopulationMember[int], ...],
        offspring: tuple[GenerationalGAPopulationMember[int], ...],
    ) -> tuple[GenerationalGAPopulationMember[int], ...]:
        return self._build_next_population(
            parents=parents,
            offspring=offspring,
            random_state=self.create_initial_state().random_state,
        ).population

    def build_diverse_backfill_for_test(
        self,
        *,
        selected_members: tuple[GenerationalGAPopulationMember[int], ...],
        overflow_members: tuple[GenerationalGAPopulationMember[int], ...],
        count: int,
    ) -> tuple[GenerationalGAPopulationMember[int], ...]:
        return self._build_diverse_backfill(
            selected_members=selected_members,
            overflow_members=overflow_members,
            count=count,
        )


class TestableEqualityHostileClearingGA(
    ClearingGeneticAlgorithmOptimizer[
        EqualityHostileCandidate,
        EqualityHostileCandidate,
    ],
):
    """Test-only clearing GA for equality-hostile backfill candidates."""

    def build_diverse_backfill_for_test(
        self,
        *,
        selected_members: tuple[
            GenerationalGAPopulationMember[EqualityHostileCandidate],
            ...,
        ],
        overflow_members: tuple[
            GenerationalGAPopulationMember[EqualityHostileCandidate],
            ...,
        ],
        count: int,
    ) -> tuple[GenerationalGAPopulationMember[EqualityHostileCandidate], ...]:
        return self._build_diverse_backfill(
            selected_members=selected_members,
            overflow_members=overflow_members,
            count=count,
        )


class TestableSpeciesGA(SpeciesConservingGeneticAlgorithmOptimizer[int, int]):
    """Test-only species GA that exposes one survival seam."""

    def build_next_population_for_test(
        self,
        *,
        parents: tuple[GenerationalGAPopulationMember[int], ...],
        offspring: tuple[GenerationalGAPopulationMember[int], ...],
    ) -> tuple[GenerationalGAPopulationMember[int], ...]:
        return self._build_next_population(
            parents=parents,
            offspring=offspring,
            random_state=self.create_initial_state().random_state,
        ).population


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

        with pytest.raises(
            ValueError, match="mutation_operator arity must be exactly 1"
        ):
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
        state = optimizer.tell(
            state, tuple(outcome.observation for outcome in outcomes)
        )
        assert tuple(member.candidate for member in state.population) == (0, 1, 8, 9)

        proposals, state = optimizer.ask(state, batch_size=4)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(
            state, tuple(outcome.observation for outcome in outcomes)
        )

        assert state.generation_index == 1
        assert tuple(member.candidate for member in state.population) == (0, 1, 7, 9)
        assert len(state.population) == 4

    def test_clearing_optimizer_rejects_failure_attempts_without_consuming_pending(
        self,
    ) -> None:
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
        failure = EvaluationFailure[int].from_exception(
            request=request,
            exception=ValueError("failed"),
        )
        attempts: EvaluationAttemptBatch[int, Observation[int]] = (
            EvaluationAttemptBatch(
                attempts=(failure,),
            )
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
            GenerationalGAPopulationMember(
                candidate=value, value=float(value), score=float(value)
            )
            for value in (0, 1, 2, 6, 7, 10)
        )
        species_pool = tuple(
            GenerationalGAPopulationMember(
                candidate=value, value=float(value), score=float(value)
            )
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

        assert tuple(member.candidate for member in clearing_population) == (
            0,
            2,
            6,
            10,
        )
        assert tuple(member.candidate for member in species_population) == (0, 1, 6, 10)

    def test_clearing_backfill_updates_running_distances_incrementally(self) -> None:
        diversity_metric = CountingIntegerDistance()
        optimizer = TestableClearingGA(
            space=IntegerSpace(0, 120),
            population_size=5,
            diversity_metric=diversity_metric,
            mutation_operator=StepTowardZeroMutation(),
        )
        selected_members = tuple(
            GenerationalGAPopulationMember(
                candidate=value,
                value=float(value),
                score=float(value),
            )
            for value in (0, 100)
        )
        overflow_members = tuple(
            GenerationalGAPopulationMember(
                candidate=value,
                value=float(value),
                score=float(value),
            )
            for value in (10, 20, 50, 70, 90, 110)
        )

        backfill = optimizer.build_diverse_backfill_for_test(
            selected_members=selected_members,
            overflow_members=overflow_members,
            count=3,
        )

        assert tuple(member.candidate for member in backfill) == (50, 20, 70)
        assert diversity_metric.count == 21

    def test_clearing_backfill_removes_by_index_not_value_equality(self) -> None:
        optimizer = TestableEqualityHostileClearingGA(
            space=EqualityHostileSpace(),
            population_size=2,
            diversity_metric=EqualityHostileDistance(),
            mutation_operator=EqualityHostileMutation(),
        )
        selected_members = (
            GenerationalGAPopulationMember(
                candidate=EqualityHostileCandidate(coordinate=0, stable_id=0),
                value=0.0,
                score=0.0,
            ),
        )
        overflow_members = tuple(
            GenerationalGAPopulationMember(
                candidate=candidate,
                value=float(candidate.coordinate),
                score=float(candidate.coordinate),
            )
            for candidate in (
                EqualityHostileCandidate(coordinate=10, stable_id=1),
                EqualityHostileCandidate(coordinate=50, stable_id=2),
                EqualityHostileCandidate(coordinate=30, stable_id=3),
            )
        )

        backfill = optimizer.build_diverse_backfill_for_test(
            selected_members=selected_members,
            overflow_members=overflow_members,
            count=1,
        )

        assert tuple(member.candidate.coordinate for member in backfill) == (50,)

    def test_clearing_backfill_handles_empty_anchor_population(self) -> None:
        optimizer = TestableClearingGA(
            space=IntegerSpace(0, 100),
            population_size=2,
            diversity_metric=IntegerDistance(),
            mutation_operator=StepTowardZeroMutation(),
        )
        overflow_members = (
            GenerationalGAPopulationMember(candidate=0, value=0.0, score=10.0),
            GenerationalGAPopulationMember(candidate=100, value=100.0, score=0.0),
            GenerationalGAPopulationMember(candidate=40, value=40.0, score=5.0),
        )

        backfill = optimizer.build_diverse_backfill_for_test(
            selected_members=(),
            overflow_members=overflow_members,
            count=2,
        )

        assert tuple(member.candidate for member in backfill) == (100, 0)

    def test_clearing_backfill_preserves_score_tiebreak(self) -> None:
        optimizer = TestableClearingGA(
            space=IntegerSpace(0, 120),
            population_size=1,
            diversity_metric=IntegerDistance(),
            mutation_operator=StepTowardZeroMutation(),
            profile=ClearingGAProfile(tournament_size=1),
        )
        selected_members = (
            GenerationalGAPopulationMember(candidate=100, value=100.0, score=0.0),
        )
        overflow_members = (
            GenerationalGAPopulationMember(candidate=90, value=90.0, score=1.0),
            GenerationalGAPopulationMember(candidate=110, value=110.0, score=2.0),
        )

        backfill = optimizer.build_diverse_backfill_for_test(
            selected_members=selected_members,
            overflow_members=overflow_members,
            count=1,
        )

        assert tuple(member.candidate for member in backfill) == (90,)

    def test_clearing_backfill_stops_when_overflow_is_exhausted(self) -> None:
        optimizer = TestableClearingGA(
            space=IntegerSpace(0, 100),
            population_size=4,
            diversity_metric=IntegerDistance(),
            mutation_operator=StepTowardZeroMutation(),
        )
        selected_members = (
            GenerationalGAPopulationMember(candidate=0, value=0.0, score=0.0),
        )
        overflow_members = (
            GenerationalGAPopulationMember(candidate=10, value=10.0, score=10.0),
            GenerationalGAPopulationMember(candidate=30, value=30.0, score=30.0),
        )

        backfill = optimizer.build_diverse_backfill_for_test(
            selected_members=selected_members,
            overflow_members=overflow_members,
            count=5,
        )

        assert tuple(member.candidate for member in backfill) == (30, 10)

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
        state = optimizer.tell(
            state, tuple(outcome.observation for outcome in outcomes)
        )

        assert len(state.population) == 6
        assert all(
            tuple(sorted(member.candidate)) == (0, 1, 2, 3, 4, 5)
            for member in state.population
        )
