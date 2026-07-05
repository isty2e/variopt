"""Tests for the native genetic algorithm optimizer."""

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
from variopt.algorithms.population.ga import (
    GAProfile,
    GeneticAlgorithmOptimizer,
)
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


class ZeroCrossover(VariationOperator[int]):
    """Binary crossover that deterministically returns zero."""

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
        _ = (parents, random_state)
        return 0


class BinaryMutation(VariationOperator[int]):
    """Invalid binary mutation used for validation tests."""

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


class UnaryCrossover(VariationOperator[int]):
    """Invalid unary crossover used for validation tests."""

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
        return parents[0]


class GeneticAlgorithmOptimizerTests:
    """Regression tests for the native GA optimizer."""

    def test_profile_rejects_invalid_fields(self) -> None:
        with pytest.raises(ValueError, match="tournament_size must be positive"):
            _ = GAProfile(tournament_size=0)

        with pytest.raises(ValueError, match="crossover_probability must be between 0.0 and 1.0"):
            _ = GAProfile(crossover_probability=1.5)

        with pytest.raises(ValueError, match="mutation_probability must be between 0.0 and 1.0"):
            _ = GAProfile(mutation_probability=-0.1)

        with pytest.raises(ValueError, match="elite_count must be non-negative"):
            _ = GAProfile(elite_count=-1)

    def test_optimizer_rejects_invalid_operator_shapes(self) -> None:
        with pytest.raises(ValueError, match="crossover_operator arity must be at least 2"):
            _ = GeneticAlgorithmOptimizer(
                space=IntegerSpace(0, 10),
                population_size=4,
                crossover_operator=UnaryCrossover(),
            )

        with pytest.raises(ValueError, match="mutation_operator arity must be exactly 1"):
            _ = GeneticAlgorithmOptimizer(
                space=IntegerSpace(0, 10),
                population_size=4,
                mutation_operator=BinaryMutation(),
            )

    def test_optimizer_buffers_initial_population_across_batches(self) -> None:
        optimizer = GeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            mutation_operator=StepTowardZeroMutation(),
            profile=GAProfile(mutation_probability=1.0, elite_count=0),
            sampler=CyclingIntegerSampler((5, 4, 3, 2)),
            random_state=0,
        )
        problem = Problem(space=IntegerSpace(0, 10), objective=SquareObjective())
        evaluator = SequentialEvaluator[int, int]()

        state = optimizer.create_initial_state()
        proposals, state = optimizer.ask(state, batch_size=2)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))

        assert len(state.population) == 0
        assert tuple(member.candidate for member in state.buffered_members) == (5, 4)

        proposals, state = optimizer.ask(state, batch_size=2)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))

        assert tuple(member.candidate for member in state.population) == (2, 3, 4, 5)
        assert len(state.buffered_members) == 0

    def test_optimizer_rejects_failure_attempts_without_consuming_pending(self) -> None:
        optimizer = GeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            mutation_operator=StepTowardZeroMutation(),
            sampler=CyclingIntegerSampler((5, 4, 3, 2)),
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

    def test_optimizer_applies_elitism_after_one_generation(self) -> None:
        optimizer = GeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=2,
            mutation_operator=StepTowardZeroMutation(),
            profile=GAProfile(
                tournament_size=2,
                crossover_probability=0.0,
                mutation_probability=1.0,
                elite_count=1,
            ),
            sampler=CyclingIntegerSampler((5, 4)),
            random_state=0,
        )
        problem = Problem(space=IntegerSpace(0, 10), objective=SquareObjective())
        evaluator = SequentialEvaluator[int, int]()

        state = optimizer.create_initial_state()
        proposals, state = optimizer.ask(state, batch_size=2)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))
        assert tuple(member.candidate for member in state.population) == (4, 5)

        proposals, state = optimizer.ask(state, batch_size=1)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))
        assert tuple(proposal.candidate for proposal in state.queued_proposals) == (3,)

        proposals, state = optimizer.ask(state, batch_size=1)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))

        assert state.generation_index == 1
        assert tuple(member.candidate for member in state.population) == (3, 4)

    def test_optimizer_can_use_crossover_without_mutation(self) -> None:
        optimizer = GeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=2,
            crossover_operator=ZeroCrossover(),
            profile=GAProfile(
                tournament_size=1,
                crossover_probability=1.0,
                mutation_probability=0.0,
                elite_count=0,
            ),
            sampler=CyclingIntegerSampler((5, 4)),
            random_state=0,
        )
        problem = Problem(space=IntegerSpace(0, 10), objective=SquareObjective())
        evaluator = SequentialEvaluator[int, int]()

        state = optimizer.create_initial_state()
        proposals, state = optimizer.ask(state, batch_size=2)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))

        proposals, state = optimizer.ask(state, batch_size=2)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))

        assert state.generation_index == 1
        assert tuple(member.candidate for member in state.population) == (0, 0)

    def test_from_permutation_space_defaults_can_optimize(self) -> None:
        space = PermutationSpace(size=6)
        optimizer = GeneticAlgorithmOptimizer.from_permutation_space_defaults(
            space=space,
            population_size=6,
            profile=GAProfile(
                crossover_probability=1.0,
                mutation_probability=1.0,
                elite_count=1,
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
