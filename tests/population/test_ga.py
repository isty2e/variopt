"""Tests for the native genetic algorithm optimizer."""

import copy
from collections.abc import Sequence
from typing import TypeGuard, TypeVar

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
from variopt.algorithms.population import (
    GAProfile,
    GenerationalGAOptimizerState,
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


def _is_integer_ga_state(
    value: object,
) -> TypeGuard[GenerationalGAOptimizerState[int]]:
    """Return whether ``value`` is the integer GA state used in tests."""
    return type(value) is GenerationalGAOptimizerState


def _deepcopy_round_trip_state(
    state: GenerationalGAOptimizerState[int],
) -> GenerationalGAOptimizerState[int]:
    """Round-trip a GA state through a detached object graph."""
    round_tripped = copy.deepcopy(state)
    if not _is_integer_ga_state(round_tripped):
        msg = "round trip did not restore a generational GA state"
        raise AssertionError(msg)

    return round_tripped


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
        assert tuple(
            member.candidate for member in state.buffered_member_buffer.materialize()
        ) == (5, 4)

        proposals, state = optimizer.ask(state, batch_size=2)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))

        assert tuple(member.candidate for member in state.population) == (2, 3, 4, 5)
        assert state.buffered_member_buffer.member_count == 0

    def test_optimizer_preserves_proposal_id_continuity_across_queue_slices(self) -> None:
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
        assert tuple(proposal.proposal_id for proposal in proposals) == ("ga-0", "ga-1")
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))

        proposals, state = optimizer.ask(state, batch_size=2)
        assert tuple(proposal.proposal_id for proposal in proposals) == ("ga-2", "ga-3")
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))

        proposals, state = optimizer.ask(state, batch_size=1)
        assert tuple(proposal.proposal_id for proposal in proposals) == ("ga-4",)
        remaining_proposals = state.queued_proposals[state.queued_proposal_index :]
        assert tuple(proposal.proposal_id for proposal in remaining_proposals) == (
            "ga-5",
            "ga-6",
            "ga-7",
        )

    def test_optimizer_compacts_mostly_consumed_generation_queue(self) -> None:
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
        proposals, state = optimizer.ask(state, batch_size=4)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))

        proposals, state = optimizer.ask(state, batch_size=3)

        assert tuple(proposal.proposal_id for proposal in proposals) == (
            "ga-4",
            "ga-5",
            "ga-6",
        )
        assert state.queued_proposal_index == 0
        assert tuple(proposal.proposal_id for proposal in state.queued_proposals) == (
            "ga-7",
        )

        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(state, tuple(outcome.observation for outcome in outcomes))
        proposals, state = optimizer.ask(state, batch_size=1)

        assert tuple(proposal.proposal_id for proposal in proposals) == ("ga-7",)

    def test_optimizer_rejects_ask_while_proposals_are_pending(self) -> None:
        optimizer = GeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            mutation_operator=StepTowardZeroMutation(),
            sampler=CyclingIntegerSampler((5, 4, 3, 2)),
            random_state=0,
        )

        state = optimizer.create_initial_state()
        proposals, state = optimizer.ask(state, batch_size=1)

        with pytest.raises(RuntimeError, match="cannot ask while proposals are still pending"):
            _ = optimizer.ask(state, batch_size=1)

        assert state.pending_proposals == proposals

    def test_optimizer_rejects_reordered_observations_without_consuming_pending(self) -> None:
        optimizer = GeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            mutation_operator=StepTowardZeroMutation(),
            sampler=CyclingIntegerSampler((5, 4, 3, 2)),
            random_state=0,
        )
        problem = Problem(space=IntegerSpace(0, 10), objective=SquareObjective())
        evaluator = SequentialEvaluator[int, int]()

        state = optimizer.create_initial_state()
        proposals, state = optimizer.ask(state, batch_size=2)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        reordered_observations = tuple(
            outcome.observation
            for outcome in reversed(outcomes)
        )

        with pytest.raises(ValueError, match="observations must align with pending proposal order"):
            _ = optimizer.tell(state, reordered_observations)

        assert state.pending_proposals == proposals

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
        failure = EvaluationFailure[int].from_exception(
            request=request,
            exception=ValueError("failed"),
        )
        attempts: EvaluationAttemptBatch[int, Observation[int]] = EvaluationAttemptBatch(
            attempts=(failure,),
        )

        with pytest.raises(UnsupportedEvaluationFailureError):
            _ = optimizer.tell_attempts(state, attempts)

        assert state.pending_proposals == proposals

    def test_optimizer_split_generation_matches_full_generation_rng_and_order(self) -> None:
        split_optimizer = GeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            mutation_operator=StepTowardZeroMutation(),
            profile=GAProfile(mutation_probability=1.0, elite_count=0),
            sampler=CyclingIntegerSampler((5, 4, 3, 2)),
            random_state=13,
        )
        full_optimizer = GeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            mutation_operator=StepTowardZeroMutation(),
            profile=GAProfile(mutation_probability=1.0, elite_count=0),
            sampler=CyclingIntegerSampler((5, 4, 3, 2)),
            random_state=13,
        )
        problem = Problem(space=IntegerSpace(0, 10), objective=SquareObjective())
        evaluator = SequentialEvaluator[int, int]()

        split_state = split_optimizer.create_initial_state()
        full_state = full_optimizer.create_initial_state()
        proposals, split_state = split_optimizer.ask(split_state, batch_size=4)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        split_state = split_optimizer.tell(
            split_state,
            tuple(outcome.observation for outcome in outcomes),
        )
        proposals, full_state = full_optimizer.ask(full_state, batch_size=4)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        full_state = full_optimizer.tell(
            full_state,
            tuple(outcome.observation for outcome in outcomes),
        )

        split_proposals: list[Proposal[int]] = []
        proposals, split_state = split_optimizer.ask(split_state, batch_size=1)
        split_proposals.extend(proposals)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        split_state = split_optimizer.tell(
            split_state,
            tuple(outcome.observation for outcome in outcomes),
        )
        proposals, split_state = split_optimizer.ask(split_state, batch_size=3)
        split_proposals.extend(proposals)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        split_state = split_optimizer.tell(
            split_state,
            tuple(outcome.observation for outcome in outcomes),
        )

        full_proposals, full_state = full_optimizer.ask(full_state, batch_size=4)
        outcomes = evaluator.evaluate(problem, _requests(full_proposals))
        full_state = full_optimizer.tell(
            full_state,
            tuple(outcome.observation for outcome in outcomes),
        )

        assert tuple(split_proposals) == full_proposals
        assert split_state.random_state == full_state.random_state
        assert split_state.proposal_index == full_state.proposal_index
        assert split_state.generation_index == full_state.generation_index
        assert split_state.population == full_state.population

    def test_optimizer_partial_generation_round_trip_preserves_continuation(self) -> None:
        split_optimizer = GeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            mutation_operator=StepTowardZeroMutation(),
            profile=GAProfile(mutation_probability=1.0, elite_count=0),
            sampler=CyclingIntegerSampler((5, 4, 3, 2)),
            random_state=13,
        )
        full_optimizer = GeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            mutation_operator=StepTowardZeroMutation(),
            profile=GAProfile(mutation_probability=1.0, elite_count=0),
            sampler=CyclingIntegerSampler((5, 4, 3, 2)),
            random_state=13,
        )
        problem = Problem(space=IntegerSpace(0, 10), objective=SquareObjective())
        evaluator = SequentialEvaluator[int, int]()

        split_state = split_optimizer.create_initial_state()
        full_state = full_optimizer.create_initial_state()
        proposals, split_state = split_optimizer.ask(split_state, batch_size=4)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        split_state = split_optimizer.tell(
            split_state,
            tuple(outcome.observation for outcome in outcomes),
        )
        proposals, full_state = full_optimizer.ask(full_state, batch_size=4)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        full_state = full_optimizer.tell(
            full_state,
            tuple(outcome.observation for outcome in outcomes),
        )

        proposals, split_state = split_optimizer.ask(split_state, batch_size=1)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        split_state = split_optimizer.tell(
            split_state,
            tuple(outcome.observation for outcome in outcomes),
        )
        assert split_state.queued_proposal_index == 1
        assert split_state.buffered_member_buffer.member_count == 1

        split_state = _deepcopy_round_trip_state(split_state)
        proposals, split_state = split_optimizer.ask(split_state, batch_size=3)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        split_state = split_optimizer.tell(
            split_state,
            tuple(outcome.observation for outcome in outcomes),
        )

        full_proposals, full_state = full_optimizer.ask(full_state, batch_size=4)
        outcomes = evaluator.evaluate(problem, _requests(full_proposals))
        full_state = full_optimizer.tell(
            full_state,
            tuple(outcome.observation for outcome in outcomes),
        )

        assert split_state.random_state == full_state.random_state
        assert split_state.proposal_index == full_state.proposal_index
        assert split_state.generation_index == full_state.generation_index
        assert split_state.population == full_state.population

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
        remaining_proposals = state.queued_proposals[state.queued_proposal_index :]
        assert tuple(proposal.candidate for proposal in remaining_proposals) == (3,)

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
