"""Tests for internal generational GA lifecycle contracts."""

from collections.abc import Sequence

import numpy as np
import pytest
from typing_extensions import override

from variopt import (
    EvaluationRequest,
    IntegerSpace,
    Objective,
    Problem,
    Proposal,
)
from variopt.algorithms.population.ga import GAProfile, GeneticAlgorithmOptimizer
from variopt.algorithms.population.generational_ga.lifecycle import (
    GenerationalGAGenerationCommit,
    ask_generational_ga,
    tell_generational_ga,
)
from variopt.algorithms.population.generational_ga.state import (
    GenerationalGAMemberBuffer,
    GenerationalGAOptimizerState,
    GenerationalGAPopulationMember,
    GenerationalGAVariant,
)
from variopt.evaluators import SequentialEvaluator
from variopt.operators import VariationOperator
from variopt.randomness import RandomStateSnapshot
from variopt.sampling import CandidateSampler


def _requests(
    proposals: Sequence[Proposal[int]],
) -> tuple[EvaluationRequest[int], ...]:
    """Lower proposal fixtures into canonical evaluation requests."""
    return tuple(EvaluationRequest(proposal=proposal) for proposal in proposals)


class SquareObjective(Objective[int]):
    """Quadratic integer objective with a minimum at zero."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


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


class TruncatingGenerationCommitter:
    """Malicious survival hook that returns the wrong population size."""

    def __call__(
        self,
        *,
        parents: tuple[GenerationalGAPopulationMember[int], ...],
        offspring: tuple[GenerationalGAPopulationMember[int], ...],
        random_state: RandomStateSnapshot,
    ) -> GenerationalGAGenerationCommit[int]:
        _ = offspring, random_state
        return GenerationalGAGenerationCommit(population=parents[:1])


class GenerationalGALifecycleTests:
    """Regression tests for shared generational GA invariants."""

    def test_population_member_rejects_nonfinite_accounting(self) -> None:
        with pytest.raises(ValueError, match="value must be finite"):
            _ = GenerationalGAPopulationMember(
                candidate=0,
                value=float("nan"),
                score=0.0,
            )

        with pytest.raises(ValueError, match="score must be finite"):
            _ = GenerationalGAPopulationMember(
                candidate=0,
                value=0.0,
                score=float("inf"),
            )

    def test_optimizer_state_rejects_negative_counters(self) -> None:
        random_state = RandomStateSnapshot.from_seed(0)

        with pytest.raises(ValueError, match="proposal_index must be non-negative"):
            _ = GenerationalGAOptimizerState(
                variant=GenerationalGAVariant.NATIVE,
                random_state=random_state,
                proposal_index=-1,
            )

        with pytest.raises(ValueError, match="generation_index must be non-negative"):
            _ = GenerationalGAOptimizerState(
                variant=GenerationalGAVariant.NATIVE,
                random_state=random_state,
                generation_index=-1,
            )

        with pytest.raises(ValueError, match="queued_proposal_index must be non-negative"):
            _ = GenerationalGAOptimizerState(
                variant=GenerationalGAVariant.NATIVE,
                random_state=random_state,
                queued_proposal_index=-1,
            )

    def test_optimizer_state_rejects_invalid_queue_cursor(self) -> None:
        random_state = RandomStateSnapshot.from_seed(0)
        proposal = Proposal(candidate=0, proposal_id="ga-0")

        with pytest.raises(ValueError, match="queued_proposal_index must not exceed"):
            _ = GenerationalGAOptimizerState(
                variant=GenerationalGAVariant.NATIVE,
                random_state=random_state,
                queued_proposals=(proposal,),
                queued_proposal_index=2,
            )

        with pytest.raises(ValueError, match="queued_proposal_index must be zero"):
            _ = GenerationalGAOptimizerState(
                variant=GenerationalGAVariant.NATIVE,
                random_state=random_state,
                queued_proposal_index=1,
            )

    def test_member_buffer_materializes_append_order(self) -> None:
        first = GenerationalGAPopulationMember(candidate=1, value=1.0, score=1.0)
        second = GenerationalGAPopulationMember(candidate=2, value=2.0, score=2.0)
        third = GenerationalGAPopulationMember(candidate=3, value=3.0, score=3.0)

        empty_buffer: GenerationalGAMemberBuffer[int] = GenerationalGAMemberBuffer()
        buffer = empty_buffer.append((first, second)).append((third,))

        assert buffer.member_count == 3
        assert buffer.materialize() == (first, second, third)

    def test_member_buffer_rejects_inconsistent_accounting(self) -> None:
        member = GenerationalGAPopulationMember(candidate=1, value=1.0, score=1.0)

        with pytest.raises(ValueError, match="member_count must match"):
            _ = GenerationalGAMemberBuffer(
                member_count=2,
                latest_batch=(member,),
            )

        empty_buffer: GenerationalGAMemberBuffer[int] = GenerationalGAMemberBuffer()
        with pytest.raises(ValueError, match="latest_batch must not be empty"):
            _ = GenerationalGAMemberBuffer(
                member_count=0,
                previous=empty_buffer,
            )

    def test_generation_commit_rejects_empty_population(self) -> None:
        with pytest.raises(ValueError, match="population must not be empty"):
            _ = GenerationalGAGenerationCommit[int](population=())

    def test_ask_rejects_empty_proposal_id_prefix(self) -> None:
        optimizer = GeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            mutation_operator=StepTowardZeroMutation(),
            sampler=CyclingIntegerSampler((5, 4, 3, 2)),
            random_state=0,
        )
        state = optimizer.create_initial_state()

        with pytest.raises(ValueError, match="proposal_id_prefix must not be empty"):
            _ = ask_generational_ga(
                optimizer,
                state,
                batch_size=1,
                proposal_id_prefix="",
                variant=GenerationalGAVariant.NATIVE,
            )

    def test_ask_rejects_state_owned_by_another_ga_variant(self) -> None:
        native_optimizer = GeneticAlgorithmOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            mutation_operator=StepTowardZeroMutation(),
            sampler=CyclingIntegerSampler((5, 4, 3, 2)),
            random_state=0,
        )
        native_state = native_optimizer.create_initial_state()
        clearing_state = GenerationalGAOptimizerState(
            variant=GenerationalGAVariant.CLEARING,
            random_state=native_state.random_state,
        )

        with pytest.raises(ValueError, match="state variant does not match optimizer variant"):
            _ = native_optimizer.ask(clearing_state, batch_size=1)

        with pytest.raises(ValueError, match="state variant does not match optimizer variant"):
            _ = native_optimizer.tell(
                clearing_state,
                (),
            )

    def test_tell_rejects_wrong_size_generation_commit(self) -> None:
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

        proposals, state = optimizer.ask(state, batch_size=4)
        outcomes = evaluator.evaluate(problem, _requests(proposals))

        with pytest.raises(RuntimeError, match="next population size must match population_size"):
            _ = tell_generational_ga(
                state,
                tuple(outcome.observation for outcome in outcomes),
                population_size=4,
                build_next_population=TruncatingGenerationCommitter(),
                variant=GenerationalGAVariant.NATIVE,
            )
