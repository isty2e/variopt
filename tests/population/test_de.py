"""Tests for the native differential-evolution optimizer."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import pytest
from typing_extensions import override

from variopt import (
    CategoricalSpace,
    EvaluationAttemptBatch,
    EvaluationFailure,
    EvaluationRequest,
    IntegerSpace,
    Objective,
    Observation,
    Problem,
    Proposal,
    UnsupportedEvaluationFailureError,
)
from variopt.algorithms.population.de import (
    DEProfile,
    DifferentialEvolutionOptimizer,
)
from variopt.evaluators import SequentialEvaluator
from variopt.sampling import CandidateSampler
from variopt.spaces import (
    LeafPath,
    StructuredLeafSpace,
    StructuredSearchSpace,
)
from variopt.spaces.types import SpaceCandidateValue

CandidateT = TypeVar("CandidateT")
ConditionalNumericCandidate = tuple[int, int]


def _requests(
    proposals: Sequence[Proposal[CandidateT]],
) -> tuple[EvaluationRequest[CandidateT], ...]:
    """Lower proposal fixtures into canonical evaluation requests."""
    return tuple(EvaluationRequest(proposal=proposal) for proposal in proposals)


@dataclass(frozen=True)
class ConditionalNumericPairSpace(
    StructuredSearchSpace[ConditionalNumericCandidate, ConditionalNumericCandidate],
):
    """Test-only numeric structured space with candidate-conditioned topology."""

    head_space: IntegerSpace
    tail_space: IntegerSpace

    @override
    def normalize(
        self,
        raw_candidate: ConditionalNumericCandidate,
    ) -> ConditionalNumericCandidate:
        return (
            self.head_space.normalize(raw_candidate[0]),
            self.tail_space.normalize(raw_candidate[1]),
        )

    @override
    def validate(self, candidate: ConditionalNumericCandidate) -> None:
        self.head_space.validate(candidate[0])
        self.tail_space.validate(candidate[1])

    @override
    def sample(
        self, random_state: np.random.RandomState
    ) -> ConditionalNumericCandidate:
        return (
            self.head_space.sample(random_state),
            self.tail_space.sample(random_state),
        )

    @override
    def leaf_paths(self) -> tuple[LeafPath, ...]:
        return (("head",), ("tail",))

    @override
    def has_static_topology(self) -> bool:
        return False

    @override
    def active_leaf_paths(
        self,
        candidate: ConditionalNumericCandidate,
    ) -> tuple[LeafPath, ...]:
        self.validate(candidate)
        if candidate[0] > 0:
            return (("head",), ("tail",))
        return (("head",),)

    @override
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        if path == ("head",):
            return self.head_space
        if path == ("tail",):
            return self.tail_space
        msg = f"invalid conditional numeric pair path: {path!r}"
        raise TypeError(msg)

    @override
    def leaf_value_at_path(
        self,
        candidate: ConditionalNumericCandidate,
        path: LeafPath,
    ) -> SpaceCandidateValue:
        self.validate(candidate)
        if path == ("head",):
            return candidate[0]
        if path == ("tail",):
            return candidate[1]
        msg = f"invalid conditional numeric pair path: {path!r}"
        raise TypeError(msg)

    @override
    def replace_leaf_values(
        self,
        candidate: ConditionalNumericCandidate,
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> ConditionalNumericCandidate:
        self.validate(candidate)
        head_value = candidate[0]
        tail_value = candidate[1]
        if ("head",) in replacements:
            replacement = replacements[("head",)]
            if type(replacement) is not int:
                msg = "conditional numeric head replacement must be a canonical integer"
                raise TypeError(msg)
            head_value = self.head_space.normalize(replacement)
        if ("tail",) in replacements:
            replacement = replacements[("tail",)]
            if type(replacement) is not int:
                msg = "conditional numeric tail replacement must be a canonical integer"
                raise TypeError(msg)
            tail_value = self.tail_space.normalize(replacement)
        return (head_value, tail_value)


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


class DifferentialEvolutionOptimizerTests:
    """Regression tests for the native DE optimizer."""

    def test_profile_rejects_invalid_fields(self) -> None:
        with pytest.raises(
            ValueError, match="mutation_range low must not exceed mutation_range high"
        ):
            _ = DEProfile(mutation_range=(1.0, 0.5))

        with pytest.raises(ValueError, match="mutation_range low must be non-negative"):
            _ = DEProfile(mutation_range=(-0.1, 0.5))

        with pytest.raises(
            ValueError, match="recombination_probability must be between 0.0 and 1.0"
        ):
            _ = DEProfile(recombination_probability=1.5)

        with pytest.raises(ValueError, match="n_cross must be positive"):
            _ = DEProfile(n_cross=0)

    def test_optimizer_rejects_invalid_space_and_shape(self) -> None:
        with pytest.raises(
            ValueError,
            match="population_size must be at least 4 for differential evolution",
        ):
            _ = DifferentialEvolutionOptimizer(
                space=IntegerSpace(0, 10),
                population_size=3,
            )

        with pytest.raises(
            TypeError,
            match="space must contain only numeric leaves for differential evolution",
        ):
            _ = DifferentialEvolutionOptimizer(
                space=CategoricalSpace(("a", "b")),
                population_size=4,
            )

        with pytest.raises(
            ValueError, match="n_cross must not exceed the number of editable leaves"
        ):
            _ = DifferentialEvolutionOptimizer(
                space=IntegerSpace(0, 10),
                population_size=4,
                profile=DEProfile(n_cross=2),
            )

        with pytest.raises(TypeError, match="static topology"):
            _ = DifferentialEvolutionOptimizer(
                space=ConditionalNumericPairSpace(
                    head_space=IntegerSpace(0, 10),
                    tail_space=IntegerSpace(0, 10),
                ),
                population_size=4,
            )

    def test_optimizer_buffers_initial_population_across_batches(self) -> None:
        optimizer = DifferentialEvolutionOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            sampler=CyclingIntegerSampler((5, 4, 3, 2)),
            random_state=0,
        )
        problem = Problem(space=IntegerSpace(0, 10), objective=SquareObjective())
        evaluator = SequentialEvaluator[int, int]()

        state = optimizer.create_initial_state()
        proposals, state = optimizer.ask(state, batch_size=2)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(
            state, tuple(outcome.observation for outcome in outcomes)
        )

        assert len(state.population) == 0
        assert tuple(
            evaluation.member.candidate for evaluation in state.buffered_evaluations
        ) == (5, 4)

        proposals, state = optimizer.ask(state, batch_size=2)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(
            state, tuple(outcome.observation for outcome in outcomes)
        )

        assert tuple(member.candidate for member in state.population) == (5, 4, 3, 2)
        assert len(state.buffered_evaluations) == 0

    def test_optimizer_rejects_failure_attempts_without_consuming_pending(self) -> None:
        optimizer = DifferentialEvolutionOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
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
        attempts: EvaluationAttemptBatch[int, Observation[int]] = (
            EvaluationAttemptBatch(
                attempts=(failure,),
            )
        )

        with pytest.raises(UnsupportedEvaluationFailureError):
            _ = optimizer.tell_attempts(state, attempts)

        assert (
            tuple(
                pending_evaluation.proposal
                for pending_evaluation in state.pending_evaluations
            )
            == proposals
        )

    def test_optimizer_replaces_targets_only_when_trials_improve(self) -> None:
        optimizer = DifferentialEvolutionOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            profile=DEProfile(
                mutation_range=(1.0, 1.0),
                recombination_probability=1.0,
                n_cross=1,
            ),
            sampler=CyclingIntegerSampler((9, 7, 5, 3)),
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
        initial_population = state.population

        proposals, state = optimizer.ask(state, batch_size=2)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(
            state, tuple(outcome.observation for outcome in outcomes)
        )
        assert len(state.population) == 4

        proposals, state = optimizer.ask(state, batch_size=2)
        outcomes = evaluator.evaluate(problem, _requests(proposals))
        state = optimizer.tell(
            state, tuple(outcome.observation for outcome in outcomes)
        )

        assert state.generation_index == 1
        assert len(state.population) == 4
        assert all(
            member.score <= initial_member.score
            for member, initial_member in zip(
                state.population,
                initial_population,
                strict=True,
            )
        )

    def test_optimizer_is_deterministic_for_one_seed(self) -> None:
        optimizer_a = DifferentialEvolutionOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            sampler=CyclingIntegerSampler((9, 7, 5, 3)),
            random_state=17,
        )
        optimizer_b = DifferentialEvolutionOptimizer(
            space=IntegerSpace(0, 10),
            population_size=4,
            sampler=CyclingIntegerSampler((9, 7, 5, 3)),
            random_state=17,
        )
        problem = Problem(space=IntegerSpace(0, 10), objective=SquareObjective())
        evaluator = SequentialEvaluator[int, int]()

        state_a = optimizer_a.create_initial_state()
        state_b = optimizer_b.create_initial_state()

        proposals_a, state_a = optimizer_a.ask(state_a, batch_size=4)
        proposals_b, state_b = optimizer_b.ask(state_b, batch_size=4)
        assert tuple(proposal.candidate for proposal in proposals_a) == tuple(
            proposal.candidate for proposal in proposals_b
        )

        outcomes_a = evaluator.evaluate(problem, _requests(proposals_a))
        outcomes_b = evaluator.evaluate(problem, _requests(proposals_b))
        state_a = optimizer_a.tell(
            state_a, tuple(outcome.observation for outcome in outcomes_a)
        )
        state_b = optimizer_b.tell(
            state_b, tuple(outcome.observation for outcome in outcomes_b)
        )

        proposals_a, state_a = optimizer_a.ask(state_a, batch_size=4)
        proposals_b, state_b = optimizer_b.ask(state_b, batch_size=4)
        assert tuple(proposal.candidate for proposal in proposals_a) == tuple(
            proposal.candidate for proposal in proposals_b
        )
