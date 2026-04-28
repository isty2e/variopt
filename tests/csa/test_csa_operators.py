"""Tests for built-in CSA variation operators."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pytest
from typing_extensions import override

from variopt import (
    ArraySpace,
    DiversityMetric,
    EvaluationRequest,
    IntegerSpace,
    Objective,
    Problem,
    Proposal,
    RealSpace,
    Study,
)
from variopt.algorithms.population.csa import (
    BoundedMutation,
    CSACutoffSchedule,
    CSAOptimizer,
    CSAPerturbationSchedule,
    CSAPerturbationSpec,
    CSAProfile,
    DifferentialEvolutionVariation,
    MixtureVariation,
    RandomResetMutation,
    UniformCrossover,
)
from variopt.evaluators import SequentialEvaluator
from variopt.spaces import (
    LeafPath,
    StructuredLeafSpace,
    StructuredSearchSpace,
)
from variopt.spaces.types import SpaceCandidateValue

ConditionalNumericCandidate = tuple[int, int]


def _requests(
    proposals: tuple[Proposal[int], ...],
) -> tuple[EvaluationRequest[int], ...]:
    """Lower scalar proposal fixtures into canonical evaluation requests."""
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
    def sample(self, random_state: np.random.RandomState) -> ConditionalNumericCandidate:
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


class IntegerAbsoluteDistance(DiversityMetric[int]):
    """Absolute-value diversity for scalar integer candidates."""

    @override
    def distance(self, left: int, right: int) -> float:
        return float(abs(left - right))


class IntegerSquareObjective(Objective[int]):
    """Scalar quadratic objective with a minimum at zero."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


class OperatorTests:
    """Unit tests for built-in CSA operator behavior."""

    def test_uniform_crossover_copies_partner_coordinates(self) -> None:
        space = ArraySpace(IntegerSpace(0, 9), length=4)
        operator = UniformCrossover(
            space=space,
            max_exchange_fraction=1.0,
        )

        child = operator.apply(
            parents=((1, 1, 1, 1), (9, 9, 9, 9)),
            random_state=rng(0),
        )

        assert len(child) == 4
        assert all(value in {1, 9} for value in child)
        assert child != (1, 1, 1, 1)

    def test_uniform_crossover_rejects_active_topology_mismatch(self) -> None:
        space = ConditionalNumericPairSpace(
            head_space=IntegerSpace(0, 9),
            tail_space=IntegerSpace(0, 9),
        )
        operator = UniformCrossover(
            space=space,
            max_exchange_fraction=1.0,
        )

        with pytest.raises(ValueError, match="matching active topology"):
            _ = operator.apply(
                parents=((0, 1), (1, 9)),
                random_state=rng(0),
            )

    def test_random_reset_mutation_resamples_within_bounds(self) -> None:
        space = ArraySpace(RealSpace(-1.0, 1.0), length=3)
        operator = RandomResetMutation(
            space=space,
            max_exchange_fraction=1.0,
        )

        child = operator.apply(
            parents=((0.0, 0.0, 0.0),),
            random_state=rng(0),
        )

        assert len(child) == 3
        assert all(-1.0 <= value <= 1.0 for value in child)
        assert child != (0.0, 0.0, 0.0)

    def test_bounded_mutation_stays_inside_declared_bounds(self) -> None:
        space = ArraySpace(RealSpace(-2.0, 2.0), length=3)
        operator = BoundedMutation(
            space=space,
            max_perturbation_fraction=0.25,
        )

        child = operator.apply(
            parents=((1.5, -1.5, 0.5),),
            random_state=rng(1),
        )

        assert len(child) == 3
        assert all(-2.0 <= value <= 2.0 for value in child)

    def test_differential_evolution_variation_matches_expected_numeric_donor(self) -> None:
        space = ArraySpace(RealSpace(-10.0, 10.0), length=2)
        operator = DifferentialEvolutionVariation(
            space=space,
            mutation_range=(0.5, 0.5),
            recombination_probability=1.0,
            n_cross=1,
        )

        child = operator.apply(
            parents=(
                (0.0, 0.0),
                (1.0, 2.0),
                (4.0, 8.0),
                (2.0, 1.0),
            ),
            random_state=rng(0),
        )

        assert child == (2.0, 5.5)

    def test_differential_evolution_variation_rejects_active_topology_mismatch(self) -> None:
        space = ConditionalNumericPairSpace(
            head_space=IntegerSpace(0, 9),
            tail_space=IntegerSpace(0, 9),
        )
        operator = DifferentialEvolutionVariation(
            space=space,
            mutation_range=(0.5, 0.5),
            recombination_probability=1.0,
            n_cross=1,
        )

        with pytest.raises(ValueError, match="matching active topology"):
            _ = operator.apply(
                parents=(
                    (0, 1),
                    (1, 1),
                    (1, 4),
                    (1, 2),
                ),
                random_state=rng(0),
            )

    def test_mixture_variation_supports_mixed_arities(self) -> None:
        space = IntegerSpace(0, 9)
        operator = MixtureVariation(
            operators=(
                RandomResetMutation(space=space, max_exchange_fraction=1.0),
                UniformCrossover(space=space, max_exchange_fraction=1.0),
            ),
            weights=(0.0, 1.0),
        )

        child = operator.apply(
            parents=(3, 9),
            random_state=rng(0),
        )

        assert operator.arity == 2
        assert child == 9


class OptimizationSmokeTests:
    """Smoke tests showing the built-in operators can reduce objective values."""

    def test_built_in_csa_variation_can_minimize_scalar_quadratic(self) -> None:
        problem, variation = build_scalar_quadratic_setup()

        initial_optimizer = CSAOptimizer(
            space=problem.space,
            diversity_metric=IntegerAbsoluteDistance(),
            bank_capacity=8,
            profile=CSAProfile(
                perturbation_schedule=CSAPerturbationSchedule(
                    mutation_family=(CSAPerturbationSpec(variation),),
                ),
                seed_count=3,
                cutoff_schedule=CSACutoffSchedule(
                    reduction_factor=0.8,
                    stagnation_update_limit=0,
                ),
            ),
            random_state=1,
        )
        evaluator = SequentialEvaluator[int, int]()
        initial_state = initial_optimizer.create_initial_state()
        initial_proposals, _ = initial_optimizer.ask(initial_state, batch_size=8)
        initial_outcomes = evaluator.evaluate(
            problem,
            _requests(initial_proposals),
        )
        initial_best_value = min(
            outcome.observation.value
            for outcome in initial_outcomes
        )

        optimizer = CSAOptimizer(
            space=problem.space,
            diversity_metric=IntegerAbsoluteDistance(),
            bank_capacity=8,
            profile=CSAProfile(
                perturbation_schedule=CSAPerturbationSchedule(
                    mutation_family=(CSAPerturbationSpec(variation),),
                ),
                seed_count=3,
                cutoff_schedule=CSACutoffSchedule(
                    reduction_factor=0.8,
                    stagnation_update_limit=0,
                ),
            ),
            random_state=1,
        )
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=SequentialEvaluator[int, int](),
        )

        result, _ = study.optimize(max_evaluations=40)

        assert result.best_observation is not None
        assert result.best_observation is not None
        assert result.best_observation.value < initial_best_value


def rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def build_scalar_quadratic_setup() -> tuple[Problem[int, int], MixtureVariation[int]]:
    problem = Problem(
        space=IntegerSpace(-10, 10),
        objective=IntegerSquareObjective(),
    )
    variation = MixtureVariation(
        operators=(
            RandomResetMutation(
                space=problem.space,
                max_exchange_fraction=1.0,
            ),
            BoundedMutation(
                space=problem.space,
                max_perturbation_fraction=0.5,
            ),
        ),
        weights=(2.0, 1.0),
    )
    return problem, variation
