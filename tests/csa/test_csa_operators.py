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
from variopt.algorithms.population.csa.engine.ask import (
    apply_variation_operator_from_validated_parents,
)
from variopt.algorithms.population.csa.operators.editing import (
    sample_exchange_count,
)
from variopt.algorithms.population.csa.operators.mutation import (
    bounded_mutation_on_paths,
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

    def seed_for_exchange_count(
        self,
        *,
        leaf_count: int,
        max_exchange_fraction: float,
        target: int,
    ) -> int:
        """Return one deterministic seed that samples the requested count."""
        for seed in range(1000):
            if (
                sample_exchange_count(
                    leaf_count=leaf_count,
                    max_exchange_fraction=max_exchange_fraction,
                    random_state=rng(seed),
                )
                == target
            ):
                return seed
        msg = f"no seed sampled exchange count {target}"
        raise AssertionError(msg)

    def test_sample_exchange_count_can_attain_fractional_cap(self) -> None:
        seed = self.seed_for_exchange_count(
            leaf_count=6,
            max_exchange_fraction=0.5,
            target=3,
        )

        exchange_count = sample_exchange_count(
            leaf_count=6,
            max_exchange_fraction=0.5,
            random_state=rng(seed),
        )

        assert exchange_count == 3

    def test_sample_exchange_count_rejects_non_positive_leaf_count(self) -> None:
        with pytest.raises(ValueError, match="leaf_count must be positive"):
            _ = sample_exchange_count(
                leaf_count=0,
                max_exchange_fraction=1.0,
                random_state=rng(0),
            )

    @pytest.mark.parametrize(
        ("leaf_count", "max_exchange_fraction", "expected"),
        (
            (1, 1.0, 1),
            (5, 0.01, 1),
            (9, 0.111, 1),
            (5, 1.0, 5),
            (5, 0.5, 2),
            (6, 0.5, 3),
            (7, 0.5, 3),
        ),
    )
    def test_sample_exchange_count_bounds_are_inclusive(
        self,
        leaf_count: int,
        max_exchange_fraction: float,
        expected: int,
    ) -> None:
        seed = self.seed_for_exchange_count(
            leaf_count=leaf_count,
            max_exchange_fraction=max_exchange_fraction,
            target=expected,
        )

        exchange_count = sample_exchange_count(
            leaf_count=leaf_count,
            max_exchange_fraction=max_exchange_fraction,
            random_state=rng(seed),
        )

        assert exchange_count == expected

    def test_uniform_crossover_full_exchange_stays_valid(self) -> None:
        space = ArraySpace(IntegerSpace(0, 9), length=4)
        operator = UniformCrossover(space=space, max_exchange_fraction=1.0)

        child = operator.apply(
            parents=((1, 2, 3, 4), (5, 6, 7, 8)),
            random_state=rng(5),
        )

        assert len(child) == 4
        assert all(value in {1, 2, 3, 4, 5, 6, 7, 8} for value in child)
        space.validate(child)

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

    def test_uniform_crossover_validated_parent_fast_path_matches_apply(self) -> None:
        space = ArraySpace(IntegerSpace(0, 9), length=4)
        operator = UniformCrossover(space=space, max_exchange_fraction=1.0)
        parents = ((1, 2, 3, 4), (5, 6, 7, 8))

        public_child = operator.apply(parents=parents, random_state=rng(5))
        fast_child = operator.apply_from_validated_parents(
            parents=parents,
            random_state=rng(5),
        )

        assert fast_child == public_child

    def test_csa_validated_parent_dispatch_matches_apply(self) -> None:
        space = ArraySpace(IntegerSpace(0, 9), length=4)
        operator = UniformCrossover(space=space, max_exchange_fraction=1.0)
        parents = ((1, 2, 3, 4), (5, 6, 7, 8))

        public_child = operator.apply(parents=parents, random_state=rng(5))
        dispatched_child = apply_variation_operator_from_validated_parents(
            operator=operator,
            parents=parents,
            random_state=rng(5),
        )

        assert dispatched_child == public_child

    def test_bounded_mutation_on_paths_does_not_revalidate_per_leaf_read(self) -> None:
        validate_count = 0

        class CountingIntegerSpace(IntegerSpace):
            """Integer space that counts validation calls."""

            @override
            def validate(self, candidate: int) -> None:
                nonlocal validate_count
                validate_count += 1
                super().validate(candidate)

        leaf_space = CountingIntegerSpace(0, 9)
        space = ArraySpace(leaf_space, length=4)
        candidate = space.normalize([1, 2, 3, 4])

        validate_count = 0
        _ = bounded_mutation_on_paths(
            space=space,
            candidate=candidate,
            selected_paths=((0,), (1,), (2,)),
            max_perturbation_fraction=0.2,
            random_state=rng(0),
        )

        assert validate_count == 10

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

    def test_uniform_crossover_fast_path_rejects_active_topology_mismatch(self) -> None:
        space = ConditionalNumericPairSpace(
            head_space=IntegerSpace(0, 9),
            tail_space=IntegerSpace(0, 9),
        )
        operator = UniformCrossover(space=space, max_exchange_fraction=1.0)

        with pytest.raises(ValueError, match="matching active topology"):
            _ = operator.apply_from_validated_parents(
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

    def test_random_reset_validated_parent_fast_path_matches_apply(self) -> None:
        space = ArraySpace(RealSpace(-1.0, 1.0), length=3)
        operator = RandomResetMutation(space=space, max_exchange_fraction=1.0)
        parents = ((0.0, 0.0, 0.0),)

        public_child = operator.apply(parents=parents, random_state=rng(0))
        fast_child = operator.apply_from_validated_parents(
            parents=parents,
            random_state=rng(0),
        )

        assert fast_child == public_child

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

    def test_bounded_mutation_validated_parent_fast_path_matches_apply(self) -> None:
        space = ArraySpace(RealSpace(-2.0, 2.0), length=3)
        operator = BoundedMutation(space=space, max_perturbation_fraction=0.25)
        parents = ((1.5, -1.5, 0.5),)

        public_child = operator.apply(parents=parents, random_state=rng(1))
        fast_child = operator.apply_from_validated_parents(
            parents=parents,
            random_state=rng(1),
        )

        assert fast_child == public_child

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
