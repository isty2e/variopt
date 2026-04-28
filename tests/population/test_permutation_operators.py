"""Tests for permutation-safe population variation operators."""


import numpy as np

from variopt import PermutationSpace
from variopt.algorithms.population.permutation import (
    InversionMutation,
    OrderCrossover,
    SwapMutation,
)


def rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


class PermutationOperatorTests:
    """Regression tests for permutation-safe operators."""

    def test_order_crossover_returns_valid_permutation(self) -> None:
        space = PermutationSpace(size=6)
        operator = OrderCrossover(space=space, max_segment_fraction=0.5)

        child = operator.apply(
            parents=((0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 0, 2)),
            random_state=rng(0),
        )

        assert tuple(sorted(child)) == (0, 1, 2, 3, 4, 5)
        assert child != (0, 1, 2, 3, 4, 5)
        assert child != (3, 5, 4, 1, 0, 2)

    def test_swap_mutation_returns_valid_permutation(self) -> None:
        space = PermutationSpace(size=5)
        operator = SwapMutation(space=space, max_swap_fraction=0.6)

        child = operator.apply(
            parents=((0, 1, 2, 3, 4),),
            random_state=rng(1),
        )

        assert tuple(sorted(child)) == (0, 1, 2, 3, 4)
        assert child != (0, 1, 2, 3, 4)

    def test_inversion_mutation_returns_valid_permutation(self) -> None:
        space = PermutationSpace(size=6)
        operator = InversionMutation(space=space, max_inversion_fraction=0.5)

        child = operator.apply(
            parents=((0, 1, 2, 3, 4, 5),),
            random_state=rng(2),
        )

        assert tuple(sorted(child)) == (0, 1, 2, 3, 4, 5)
        assert child != (0, 1, 2, 3, 4, 5)
