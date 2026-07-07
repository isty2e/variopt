"""Tests for permutation-safe population variation operators."""

import numpy as np
import pytest

from variopt import PermutationSpace
from variopt.algorithms.population.permutation import (
    InversionMutation,
    OrderCrossover,
    SwapMutation,
)
from variopt.algorithms.population.permutation.operators import (
    sample_exchange_count,
    sample_segment_bounds,
)


def rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


class PermutationOperatorTests:
    """Regression tests for permutation-safe operators."""

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

    def test_sample_segment_bounds_allows_full_segment(self) -> None:
        matching_seed = None
        for seed in range(1000):
            if sample_segment_bounds(
                size=6,
                max_segment_fraction=1.0,
                random_state=rng(seed),
            ) == (0, 6):
                matching_seed = seed
                break
        assert matching_seed is not None

        bounds = sample_segment_bounds(
            size=6,
            max_segment_fraction=1.0,
            random_state=rng(matching_seed),
        )

        assert bounds == (0, 6)

    def test_sample_segment_bounds_stays_inside_candidate_length(self) -> None:
        for seed in range(100):
            start_index, end_index = sample_segment_bounds(
                size=7,
                max_segment_fraction=0.5,
                random_state=rng(seed),
            )
            assert 0 <= start_index < end_index <= 7
            assert end_index - start_index <= 3

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
