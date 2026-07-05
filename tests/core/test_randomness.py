"""Tests for the variopt randomness contract."""

from collections.abc import Sequence

import numpy as np
import pytest
from typing_extensions import override

from tests import conformance as contract_cases
from variopt import SearchSpace, VariationOperator
from variopt.randomness import (
    RandomStateSnapshot,
    derive_random_state_snapshot,
    normalize_random_state,
    random_state_randint,
)


class DummySpace(SearchSpace[int, int]):
    """Minimal search space used to verify explicit RNG sampling."""

    @override
    def normalize(self, raw_candidate: int) -> int:
        return raw_candidate

    @override
    def validate(self, candidate: int) -> None:
        if candidate < 0:
            msg = "candidate must be non-negative"
            raise ValueError(msg)

    @override
    def sample(self, random_state: np.random.RandomState) -> int:
        return int(random_state.random_sample() * 1000.0)


class DummyVariation(VariationOperator[int]):
    """Minimal variation operator used to verify explicit RNG usage."""

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
        return parents[0] + int(random_state.random_sample() * 100.0)


class SearchSpaceSamplingConformanceTests(
    contract_cases.ExplicitRandomnessConformanceCase[int],
):
    """Randomness conformance for SearchSpace sampling."""

    @override
    def exercise_with_rng(self, random_state: np.random.RandomState) -> int:
        return DummySpace().sample(random_state)


class VariationOperatorConformanceTests(
    contract_cases.ExplicitRandomnessConformanceCase[int],
):
    """Randomness conformance for variation operators."""

    @override
    def exercise_with_rng(self, random_state: np.random.RandomState) -> int:
        return DummyVariation().apply([5], random_state)


class RandomnessContractTests:
    """Conformance tests for normalized RNG handling."""

    def test_normalize_random_state_returns_local_rng(self) -> None:
        rng = normalize_random_state(123)

        assert isinstance(rng, np.random.RandomState)

    def test_repeated_seed_reproduces_same_sequence(self) -> None:
        rng_one = normalize_random_state(123)
        rng_two = normalize_random_state(123)

        seq_one = rng_one.random_sample(8)
        seq_two = rng_two.random_sample(8)

        np.testing.assert_array_equal(seq_one, seq_two)

    def test_random_state_normalization_does_not_touch_global_rng(self) -> None:
        np.random.seed(999)

        local_rng = normalize_random_state(123)
        _ = local_rng.random_sample(8)

        after = np.random.random_sample(8)

        np.random.seed(999)
        expected = np.random.random_sample(8)

        np.testing.assert_array_equal(after, expected)

    def test_search_space_sampling_uses_explicit_rng(self) -> None:
        space = DummySpace()

        rng_one = normalize_random_state(7)
        rng_two = normalize_random_state(7)

        sample_one = space.sample(rng_one)
        sample_two = space.sample(rng_two)

        assert sample_one == sample_two

    def test_variation_operator_uses_explicit_rng(self) -> None:
        operator = DummyVariation()

        rng_one = normalize_random_state(11)
        rng_two = normalize_random_state(11)

        child_one = operator.apply([5], rng_one)
        child_two = operator.apply([5], rng_two)

        assert child_one == child_two

    def test_bool_seed_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            _ = normalize_random_state(True)

    def test_spawn_seeds_matches_repeated_scalar_advances(self) -> None:
        snapshot = RandomStateSnapshot.from_seed(123)

        spawned_seeds, spawned_snapshot = snapshot.spawn_seeds(8)

        advanced_snapshot = snapshot
        advanced_seeds: list[int] = []
        for _ in range(8):
            seed, advanced_snapshot = advanced_snapshot.advance(
                lambda random_state: random_state_randint(
                    random_state,
                    low=0,
                    high=int(np.iinfo(np.int32).max),
                ),
            )
            advanced_seeds.append(seed)

        assert spawned_seeds == tuple(advanced_seeds)
        assert spawned_snapshot == advanced_snapshot

    def test_spawn_seeds_zero_count_returns_original_snapshot(self) -> None:
        snapshot = RandomStateSnapshot.from_seed(7)

        seeds, next_snapshot = snapshot.spawn_seeds(0)

        assert seeds == ()
        assert next_snapshot == snapshot

    def test_random_state_snapshot_from_dict_rejects_bool_position(self) -> None:
        snapshot = RandomStateSnapshot.from_seed(7).to_dict()
        snapshot["position"] = True

        with pytest.raises(TypeError, match="position"):
            _ = RandomStateSnapshot.from_dict(snapshot)

    def test_random_state_snapshot_from_dict_rejects_bool_has_gaussian(self) -> None:
        snapshot = RandomStateSnapshot.from_seed(7).to_dict()
        snapshot["has_gaussian"] = True

        with pytest.raises(TypeError, match="has_gaussian"):
            _ = RandomStateSnapshot.from_dict(snapshot)

    @pytest.mark.parametrize("cached_gaussian", [float("nan"), float("inf")])
    def test_random_state_snapshot_from_dict_rejects_non_finite_cached_gaussian(
        self,
        cached_gaussian: float,
    ) -> None:
        snapshot = RandomStateSnapshot.from_seed(7).to_dict()
        snapshot["cached_gaussian"] = cached_gaussian

        with pytest.raises(ValueError, match="cached_gaussian"):
            _ = RandomStateSnapshot.from_dict(snapshot)

    def test_random_state_snapshot_from_dict_rejects_malformed_key_hex(self) -> None:
        snapshot = RandomStateSnapshot.from_seed(7).to_dict()
        snapshot["key_hex"] = "not-hex"

        with pytest.raises(ValueError, match="key_hex"):
            _ = RandomStateSnapshot.from_dict(snapshot)

    def test_random_state_snapshot_rejects_runtime_bool_position(self) -> None:
        snapshot = RandomStateSnapshot.from_seed(7)

        with pytest.raises(TypeError, match="position"):
            _ = RandomStateSnapshot(
                algorithm=snapshot.algorithm,
                key_bytes=snapshot.key_bytes,
                position=True,
                has_gaussian=snapshot.has_gaussian,
                cached_gaussian=snapshot.cached_gaussian,
            )

    def test_derived_random_state_snapshot_is_stable_for_same_keys(self) -> None:
        snapshot = RandomStateSnapshot.from_seed(7)

        first_child = derive_random_state_snapshot(
            snapshot,
            namespace="test.local_search",
            keys=("proposal-1",),
        )
        second_child = derive_random_state_snapshot(
            snapshot,
            namespace="test.local_search",
            keys=("proposal-1",),
        )

        assert first_child == second_child

    def test_derived_random_state_snapshot_separates_namespaces_and_keys(
        self,
    ) -> None:
        snapshot = RandomStateSnapshot.from_seed(7)

        baseline_child = derive_random_state_snapshot(
            snapshot,
            namespace="test.local_search",
            keys=("proposal-1",),
        )
        other_namespace_child = derive_random_state_snapshot(
            snapshot,
            namespace="test.other",
            keys=("proposal-1",),
        )
        other_key_child = derive_random_state_snapshot(
            snapshot,
            namespace="test.local_search",
            keys=("proposal-2",),
        )

        assert baseline_child != other_namespace_child
        assert baseline_child != other_key_child

    def test_derived_random_state_snapshot_separates_ambiguous_key_boundaries(
        self,
    ) -> None:
        snapshot = RandomStateSnapshot.from_seed(7)

        first_child = derive_random_state_snapshot(
            snapshot,
            namespace="test.local_search",
            keys=("ab", "c"),
        )
        second_child = derive_random_state_snapshot(
            snapshot,
            namespace="test.local_search",
            keys=("a", "bc"),
        )

        assert first_child != second_child

    def test_derived_random_state_snapshot_rejects_empty_namespace(self) -> None:
        snapshot = RandomStateSnapshot.from_seed(7)

        with pytest.raises(ValueError, match="namespace"):
            _ = derive_random_state_snapshot(
                snapshot,
                namespace="",
                keys=("proposal-1",),
            )
