"""Tests for built-in variopt search spaces."""

from collections.abc import Sequence
from math import exp, log
from typing import cast

import numpy as np
import pytest
from typing_extensions import override

from tests import conformance as contract_cases
from variopt import (
    ArraySpace,
    CategoricalSpace,
    IntegerSpace,
    PermutationSpace,
    RealSpace,
    RecordSpace,
    TupleSpace,
)
from variopt.randomness import normalize_random_state
from variopt.spaces import (
    RecordCandidate,
    SpaceBoundaryValue,
    SpaceCandidateValue,
)


class RealSpaceConformanceTests(
    contract_cases.SearchSpaceConformanceCase[int | float, float],
):
    """SearchSpace conformance for RealSpace."""

    @override
    def make_space(self) -> RealSpace:
        return RealSpace(low=-1.0, high=1.0)

    @override
    def make_boundary_candidate(self) -> int:
        return 1

    @override
    def make_expected_candidate(self) -> float:
        return 1.0

    @override
    def make_invalid_boundary_candidate(self) -> bool:
        return True


class IntegerSpaceConformanceTests(
    contract_cases.SearchSpaceConformanceCase[int, int],
):
    """SearchSpace conformance for IntegerSpace."""

    @override
    def make_space(self) -> IntegerSpace:
        return IntegerSpace(low=0, high=5)

    @override
    def make_boundary_candidate(self) -> int:
        return 3

    @override
    def make_expected_candidate(self) -> int:
        return 3

    @override
    def make_invalid_boundary_candidate(self) -> bool:
        return True


class CategoricalSpaceConformanceTests(
    contract_cases.SearchSpaceConformanceCase[str, str],
):
    """SearchSpace conformance for CategoricalSpace."""

    @override
    def make_space(self) -> CategoricalSpace[str]:
        return CategoricalSpace(("a", "b", "c"))

    @override
    def make_boundary_candidate(self) -> str:
        return "b"

    @override
    def make_expected_candidate(self) -> str:
        return "b"

    @override
    def make_invalid_boundary_candidate(self) -> str:
        return "z"


class TupleSpaceConformanceTests(
    contract_cases.SearchSpaceConformanceCase[
        Sequence[SpaceBoundaryValue],
        tuple[SpaceCandidateValue, ...],
    ],
):
    """SearchSpace conformance for TupleSpace."""

    @override
    def make_space(self) -> TupleSpace:
        return TupleSpace(IntegerSpace(0, 5), RealSpace(-1.0, 1.0))

    @override
    def make_boundary_candidate(self) -> Sequence[SpaceBoundaryValue]:
        return [3, 0]

    @override
    def make_expected_candidate(self) -> tuple[SpaceCandidateValue, ...]:
        return (3, 0.0)

    @override
    def make_invalid_boundary_candidate(self) -> str:
        return "bad"


class RecordSpaceConformanceTests(
    contract_cases.SearchSpaceConformanceCase[
        dict[str, SpaceBoundaryValue],
        RecordCandidate,
    ],
):
    """SearchSpace conformance for RecordSpace."""

    @override
    def make_space(self) -> RecordSpace:
        return RecordSpace(
            depth=IntegerSpace(1, 4),
            scale=RealSpace(0.0, 2.0),
        )

    @override
    def make_boundary_candidate(self) -> dict[str, SpaceBoundaryValue]:
        return {"scale": 1, "depth": 2}

    @override
    def make_expected_candidate(self) -> RecordCandidate:
        return RecordCandidate(entries=(("depth", 2), ("scale", 1.0)))

    @override
    def make_invalid_boundary_candidate(self) -> dict[str, int]:
        return {"depth": 2}


class ArraySpaceConformanceTests(
    contract_cases.SearchSpaceConformanceCase[list[int], tuple[int, ...]],
):
    """SearchSpace conformance for ArraySpace."""

    @override
    def make_space(self) -> ArraySpace[int, int]:
        return ArraySpace(IntegerSpace(0, 9), length=3)

    @override
    def make_boundary_candidate(self) -> list[int]:
        return [1, 2, 3]

    @override
    def make_expected_candidate(self) -> tuple[int, ...]:
        return (1, 2, 3)

    @override
    def make_invalid_boundary_candidate(self) -> list[int]:
        return [1, 2]


class PermutationSpaceConformanceTests(
    contract_cases.SearchSpaceConformanceCase[list[int], tuple[int, ...]],
):
    """SearchSpace conformance for PermutationSpace."""

    @override
    def make_space(self) -> PermutationSpace:
        return PermutationSpace(size=4)

    @override
    def make_boundary_candidate(self) -> list[int]:
        return [2, 0, 3, 1]

    @override
    def make_expected_candidate(self) -> tuple[int, ...]:
        return (2, 0, 3, 1)

    @override
    def make_invalid_boundary_candidate(self) -> list[int]:
        return [0, 1, 1, 3]


class SearchSpaceTests:
    """Conformance checks for the built-in search space family."""

    def test_real_space_normalizes_to_float(self) -> None:
        space = RealSpace(low=-1.0, high=1.0)

        candidate = space.normalize(1)

        assert candidate == 1.0
        assert isinstance(candidate, float)

    def test_real_space_sample_matches_random_state_uniform(self) -> None:
        space = RealSpace(low=-2.0, high=3.0)
        actual = space.sample(normalize_random_state(7))
        expected = float(np.random.RandomState(7).uniform(-2.0, 3.0))

        assert actual == expected

    def test_real_space_log_sample_matches_log_uniform(self) -> None:
        space = RealSpace(low=1e-3, high=1e1, scale="log")
        actual = space.sample(normalize_random_state(7))
        expected = float(exp(np.random.RandomState(7).uniform(log(1e-3), log(1e1))))

        assert actual == expected

    def test_real_space_log_scale_requires_positive_bounds(self) -> None:
        with pytest.raises(ValueError):
            _ = RealSpace(low=0.0, high=1.0, scale="log")

    def test_integer_space_rejects_bool(self) -> None:
        space = IntegerSpace(low=0, high=3)

        with pytest.raises(TypeError):
            _ = space.normalize(True)

    def test_integer_space_rejects_non_integer_bounds(self) -> None:
        with pytest.raises(TypeError):
            _ = IntegerSpace(low=cast(int, 0.0), high=3)

    def test_integer_space_sample_matches_random_state_randint(self) -> None:
        space = IntegerSpace(low=2, high=6)
        actual = space.sample(normalize_random_state(11))

        assert actual == 3

    def test_integer_space_log_sample_matches_coordinate_projection(self) -> None:
        space = IntegerSpace(low=2, high=6, scale="log")
        actual = space.sample(normalize_random_state(11))
        expected = min(
            6,
            max(
                2,
                int(round(exp(np.random.RandomState(11).uniform(log(2.0), log(6.0))))),
            ),
        )

        assert actual == expected

    def test_integer_space_log_scale_requires_positive_bounds(self) -> None:
        with pytest.raises(ValueError):
            _ = IntegerSpace(low=0, high=3, scale="log")

    def test_categorical_space_rejects_unknown_choice(self) -> None:
        space: CategoricalSpace[str] = CategoricalSpace(("a", "b", "c"))

        with pytest.raises(ValueError):
            _ = space.normalize("z")

    def test_categorical_space_rejects_duplicate_choices(self) -> None:
        with pytest.raises(ValueError):
            _ = CategoricalSpace(("a", "a"))

    def test_tuple_space_normalizes_heterogeneous_sequence(self) -> None:
        space = TupleSpace(IntegerSpace(0, 5), RealSpace(-1.0, 1.0))

        candidate = space.normalize([3, 0])

        assert candidate == (3, 0.0)

    def test_permutation_space_rejects_duplicates(self) -> None:
        space = PermutationSpace(size=4)

        with pytest.raises(ValueError):
            _ = space.normalize([0, 1, 1, 3])

    def test_permutation_space_sample_is_a_valid_permutation(self) -> None:
        space = PermutationSpace(size=5)

        first_candidate = space.sample(normalize_random_state(7))
        second_candidate = space.sample(normalize_random_state(7))

        assert first_candidate == second_candidate
        assert tuple(sorted(first_candidate)) == (0, 1, 2, 3, 4)

    def test_permutation_space_replace_leaf_values_preserves_global_constraint(self) -> None:
        space = PermutationSpace(size=4)
        candidate = space.normalize([0, 1, 2, 3])

        with pytest.raises(ValueError):
            _ = space.replace_leaf_values(candidate, {(0,): 1})
        space.validate(candidate)

    def test_record_space_returns_record_candidate(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 4),
            scale=RealSpace(0.0, 2.0),
        )

        candidate = space.normalize({"depth": 2, "scale": 1})

        assert isinstance(candidate, RecordCandidate)
        assert candidate["depth"] == 2
        assert candidate["scale"] == 1.0
        assert candidate.as_dict() == {"depth": 2, "scale": 1.0}

    def test_record_space_normalizes_mapping_without_order_dependency(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 4),
            scale=RealSpace(0.0, 2.0),
        )

        candidate = space.normalize({"scale": 1, "depth": 2})

        assert candidate.entries == (("depth", 2), ("scale", 1.0))

    def test_record_space_is_idempotent_for_canonical_candidate(self) -> None:
        space = RecordSpace(level=IntegerSpace(0, 3))
        canonical = RecordCandidate(entries=(("level", 1),))

        candidate = space.normalize(canonical)

        assert candidate == canonical

    def test_record_space_rejects_malformed_record_candidate_input(self) -> None:
        space = RecordSpace(level=IntegerSpace(0, 3))
        malformed = RecordCandidate(entries=(("level", 1), ("level", 2)))

        with pytest.raises(ValueError):
            _ = space.normalize(malformed)

    def test_array_space_normalizes_fixed_length_sequence(self) -> None:
        space = ArraySpace(IntegerSpace(0, 9), length=3)

        candidate = space.normalize([1, 2, 3])

        assert candidate == (1, 2, 3)
        space.validate(candidate)

    def test_array_space_rejects_non_integer_length(self) -> None:
        with pytest.raises(TypeError):
            _ = ArraySpace(IntegerSpace(0, 9), length=cast(int, True))

        with pytest.raises(TypeError):
            _ = ArraySpace(IntegerSpace(0, 9), length=cast(int, 2.5))

    def test_array_space_sampling_is_deterministic_for_same_seed(self) -> None:
        space = ArraySpace(RealSpace(-1.0, 1.0), length=4)

        rng_one = normalize_random_state(17)
        rng_two = normalize_random_state(17)

        sample_one = space.sample(rng_one)
        sample_two = space.sample(rng_two)

        assert sample_one == sample_two

    def test_built_in_structured_spaces_report_all_declared_paths_as_active(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 4),
            schedule=TupleSpace(
                IntegerSpace(0, 3),
                CategoricalSpace(("x", "y")),
            ),
        )
        candidate = space.normalize(
            {
                "depth": 2,
                "schedule": [1, "x"],
            },
        )

        assert space.has_static_topology()
        assert space.active_leaf_paths(candidate) == space.leaf_paths()
        for path in space.leaf_paths():
            assert space.is_active_leaf_path(candidate, path)

    def test_tuple_space_replace_leaf_values_updates_only_touched_child(self) -> None:
        space = TupleSpace(IntegerSpace(0, 5), RealSpace(-1.0, 1.0))
        candidate = space.normalize([3, 0])

        replaced = space.replace_leaf_values(candidate, {(1,): 0.5})

        assert replaced == (3, 0.5)

    def test_record_space_replace_leaf_values_updates_only_touched_field(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 4),
            scale=RealSpace(0.0, 2.0),
        )
        candidate = space.normalize({"depth": 2, "scale": 1})

        replaced = space.replace_leaf_values(candidate, {("depth",): 3})

        assert replaced.entries == (("depth", 3), ("scale", 1.0))

    def test_record_space_uses_field_indices_for_internal_candidate_access(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lookup_count = 0
        original_getitem = RecordCandidate.__getitem__

        def count_getitem(
            candidate: RecordCandidate,
            key: str,
        ) -> SpaceCandidateValue:
            nonlocal lookup_count
            lookup_count += 1
            return original_getitem(candidate, key)

        monkeypatch.setattr(RecordCandidate, "__getitem__", count_getitem)
        space = RecordSpace(
            depth=IntegerSpace(1, 4),
            scale=RealSpace(0.0, 2.0),
        )
        candidate = RecordCandidate(entries=(("depth", 2), ("scale", 1.0)))

        space.validate(candidate)
        value = space.leaf_value_at_path(candidate, ("depth",))
        replaced = space.replace_leaf_values(candidate, {("scale",): 1.5})

        assert value == 2
        assert replaced.entries == (("depth", 2), ("scale", 1.5))
        assert lookup_count == 0

    def test_array_space_replace_leaf_values_updates_only_touched_index(self) -> None:
        space = ArraySpace(IntegerSpace(0, 9), length=3)
        candidate = space.normalize([1, 2, 3])

        replaced = space.replace_leaf_values(candidate, {(1,): 7})

        assert replaced == (1, 7, 3)

    def test_array_space_replace_leaf_values_supports_pair_update(self) -> None:
        space = ArraySpace(IntegerSpace(0, 9), length=4)
        candidate = space.normalize([1, 2, 3, 4])

        replaced = space.replace_leaf_values(candidate, {(0,): 8, (3,): 5})

        assert replaced == (8, 2, 3, 5)

    def test_record_space_sampling_is_valid(self) -> None:
        space = RecordSpace(
            width=IntegerSpace(1, 3),
            mode=CategoricalSpace(("x", "y")),
        )

        candidate = space.sample(normalize_random_state(5))

        assert isinstance(candidate, RecordCandidate)
        space.validate(candidate)

    def test_real_space_sample_is_within_bounds(self) -> None:
        space = RealSpace(low=-2.0, high=-1.0)

        candidate = space.sample(normalize_random_state(1))

        assert candidate >= -2.0
        assert candidate <= -1.0
