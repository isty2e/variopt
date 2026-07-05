"""Tests for built-in variopt search spaces."""

from collections.abc import Mapping, Sequence
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
    SearchSpace,
    TupleSpace,
)
from variopt.randomness import normalize_random_state
from variopt.spaces import (
    LeafPath,
    RecordCandidate,
    SpaceBoundaryValue,
    SpaceCandidateValue,
    SpaceScalarValue,
    StructuredLeafSpace,
    StructuredSearchSpace,
)
from variopt.spaces.structured import (
    is_space_candidate_value,
    is_space_scalar_value,
    require_space_candidate_value,
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


def require_sum_tuple_candidate(candidate: SpaceCandidateValue) -> tuple[int, ...]:
    """Return a canonical tuple-of-int candidate for sum-equality tests."""
    if type(candidate) is not tuple:
        msg = "sum tuple candidate must be canonical tuple"
        raise TypeError(msg)

    values: list[int] = []
    for value in candidate:
        if type(value) is not int:
            msg = "sum tuple candidate values must be canonical integers"
            raise TypeError(msg)
        values.append(value)
    return tuple(values)


class SumTupleSpace(SearchSpace[SpaceBoundaryValue, SpaceCandidateValue]):
    """Test space with equality semantics that differ from raw tuple equality."""

    @override
    def normalize(self, raw_candidate: SpaceBoundaryValue) -> SpaceCandidateValue:
        if (
            isinstance(raw_candidate, (bytes, bytearray, str))
            or not isinstance(raw_candidate, Sequence)
        ):
            msg = "sum tuple boundary candidate must be a non-string sequence"
            raise TypeError(msg)
        values: list[int] = []
        for value in raw_candidate:
            if type(value) is not int:
                msg = "sum tuple boundary values must be canonical integers"
                raise TypeError(msg)
            values.append(value)
        return tuple(values)

    @override
    def validate(self, candidate: SpaceCandidateValue) -> None:
        _ = require_sum_tuple_candidate(candidate)

    @override
    def sample(self, random_state: np.random.RandomState) -> SpaceCandidateValue:
        _ = random_state
        return (1,)

    @override
    def candidates_equal(
        self,
        left_candidate: SpaceCandidateValue,
        right_candidate: SpaceCandidateValue,
    ) -> bool:
        left_tuple = require_sum_tuple_candidate(left_candidate)
        right_tuple = require_sum_tuple_candidate(right_candidate)
        return sum(left_tuple) == sum(right_tuple)


class SearchSpaceTests:
    """Conformance checks for the built-in search space family."""

    def test_structured_candidate_predicate_rejects_numpy_scalars(self) -> None:
        scalar = np.float64(1.0)

        assert not is_space_scalar_value(scalar)
        assert not is_space_candidate_value(scalar)
        with pytest.raises(TypeError):
            _ = require_space_candidate_value(
                scalar,
                operation="test structured candidate gate",
            )

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

    def test_categorical_space_sample_uses_canonical_randint_index(self) -> None:
        space: CategoricalSpace[str] = CategoricalSpace(("a", "b", "c"))

        candidate = space.sample(normalize_random_state(3))

        assert candidate == "c"

    def test_categorical_space_normalizes_to_declared_choice_type(self) -> None:
        numeric_space: CategoricalSpace[SpaceScalarValue] = CategoricalSpace((1.0, 2.0))
        binary_space: CategoricalSpace[SpaceScalarValue] = CategoricalSpace((0, 1))

        numeric_candidate = numeric_space.normalize(1)
        binary_candidate = binary_space.normalize(True)

        assert numeric_candidate == 1.0
        assert type(numeric_candidate) is float
        assert binary_candidate == 1
        assert type(binary_candidate) is int

    def test_categorical_space_validate_rejects_equal_noncanonical_value(self) -> None:
        numeric_space: CategoricalSpace[SpaceScalarValue] = CategoricalSpace((1.0, 2.0))
        binary_space: CategoricalSpace[SpaceScalarValue] = CategoricalSpace((0, 1))

        with pytest.raises(TypeError):
            numeric_space.validate(1)

        with pytest.raises(TypeError):
            binary_space.validate(True)

    def test_categorical_space_canonicalizes_equal_bytearray_boundary_value(self) -> None:
        space: CategoricalSpace[SpaceScalarValue] = CategoricalSpace((b"a", b"b"))

        candidate = space.normalize(bytearray(b"a"))

        assert candidate == b"a"
        assert type(candidate) is bytes
        with pytest.raises(TypeError):
            space.validate(bytearray(b"a"))

    def test_categorical_space_alternatives_reject_equal_noncanonical_value(self) -> None:
        space: CategoricalSpace[SpaceScalarValue] = CategoricalSpace((0, 1))

        with pytest.raises(TypeError):
            _ = space.alternatives(True)

    def test_categorical_space_rejects_duplicate_choices(self) -> None:
        with pytest.raises(ValueError):
            _ = CategoricalSpace(("a", "a"))

    def test_categorical_space_rejects_non_scalar_choices(self) -> None:
        choices = cast(Sequence[SpaceScalarValue], (("nested",),))

        with pytest.raises(TypeError):
            _ = CategoricalSpace(choices)

    def test_categorical_space_rejects_numpy_scalar_choices(self) -> None:
        choices = cast(Sequence[SpaceScalarValue], (np.float64(1.0),))

        with pytest.raises(TypeError):
            _ = CategoricalSpace(choices)

    def test_categorical_space_normalize_rejects_non_scalar_equal_object(self) -> None:
        class EqualToOne:
            @override
            def __eq__(self, other: object) -> bool:
                return other == 1

        space: CategoricalSpace[SpaceScalarValue] = CategoricalSpace((1,))
        boundary_space = cast(SearchSpace[object, SpaceScalarValue], space)

        with pytest.raises(TypeError):
            _ = boundary_space.normalize(EqualToOne())

    @pytest.mark.parametrize("choice", [float("inf"), float("-inf"), float("nan")])
    def test_categorical_space_rejects_nonfinite_float_choices(
        self,
        choice: float,
    ) -> None:
        with pytest.raises(ValueError):
            _ = CategoricalSpace((1.0, choice))

    def test_tuple_space_normalizes_heterogeneous_sequence(self) -> None:
        space = TupleSpace(IntegerSpace(0, 5), RealSpace(-1.0, 1.0))

        candidate = space.normalize([3, 0])

        assert candidate == (3, 0.0)

    def test_tuple_space_validate_rejects_noncanonical_real_child(self) -> None:
        space = TupleSpace(RealSpace(0.0, 10.0))

        with pytest.raises(TypeError):
            space.validate((3,))

    def test_tuple_space_equality_rejects_noncanonical_real_child(self) -> None:
        space = TupleSpace(RealSpace(0.0, 10.0))

        with pytest.raises(TypeError):
            _ = space.candidates_equal((3,), (3.0,))

    def test_tuple_space_leaf_access_rejects_noncanonical_real_child(self) -> None:
        space = TupleSpace(RealSpace(0.0, 10.0))

        with pytest.raises(TypeError):
            _ = space.leaf_value_at_path((3,), (0,))

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values((3,), {(0,): 4.0})

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

    def test_permutation_space_rejects_bool_leaf_path_index(self) -> None:
        space = PermutationSpace(size=4)
        candidate = space.normalize([0, 1, 2, 3])

        with pytest.raises(TypeError):
            _ = space.leaf_space_at_path((True,))

        with pytest.raises(TypeError):
            _ = space.leaf_value_at_path(candidate, (True,))

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {(True,): 2})

    def test_permutation_space_rejects_empty_leaf_path(self) -> None:
        space = PermutationSpace(size=4)
        candidate = space.normalize([0, 1, 2, 3])

        with pytest.raises(TypeError):
            _ = space.leaf_space_at_path(())

        with pytest.raises(TypeError):
            _ = space.leaf_value_at_path(candidate, ())

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {(): 2})

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

    def test_record_space_rejects_reordered_canonical_candidate(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 4),
            scale=RealSpace(0.0, 2.0),
        )
        candidate = RecordCandidate(entries=(("scale", 1.0), ("depth", 2)))

        with pytest.raises(ValueError):
            space.validate(candidate)

    def test_record_space_validate_rejects_noncanonical_real_field(self) -> None:
        space = RecordSpace(scale=RealSpace(0.0, 2.0))
        candidate = RecordCandidate(entries=(("scale", 1),))

        with pytest.raises(TypeError):
            space.validate(candidate)

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

    def test_array_space_uses_element_candidate_equality(self) -> None:
        space = ArraySpace(SumTupleSpace(), length=2)

        assert space.candidates_equal(((1,), (2,)), ((1, 0), (0, 2)))
        assert not space.candidates_equal(((1,), (2,)), ((2,), (2,)))

    def test_tuple_space_recurses_into_child_candidate_equality(self) -> None:
        array_space = ArraySpace(SumTupleSpace(), length=1)
        space = TupleSpace(array_space, IntegerSpace(0, 5))

        assert space.candidates_equal((((1,),), 3), (((1, 0),), 3))
        assert not space.candidates_equal((((1,),), 3), (((1, 1),), 3))

    def test_record_space_recurses_into_child_candidate_equality(self) -> None:
        space = RecordSpace(
            items=ArraySpace(SumTupleSpace(), length=1),
            depth=IntegerSpace(0, 5),
        )
        left_candidate = space.normalize({"items": [(1,)], "depth": 3})
        right_candidate = space.normalize({"items": [(1, 0)], "depth": 3})
        different_candidate = space.normalize({"items": [(2,)], "depth": 3})

        assert space.candidates_equal(left_candidate, right_candidate)
        assert not space.candidates_equal(left_candidate, different_candidate)

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

    def test_tuple_space_replace_leaf_values_rejects_unknown_index(self) -> None:
        space = TupleSpace(IntegerSpace(0, 5), RealSpace(-1.0, 1.0))
        candidate = space.normalize([3, 0])

        with pytest.raises(TypeError):
            _ = space.leaf_space_at_path((2,))

        with pytest.raises(TypeError):
            _ = space.leaf_value_at_path(candidate, (2,))

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {(2,): 4})

    def test_tuple_space_rejects_bool_leaf_path_index(self) -> None:
        space = TupleSpace(IntegerSpace(0, 5), RealSpace(-1.0, 1.0))
        candidate = space.normalize([3, 0])

        with pytest.raises(TypeError):
            _ = space.leaf_space_at_path((True,))

        with pytest.raises(TypeError):
            _ = space.leaf_value_at_path(candidate, (True,))

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {(True,): 4})

    def test_tuple_space_replace_leaf_values_rejects_empty_path(self) -> None:
        space = TupleSpace(IntegerSpace(0, 5), RealSpace(-1.0, 1.0))
        candidate = space.normalize([3, 0])

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {(): 4})

    def test_record_space_replace_leaf_values_updates_only_touched_field(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 4),
            scale=RealSpace(0.0, 2.0),
        )
        candidate = space.normalize({"depth": 2, "scale": 1})

        replaced = space.replace_leaf_values(candidate, {("depth",): 3})

        assert replaced.entries == (("depth", 3), ("scale", 1.0))

    def test_record_space_replace_leaf_values_rejects_unknown_field_path(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 4),
            scale=RealSpace(0.0, 2.0),
        )
        candidate = space.normalize({"depth": 2, "scale": 1})

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {("missing",): 3})

    def test_record_space_rejects_non_string_leaf_path_segment(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 4),
            scale=RealSpace(0.0, 2.0),
        )
        candidate = space.normalize({"depth": 2, "scale": 1})

        with pytest.raises(TypeError):
            _ = space.leaf_space_at_path((0,))

        with pytest.raises(TypeError):
            _ = space.leaf_value_at_path(candidate, (0,))

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {(0,): 3})

    def test_record_space_replace_leaf_values_rejects_empty_path(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 4),
            scale=RealSpace(0.0, 2.0),
        )
        candidate = space.normalize({"depth": 2, "scale": 1})

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {(): 3})

    def test_record_space_nested_replace_leaf_values_rejects_unknown_child_path(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 4),
            schedule=TupleSpace(IntegerSpace(0, 3), RealSpace(0.0, 2.0)),
        )
        candidate = space.normalize({"depth": 2, "schedule": [1, 1.0]})

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {("schedule", 2): 1.5})

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {("schedule",): 1.5})

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

    def test_array_space_replace_leaf_values_rejects_unknown_index(self) -> None:
        space = ArraySpace(IntegerSpace(0, 9), length=3)
        candidate = space.normalize([1, 2, 3])

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {(3,): 7})

    def test_array_space_rejects_negative_leaf_path_index(self) -> None:
        space = ArraySpace(IntegerSpace(0, 9), length=3)
        candidate = space.normalize([1, 2, 3])

        with pytest.raises(TypeError):
            _ = space.leaf_space_at_path((-1,))

        with pytest.raises(TypeError):
            _ = space.leaf_value_at_path(candidate, (-1,))

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {(-1,): 7})

    def test_array_space_rejects_bool_leaf_path_index(self) -> None:
        space = ArraySpace(IntegerSpace(0, 9), length=3)
        candidate = space.normalize([1, 2, 3])

        with pytest.raises(TypeError):
            _ = space.leaf_space_at_path((True,))

        with pytest.raises(TypeError):
            _ = space.leaf_value_at_path(candidate, (True,))

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {(True,): 7})

    def test_array_space_replace_leaf_values_rejects_empty_path(self) -> None:
        space = ArraySpace(IntegerSpace(0, 9), length=3)
        candidate = space.normalize([1, 2, 3])

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values(candidate, {(): 7})

    def test_array_space_replace_leaf_values_supports_pair_update(self) -> None:
        space = ArraySpace(IntegerSpace(0, 9), length=4)
        candidate = space.normalize([1, 2, 3, 4])

        replaced = space.replace_leaf_values(candidate, {(0,): 8, (3,): 5})

        assert replaced == (8, 2, 3, 5)

    def test_composite_replace_leaf_values_empty_mapping_returns_same_candidate(self) -> None:
        tuple_space = TupleSpace(IntegerSpace(0, 5), RealSpace(-1.0, 1.0))
        record_space = RecordSpace(depth=IntegerSpace(1, 4), scale=RealSpace(0.0, 2.0))
        array_space = ArraySpace(IntegerSpace(0, 9), length=3)
        tuple_candidate = tuple_space.normalize([3, 0])
        record_candidate = record_space.normalize({"depth": 2, "scale": 1})
        array_candidate = array_space.normalize([1, 2, 3])

        assert tuple_space.replace_leaf_values(tuple_candidate, {}) is tuple_candidate
        assert record_space.replace_leaf_values(record_candidate, {}) is record_candidate
        assert array_space.replace_leaf_values(array_candidate, {}) is array_candidate

    def test_validated_composite_leaf_reads_do_not_revalidate_children(self) -> None:
        validate_count = 0

        class CountingIntegerSpace(IntegerSpace):
            """Integer space that counts validation calls."""

            @override
            def validate(self, candidate: int) -> None:
                nonlocal validate_count
                validate_count += 1
                super().validate(candidate)

        leaf_space = CountingIntegerSpace(0, 9)
        record_space = RecordSpace(
            pair=TupleSpace(leaf_space, leaf_space),
            order=PermutationSpace(3),
        )
        record_candidate = record_space.normalize(
            {
                "pair": [1, 2],
                "order": [0, 1, 2],
            },
        )
        array_space = ArraySpace(TupleSpace(leaf_space, leaf_space), length=2)
        array_candidate = array_space.normalize([[3, 4], [5, 6]])

        validate_count = 0
        record_space.validate(record_candidate)
        for path in record_space.leaf_paths():
            _ = record_space.leaf_value_at_validated_path(record_candidate, path)
        array_space.validate(array_candidate)
        for path in array_space.leaf_paths():
            _ = array_space.leaf_value_at_validated_path(array_candidate, path)

        assert validate_count == 6

    def test_validated_composite_replacement_validates_each_changed_leaf_once(self) -> None:
        validate_count = 0

        class CountingIntegerSpace(IntegerSpace):
            """Integer space that counts validation calls."""

            @override
            def validate(self, candidate: int) -> None:
                nonlocal validate_count
                validate_count += 1
                super().validate(candidate)

        leaf_space = CountingIntegerSpace(0, 9)
        record_space = RecordSpace(
            pair=TupleSpace(leaf_space, leaf_space),
        )
        record_candidate = record_space.normalize({"pair": [1, 2]})
        array_space = ArraySpace(TupleSpace(leaf_space, leaf_space), length=2)
        array_candidate = array_space.normalize([[3, 4], [5, 6]])

        validate_count = 0
        replaced_record = record_space.replace_leaf_values(
            record_candidate,
            {("pair", 0): 8},
        )
        replaced_array = array_space.replace_leaf_values(array_candidate, {(1, 0): 7})

        assert replaced_record.entries == (("pair", (8, 2)),)
        assert replaced_array == ((3, 4), (7, 6))
        assert validate_count == 8

    def test_validated_permutation_replacement_preserves_global_constraint(self) -> None:
        space = PermutationSpace(4)
        candidate = space.normalize([0, 1, 2, 3])
        space.validate(candidate)

        replaced = space.replace_leaf_values_in_validated_candidate(
            candidate,
            {
                (0,): 1,
                (1,): 0,
            },
        )

        assert replaced == (1, 0, 2, 3)
        with pytest.raises(ValueError):
            _ = space.replace_leaf_values_in_validated_candidate(candidate, {(0,): 1})

    def test_validated_array_replacement_rejects_bool_path_index(self) -> None:
        space = ArraySpace(IntegerSpace(0, 9), length=2)
        candidate = space.normalize([1, 2])
        space.validate(candidate)

        with pytest.raises(TypeError):
            _ = space.replace_leaf_values_in_validated_candidate(
                candidate,
                {(True,): 7},
            )

    def test_custom_space_validated_fallback_preserves_public_validation(self) -> None:
        validate_count = 0

        class CustomRootSpace(StructuredSearchSpace[int, int]):
            """Custom structured space without built-in validated traversal hooks."""

            @override
            def normalize(self, raw_candidate: int) -> int:
                self.validate(raw_candidate)
                return raw_candidate

            @override
            def validate(self, candidate: int) -> None:
                nonlocal validate_count
                validate_count += 1
                IntegerSpace(0, 9).validate(candidate)

            @override
            def sample(self, random_state: np.random.RandomState) -> int:
                return IntegerSpace(0, 9).sample(random_state)

            @override
            def leaf_paths(self) -> tuple[LeafPath, ...]:
                return ((),)

            @override
            def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
                return IntegerSpace(0, 9).leaf_space_at_path(path)

            @override
            def leaf_value_at_path(
                self,
                candidate: int,
                path: LeafPath,
            ) -> SpaceCandidateValue:
                self.validate(candidate)
                if path != ():
                    msg = f"invalid custom path: {path!r}"
                    raise TypeError(msg)
                return candidate

            @override
            def replace_leaf_values(
                self,
                candidate: int,
                replacements: Mapping[LeafPath, SpaceCandidateValue],
            ) -> int:
                self.validate(candidate)
                if () not in replacements:
                    return candidate
                replacement = replacements[()]
                if type(replacement) is not int:
                    msg = "custom replacement must be an integer"
                    raise TypeError(msg)
                return IntegerSpace(0, 9).normalize(replacement)

        space = CustomRootSpace()
        candidate = space.normalize(3)

        validate_count = 0
        space.validate(candidate)
        assert space.active_leaf_paths_for_validated_candidate(candidate) == ((),)
        assert space.leaf_value_at_validated_path(candidate, ()) == 3
        assert space.replace_leaf_values_in_validated_candidate(candidate, {(): 4}) == 4

        assert validate_count == 4

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
