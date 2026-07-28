"""Tests for reusable encoded structured-space geometry plans."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import numpy as np
import pytest

from variopt import (
    ArraySpace,
    CategoricalSpace,
    IntegerSpace,
    PermutationSpace,
    RealSpace,
    RecordSpace,
    TupleSpace,
)
from variopt.spaces.geometry.compile import compile_structured_geometry
from variopt.spaces.geometry.plan import compile_builtin_geometry_plan
from variopt.spaces.geometry.taxonomy import BUILTIN_GEOMETRY_SPACE_TYPES
from variopt.spaces.structured import StructuredSearchSpace
from variopt.spaces.types import SpaceCandidateValue

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)


class CustomIntegerSpace(IntegerSpace):
    """Integer-space subtype with potentially distinct geometry behavior."""


def test_builtin_geometry_taxonomy_is_closed_and_explicit() -> None:
    assert frozenset(BUILTIN_GEOMETRY_SPACE_TYPES) == frozenset(
        (
            ArraySpace,
            CategoricalSpace,
            IntegerSpace,
            PermutationSpace,
            RealSpace,
            RecordSpace,
            TupleSpace,
        )
    )


def assert_plan_matches_existing_geometry(
    space: StructuredSearchSpace[BoundaryT, CandidateT],
    *,
    seed: int = 0,
) -> None:
    """Assert encoded and existing compiled geometry agree for sampled candidates."""
    plan = compile_builtin_geometry_plan(space)
    geometry = compile_structured_geometry(space)
    assert plan is not None
    assert geometry is not None

    random_state = np.random.RandomState(seed)
    candidates = tuple(space.sample(random_state) for _index in range(8))
    encodings = plan.encode_many(candidates)
    assert encodings == plan.encode_many(candidates)

    for left_index, left in enumerate(candidates):
        for right_index, right in enumerate(candidates):
            expected_parts = geometry.distance_parts(left, right)
            actual_parts = plan.distance_parts(left, right)
            assert actual_parts == expected_parts
            assert (
                plan.squared_distance(
                    encodings[left_index],
                    encodings[right_index],
                )
                == expected_parts.overlap_squared_distance
            )

    expected_to_many = tuple(
        plan.squared_distance(encodings[0], reference) for reference in encodings[1:]
    )
    assert plan.squared_distances_to_many(encodings[0], encodings[1:]) == (
        expected_to_many
    )

    pairwise_distances = plan.pairwise_squared_distances(encodings)
    assert len(pairwise_distances) == len(encodings)
    for left_index in range(len(encodings)):
        for right_index in range(len(encodings)):
            assert pairwise_distances[left_index][right_index] == plan.squared_distance(
                encodings[left_index],
                encodings[right_index],
            )


def test_plan_matches_scalar_geometry() -> None:
    assert_plan_matches_existing_geometry(RealSpace(-5.0, 5.0))
    assert_plan_matches_existing_geometry(RealSpace(0.1, 10.0, scale="log"))
    assert_plan_matches_existing_geometry(RealSpace(2.0, 2.0))
    assert_plan_matches_existing_geometry(IntegerSpace(-8, 8))
    assert_plan_matches_existing_geometry(IntegerSpace(1, 100, scale="log"))
    assert_plan_matches_existing_geometry(IntegerSpace(0, 1))
    assert_plan_matches_existing_geometry(IntegerSpace(3, 3))
    assert_plan_matches_existing_geometry(CategoricalSpace(("a", "b", "c")))
    assert_plan_matches_existing_geometry(CategoricalSpace((True, False)))
    assert_plan_matches_existing_geometry(PermutationSpace(7))


def test_direct_geometry_plans_do_not_claim_candidate_structure_reuse() -> None:
    plans = (
        compile_builtin_geometry_plan(RealSpace(-5.0, 5.0)),
        compile_builtin_geometry_plan(IntegerSpace(-8, 8)),
        compile_builtin_geometry_plan(CategoricalSpace(("a", "b", "c"))),
        compile_builtin_geometry_plan(PermutationSpace(4)),
    )

    for plan in plans:
        assert plan is not None
        assert not plan.reuses_candidate_structure


def test_container_plans_claim_candidate_structure_reuse() -> None:
    plans = (
        compile_builtin_geometry_plan(ArraySpace(RealSpace(-1.0, 1.0), length=1)),
        compile_builtin_geometry_plan(TupleSpace(IntegerSpace(-2, 2))),
        compile_builtin_geometry_plan(RecordSpace(value=IntegerSpace(-2, 2))),
    )

    for plan in plans:
        assert plan is not None
        assert plan.reuses_candidate_structure


def test_plan_matches_nested_composite_geometry() -> None:
    assert_plan_matches_existing_geometry(
        TupleSpace(
            RealSpace(-1.0, 1.0),
            IntegerSpace(-4, 4),
            CategoricalSpace(("red", "green", "blue")),
            PermutationSpace(4),
        )
    )
    assert_plan_matches_existing_geometry(
        RecordSpace(
            temperature=RealSpace(0.1, 10.0, scale="log"),
            mode=CategoricalSpace(("a", "b", "c")),
            order=PermutationSpace(5),
            count=IntegerSpace(-8, 8),
        )
    )
    assert_plan_matches_existing_geometry(
        ArraySpace(
            TupleSpace(
                RealSpace(-2.0, 2.0),
                CategoricalSpace((1, 2, 3)),
            ),
            length=5,
        )
    )
    assert_plan_matches_existing_geometry(
        ArraySpace(
            ArraySpace(IntegerSpace(-4, 4), length=3),
            length=2,
        )
    )


def test_plan_matches_single_element_array_geometry() -> None:
    assert_plan_matches_existing_geometry(ArraySpace(RealSpace(-2.0, 2.0), length=1))
    assert_plan_matches_existing_geometry(ArraySpace(IntegerSpace(-8, 8), length=1))
    assert_plan_matches_existing_geometry(
        ArraySpace(CategoricalSpace(("a", "b")), length=1)
    )
    assert_plan_matches_existing_geometry(ArraySpace(IntegerSpace(3, 3), length=1))


def test_plan_preserves_large_integer_distance_arithmetic() -> None:
    space = IntegerSpace(-(10**30), 10**30)
    plan = compile_builtin_geometry_plan(space)
    geometry = compile_structured_geometry(space)
    assert plan is not None
    assert geometry is not None

    left = -(10**30) + 1
    right = 10**30 - 1

    assert plan.distance_parts(left, right) == geometry.distance_parts(left, right)


def test_plan_supports_empty_batches() -> None:
    plan = compile_builtin_geometry_plan(IntegerSpace(-8, 8))
    assert plan is not None
    encoded = plan.encode(0)

    assert plan.encode_many(()) == ()
    assert plan.squared_distances_to_many(encoded, ()) == ()
    assert plan.pairwise_squared_distances(()) == ()


def test_plan_can_encode_candidate_validated_by_operation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = compile_builtin_geometry_plan(IntegerSpace(-8, 8))
    assert plan is not None

    def reject_revalidation(_space: IntegerSpace, _candidate: int) -> None:
        raise AssertionError("validated encoding must not repeat space validation")

    monkeypatch.setattr(IntegerSpace, "validate", reject_revalidation)

    encoded = plan.encode_validated(3)

    assert encoded.integer_values == (3,)
    with pytest.raises(AssertionError, match="must not repeat"):
        _ = plan.encode(3)


def test_plan_rejects_encodings_from_another_plan() -> None:
    space = ArraySpace(IntegerSpace(-8, 8), length=4)
    left_plan = compile_builtin_geometry_plan(space)
    right_plan = compile_builtin_geometry_plan(space)
    assert left_plan is not None
    assert right_plan is not None

    candidate = (0, 1, 2, 3)
    left_encoding = left_plan.encode(candidate)
    right_encoding = right_plan.encode(candidate)

    with pytest.raises(
        ValueError,
        match="encoded candidate belongs to a different geometry plan",
    ):
        _ = left_plan.squared_distance(left_encoding, right_encoding)
    with pytest.raises(
        ValueError,
        match="encoded candidate belongs to a different geometry plan",
    ):
        _ = left_plan.squared_distances_to_many(
            left_encoding,
            (right_encoding,),
        )
    with pytest.raises(
        ValueError,
        match="encoded candidate belongs to a different geometry plan",
    ):
        _ = left_plan.pairwise_squared_distances(
            (left_encoding, right_encoding),
        )


def test_plan_declines_space_subclasses_and_nested_custom_spaces() -> None:
    custom_space = CustomIntegerSpace(-2, 2)
    assert compile_builtin_geometry_plan(custom_space) is None
    assert (
        compile_builtin_geometry_plan(
            ArraySpace(custom_space, length=3),
        )
        is None
    )
    assert compile_builtin_geometry_plan(TupleSpace(custom_space)) is None
    assert compile_builtin_geometry_plan(RecordSpace(value=custom_space)) is None


def test_plan_keeps_validation_at_the_encoding_boundary() -> None:
    real_plan = compile_builtin_geometry_plan(RealSpace(-1.0, 1.0))
    array_plan = compile_builtin_geometry_plan(
        ArraySpace(IntegerSpace(-2, 2), length=2)
    )
    assert real_plan is not None
    assert array_plan is not None

    with pytest.raises(TypeError):
        _ = real_plan.encode(1)
    with pytest.raises(ValueError):
        _ = array_plan.encode((0, 1, 2))


def test_plan_is_safe_for_parallel_read_only_use() -> None:
    space = RecordSpace(
        scale=RealSpace(0.1, 10.0, scale="log"),
        mode=CategoricalSpace(("a", "b", "c")),
        count=IntegerSpace(-8, 8),
        order=PermutationSpace(5),
    )
    plan = compile_builtin_geometry_plan(space)
    assert plan is not None
    random_state = np.random.RandomState(7)
    encodings = plan.encode_many(
        tuple(space.sample(random_state) for _index in range(16))
    )
    index_pairs: Sequence[tuple[int, int]] = tuple(
        (left_index, right_index)
        for left_index in range(len(encodings))
        for right_index in range(left_index)
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel_distances = tuple(
            executor.map(
                lambda indices: plan.squared_distance(
                    encodings[indices[0]],
                    encodings[indices[1]],
                ),
                index_pairs,
            )
        )

    expected_distances = tuple(
        plan.squared_distance(encodings[left_index], encodings[right_index])
        for left_index, right_index in index_pairs
    )
    assert parallel_distances == expected_distances
