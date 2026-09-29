"""Exact arithmetic and lifecycle contracts for encoded distance batches."""

import pickle
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from sys import float_info
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
from variopt.spaces.structured import StructuredSearchSpace
from variopt.spaces.types import SpaceCandidateValue

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)


def assert_batch_geometry_parity(
    space: StructuredSearchSpace[BoundaryT, CandidateT],
) -> None:
    """Compare batched results with both scalar implementations exactly."""
    plan = compile_builtin_geometry_plan(space)
    geometry = compile_structured_geometry(space)
    assert plan is not None
    assert geometry is not None
    random_state = np.random.RandomState(29)
    candidates = tuple(space.sample(random_state) for _ in range(128))
    encodings = plan.encode_many(candidates)

    expected = tuple(
        geometry.distance_parts(candidates[0], candidate).overlap_squared_distance
        for candidate in candidates
    )
    assert plan.squared_distances_to_many(encodings[0], encodings) == expected
    assert expected == tuple(
        plan.squared_distance(encodings[0], encoding) for encoding in encodings
    )
    assert (
        plan.squared_distances_to_many(encodings[0], encodings[::-1]) == expected[::-1]
    )
    assert (
        plan.squared_distances_to_many(encodings[0], (encodings[1],) * 128)
        == (expected[1],) * 128
    )

    pairwise = plan.pairwise_squared_distances(encodings)
    for index, encoding in enumerate(encodings):
        assert pairwise[index] == tuple(
            plan.squared_distance(encoding, other) for other in encodings
        )


def test_batch_real_and_log_geometry_matches_scalar_arithmetic() -> None:
    assert_batch_geometry_parity(ArraySpace(RealSpace(-5.0, 5.0), length=32))
    assert_batch_geometry_parity(
        ArraySpace(RealSpace(1e-100, 1e100, scale="log"), length=16)
    )


def test_batch_integer_and_discrete_geometry_matches_scalar_arithmetic() -> None:
    assert_batch_geometry_parity(ArraySpace(IntegerSpace(-8, 8), length=32))
    assert_batch_geometry_parity(
        ArraySpace(IntegerSpace(1, 1000, scale="log"), length=16)
    )
    assert_batch_geometry_parity(ArraySpace(IntegerSpace(0, 1), length=32))
    assert_batch_geometry_parity(
        ArraySpace(CategoricalSpace((b"a", b"b", b"c")), length=32)
    )
    assert_batch_geometry_parity(PermutationSpace(32))


def test_batch_nested_mixed_geometry_preserves_child_subtotals() -> None:
    assert_batch_geometry_parity(
        ArraySpace(
            RecordSpace(
                scale=RealSpace(0.1, 10.0, scale="log"),
                values=TupleSpace(
                    IntegerSpace(-8, 8),
                    CategoricalSpace((True, False)),
                    RealSpace(-3.0, 3.0),
                ),
                order=PermutationSpace(7),
                fixed=IntegerSpace(5, 5),
            ),
            length=8,
        )
    )


@pytest.mark.parametrize("reference_count", [0, 1, 31, 32, 33, 255, 256, 257, 513])
def test_batch_reference_and_chunk_boundaries(reference_count: int) -> None:
    space = ArraySpace(RealSpace(-5.0, 5.0), length=64)
    plan = compile_builtin_geometry_plan(space)
    assert plan is not None
    random_state = np.random.RandomState(31)
    candidate = plan.encode(space.sample(random_state))
    references = plan.encode_many(
        tuple(space.sample(random_state) for _ in range(reference_count))
    )

    assert plan.squared_distances_to_many(candidate, references) == tuple(
        plan.squared_distance(candidate, reference) for reference in references
    )


@pytest.mark.parametrize("reference_count", [1, 31, 32, 128, 300])
def test_batch_accepts_non_sliceable_sequences(reference_count: int) -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(IntegerSpace(-8, 8), length=4))
    assert plan is not None
    candidate = plan.encode((0, 1, 2, 3))
    references = deque(
        plan.encode((index % 17 - 8,) * 4) for index in range(reference_count)
    )
    before = tuple(references)

    assert plan.squared_distances_to_many(candidate, references) == tuple(
        plan.squared_distance(candidate, reference) for reference in references
    )
    assert tuple(references) == before


def test_pairwise_batch_accepts_non_sliceable_sequences() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(IntegerSpace(-8, 8), length=4))
    assert plan is not None
    candidates = deque(plan.encode((index % 17 - 8,) * 4) for index in range(128))
    expected = plan.pairwise_squared_distances(tuple(candidates))

    assert plan.pairwise_squared_distances(candidates) == expected


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (-(2**63), -1),
        (0, 2**63 - 1),
        (2**53, 2**53 + 100),
        (-(2**63), 2**63 - 1),
        (-(2**62), 2**62),
        (2**60, 2**60 + 100),
    ],
)
def test_batch_integer_subtraction_precedes_float_conversion(
    low: int, high: int
) -> None:
    if low < np.iinfo("l").min or high > np.iinfo("l").max:
        pytest.skip("requires a 64-bit C-long sampler")
    plan = compile_builtin_geometry_plan(ArraySpace(IntegerSpace(low, high), length=3))
    assert plan is not None
    left = plan.encode((low, low + 1, high))
    right = plan.encode((high, low + 2, low))
    expected = plan.squared_distance(left, right)

    assert plan.squared_distances_to_many(left, (right,) * 128) == (expected,) * 128
    assert plan.squared_distances_to_many(right, (left,) * 128) == (expected,) * 128


@pytest.mark.skipif(np.iinfo("l").bits < 64, reason="requires a 64-bit C-long sampler")
def test_batch_mixed_plan_keeps_integer_differences_beyond_float_precision() -> None:
    plan = compile_builtin_geometry_plan(
        TupleSpace(RealSpace(-2.0, 2.0), IntegerSpace(2**60, 2**60 + 2))
    )
    assert plan is not None
    left = plan.encode((1.0, 2**60 + 1))
    right = plan.encode((-1.0, 2**60 + 2))

    assert plan.squared_distance(left, right) == 0.5
    assert plan.squared_distances_to_many(left, (right,) * 128) == (0.5,) * 128


def test_batch_very_wide_candidate_keeps_coordinate_order() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(RealSpace(0.0, 1.0), length=8193))
    assert plan is not None
    left = plan.encode((1.0,) + (1e-8,) * 8192)
    right = plan.encode((0.0,) * 8193)

    assert plan.squared_distances_to_many(left, (right,) * 33) == (1.0,) * 33


def test_batch_does_not_replace_sequential_sum_with_pairwise_reduction() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(RealSpace(0.0, 1.0), length=17))
    assert plan is not None
    left = plan.encode((1.0,) + (1e-8,) * 16)
    right = plan.encode((0.0,) * 17)

    assert plan.squared_distance(left, right) == 1.0
    assert plan.squared_distances_to_many(left, (right,) * 128) == (1.0,) * 128


def test_batch_does_not_flatten_nested_sum_groups() -> None:
    plan = compile_builtin_geometry_plan(
        TupleSpace(
            RealSpace(0.0, 1.0), TupleSpace(RealSpace(0.0, 1.0), RealSpace(0.0, 1.0))
        )
    )
    assert plan is not None
    left = plan.encode((1.0, (1e-8, 1e-8)))
    right = plan.encode((0.0, (0.0, 0.0)))
    expected = float.fromhex("0x1.0000000000001p+0")

    assert plan.squared_distance(left, right) == expected
    assert plan.squared_distances_to_many(left, (right,) * 128) == (expected,) * 128


def test_batch_preserves_scalar_underflow_with_numpy_errors_enabled() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(RealSpace(0.0, 1.0), length=4))
    assert plan is not None
    left = plan.encode((1e-200, 1e-160, 0.0, -0.0))
    right = plan.encode((0.0, 0.0, -0.0, 0.0))
    expected = plan.squared_distance(left, right)

    with np.errstate(all="raise"):
        previous = np.geterr()
        assert plan.squared_distances_to_many(left, (right,) * 128) == (expected,) * 128
        assert np.geterr() == previous


def test_batch_preserves_scalar_behavior_for_largest_real_span() -> None:
    plan = compile_builtin_geometry_plan(
        ArraySpace(RealSpace(0.0, float_info.max), length=4)
    )
    assert plan is not None
    left = plan.encode((0.0,) * 4)
    right = plan.encode((float_info.max,) * 4)
    assert plan.squared_distance(left, right) == 4.0

    with np.errstate(all="raise"):
        assert plan.squared_distances_to_many(left, (right,) * 128) == (4.0,) * 128


@pytest.mark.skipif(np.iinfo("l").bits < 64, reason="requires a 64-bit C-long sampler")
def test_batch_preserves_small_positive_log_span() -> None:
    plan = compile_builtin_geometry_plan(IntegerSpace(2**53, 2**53 + 1024, scale="log"))
    assert plan is not None
    left = plan.encode(2**53)
    right = plan.encode(2**53 + 1024)

    assert plan.squared_distance(left, right) == 1.0
    assert plan.squared_distances_to_many(left, (right,) * 128) == (1.0,) * 128


def test_batch_constant_geometry_keeps_zero_subtotals() -> None:
    assert_batch_geometry_parity(
        RecordSpace(
            real=RealSpace(2.0, 2.0),
            integer=IntegerSpace(4, 4),
            category=CategoricalSpace(("fixed",)),
        )
    )
    assert_batch_geometry_parity(ArraySpace(RealSpace(1.0, 1.0), length=4))


def test_batch_rejects_foreign_plan_in_later_chunk() -> None:
    space = ArraySpace(IntegerSpace(-2, 2), length=4)
    plan = compile_builtin_geometry_plan(space)
    foreign_plan = compile_builtin_geometry_plan(space)
    assert plan is not None
    assert foreign_plan is not None
    candidate = plan.encode((0, 0, 0, 0))
    foreign = foreign_plan.encode((0, 0, 0, 0))

    with pytest.raises(ValueError, match="different geometry plan"):
        plan.squared_distances_to_many(candidate, (candidate,) * 300 + (foreign,))


def test_batch_plan_supports_concurrent_calls_without_mutating_encodings() -> None:
    space = ArraySpace(IntegerSpace(-8, 8), length=32)
    plan = compile_builtin_geometry_plan(space)
    assert plan is not None
    random_state = np.random.RandomState(37)
    encodings = plan.encode_many(tuple(space.sample(random_state) for _ in range(65)))
    before = pickle.dumps((plan, encodings))
    expected = tuple(
        tuple(plan.squared_distance(left, right) for right in encodings)
        for left in encodings[:8]
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = tuple(
            executor.map(
                lambda left: plan.squared_distances_to_many(left, encodings),
                encodings[:8],
            )
        )

    assert actual == expected
    assert pickle.dumps((plan, encodings)) == before


def test_pickled_plan_and_encodings_keep_batch_identity_alignment() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(IntegerSpace(-8, 8), length=4))
    assert plan is not None
    left = plan.encode((0, 1, 2, 3))
    references = (plan.encode((1, 2, 3, 4)),) * 128
    expected = plan.squared_distances_to_many(left, references)

    restored_plan, restored_left, restored_references = pickle.loads(
        pickle.dumps((plan, left, references))
    )

    assert (
        restored_plan.squared_distances_to_many(restored_left, restored_references)
        == expected
    )


def test_parallel_batches_preserve_each_callers_numpy_error_policy() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(RealSpace(0.0, 1.0), length=4))
    assert plan is not None
    left = plan.encode((1e-200, 1e-160, 0.0, -0.0))
    right = plan.encode((0.0, 0.0, -0.0, 0.0))
    expected = (plan.squared_distance(left, right),) * 128
    outer_policy = np.geterr()

    def query(index: int) -> tuple[float, ...]:
        with np.errstate(all="raise" if index % 2 else "warn"):
            previous = np.geterr()
            result = plan.squared_distances_to_many(left, (right,) * 128)
            assert np.geterr() == previous
            return result

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(executor.map(query, range(32)))

    assert results == (expected,) * 32
    assert np.geterr() == outer_policy
