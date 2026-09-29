"""Admission and numerical contracts for built-in numeric bounds."""

from dataclasses import replace
from math import isfinite, nextafter, ulp
from sys import float_info
from typing import Literal

import numpy as np
import pytest

from variopt import ArraySpace, IntegerSpace, RealSpace, TupleSpace
from variopt.diversity import StructuredSpaceDiversityMetric
from variopt.randomness import RandomStateSnapshot
from variopt.spaces.geometry.compile import compile_structured_geometry
from variopt.spaces.geometry.plan import compile_builtin_geometry_plan

INTEGER_LOW = int(np.iinfo("l").min)
INTEGER_HIGH = int(np.iinfo("l").max)


@pytest.mark.parametrize(
    ("low", "high", "scale"),
    [
        (-1e308, 1e308, "linear"),
        (-float_info.max, float_info.max, "linear"),
        (1e100, nextafter(1e100, float("inf")), "log"),
        (1e-100, nextafter(1e-100, float("inf")), "log"),
    ],
)
def test_real_bounds_reject_unusable_coordinate_spans(
    low: float, high: float, scale: Literal["linear", "log"]
) -> None:
    with pytest.raises(ValueError, match="RealSpace.*coordinate span"):
        RealSpace(low, high, scale=scale)


@pytest.mark.parametrize("scale", ["linear", "log"])
@pytest.mark.parametrize(
    ("low", "high"),
    [(1, INTEGER_HIGH + 1), (1, 10**400), (10**400, 10**400)],
)
def test_integer_bounds_reject_values_outside_sampler_dtype(
    low: int, high: int, scale: Literal["linear", "log"]
) -> None:
    with pytest.raises(ValueError, match="IntegerSpace.*sampling range"):
        IntegerSpace(low, high, scale=scale)


def test_integer_bounds_reject_values_below_sampler_dtype() -> None:
    with pytest.raises(ValueError, match="IntegerSpace.*sampling range"):
        IntegerSpace(INTEGER_LOW - 1, 0)


@pytest.mark.skipif(INTEGER_HIGH < 2**60, reason="requires a 64-bit C-long sampler")
@pytest.mark.parametrize("low", [2**53, 2**60])
def test_integer_log_bounds_reject_collapsed_coordinates(low: int) -> None:
    with pytest.raises(ValueError, match="IntegerSpace.*coordinate span"):
        IntegerSpace(low, low + 1, scale="log")


@pytest.mark.parametrize(
    ("low", "high", "scale"),
    [
        (0.0, float_info.max, "linear"),
        (-float_info.max, 0.0, "linear"),
        (0.0, ulp(0.0), "linear"),
        (1.0, nextafter(1.0, 2.0), "linear"),
        (1.0, nextafter(1.0, 2.0), "log"),
        (ulp(0.0), float_info.max, "log"),
    ],
)
def test_admitted_real_bounds_support_sampling_projection_and_geometry(
    low: float, high: float, scale: Literal["linear", "log"]
) -> None:
    space = RealSpace(low, high, scale=scale)
    geometry = compile_structured_geometry(space)
    plan = compile_builtin_geometry_plan(space)
    metric = StructuredSpaceDiversityMetric(space)
    assert geometry is not None
    assert plan is not None

    assert geometry.distance_parts(low, high).overlap_squared_distance == 1.0
    assert metric.distance(low, high) == metric.distance(high, low) == 1.0
    assert metric.distance(low, low) == metric.distance(high, high) == 0.0
    left, right = plan.encode(low), plan.encode(high)
    assert plan.squared_distances_to_many(left, (right,) * 128) == (1.0,) * 128
    array_plan = compile_builtin_geometry_plan(ArraySpace(space, length=4))
    assert array_plan is not None
    array_left = array_plan.encode((low,) * 4)
    array_right = array_plan.encode((high,) * 4)
    packed = array_plan.pack_references((array_right,) * 128)
    assert packed is not None
    assert array_plan.squared_distances_to_packed(array_left, packed) == (4.0,) * 128

    random_state = np.random.RandomState(19)
    for value in (low, high, *(space.sample(random_state) for _ in range(32))):
        space.validate(value)
        coordinate = space.to_coordinate(value)
        assert isfinite(coordinate)
        space.validate(space.project_coordinate(coordinate))


@pytest.mark.parametrize(
    ("low", "high", "scale"),
    [
        (INTEGER_LOW, INTEGER_HIGH, "linear"),
        (INTEGER_LOW, INTEGER_LOW + 1, "linear"),
        (INTEGER_HIGH - 1, INTEGER_HIGH, "linear"),
        (1, INTEGER_HIGH, "log"),
    ],
)
def test_admitted_integer_bounds_support_sampling_projection_and_geometry(
    low: int, high: int, scale: Literal["linear", "log"]
) -> None:
    space = IntegerSpace(low, high, scale=scale)
    geometry = compile_structured_geometry(space)
    plan = compile_builtin_geometry_plan(space)
    metric = StructuredSpaceDiversityMetric(space)
    assert geometry is not None
    assert plan is not None

    assert geometry.distance_parts(low, high).overlap_squared_distance == 1.0
    assert metric.distance(low, high) == metric.distance(high, low) == 1.0
    assert metric.distance(low, low) == metric.distance(high, high) == 0.0
    left, right = plan.encode(low), plan.encode(high)
    assert plan.squared_distances_to_many(left, (right,) * 128) == (1.0,) * 128

    random_state = np.random.RandomState(19)
    for value in (low, high, *(space.sample(random_state) for _ in range(32))):
        space.validate(value)
        coordinate = space.to_coordinate(value)
        assert isfinite(coordinate)
        space.validate(space.project_coordinate(coordinate))


def test_replacement_cannot_bypass_numeric_bound_admission() -> None:
    with pytest.raises(ValueError, match="coordinate span"):
        replace(RealSpace(0.0, 1e308), low=-1e308)
    with pytest.raises(ValueError, match="sampling range"):
        replace(IntegerSpace(0, 1), high=10**400)


def test_mixed_composite_keeps_extreme_admitted_leaf_geometry() -> None:
    space = TupleSpace(
        RealSpace(0.0, float_info.max), IntegerSpace(INTEGER_LOW, INTEGER_HIGH)
    )
    metric = StructuredSpaceDiversityMetric(space)
    left = (0.0, INTEGER_LOW)
    right = (float_info.max, INTEGER_HIGH)

    assert metric.distance(left, right) == metric.distance(right, left) == 1.0
    assert metric.distance(left, left) == metric.distance(right, right) == 0.0


@pytest.mark.parametrize(
    ("value", "scale"),
    [
        (-float_info.max, "linear"),
        (-0.0, "linear"),
        (float_info.max, "linear"),
        (ulp(0.0), "log"),
        (float_info.max, "log"),
    ],
)
def test_constant_real_extremes_do_not_consume_rng(
    value: float, scale: Literal["linear", "log"]
) -> None:
    space = RealSpace(value, value, scale=scale)
    random_state = np.random.RandomState(31)
    before = RandomStateSnapshot.from_random_state(random_state)

    assert space.sample(random_state).hex() == value.hex()
    assert RandomStateSnapshot.from_random_state(random_state) == before
    assert StructuredSpaceDiversityMetric(space).distance(value, value) == 0.0
    assert space.project_coordinate(space.to_coordinate(value)) == value


@pytest.mark.parametrize(
    ("value", "scale"),
    [(INTEGER_LOW, "linear"), (INTEGER_HIGH, "linear"), (INTEGER_HIGH, "log")],
)
def test_constant_integer_limits_do_not_consume_rng(
    value: int, scale: Literal["linear", "log"]
) -> None:
    space = IntegerSpace(value, value, scale=scale)
    random_state = np.random.RandomState(31)
    before = RandomStateSnapshot.from_random_state(random_state)

    assert space.sample(random_state) == value
    assert RandomStateSnapshot.from_random_state(random_state) == before
    assert StructuredSpaceDiversityMetric(space).distance(value, value) == 0.0
    assert space.project_coordinate(space.to_coordinate(value)) == value


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_real_sampling_preserves_randomstate_calls(
    scale: Literal["linear", "log"],
) -> None:
    space = RealSpace(0.01, 100.0, scale=scale)
    actual_rng = np.random.RandomState(31)
    expected_rng = np.random.RandomState(31)
    low, high = space.coordinate_bounds()
    expected = tuple(
        space.project_coordinate(float(expected_rng.uniform(low, high)))
        for _ in range(64)
    )

    assert tuple(space.sample(actual_rng) for _ in range(64)) == expected
    assert RandomStateSnapshot.from_random_state(actual_rng) == (
        RandomStateSnapshot.from_random_state(expected_rng)
    )


@pytest.mark.parametrize(
    ("low", "high"),
    [(-10, 10), (INTEGER_LOW, INTEGER_HIGH), (INTEGER_HIGH - 1, INTEGER_HIGH)],
)
def test_integer_sampling_preserves_inclusive_default_dtype_rng_calls(
    low: int,
    high: int,
) -> None:
    space = IntegerSpace(low, high)
    actual_rng = np.random.RandomState(31)
    expected_rng = np.random.RandomState(31)
    expected = tuple(int(expected_rng.randint(low, high + 1)) for _ in range(64))

    assert tuple(space.sample(actual_rng) for _ in range(64)) == expected
    assert RandomStateSnapshot.from_random_state(actual_rng) == (
        RandomStateSnapshot.from_random_state(expected_rng)
    )


def test_log_integer_sampling_preserves_randomstate_calls() -> None:
    space = IntegerSpace(1, INTEGER_HIGH, scale="log")
    actual_rng = np.random.RandomState(31)
    expected_rng = np.random.RandomState(31)
    low, high = space.coordinate_bounds()
    expected = tuple(
        space.project_coordinate(float(expected_rng.uniform(low, high)))
        for _ in range(64)
    )

    assert tuple(space.sample(actual_rng) for _ in range(64)) == expected
    assert RandomStateSnapshot.from_random_state(actual_rng) == (
        RandomStateSnapshot.from_random_state(expected_rng)
    )
