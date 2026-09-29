"""Exact arithmetic and consumption contracts for scalar geometry queries."""

import math
import pickle
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest
from typing_extensions import override

from variopt import CategoricalSpace, IntegerSpace, RealSpace
from variopt.diversity import StructuredSpaceDiversityMetric
from variopt.diversity.space_metric import structured_distances_to_validated_candidates
from variopt.spaces.geometry import (
    CompiledStructuredGeometryProvider,
    StructuredSpaceGeometry,
)
from variopt.spaces.geometry.scalar import (
    IntegerSpaceGeometry,
    RealSpaceGeometry,
)
from variopt.spaces.types import SpaceCandidateValue, SpaceScalarValue


@pytest.mark.parametrize(
    ("space", "candidate", "references"),
    [
        (RealSpace(-1.0, 1.0), -0.0, (0.0, -1.0, 1.0, -0.0)),
        (RealSpace(0.0, 1.0), 0.0, (math.ulp(0.0), 1e-160, 1e-150, 1.0)),
        (RealSpace(0.0, math.ulp(0.0)), 0.0, (0.0, math.ulp(0.0))),
        (RealSpace(-1e100, 1e100), 1.0, (1.0, -1e100, 1e100)),
        (RealSpace(1e-200, 1e200, scale="log"), 1e-50, (1e-200, 1.0, 1e200)),
        (RealSpace(1.0, math.nextafter(1.0, 2.0), scale="log"), 1.0, (1.0,)),
        (RealSpace(7.0, 7.0), 7.0, (7.0, 7.0)),
        (RealSpace(7.0, 7.0, scale="log"), 7.0, (7.0, 7.0)),
    ],
)
def test_real_query_matches_pairwise_hex(
    space: RealSpace, candidate: float, references: tuple[float, ...]
) -> None:
    metric = StructuredSpaceDiversityMetric(space)
    expected = tuple(metric.distance(candidate, value).hex() for value in references)

    actual = structured_distances_to_validated_candidates(
        metric, candidate, iter(references)
    )

    assert tuple(value.hex() for value in actual) == expected


@pytest.mark.parametrize(
    ("low", "high", "scale", "candidate", "references"),
    [
        (-(2**62), 2**62, "linear", 2**61, (2**61 + 1, -(2**62), 0)),
        (1, 2**62, "log", 2**61, (1, 2**61 + 1, 2**62)),
        (3, 3, "linear", 3, (3, 3)),
        (3, 3, "log", 3, (3, 3)),
    ],
)
def test_integer_query_matches_pairwise_hex(
    low: int,
    high: int,
    scale: Literal["linear", "log"],
    candidate: int,
    references: tuple[int, ...],
) -> None:
    if high > np.iinfo("l").max:
        pytest.skip("requires a 64-bit C-long sampler")
    space = IntegerSpace(low, high, scale=scale)
    metric = StructuredSpaceDiversityMetric(space)
    expected = tuple(metric.distance(candidate, value).hex() for value in references)

    actual = structured_distances_to_validated_candidates(
        metric, candidate, iter(references)
    )

    assert tuple(value.hex() for value in actual) == expected


@pytest.mark.parametrize(
    "choices", [(False, True), (1, 2), (1.0, 2.0), (b"\x00", b"\xff"), ("", "a")]
)
def test_categorical_query_preserves_order_and_duplicates(
    choices: tuple[SpaceScalarValue, ...],
) -> None:
    metric = StructuredSpaceDiversityMetric(CategoricalSpace(choices))
    references = (choices[1], choices[0], choices[1], choices[0])

    assert structured_distances_to_validated_candidates(
        metric, choices[0], iter(references)
    ) == (1.0, 0.0, 1.0, 0.0)


def test_query_preserves_squared_underflow_instead_of_absolute_distance() -> None:
    metric = StructuredSpaceDiversityMetric(RealSpace(0.0, 1.0))

    assert structured_distances_to_validated_candidates(metric, 0.0, (1e-200,)) == (
        0.0,
    )
    assert metric.distance(0.0, 1e-200) == 0.0


@pytest.mark.skipif(np.iinfo("l").bits < 64, reason="requires a 64-bit C-long sampler")
def test_large_integer_query_does_not_round_coordinates_before_subtraction() -> None:
    metric = StructuredSpaceDiversityMetric(IntegerSpace(-(2**62), 2**62))

    assert structured_distances_to_validated_candidates(
        metric, 2**61, (2**61 + 1,)
    ) == (2.0**-63,)


def test_degenerate_large_integer_query_has_zero_distance() -> None:
    value = int(np.iinfo("l").max)
    metric = StructuredSpaceDiversityMetric(IntegerSpace(value, value))

    assert structured_distances_to_validated_candidates(
        metric, value, (value, value)
    ) == (0.0, 0.0)


def test_numeric_failure_does_not_consume_later_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonfinite_geometry(
        self: RealSpaceGeometry,
        candidate: SpaceCandidateValue,
        references: Iterable[SpaceCandidateValue],
    ) -> Iterator[float]:
        for reference in references:
            yield math.nan if reference == 1.0 else 0.0

    monkeypatch.setattr(
        RealSpaceGeometry,
        "iter_squared_distances_for_validated_candidates",
        nonfinite_geometry,
    )
    metric = StructuredSpaceDiversityMetric(RealSpace(0.0, 1.0))
    references = iter((0.0, 1.0, 0.5))

    with pytest.raises(ValueError, match="finite"):
        structured_distances_to_validated_candidates(metric, 0.0, references)

    assert tuple(references) == (0.5,)


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_empty_query_accepts_wide_integer_span(
    scale: Literal["linear", "log"],
) -> None:
    space = IntegerSpace(1, int(np.iinfo("l").max), scale=scale)
    geometry = IntegerSpaceGeometry(space)

    assert tuple(geometry.iter_squared_distances_for_validated_candidates(1, ())) == ()


def test_iterator_failure_propagates_without_restarting_input() -> None:
    visited: list[float] = []

    def references() -> Iterator[float]:
        visited.append(0.0)
        yield 0.0
        raise RuntimeError("reference stream failed")

    metric = StructuredSpaceDiversityMetric(RealSpace(0.0, 1.0))
    with pytest.raises(RuntimeError, match="reference stream failed"):
        structured_distances_to_validated_candidates(metric, 0.0, references())
    assert visited == [0.0]


def test_log_query_prepares_bounds_and_left_coordinate_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = StructuredSpaceDiversityMetric(RealSpace(1.0, 100.0, scale="log"))
    calls: list[float] = []

    def counted_log(value: float) -> float:
        calls.append(value)
        return math.log(value)

    monkeypatch.setattr("variopt.spaces.geometry.scalar.log", counted_log)
    values = structured_distances_to_validated_candidates(
        metric, 10.0, (1.0, 10.0, 100.0)
    )

    assert values == (0.5, 0.0, 0.5)
    assert calls == [100.0, 1.0, 10.0, 1.0, 10.0, 100.0]


@dataclass(frozen=True, slots=True)
class CustomRealGeometry(RealSpaceGeometry):
    @override
    def distance_part_values_for_validated_candidates(
        self, left: SpaceCandidateValue, right: SpaceCandidateValue
    ) -> tuple[float, int, int]:
        return (0.25, 1, 0)


@dataclass(frozen=True, slots=True)
class CustomGeometryRealSpace(RealSpace, CompiledStructuredGeometryProvider):
    @override
    def compile_structured_geometry(self) -> StructuredSpaceGeometry:
        return CustomRealGeometry(self)


def test_geometry_subclass_override_is_not_bypassed() -> None:
    metric = StructuredSpaceDiversityMetric(CustomGeometryRealSpace(0.0, 1.0))

    assert structured_distances_to_validated_candidates(metric, 0.0, (0.0, 1.0)) == (
        0.5,
        0.5,
    )


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_metric_pickle_rebuild_preserves_scalar_queries(
    scale: Literal["linear", "log"],
) -> None:
    metric = StructuredSpaceDiversityMetric(RealSpace(1.0, 100.0, scale=scale))
    restored = pickle.loads(pickle.dumps(metric))
    references = (1.0, 5.0, 20.0, 100.0)

    assert structured_distances_to_validated_candidates(
        restored, 10.0, references
    ) == structured_distances_to_validated_candidates(metric, 10.0, references)


@pytest.mark.parametrize(
    "geometry",
    [RealSpaceGeometry(RealSpace(0.0, 1.0)), IntegerSpaceGeometry(IntegerSpace(0, 1))],
)
def test_scalar_query_wrong_type_fails_without_consuming_tail(
    geometry: RealSpaceGeometry | IntegerSpaceGeometry,
) -> None:
    references: Iterator[SpaceCandidateValue] = iter(("not numeric", "tail"))
    candidate = 0.0 if isinstance(geometry, RealSpaceGeometry) else 0

    with pytest.raises(TypeError):
        tuple(
            geometry.iter_squared_distances_for_validated_candidates(
                candidate, references
            )
        )
    assert tuple(references) == ("tail",)
