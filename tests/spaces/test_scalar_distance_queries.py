"""Exact arithmetic and consumption contracts for scalar geometry queries."""

import math
import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

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
    ("space", "candidate", "references"),
    [
        (IntegerSpace(-(2**100), 2**100), 2**99, (2**99 + 1, -(2**100), 0)),
        (IntegerSpace(1, 2**100, scale="log"), 2**99, (1, 2**99 + 1, 2**100)),
        (IntegerSpace(3, 3), 3, (3, 3)),
        (IntegerSpace(3, 3, scale="log"), 3, (3, 3)),
    ],
)
def test_integer_query_matches_pairwise_hex(
    space: IntegerSpace, candidate: int, references: tuple[int, ...]
) -> None:
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


def test_large_integer_query_does_not_round_coordinates_before_subtraction() -> None:
    metric = StructuredSpaceDiversityMetric(IntegerSpace(-(2**100), 2**100))

    assert structured_distances_to_validated_candidates(
        metric, 2**99, (2**99 + 1,)
    ) == (2.0**-101,)


def test_degenerate_huge_integer_query_needs_no_float_conversion() -> None:
    metric = StructuredSpaceDiversityMetric(IntegerSpace(10**400, 10**400))

    assert structured_distances_to_validated_candidates(
        metric, 10**400, (10**400, 10**400)
    ) == (0.0, 0.0)


def test_numeric_failure_does_not_consume_later_references() -> None:
    metric = StructuredSpaceDiversityMetric(RealSpace(-1e308, 1e308))
    references = iter((0.0, 1e308, -1e308))

    with pytest.raises(ValueError, match="finite"):
        structured_distances_to_validated_candidates(metric, -1e308, references)

    assert tuple(references) == (-1e308,)


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_empty_query_does_not_evaluate_unrepresentable_integer_span(
    scale: Literal["linear", "log"],
) -> None:
    space = IntegerSpace(1, 10**400, scale=scale)
    geometry = IntegerSpaceGeometry(space)

    assert tuple(geometry.iter_squared_distances_for_validated_candidates(1, ())) == ()
    references: Iterator[SpaceCandidateValue] = iter((1, 2))
    with pytest.raises(OverflowError):
        tuple(geometry.iter_squared_distances_for_validated_candidates(1, references))
    assert tuple(references) == (2,)


def test_collapsed_log_span_preserves_error_and_consumption() -> None:
    metric = StructuredSpaceDiversityMetric(IntegerSpace(2**60, 2**60 + 1, scale="log"))
    references = iter((2**60, 2**60 + 1))

    with pytest.raises(ZeroDivisionError):
        structured_distances_to_validated_candidates(metric, 2**60, references)
    assert tuple(references) == (2**60 + 1,)


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
