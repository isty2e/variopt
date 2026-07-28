"""Tests for internal distance normalization."""

from math import copysign

import numpy as np
import pytest

from variopt.distance import require_valid_distance


@pytest.mark.parametrize(
    ("distance", "expected"),
    (
        (0.0, 0.0),
        (1, 1.0),
        (np.float64(1.5), 1.5),
    ),
)
def test_require_valid_distance_normalizes_finite_non_negative_values(
    distance: float,
    expected: float,
) -> None:
    assert require_valid_distance(distance) == expected


def test_require_valid_distance_preserves_negative_zero() -> None:
    normalized_distance = require_valid_distance(-0.0)

    assert copysign(1.0, normalized_distance) == -1.0


@pytest.mark.parametrize("distance", (float("nan"), float("inf"), float("-inf")))
def test_require_valid_distance_rejects_non_finite_values(distance: float) -> None:
    with pytest.raises(ValueError, match="distance must be finite"):
        _ = require_valid_distance(distance)


def test_require_valid_distance_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="distance must be non-negative"):
        _ = require_valid_distance(-1.0)
