"""Typed numeric helpers for pytest-based test assertions."""

import math
from typing import SupportsFloat


def approx_equal(
    actual: SupportsFloat,
    expected: SupportsFloat,
    *,
    rel: float = 1e-6,
    abs: float = 1e-12,
) -> bool:
    """Return whether two numeric values are approximately equal."""

    return math.isclose(
        float(actual),
        float(expected),
        rel_tol=rel,
        abs_tol=abs,
    )
