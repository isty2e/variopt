"""Internal helpers for validating distance-like quantities."""

import numpy as np


def require_valid_distance(distance: float) -> float:
    """Validate and normalize a distance-like quantity.

    Parameters
    ----------
    distance : float
        Distance-like quantity to validate.

    Returns
    -------
    float
        Canonical non-negative finite distance.

    Raises
    ------
    ValueError
        If ``distance`` is non-finite or negative.
    """
    normalized_distance = float(distance)
    if not np.isfinite(normalized_distance):
        msg = "distance must be finite"
        raise ValueError(msg)

    if normalized_distance < 0.0:
        msg = "distance must be non-negative"
        raise ValueError(msg)

    return normalized_distance
