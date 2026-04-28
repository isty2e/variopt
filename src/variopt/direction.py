"""Optimization direction definitions."""

from enum import Enum


class OptimizationDirection(str, Enum):
    """Optimization direction for raw objective values.

    Notes
    -----
    ``variopt`` internally compares scalar outcomes under a canonical
    minimization convention. This enum captures the boundary-level direction
    and provides the normalization hook that converts raw values into canonical
    scores.
    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

    def normalize_objective_value(self, value: float) -> float:
        """Convert a raw objective value into canonical minimization form.

        Parameters
        ----------
        value : float
            Raw objective value produced by the user-facing objective or
            evaluation protocol.

        Returns
        -------
        float
            Canonical minimization score used for internal ordering.
        """
        if self is type(self).MINIMIZE:
            return value

        return -value
