"""Normalized result nouns for SciPy local-search integration."""

from dataclasses import dataclass

import numpy as np
from typing_extensions import Self

from ....kernel import KernelDiagnostics, KernelStatus
from .contracts import ScipyOptimizeResult


@dataclass(frozen=True, slots=True)
class ScipyMinimizeResult:
    """Canonical normalized result returned by ``scipy.optimize.minimize``.

    Parameters
    ----------
    coordinates : tuple[float, ...]
        Final coordinate vector reported by SciPy.
    function_value : float
        Final minimized function value reported by SciPy.
    evaluation_count : int
        Total number of objective evaluations consumed by the SciPy run.
    converged : bool
        Whether SciPy reported successful convergence.
    message : str | None, optional
        Optional backend message carried through for diagnostics.
    """

    coordinates: tuple[float, ...]
    function_value: float
    evaluation_count: int
    converged: bool
    message: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid normalized SciPy minimize results."""
        if len(self.coordinates) == 0:
            msg = "coordinates must not be empty"
            raise ValueError(msg)

        if not all(np.isfinite(coordinate) for coordinate in self.coordinates):
            msg = "coordinates must be finite"
            raise ValueError(msg)

        if not np.isfinite(self.function_value):
            msg = "function_value must be finite"
            raise ValueError(msg)

        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)

        if self.message == "":
            msg = "message must not be empty"
            raise ValueError(msg)

    @classmethod
    def from_optimize_result(cls, optimize_result: ScipyOptimizeResult) -> Self:
        """Normalize one raw SciPy optimize result.

        Parameters
        ----------
        optimize_result : ScipyOptimizeResult
            Raw result object returned by ``scipy.optimize.minimize``.

        Returns
        -------
        Self
            Canonical normalized minimize result.
        """
        coordinates = tuple(float(coordinate) for coordinate in optimize_result.x)
        function_value = float(optimize_result.fun)
        evaluation_count = int(optimize_result.nfev)
        converged = bool(optimize_result.success)
        message_object = optimize_result.message
        message = None if message_object is None else str(message_object)
        return cls(
            coordinates=coordinates,
            function_value=function_value,
            evaluation_count=evaluation_count,
            converged=converged,
            message=message,
        )

    def diagnostics(self, *, method: str) -> KernelDiagnostics:
        """Build kernel diagnostics from the normalized SciPy result.

        Parameters
        ----------
        method : str
            SciPy method name used for the local-search run.

        Returns
        -------
        KernelDiagnostics
            Diagnostics payload aligned with the normalized result.
        """
        status = KernelStatus.STOPPED
        if self.converged:
            status = KernelStatus.CONVERGED

        return KernelDiagnostics(
            backend="scipy.optimize.minimize",
            method=method,
            status=status,
            message=self.message,
        )
