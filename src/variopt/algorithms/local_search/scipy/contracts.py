"""Foreign boundary contracts for SciPy local-search integration."""

from collections.abc import Sequence
from typing import Literal, Protocol, TypeAlias

ScipyMinimizeMethod: TypeAlias = Literal["L-BFGS-B", "Powell"]


class ScipyOptimizeResult(Protocol):
    """Typed view of the subset of SciPy optimize results used by variopt.

    Notes
    -----
    The local-search wrapper only depends on a narrow result surface, so this
    protocol keeps the SciPy boundary explicit and easy to mock.
    """

    x: Sequence[float]
    fun: float
    nfev: int
    success: bool
    message: str | None
