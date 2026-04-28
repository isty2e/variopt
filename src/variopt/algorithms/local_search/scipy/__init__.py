"""SciPy-backed kernels for continuous local search."""

from .contracts import ScipyMinimizeMethod, ScipyOptimizeResult
from .kernel import ScipyMinimizeKernel
from .results import ScipyMinimizeResult
from .runner import run_scipy_minimize

__all__ = [
    "ScipyMinimizeKernel",
    "ScipyMinimizeMethod",
    "ScipyMinimizeResult",
    "ScipyOptimizeResult",
    "run_scipy_minimize",
]
