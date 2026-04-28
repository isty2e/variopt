"""Tests for SciPy local-search package exports."""


from variopt.algorithms.local_search import ScipyMinimizeKernel
from variopt.algorithms.local_search.scipy import (
    ScipyMinimizeKernel as PackageScipyMinimizeKernel,
)
from variopt.algorithms.local_search.scipy import (
    ScipyMinimizeMethod,
    ScipyMinimizeResult,
    ScipyOptimizeResult,
    run_scipy_minimize,
)


class ScipyLocalSearchExportsTests:
    """Regression tests for the SciPy local-search package facade."""

    def test_package_facade_exports_kernel_surface(self) -> None:
        assert ScipyMinimizeKernel is PackageScipyMinimizeKernel
        assert ScipyMinimizeMethod is not None
        assert ScipyMinimizeResult is not None
        assert ScipyOptimizeResult is not None
        assert callable(run_scipy_minimize)
