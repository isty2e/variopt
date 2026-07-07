"""Facade regressions for the structured local-search package."""

from variopt.algorithms.local_search import (
    StructuredHillClimbKernel as LocalSearchHillClimbKernel,
)
from variopt.algorithms.local_search.structured import (
    StructuredHillClimbKernel,
    StructuredIteratedLocalSearchKernel,
    StructuredKickPolicy,
    StructuredScheduledLocalSearchKernel,
    StructuredStochasticNeighborhoodKernel,
    StructuredVariableNeighborhoodKernel,
    StructuredVariableNeighborhoodStage,
)


class StructuredLocalSearchPackageExportTests:
    """Lock the public facade for the structured local-search package."""

    def test_structured_package_re_exports_kernel_symbols(self) -> None:
        """The structured package should remain the canonical family facade."""
        assert StructuredHillClimbKernel is LocalSearchHillClimbKernel
        assert callable(StructuredScheduledLocalSearchKernel)
        assert callable(StructuredStochasticNeighborhoodKernel)
        assert callable(StructuredVariableNeighborhoodKernel)
        assert callable(StructuredIteratedLocalSearchKernel)
        assert callable(StructuredVariableNeighborhoodStage)
        assert callable(StructuredKickPolicy)
