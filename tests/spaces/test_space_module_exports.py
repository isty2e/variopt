"""Facade regressions for the semantic spaces modules."""


from variopt.spaces import ArraySpace as RootArraySpace
from variopt.spaces import IntegerSpace as RootIntegerSpace
from variopt.spaces.composites import (
    ArraySpace,
    RecordCandidate,
    RecordSpace,
    TupleSpace,
)
from variopt.spaces.permutation import PermutationSpace
from variopt.spaces.scalar import CategoricalSpace, IntegerSpace, RealSpace


class SpacesModuleExportTests:
    """Lock the semantic top-level spaces modules."""

    def test_top_level_space_modules_re_export_canonical_space_types(self) -> None:
        """Top-level semantic space modules should align with the root facade."""
        assert IntegerSpace is RootIntegerSpace
        assert ArraySpace is RootArraySpace
        assert callable(RealSpace)
        assert callable(CategoricalSpace)
        assert callable(PermutationSpace)
        assert callable(RecordCandidate)
        assert callable(RecordSpace)
        assert callable(TupleSpace)
