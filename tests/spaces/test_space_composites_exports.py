"""Facade regressions for the composite spaces package."""


from variopt.spaces.composites import (
    ArraySpace,
    RecordCandidate,
    RecordSpace,
    TupleSpace,
)
from variopt.spaces.composites.array_space import ArraySpace as ArraySpaceModule
from variopt.spaces.composites.record_space import RecordSpace as RecordSpaceModule
from variopt.spaces.composites.records import RecordCandidate as RecordCandidateModule
from variopt.spaces.composites.tuple_space import TupleSpace as TupleSpaceModule


class CompositeSpacesFacadeExportTests:
    """Lock composite-spaces facade identity."""

    def test_composites_facade_reexports_semantic_modules(self) -> None:
        """The composites facade should align with the semantic submodules."""
        assert ArraySpace is ArraySpaceModule
        assert RecordCandidate is RecordCandidateModule
        assert RecordSpace is RecordSpaceModule
        assert TupleSpace is TupleSpaceModule
