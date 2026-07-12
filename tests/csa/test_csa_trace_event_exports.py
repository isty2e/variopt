"""Regression tests for CSA trace event exports."""

from variopt.algorithms.population.csa.trace.events import (
    CSAEventTraceRecorder,
    CSAEventTraceState,
    CSAProposalFamilyTrace,
)
from variopt.algorithms.population.csa.trace.events.artifacts import (
    CSAChildEmissionTrace,
)
from variopt.algorithms.population.csa.trace.events.recorder import (
    CSAEventTraceRecorder as CSAEventTraceRecorderDirect,
)
from variopt.algorithms.population.csa.trace.events.state import (
    CSAEventTraceState as CSAEventTraceStateDirect,
)


class CSATraceEventExportTests:
    """Lock the canonical and semantic CSA trace event import paths."""

    def test_facade_re_exports_canonical_trace_symbols(self) -> None:
        assert CSAEventTraceState is CSAEventTraceStateDirect
        assert CSAEventTraceRecorder is CSAEventTraceRecorderDirect

    def test_semantic_submodules_are_importable(self) -> None:
        trace_state = CSAEventTraceStateDirect[tuple[float, ...]]()
        recorder = CSAEventTraceRecorderDirect(trace_state=trace_state)
        family_trace = CSAProposalFamilyTrace(
            family_key="mutation:0",
            observation_count=1,
            effective_credit_rate=0.5,
            mutation_weight=1.0,
        )
        child_trace = CSAChildEmissionTrace(
            family="mutation",
            proposal_family_key="mutation:0",
            seed_index=0,
            primary_source="bank",
            partner_indices=(),
            candidate=(0.0, 1.0),
        )

        assert recorder.completed_snapshot() == ()
        assert family_trace.family_key == "mutation:0"
        assert child_trace.family == "mutation"
