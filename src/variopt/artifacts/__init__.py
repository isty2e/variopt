"""Runtime artifact definitions.

Public artifact names continue to live under ``variopt.artifacts`` while the
implementation is split by ontology tier:

- request-plane artifacts
- evaluation-record artifacts
- candidate-refinement provenance artifacts
- terminal/report surfaces
"""

from .records import (
    EvaluationRecord,
    InteractionEvaluationRecord,
    ObjectiveVectorRecord,
    Observation,
    RequestAlignedEvaluationRecord,
)
from .refinement import CandidateRefinement
from .requests import (
    EvaluationRequest,
    InteractionEvaluationSpec,
    InteractionEvaluationUnit,
    Proposal,
    ProposalEvaluationSpec,
)
from .terminal import (
    NondominatedRunSurface,
    RunReport,
    RunResult,
    Trace,
    TraceEvent,
)

__all__ = [
    "CandidateRefinement",
    "EvaluationRecord",
    "EvaluationRequest",
    "InteractionEvaluationRecord",
    "InteractionEvaluationSpec",
    "InteractionEvaluationUnit",
    "NondominatedRunSurface",
    "ObjectiveVectorRecord",
    "Observation",
    "Proposal",
    "ProposalEvaluationSpec",
    "RequestAlignedEvaluationRecord",
    "RunReport",
    "RunResult",
    "Trace",
    "TraceEvent",
]
