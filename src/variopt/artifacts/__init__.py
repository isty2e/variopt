"""Runtime artifact definitions.

Public artifact names continue to live under ``variopt.artifacts`` while the
implementation is split by ontology tier:

- request-plane artifacts
- evaluation-attempt failure artifacts
- compatibility projection artifacts
- candidate-refinement provenance artifacts
- terminal/report surfaces
"""

from .attempts import (
    EvaluationAttempt,
    EvaluationAttemptBatch,
    EvaluationExceptionSnapshot,
    EvaluationFailure,
    EvaluationSuccess,
)
from .records import (
    ObjectiveVectorPayload,
    ObjectiveVectorRecord,
    Observation,
    ObservationPayload,
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
    "EvaluationAttempt",
    "EvaluationAttemptBatch",
    "EvaluationExceptionSnapshot",
    "EvaluationFailure",
    "EvaluationRequest",
    "EvaluationSuccess",
    "InteractionEvaluationSpec",
    "InteractionEvaluationUnit",
    "NondominatedRunSurface",
    "ObjectiveVectorRecord",
    "ObjectiveVectorPayload",
    "Observation",
    "ObservationPayload",
    "Proposal",
    "ProposalEvaluationSpec",
    "RunReport",
    "RunResult",
    "Trace",
    "TraceEvent",
]
