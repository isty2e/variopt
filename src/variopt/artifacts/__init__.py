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
    DefaultEvaluationAttemptMaterializer,
    EvaluationAttempt,
    EvaluationAttemptBatch,
    EvaluationAttemptMaterializer,
    EvaluationExceptionSnapshot,
    EvaluationFailure,
    EvaluationSuccess,
    materialize_attempt_batch_records,
    materialize_success_record,
    materialize_success_records,
)
from .diagnostics import KernelDiagnostics, KernelStatus
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
    "DefaultEvaluationAttemptMaterializer",
    "EvaluationAttempt",
    "EvaluationAttemptBatch",
    "EvaluationExceptionSnapshot",
    "EvaluationFailure",
    "EvaluationAttemptMaterializer",
    "EvaluationRequest",
    "EvaluationSuccess",
    "InteractionEvaluationSpec",
    "InteractionEvaluationUnit",
    "KernelDiagnostics",
    "KernelStatus",
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
    "materialize_attempt_batch_records",
    "materialize_success_record",
    "materialize_success_records",
]
