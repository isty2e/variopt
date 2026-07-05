"""Public interfaces for the variopt package."""

from .artifacts import (
    CandidateRefinement,
    EvaluationExceptionSnapshot,
    EvaluationFailure,
    EvaluationRequest,
    NondominatedRunSurface,
    ObjectiveVectorRecord,
    Observation,
    Proposal,
    RunReport,
    RunResult,
)
from .direction import OptimizationDirection
from .diversity import DiversityMetric
from .evaluators.base import Evaluator
from .execution import (
    EXACT_ASYNC_EXECUTION_MODEL,
    SEQUENTIAL_EXECUTION_MODEL,
    STALE_ASYNC_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
    EvaluationBudget,
    EvaluationBudgetExhausted,
    ExecutionAssimilationMode,
    ExecutionCompletionMode,
    ExecutionModel,
    ExecutionResources,
    NestedParallelismPolicy,
)
from .kernel import (
    Kernel,
    KernelDiagnostics,
    KernelStatus,
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
)
from .methods import RunMethod, UnsupportedEvaluationFailureError
from .objective import (
    EvaluationProtocol,
    InteractionEvaluationProtocol,
    Objective,
    ObservationEvaluationProtocol,
    ScalarEvaluationProtocol,
)
from .operators import VariationOperator
from .outcomes import EvaluationAttemptBatch, EvaluationOutcome
from .problem import InteractionProblem, Problem
from .spaces import (
    ArraySpace,
    CandidateEquality,
    CategoricalSpace,
    IntegerSpace,
    PermutationSpace,
    RealSpace,
    RecordSpace,
    SearchSpace,
    TupleSpace,
)
from .study import RunExecutionFailed, Study

__all__ = [
    "ArraySpace",
    "CandidateEquality",
    "CandidateRefinement",
    "CategoricalSpace",
    "DiversityMetric",
    "EvaluationAttemptBatch",
    "EvaluationExceptionSnapshot",
    "EvaluationFailure",
    "EvaluationOutcome",
    "EvaluationBudget",
    "EvaluationBudgetExhausted",
    "EvaluationProtocol",
    "EvaluationRequest",
    "Evaluator",
    "EXACT_ASYNC_EXECUTION_MODEL",
    "ExecutionAssimilationMode",
    "ExecutionCompletionMode",
    "ExecutionModel",
    "ExecutionResources",
    "InteractionEvaluationProtocol",
    "InteractionProblem",
    "IntegerSpace",
    "Kernel",
    "KernelDiagnostics",
    "KernelStatus",
    "NondominatedRunSurface",
    "NestedParallelismPolicy",
    "Objective",
    "ObjectiveVectorRecord",
    "Observation",
    "ObservationEvaluationProtocol",
    "OptimizationDirection",
    "PermutationSpace",
    "Problem",
    "Proposal",
    "ProposalBatchQuery",
    "ProposalKernelHint",
    "ProposalLocalSearchContext",
    "RealSpace",
    "RecordSpace",
    "RunMethod",
    "RunExecutionFailed",
    "RunReport",
    "RunResult",
    "ScalarEvaluationProtocol",
    "SearchSpace",
    "SEQUENTIAL_EXECUTION_MODEL",
    "STALE_ASYNC_EXECUTION_MODEL",
    "Study",
    "SYNC_BATCH_EXECUTION_MODEL",
    "TupleSpace",
    "UnsupportedEvaluationFailureError",
    "VariationOperator",
]
