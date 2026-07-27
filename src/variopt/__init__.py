"""Public interfaces for the variopt package."""

from .artifacts import (
    CandidateRefinement,
    EvaluationAttemptBatch,
    EvaluationExceptionSnapshot,
    EvaluationFailure,
    EvaluationRequest,
    KernelDiagnostics,
    KernelStatus,
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
from .outcomes import EvaluationOutcome
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
    "EXACT_ASYNC_EXECUTION_MODEL",
    "SEQUENTIAL_EXECUTION_MODEL",
    "STALE_ASYNC_EXECUTION_MODEL",
    "SYNC_BATCH_EXECUTION_MODEL",
    "ArraySpace",
    "CandidateEquality",
    "CandidateRefinement",
    "CategoricalSpace",
    "DiversityMetric",
    "EvaluationAttemptBatch",
    "EvaluationBudget",
    "EvaluationBudgetExhausted",
    "EvaluationExceptionSnapshot",
    "EvaluationFailure",
    "EvaluationOutcome",
    "EvaluationProtocol",
    "EvaluationRequest",
    "Evaluator",
    "ExecutionAssimilationMode",
    "ExecutionCompletionMode",
    "ExecutionModel",
    "ExecutionResources",
    "IntegerSpace",
    "InteractionEvaluationProtocol",
    "InteractionProblem",
    "Kernel",
    "KernelDiagnostics",
    "KernelStatus",
    "NestedParallelismPolicy",
    "NondominatedRunSurface",
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
    "RunExecutionFailed",
    "RunMethod",
    "RunReport",
    "RunResult",
    "ScalarEvaluationProtocol",
    "SearchSpace",
    "Study",
    "TupleSpace",
    "UnsupportedEvaluationFailureError",
    "VariationOperator",
]
