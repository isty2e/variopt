"""Regression coverage for the trimmed variopt root facade."""


import variopt
from variopt.artifacts import (
    CandidateRefinement,
    EvaluationRequest,
    NondominatedRunSurface,
    Observation,
    Proposal,
    RunReport,
    RunResult,
)
from variopt.evaluators import Evaluator
from variopt.execution import (
    EXACT_ASYNC_EXECUTION_MODEL,
    SEQUENTIAL_EXECUTION_MODEL,
    STALE_ASYNC_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
    ExecutionAssimilationMode,
    ExecutionCompletionMode,
    ExecutionModel,
    ExecutionResources,
    NestedParallelismPolicy,
)
from variopt.kernel import (
    Kernel,
    KernelDiagnostics,
    KernelStatus,
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
)
from variopt.methods import RunMethod
from variopt.objective import Objective
from variopt.outcomes import EvaluationOutcome
from variopt.problem import Problem
from variopt.sampling import CandidateSampler
from variopt.spaces import (
    CandidateEquality,
    SearchSpace,
    SpaceBoundaryValue,
    SpaceCandidateValue,
)
from variopt.study import Study


class RootFacadeExportTests:
    """Lock the 0.1.0 variopt root facade to direct-use and common contract nouns."""

    def test_root_facade_reexports_common_direct_use_and_contract_nouns(self) -> None:
        assert variopt.CandidateRefinement is CandidateRefinement
        assert variopt.CandidateEquality is CandidateEquality
        assert variopt.EvaluationOutcome is EvaluationOutcome
        assert variopt.EvaluationRequest is EvaluationRequest
        assert variopt.Evaluator is Evaluator
        assert variopt.EXACT_ASYNC_EXECUTION_MODEL is EXACT_ASYNC_EXECUTION_MODEL
        assert variopt.ExecutionAssimilationMode is ExecutionAssimilationMode
        assert variopt.ExecutionCompletionMode is ExecutionCompletionMode
        assert variopt.ExecutionModel is ExecutionModel
        assert variopt.ExecutionResources is ExecutionResources
        assert variopt.Kernel is Kernel
        assert variopt.KernelDiagnostics is KernelDiagnostics
        assert variopt.KernelStatus is KernelStatus
        assert variopt.NondominatedRunSurface is NondominatedRunSurface
        assert variopt.NestedParallelismPolicy is NestedParallelismPolicy
        assert variopt.Objective is Objective
        assert variopt.Observation is Observation
        assert variopt.Problem is Problem
        assert variopt.Proposal is Proposal
        assert variopt.ProposalBatchQuery is ProposalBatchQuery
        assert variopt.ProposalKernelHint is ProposalKernelHint
        assert variopt.ProposalLocalSearchContext is ProposalLocalSearchContext
        assert variopt.RunMethod is RunMethod
        assert variopt.RunReport is RunReport
        assert variopt.RunResult is RunResult
        assert variopt.SearchSpace is SearchSpace
        assert variopt.SEQUENTIAL_EXECUTION_MODEL is SEQUENTIAL_EXECUTION_MODEL
        assert variopt.STALE_ASYNC_EXECUTION_MODEL is STALE_ASYNC_EXECUTION_MODEL
        assert variopt.Study is Study
        assert variopt.SYNC_BATCH_EXECUTION_MODEL is SYNC_BATCH_EXECUTION_MODEL

    def test_root_facade_omits_family_specific_helper_contracts(self) -> None:
        removed_names = (
            "CandidateSampler",
            "RandomSeed",
            "SearchMethod",
            "SpaceBoundaryValue",
            "SpaceCandidateValue",
            "Trace",
        )

        assert all(name not in variopt.__all__ for name in removed_names)
        assert all(not hasattr(variopt, name) for name in removed_names)

        assert CandidateSampler is not None
        assert SpaceBoundaryValue is not None
        assert SpaceCandidateValue is not None
