"""Regression coverage for the trimmed variopt root facade."""

import pytest

import variopt
import variopt.artifacts as artifact_facade
from variopt.artifacts import (
    CandidateRefinement,
    EvaluationAttemptBatch,
    EvaluationExceptionSnapshot,
    EvaluationFailure,
    EvaluationRequest,
    KernelDiagnostics,
    KernelStatus,
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
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
)
from variopt.methods import RunMethod, UnsupportedEvaluationFailureError
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
from variopt.study import RunExecutionFailed, Study


class RootFacadeExportTests:
    """Lock the 0.1.0 variopt root facade to direct-use and common contract nouns."""

    def test_root_facade_reexports_common_direct_use_and_contract_nouns(self) -> None:
        assert variopt.CandidateRefinement is CandidateRefinement
        assert variopt.CandidateEquality is CandidateEquality
        assert variopt.EvaluationAttemptBatch is EvaluationAttemptBatch
        assert variopt.EvaluationExceptionSnapshot is EvaluationExceptionSnapshot
        assert variopt.EvaluationFailure is EvaluationFailure
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
        assert variopt.RunExecutionFailed is RunExecutionFailed
        assert variopt.RunReport is RunReport
        assert variopt.RunResult is RunResult
        assert variopt.SearchSpace is SearchSpace
        assert variopt.SEQUENTIAL_EXECUTION_MODEL is SEQUENTIAL_EXECUTION_MODEL
        assert variopt.STALE_ASYNC_EXECUTION_MODEL is STALE_ASYNC_EXECUTION_MODEL
        assert variopt.Study is Study
        assert variopt.SYNC_BATCH_EXECUTION_MODEL is SYNC_BATCH_EXECUTION_MODEL
        assert (
            variopt.UnsupportedEvaluationFailureError
            is UnsupportedEvaluationFailureError
        )

    def test_root_facade_omits_family_specific_helper_contracts(self) -> None:
        removed_names = (
            "CandidateSampler",
            "DefaultEvaluationAttemptMaterializer",
            "EvaluationAttempt",
            "EvaluationAttemptMaterializer",
            "EvaluationRecord",
            "EvaluationSuccess",
            "InteractionEvaluationRecord",
            "ObjectiveVectorPayload",
            "ObservationPayload",
            "RandomSeed",
            "RequestAlignedEvaluationRecord",
            "SearchMethod",
            "SpaceBoundaryValue",
            "SpaceCandidateValue",
            "Trace",
            "materialize_attempt_batch_records",
            "materialize_success_record",
            "materialize_success_records",
        )

        assert all(name not in variopt.__all__ for name in removed_names)
        assert all(not hasattr(variopt, name) for name in removed_names)

        assert CandidateSampler is not None
        assert SpaceBoundaryValue is not None
        assert SpaceCandidateValue is not None

    def test_artifact_facade_omits_obsolete_generic_record_api(self) -> None:
        removed_names = (
            "EvaluationRecord",
            "InteractionEvaluationRecord",
            "RequestAlignedEvaluationRecord",
        )

        assert all(name not in artifact_facade.__all__ for name in removed_names)
        assert all(not hasattr(artifact_facade, name) for name in removed_names)

    def test_obsolete_generic_record_api_cannot_be_imported_from_facades(
        self,
    ) -> None:
        removed_imports = (
            "from variopt import EvaluationRecord",
            "from variopt import InteractionEvaluationRecord",
            "from variopt import RequestAlignedEvaluationRecord",
            "from variopt.artifacts import EvaluationRecord",
            "from variopt.artifacts import InteractionEvaluationRecord",
            "from variopt.artifacts import RequestAlignedEvaluationRecord",
        )

        for import_statement in removed_imports:
            with pytest.raises(ImportError):
                exec(import_statement, {})
