"""Concrete sequential evaluator for local execution."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from typing_extensions import TypeVar as DefaultTypeVar
from typing_extensions import override

from ..artifacts import (
    EvaluationAttemptBatch,
    EvaluationRequest,
    ObservationPayload,
)
from ..artifacts.records import RequestAlignedEvaluationRecord
from ..evaluation_pipeline import (
    CompatibilityEvaluationPayload,
    evaluate_request_attempt,
    evaluate_request_outcome,
)
from ..execution import ExecutionResources, NestedParallelismPolicy
from ..outcomes import EvaluationOutcome
from ..problem import Problem
from ..typevars import CandidateT
from .base import Evaluator

BoundaryT = TypeVar("BoundaryT")
SequentialEvaluationRecordT = DefaultTypeVar(
    "SequentialEvaluationRecordT",
    bound=CompatibilityEvaluationPayload,
    default=ObservationPayload,
)


@dataclass(slots=True)
class SequentialEvaluator(
    Evaluator[
        Problem[BoundaryT, CandidateT, SequentialEvaluationRecordT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord],
    ],
    Generic[BoundaryT, CandidateT, SequentialEvaluationRecordT],
):
    """Sequential evaluator that executes requests in-process.

    Notes
    -----
    This is the simplest evaluator implementation. It preserves canonical batch
    ordering and performs no parallel execution.
    """

    @override
    def execution_resources(self) -> ExecutionResources:
        """Return execution resources for sequential evaluation.

        Returns
        -------
        ExecutionResources
            Evaluator-owned resource contract with a single in-process worker.
        """
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
            owner_worker_count=1,
            owner_backend="sequential",
        )

    @override
    def evaluate(
        self,
        problem: Problem[BoundaryT, CandidateT, SequentialEvaluationRecordT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> tuple[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord], ...]:
        """Execute a request batch sequentially.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, SequentialEvaluationRecordT]
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequest[CandidateT]]
            Request batch to execute.

        Returns
        -------
        tuple[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord], ...]
            Outcomes aligned one-to-one with ``requests`` in batch order.
        """
        return tuple(
            evaluate_request_outcome(
                problem=problem,
                request=request,
            )
            for request in requests
        )

    def evaluate_attempts(
        self,
        problem: Problem[BoundaryT, CandidateT, SequentialEvaluationRecordT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationAttemptBatch[CandidateT, SequentialEvaluationRecordT]:
        """Execute a request batch into a dense success/failure attempt batch.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, SequentialEvaluationRecordT]
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequest[CandidateT]]
            Request batch to execute.

        Returns
        -------
        EvaluationAttemptBatch[CandidateT, SequentialEvaluationRecordT]
            Ordered request-owned attempt batch aligned to ``requests``.
        """
        return EvaluationAttemptBatch[
            CandidateT,
            SequentialEvaluationRecordT,
        ].from_single_request_attempts(
            evaluate_request_attempt(
                problem=problem,
                request=request,
            )
            for request in requests
        )
