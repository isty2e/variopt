"""Synchronous joblib-backed evaluator."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Literal, cast

import joblib  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import override

from ...artifacts import EvaluationRequest
from ...artifacts.records import RequestAlignedEvaluationRecord
from ...evaluation_pipeline import evaluate_request_attempt, evaluate_request_outcome
from ...execution import ExecutionResources
from ...outcomes import EvaluationAttemptBatch, EvaluationOutcome
from ...problem import Problem
from ...typevars import CandidateT
from ..base import Evaluator
from .contracts import (
    BoundaryT,
    JoblibDelayedFactory,
    JoblibEvaluationPayloadT,
    JoblibListParallelFactory,
)
from .execution import build_execution_resources, validate_joblib_configuration


@dataclass(slots=True)
class JoblibEvaluator(
    Evaluator[
        Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord],
    ],
    Generic[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
):
    """Joblib-backed evaluator that preserves canonical batch ordering.

    Parameters
    ----------
    n_jobs : int, default=-1
        Joblib worker count. ``-1`` delegates to joblib's default
        all-available-worker behavior.
    backend : {"loky", "threading"}, default="loky"
        Joblib backend used for request execution.
    """

    n_jobs: int = -1
    backend: Literal["loky", "threading"] = "loky"

    def __post_init__(self) -> None:
        """Validate joblib evaluator configuration."""
        validate_joblib_configuration(
            n_jobs=self.n_jobs,
            backend=self.backend,
        )

    @override
    def execution_resources(self) -> ExecutionResources:
        """Return evaluator-owned execution resources for a joblib batch.

        Returns
        -------
        ExecutionResources
            Resource contract describing evaluator-owned joblib parallelism.
        """
        return build_execution_resources(
            n_jobs=self.n_jobs,
            backend=self.backend,
        )

    @override
    def evaluate(
        self,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> tuple[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord], ...]:
        """Execute a request batch through joblib.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequest[CandidateT]]
            Request batch to execute.

        Returns
        -------
        tuple[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord], ...]
            Ordered outcomes aligned one-to-one with ``requests``.
        """
        parallel_factory = cast(
            JoblibListParallelFactory[
                EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]
            ],
            getattr(joblib, "Parallel"),
        )
        delayed_factory = cast(
            JoblibDelayedFactory,
            getattr(joblib, "delayed"),
        )
        outcomes = parallel_factory(
            n_jobs=self.n_jobs,
            backend=self.backend,
        )(
            delayed_factory(evaluate_request_outcome)(
                problem=problem,
                request=request,
            )
            for request in requests
        )
        return tuple(outcomes)

    def evaluate_attempts(
        self,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord]:
        """Execute a request batch through joblib into a dense attempt batch.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequest[CandidateT]]
            Request batch to execute.

        Returns
        -------
        EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord]
            Dense attempt batch aligned to ``requests``.
        """
        parallel_factory = cast(
            JoblibListParallelFactory[
                EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord]
            ],
            getattr(joblib, "Parallel"),
        )
        delayed_factory = cast(
            JoblibDelayedFactory,
            getattr(joblib, "delayed"),
        )
        attempts = parallel_factory(
            n_jobs=self.n_jobs,
            backend=self.backend,
        )(
            delayed_factory(evaluate_request_attempt)(
                problem=problem,
                request=request,
            )
            for request in requests
        )
        return EvaluationAttemptBatch[
            CandidateT,
            RequestAlignedEvaluationRecord,
        ].from_single_request_attempts(tuple(attempts))
