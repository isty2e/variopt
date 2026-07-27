"""Synchronous joblib-backed evaluator."""

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, cast

import joblib  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import Self, override

from ...artifacts import EvaluationAttemptBatch, EvaluationRequest
from ...artifacts.records import RequestAlignedEvaluationRecord
from ...evaluation_pipeline import evaluate_request_attempt, evaluate_request_outcome
from ...execution import ExecutionResources
from ...outcomes import EvaluationOutcome
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
from .session import JoblibWorkerSession

JoblibProblemTransportMode: TypeAlias = Literal["per_request", "worker_session"]


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
    problem_transport : {"per_request", "worker_session"}, default="per_request"
        Problem transport contract. ``per_request`` sends the supplied problem
        with every request batch. ``worker_session`` serializes one bound
        problem snapshot per synchronous run scope and reuses it worker-locally.
    """

    n_jobs: int = -1
    backend: Literal["loky", "threading"] = "loky"
    problem_transport: JoblibProblemTransportMode = "per_request"

    def __post_init__(self) -> None:
        """Validate joblib evaluator configuration."""
        validate_joblib_configuration(
            n_jobs=self.n_jobs,
            backend=self.backend,
        )
        if self.problem_transport not in {"per_request", "worker_session"}:
            msg = "problem_transport must be 'per_request' or 'worker_session'"
            raise ValueError(msg)

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
        if len(requests) == 0:
            return ()
        if self.problem_transport == "worker_session":
            with JoblibWorkerSession[
                BoundaryT,
                CandidateT,
                JoblibEvaluationPayloadT,
            ](
                problem=problem,
                n_jobs=self.n_jobs,
                backend=self.backend,
            ) as session:
                return session.evaluate(problem, requests)

        parallel_factory = cast(
            JoblibListParallelFactory[
                EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]
            ],
            joblib.Parallel,
        )
        delayed_factory = cast(
            JoblibDelayedFactory,
            joblib.delayed,
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
    ) -> EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]:
        """Execute a request batch through joblib into a dense attempt batch.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequest[CandidateT]]
            Request batch to execute.

        Returns
        -------
        EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]
            Dense attempt batch aligned to ``requests``.
        """
        if len(requests) == 0:
            return EvaluationAttemptBatch[
                CandidateT,
                JoblibEvaluationPayloadT,
            ].from_single_request_attempts(())
        if self.problem_transport == "worker_session":
            with JoblibWorkerSession[
                BoundaryT,
                CandidateT,
                JoblibEvaluationPayloadT,
            ](
                problem=problem,
                n_jobs=self.n_jobs,
                backend=self.backend,
            ) as session:
                return session.evaluate_attempts(problem, requests)

        parallel_factory = cast(
            JoblibListParallelFactory[
                EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]
            ],
            joblib.Parallel,
        )
        delayed_factory = cast(
            JoblibDelayedFactory,
            joblib.delayed,
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
            JoblibEvaluationPayloadT,
        ].from_single_request_attempts(attempts)

    def _open_attempt_run_scope(
        self,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
    ) -> AbstractContextManager[
        Self | JoblibWorkerSession[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
    ]:
        """Open one internal problem-bound synchronous evaluation scope.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]
            Exact problem instance owned by the run.

        Returns
        -------
        AbstractContextManager[JoblibEvaluator | JoblibWorkerSession]
            Default per-request evaluator scope or an isolated worker session.

        Notes
        -----
        The reusable evaluator stores configuration only. Runtime transport,
        worker generations, and persistent Joblib resources belong exclusively
        to the returned scope.
        """
        if self.problem_transport == "per_request":
            return nullcontext(self)
        return JoblibWorkerSession[
            BoundaryT,
            CandidateT,
            JoblibEvaluationPayloadT,
        ](
            problem=problem,
            n_jobs=self.n_jobs,
            backend=self.backend,
        )
