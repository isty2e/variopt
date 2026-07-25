"""Coordinator-owned synchronous Joblib worker-session lifecycle."""

from collections.abc import Sequence
from contextlib import ExitStack
from math import isfinite
from mmap import mmap
from os import O_CREAT, O_EXCL, O_WRONLY, chmod, fdopen
from os import open as open_file
from pathlib import Path
from secrets import token_bytes
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Generic, Literal, cast

import joblib  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Never, Self

from ...artifacts import EvaluationAttemptBatch, EvaluationRequest
from ...artifacts.records import RequestAlignedEvaluationRecord
from ...evaluation_pipeline import evaluate_request_attempt, evaluate_request_outcome
from ...execution import ExecutionResources
from ...outcomes import EvaluationOutcome
from ...problem import Problem
from ...typevars import CandidateT
from .contracts import (
    BoundaryT,
    JoblibDelayedFactory,
    JoblibEffectiveJobs,
    JoblibEvaluationPayloadT,
    JoblibListParallelFactory,
    JoblibListParallelRunner,
    JoblibParallelConfigurationFactory,
)
from .execution import build_execution_resources
from .worker_context import (
    SESSION_TOKEN_BYTES,
    JoblibProblemEnvelope,
    evaluate_worker_session_request,
    evaluate_worker_session_request_attempt,
)


class JoblibProblemSnapshotTransport(
    Generic[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
):
    """Coordinator-owned private mmap containing one serialized problem snapshot."""

    __slots__ = ("_mapping", "_temporary_directory", "token")

    def __init__(
        self,
        *,
        token: bytes,
        mapping: np.memmap,
        temporary_directory: TemporaryDirectory[str],
    ) -> None:
        self.token = token
        self._mapping: np.memmap | None = mapping
        self._temporary_directory: TemporaryDirectory[str] | None = temporary_directory

    @classmethod
    def create(
        cls,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
    ) -> Self:
        """Serialize one problem into a private read-only mmap transport."""
        token = token_bytes(SESSION_TOKEN_BYTES)
        envelope = JoblibProblemEnvelope(token=token, problem=problem)
        temporary_directory = TemporaryDirectory(prefix="variopt-joblib-")
        chmod(temporary_directory.name, 0o700)
        transport_path = Path(temporary_directory.name) / "problem.snapshot"
        try:
            descriptor = open_file(
                transport_path,
                O_CREAT | O_EXCL | O_WRONLY,
                0o600,
            )
            with fdopen(descriptor, "wb") as stream:
                # Keep the direct dependency off the default import path.
                import cloudpickle

                cloudpickle.dump(envelope, stream)
            mapping = np.memmap(transport_path, dtype=np.uint8, mode="r")
        except BaseException:
            temporary_directory.cleanup()
            raise

        return cls(
            token=token,
            mapping=mapping,
            temporary_directory=temporary_directory,
        )

    @property
    def mapping(self) -> NDArray[np.uint8]:
        """Return the active read-only mapping."""
        mapping = self._mapping
        if mapping is None:
            msg = "worker-session transport is closed"
            raise RuntimeError(msg)
        return mapping

    def close(self) -> None:
        """Release the mapping and remove the private temporary directory."""
        mapping = self._mapping
        temporary_directory = self._temporary_directory
        self._mapping = None
        self._temporary_directory = None

        if mapping is not None:
            # NumPy's stubs omit the mmap base used by np.memmap at runtime.
            mapping_base: object = mapping.base
            if isinstance(mapping_base, mmap):
                mapping_base.close()
        if temporary_directory is not None:
            temporary_directory.cleanup()

    def __enter__(self) -> Self:
        """Return the active transport."""
        _ = self.mapping
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the transport regardless of scope outcome."""
        _ = exception_type, exception, traceback
        self.close()


class JoblibWorkerSession(
    Generic[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
):
    """Problem-bound synchronous Joblib execution scope."""

    __slots__ = (
        "_backend",
        "_closed",
        "_entered",
        "_n_jobs",
        "_parallel_runner",
        "_problem",
        "_resource_stack",
        "_transport",
        "_worker_idle_timeout",
    )
    _backend: Literal["loky", "threading"]

    def __init__(
        self,
        *,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        n_jobs: int,
        backend: Literal["loky", "threading"],
        worker_idle_timeout: float | None = None,
    ) -> None:
        if worker_idle_timeout is not None:
            if type(worker_idle_timeout) is not float:
                msg = "worker_idle_timeout must be a float when provided"
                raise TypeError(msg)
            if not isfinite(worker_idle_timeout) or worker_idle_timeout <= 0.0:
                msg = "worker_idle_timeout must be finite and positive"
                raise ValueError(msg)
            if backend != "loky":
                msg = "worker_idle_timeout is supported only by the loky backend"
                raise ValueError(msg)
        self._problem = problem
        self._n_jobs = n_jobs
        self._backend = backend
        self._worker_idle_timeout = worker_idle_timeout
        self._closed = False
        self._entered = False
        self._parallel_runner: JoblibListParallelRunner[object] | None = None
        self._resource_stack: ExitStack | None = None
        self._transport: (
            JoblibProblemSnapshotTransport[
                BoundaryT,
                CandidateT,
                JoblibEvaluationPayloadT,
            ]
            | None
        ) = None

    def __enter__(self) -> Self:
        """Open the problem transport and persistent Joblib runner."""
        if self._entered:
            msg = "worker session cannot be entered more than once"
            raise RuntimeError(msg)
        if self._closed:
            msg = "worker session cannot be reopened after close"
            raise RuntimeError(msg)

        self._entered = True
        parallel_config_factory = cast(
            JoblibParallelConfigurationFactory,
            getattr(joblib, "parallel_config"),
        )
        if self._worker_idle_timeout is not None:
            parallel_configuration = parallel_config_factory(
                backend=self._backend,
                idle_worker_timeout=self._worker_idle_timeout,
            )
        else:
            parallel_configuration = parallel_config_factory(backend=self._backend)
        resource_stack = ExitStack()
        try:
            _ = resource_stack.enter_context(parallel_configuration)
            effective_jobs = cast(
                JoblibEffectiveJobs,
                getattr(joblib, "effective_n_jobs"),
            )(self._n_jobs)
            if effective_jobs <= 1:
                resource_stack.close()
                return self

            if self._backend == "loky":
                self._transport = resource_stack.enter_context(
                    JoblibProblemSnapshotTransport[
                        BoundaryT,
                        CandidateT,
                        JoblibEvaluationPayloadT,
                    ].create(self._problem)
                )

            parallel_factory = cast(
                JoblibListParallelFactory[object],
                getattr(joblib, "Parallel"),
            )
            self._parallel_runner = resource_stack.enter_context(
                parallel_factory(n_jobs=self._n_jobs)
            )
            self._resource_stack = resource_stack
        except BaseException:
            self._entered = False
            self._closed = True
            self._parallel_runner = None
            self._transport = None
            resource_stack.close()
            raise

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the persistent runner before removing its transport."""
        if not self._entered:
            return
        self._entered = False
        self._closed = True

        resource_stack = self._resource_stack
        self._parallel_runner = None
        self._resource_stack = None
        self._transport = None
        if resource_stack is not None:
            resource_stack.__exit__(exception_type, exception, traceback)

    def __reduce__(self) -> Never:
        """Reject persistence of runtime-only evaluator session state."""
        msg = "Joblib worker sessions are runtime-only and cannot be pickled"
        raise TypeError(msg)

    def execution_resources(self) -> ExecutionResources:
        """Return the execution resources owned by this scope."""
        return build_execution_resources(
            n_jobs=self._n_jobs,
            backend=self._backend,
        )

    def evaluate(
        self,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> tuple[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord], ...]:
        """Evaluate one ordered batch against the bound problem."""
        self._require_active_problem(problem)
        if len(requests) == 0:
            return ()

        parallel_runner = self._parallel_runner
        if parallel_runner is None:
            return tuple(
                evaluate_request_outcome(problem=problem, request=request)
                for request in requests
            )

        delayed_factory = cast(
            JoblibDelayedFactory,
            getattr(joblib, "delayed"),
        )
        transport = self._transport
        if transport is None:
            outcomes = parallel_runner(
                delayed_factory(evaluate_request_outcome)(
                    problem=problem,
                    request=request,
                )
                for request in requests
            )
        else:
            outcomes = parallel_runner(
                delayed_factory(evaluate_worker_session_request)(
                    token=transport.token,
                    transport=transport.mapping,
                    request=request,
                )
                for request in requests
            )
        return cast(
            tuple[
                EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord],
                ...,
            ],
            tuple(outcomes),
        )

    def evaluate_attempts(
        self,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]:
        """Evaluate one dense attempt batch against the bound problem."""
        self._require_active_problem(problem)
        if len(requests) == 0:
            return EvaluationAttemptBatch[
                CandidateT,
                JoblibEvaluationPayloadT,
            ].from_single_request_attempts(())

        parallel_runner = self._parallel_runner
        if parallel_runner is None:
            attempts = tuple(
                evaluate_request_attempt(problem=problem, request=request)
                for request in requests
            )
        else:
            delayed_factory = cast(
                JoblibDelayedFactory,
                getattr(joblib, "delayed"),
            )
            transport = self._transport
            if transport is None:
                raw_attempts = parallel_runner(
                    delayed_factory(evaluate_request_attempt)(
                        problem=problem,
                        request=request,
                    )
                    for request in requests
                )
            else:
                raw_attempts = parallel_runner(
                    delayed_factory(evaluate_worker_session_request_attempt)(
                        token=transport.token,
                        transport=transport.mapping,
                        request=request,
                    )
                    for request in requests
                )
            attempts = cast(
                tuple[
                    EvaluationAttemptBatch[
                        CandidateT,
                        JoblibEvaluationPayloadT,
                    ],
                    ...,
                ],
                tuple(raw_attempts),
            )

        return EvaluationAttemptBatch[
            CandidateT,
            JoblibEvaluationPayloadT,
        ].from_single_request_attempts(attempts)

    def _require_active_problem(
        self,
        problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
    ) -> None:
        if not self._entered:
            msg = "worker session is not active"
            raise RuntimeError(msg)
        if problem is not self._problem:
            msg = "worker session can evaluate only its bound Problem instance"
            raise ValueError(msg)
