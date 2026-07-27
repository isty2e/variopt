"""Process-local problem context for synchronous Joblib run scopes."""

from dataclasses import dataclass, field
from threading import Lock
from typing import Generic, cast

import numpy as np
from numpy.typing import NDArray

from ...artifacts import EvaluationAttemptBatch, EvaluationRequest
from ...artifacts.records import RequestAlignedEvaluationRecord
from ...evaluation_pipeline import evaluate_request_attempt, evaluate_request_outcome
from ...outcomes import EvaluationOutcome
from ...problem import Problem
from ...typevars import CandidateT
from .contracts import (
    BoundaryT,
    JoblibEvaluationPayloadT,
)

SESSION_TOKEN_BYTES = 16


def require_session_token(token: bytes) -> None:
    """Validate one opaque worker-session generation token."""
    if type(token) is not bytes:
        msg = "worker-session token must be bytes"
        raise TypeError(msg)
    if len(token) != SESSION_TOKEN_BYTES:
        msg = "worker-session token has an invalid length"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class JoblibProblemEnvelope(
    Generic[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
):
    """Serialized token-qualified problem snapshot."""

    token: bytes = field(repr=False)
    problem: Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]

    def __post_init__(self) -> None:
        """Validate the opaque generation token."""
        require_session_token(self.token)


def decode_problem_envelope(
    *,
    token: bytes,
    transport: NDArray[np.uint8],
) -> Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT]:
    """Decode and validate one token-qualified problem snapshot.

    This is the sole generic type-erasure boundary for the worker transport.
    Runtime validation recovers the concrete envelope and problem classes;
    Python cannot recover erased generic arguments from a pickle stream.
    """
    require_session_token(token)
    if not isinstance(transport, np.memmap):
        msg = "worker-session transport must be a read-only NumPy memory map"
        raise TypeError(msg)
    if transport.flags.writeable:
        msg = "worker-session transport must be read-only"
        raise ValueError(msg)

    # Import only on the explicit worker-session cold path so the default
    # evaluator facade does not eagerly import the serializer.
    import cloudpickle

    decoded: object = cloudpickle.loads(transport.tobytes())
    if not isinstance(decoded, JoblibProblemEnvelope):
        msg = "worker-session transport did not contain a problem envelope"
        raise TypeError(msg)
    if decoded.token != token:
        msg = "worker-session transport generation does not match the task"
        raise ValueError(msg)
    decoded_problem: object = decoded.problem
    if not isinstance(decoded_problem, Problem):
        msg = "worker-session envelope did not contain a Problem"
        raise TypeError(msg)

    return cast(
        Problem[BoundaryT, CandidateT, JoblibEvaluationPayloadT],
        decoded_problem,
    )


class JoblibWorkerProblemRegistry:
    """Process-local single-generation problem registry."""

    __slots__ = ("_lock", "_problem", "_token")

    def __init__(self) -> None:
        self._lock = Lock()
        self._problem: object | None = None
        self._token: bytes | None = None

    def problem_for(
        self,
        *,
        token: bytes,
        transport: NDArray[np.uint8],
    ) -> Problem[object, CandidateT, JoblibEvaluationPayloadT]:
        """Return the installed problem, replacing stale generations atomically."""
        require_session_token(token)
        with self._lock:
            if self._token != token:
                decoded_problem: Problem[
                    object,
                    CandidateT,
                    JoblibEvaluationPayloadT,
                ] = decode_problem_envelope(
                    token=token,
                    transport=transport,
                )
                self._problem = decoded_problem
                self._token = token

            problem = self._problem
            if not isinstance(problem, Problem):
                msg = "worker-session problem registry is not initialized"
                raise RuntimeError(msg)  # noqa: TRY004 - invalid worker state
            return cast(
                Problem[object, CandidateT, JoblibEvaluationPayloadT],
                problem,
            )


WORKER_PROBLEM_REGISTRY = JoblibWorkerProblemRegistry()


def evaluate_worker_session_request(
    *,
    token: bytes,
    transport: NDArray[np.uint8],
    request: EvaluationRequest[CandidateT],
) -> EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]:
    """Evaluate one request against the token-qualified worker problem."""
    problem = WORKER_PROBLEM_REGISTRY.problem_for(
        token=token,
        transport=transport,
    )
    return evaluate_request_outcome(problem=problem, request=request)


def evaluate_worker_session_request_attempt(
    *,
    token: bytes,
    transport: NDArray[np.uint8],
    request: EvaluationRequest[CandidateT],
) -> EvaluationAttemptBatch[CandidateT, JoblibEvaluationPayloadT]:
    """Evaluate one request attempt against the token-qualified worker problem."""
    problem: Problem[object, CandidateT, JoblibEvaluationPayloadT] = (
        WORKER_PROBLEM_REGISTRY.problem_for(
            token=token,
            transport=transport,
        )
    )
    return evaluate_request_attempt(problem=problem, request=request)
