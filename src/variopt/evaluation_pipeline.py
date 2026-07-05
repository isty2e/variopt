"""Canonical request-local evaluation pipeline."""

from typing import Protocol, TypeAlias, TypeGuard, TypeVar, overload, runtime_checkable

from .artifacts import (
    EvaluationAttemptBatch as PayloadEvaluationAttemptBatch,
)
from .artifacts import (
    EvaluationFailure,
    EvaluationRequest,
    EvaluationSuccess,
    ObjectiveVectorPayload,
    ObjectiveVectorRecord,
    Observation,
    ObservationPayload,
    Proposal,
    ProposalEvaluationSpec,
)
from .artifacts.records import RequestAlignedEvaluationRecord
from .outcomes import EvaluationOutcome
from .problem import Problem
from .typevars import CandidateT

BoundaryT = TypeVar("BoundaryT")
PayloadT = TypeVar("PayloadT")
CompatibilityEvaluationPayload: TypeAlias = (
    RequestAlignedEvaluationRecord | ObservationPayload | ObjectiveVectorPayload
)
CompatibilityEvaluationPayloadT = TypeVar(
    "CompatibilityEvaluationPayloadT",
    bound=CompatibilityEvaluationPayload,
)
ProposalEvaluationRecordT = TypeVar(
    "ProposalEvaluationRecordT",
    bound=RequestAlignedEvaluationRecord,
)


@runtime_checkable
class _RequestAlignedCompatibilityShape(Protocol):
    """Runtime shape for legacy request-aligned compatibility payloads."""

    @property
    def request(self) -> object:
        """Return the payload's request-like slot."""
        ...

    @property
    def candidate(self) -> object:
        """Return the payload's candidate-like slot."""
        ...


def evaluate_request_payload(
    *,
    problem: Problem[BoundaryT, CandidateT, PayloadT],
    request: EvaluationRequest[CandidateT],
) -> PayloadT:
    """Execute one validated request into a request-free payload.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, PayloadT]
        Problem that validates candidates and evaluates requests.
    request : EvaluationRequest[CandidateT]
        Request to execute.

    Returns
    -------
    PayloadT
        Request-free payload produced by the problem protocol.
    """
    candidate = request.candidate
    problem.space.validate(candidate)
    return problem.evaluation_protocol.evaluate_request(request)


def evaluate_request_success(
    *,
    problem: Problem[BoundaryT, CandidateT, PayloadT],
    request: EvaluationRequest[CandidateT],
) -> EvaluationSuccess[CandidateT, PayloadT]:
    """Execute one request into a successful request-owned attempt.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, PayloadT]
        Problem that validates candidates and evaluates requests.
    request : EvaluationRequest[CandidateT]
        Request to execute.

    Returns
    -------
    EvaluationSuccess[CandidateT, PayloadT]
        Request-owned successful attempt carrying a request-free payload.
    """
    return EvaluationSuccess(
        request=request,
        payload=evaluate_request_payload(problem=problem, request=request),
        evaluation_count=1,
    )


def evaluate_request_payload_attempt(
    *,
    problem: Problem[BoundaryT, CandidateT, PayloadT],
    request: EvaluationRequest[CandidateT],
) -> PayloadEvaluationAttemptBatch[CandidateT, PayloadT]:
    """Execute one request into a payload success-or-failure attempt batch.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, PayloadT]
        Problem that validates candidates and evaluates requests.
    request : EvaluationRequest[CandidateT]
        Request to execute.

    Returns
    -------
    PayloadEvaluationAttemptBatch[CandidateT, PayloadT]
        One-slot attempt batch containing either a request-owned success or a
        recorded user-code evaluation failure.

    Notes
    -----
    Candidate validation happens before the user-code call and remains a hard
    failure. Only ``Exception`` raised while invoking the user evaluation
    protocol is recorded; ``BaseException`` subclasses such as
    ``KeyboardInterrupt`` and ``SystemExit`` escape.
    """
    candidate = request.candidate
    problem.space.validate(candidate)
    try:
        payload = problem.evaluation_protocol.evaluate_request(request)
    except Exception as exception:
        return PayloadEvaluationAttemptBatch(
            attempts=(
                EvaluationFailure[CandidateT].from_exception(
                    request=request,
                    exception=exception,
                ),
            ),
        )

    return PayloadEvaluationAttemptBatch(
        attempts=(
            EvaluationSuccess(
                request=request,
                payload=payload,
                evaluation_count=1,
            ),
        ),
    )


@overload
def _compatibility_record_from_payload(
    *,
    request: EvaluationRequest[CandidateT],
    payload: ObservationPayload,
) -> Observation[CandidateT]: ...


@overload
def _compatibility_record_from_payload(
    *,
    request: EvaluationRequest[CandidateT],
    payload: ObjectiveVectorPayload,
) -> ObjectiveVectorRecord[CandidateT]: ...


@overload
def _compatibility_record_from_payload(
    *,
    request: EvaluationRequest[CandidateT],
    payload: ProposalEvaluationRecordT,
) -> ProposalEvaluationRecordT: ...


def _compatibility_record_from_payload(
    *,
    request: EvaluationRequest[CandidateT],
    payload: ProposalEvaluationRecordT | ObservationPayload | ObjectiveVectorPayload,
) -> ProposalEvaluationRecordT | Observation[CandidateT] | ObjectiveVectorRecord[CandidateT]:
    if isinstance(payload, ObservationPayload):
        return Observation(
            request=request,
            candidate=request.candidate,
            value=payload.value,
            score=payload.score,
            elapsed_seconds=payload.elapsed_seconds,
        )

    if isinstance(payload, ObjectiveVectorPayload):
        return ObjectiveVectorRecord(
            request=request,
            candidate=request.candidate,
            objective_values=payload.objective_values,
            objective_scores=payload.objective_scores,
            elapsed_seconds=payload.elapsed_seconds,
        )

    return payload


def _is_request_aligned_compatibility_payload(
    payload: object,
) -> TypeGuard[RequestAlignedEvaluationRecord[object]]:
    if isinstance(payload, (ObservationPayload, ObjectiveVectorPayload)):
        return False
    if not isinstance(payload, _RequestAlignedCompatibilityShape):
        return False

    return type(payload.request) is EvaluationRequest


def _validate_compatibility_record_alignment(
    *,
    request: EvaluationRequest[CandidateT],
    record: RequestAlignedEvaluationRecord,
) -> None:
    if record.request is not request:
        msg = "outcome record request must match the evaluation request"
        raise ValueError(msg)
    if record.candidate is not request.candidate:
        msg = "outcome record candidate must match the evaluation request candidate"
        raise ValueError(msg)


def evaluate_request_compatibility_record(
    *,
    problem: Problem[BoundaryT, CandidateT, CompatibilityEvaluationPayloadT],
    request: EvaluationRequest[CandidateT],
) -> RequestAlignedEvaluationRecord:
    """Execute one request and project its payload into a legacy record.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, CompatibilityEvaluationPayloadT]
        Problem whose protocol returns either a request-free payload or a
        legacy request-aligned record.
    request : EvaluationRequest[CandidateT]
        Request to execute.

    Returns
    -------
    RequestAlignedEvaluationRecord
        Request-aligned compatibility record for old outcome-aware callers.
    """
    payload = evaluate_request_payload(problem=problem, request=request)
    if isinstance(payload, ObservationPayload):
        scalar_record = _compatibility_record_from_payload(
            request=request,
            payload=payload,
        )
        _validate_compatibility_record_alignment(request=request, record=scalar_record)
        return scalar_record

    if isinstance(payload, ObjectiveVectorPayload):
        vector_record = _compatibility_record_from_payload(
            request=request,
            payload=payload,
        )
        _validate_compatibility_record_alignment(request=request, record=vector_record)
        return vector_record

    if _is_request_aligned_compatibility_payload(payload):
        _validate_compatibility_record_alignment(request=request, record=payload)
        return payload

    msg = "compatibility payload could not be projected to a request-aligned record"
    raise TypeError(msg)


def evaluate_request_outcome(
    *,
    problem: Problem[BoundaryT, CandidateT, CompatibilityEvaluationPayloadT],
    request: EvaluationRequest[CandidateT],
) -> EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]:
    """Execute one canonical evaluation request.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, CompatibilityEvaluationPayloadT]
        Problem that validates candidates and evaluates requests.
    request : EvaluationRequest[CandidateT]
        Request to execute.

    Returns
    -------
    EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]
        Compatibility evaluation outcome for the request.
    """
    record = evaluate_request_compatibility_record(problem=problem, request=request)

    return EvaluationOutcome(
        record=record,
        evaluation_count=1,
    )


def evaluate_request_attempt(
    *,
    problem: Problem[BoundaryT, CandidateT, PayloadT],
    request: EvaluationRequest[CandidateT],
) -> PayloadEvaluationAttemptBatch[CandidateT, PayloadT]:
    """Execute one canonical request into a success-or-failure attempt batch.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, PayloadT]
        Problem that validates candidates and evaluates requests.
    request : EvaluationRequest[CandidateT]
        Request to execute.

    Returns
    -------
    EvaluationAttemptBatch[CandidateT, PayloadT]
        One-slot attempt batch containing either a request-owned success with the
        request-free payload or a recorded user-code evaluation failure.

    Notes
    -----
    Candidate validation happens before the user-code call and remains a hard
    failure. Only ``Exception`` raised while invoking the user evaluation
    protocol is recorded; ``BaseException`` subclasses such as
    ``KeyboardInterrupt`` and ``SystemExit`` escape.
    """
    candidate = request.candidate
    problem.space.validate(candidate)
    try:
        payload = problem.evaluation_protocol.evaluate_request(request)
    except Exception as exception:
        return PayloadEvaluationAttemptBatch(
            attempts=(
                EvaluationFailure[CandidateT].from_exception(
                    request=request,
                    exception=exception,
                ),
            ),
        )

    return PayloadEvaluationAttemptBatch(
        attempts=(
            EvaluationSuccess(
                request=request,
                payload=payload,
                evaluation_count=1,
            ),
        ),
    )


def evaluate_proposal_outcome(
    *,
    problem: Problem[BoundaryT, CandidateT, CompatibilityEvaluationPayloadT],
    proposal: Proposal[CandidateT],
    proposal_evaluation_spec: ProposalEvaluationSpec | None = None,
) -> EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]:
    """Execute one proposal through the request-first evaluation pipeline.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, CompatibilityEvaluationPayloadT]
        Problem that validates candidates and evaluates requests.
    proposal : Proposal[CandidateT]
        Proposal to evaluate.
    proposal_evaluation_spec : ProposalEvaluationSpec | None, optional
        Optional proposal-side metadata attached to the synthesized request.

    Returns
    -------
    EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]
        Compatibility evaluation outcome for the proposal.
    """
    return evaluate_request_outcome(
        problem=problem,
        request=EvaluationRequest(
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
        ),
    )
