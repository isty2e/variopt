"""Canonical request-local evaluation pipeline."""

from typing import TypeVar

from .artifacts import (
    EvaluationFailure,
    EvaluationRequest,
    Proposal,
    ProposalEvaluationSpec,
    RequestAlignedEvaluationRecord,
)
from .outcomes import EvaluationAttemptBatch, EvaluationOutcome
from .problem import Problem
from .typevars import CandidateT

BoundaryT = TypeVar("BoundaryT")
ProposalEvaluationRecordT = TypeVar(
    "ProposalEvaluationRecordT",
    bound=RequestAlignedEvaluationRecord,
)


def evaluate_request_outcome(
    *,
    problem: Problem[BoundaryT, CandidateT, ProposalEvaluationRecordT],
    request: EvaluationRequest[CandidateT],
) -> EvaluationOutcome[CandidateT, ProposalEvaluationRecordT]:
    """Execute one canonical evaluation request.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, ProposalEvaluationRecordT]
        Problem that validates candidates and evaluates requests.
    request : EvaluationRequest[CandidateT]
        Request to execute.

    Returns
    -------
    EvaluationOutcome[CandidateT, ProposalEvaluationRecordT]
        Canonical evaluation outcome for the request.
    """
    candidate = request.candidate
    problem.space.validate(candidate)
    record = problem.evaluation_protocol.evaluate_request(request)

    return EvaluationOutcome(
        record=record,
        evaluation_count=1,
    )


def evaluate_request_attempt(
    *,
    problem: Problem[BoundaryT, CandidateT, ProposalEvaluationRecordT],
    request: EvaluationRequest[CandidateT],
) -> EvaluationAttemptBatch[CandidateT, ProposalEvaluationRecordT]:
    """Execute one canonical request into a success-or-failure attempt batch.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, ProposalEvaluationRecordT]
        Problem that validates candidates and evaluates requests.
    request : EvaluationRequest[CandidateT]
        Request to execute.

    Returns
    -------
    EvaluationAttemptBatch[CandidateT, ProposalEvaluationRecordT]
        One-slot attempt batch containing either a successful outcome or a
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
        record = problem.evaluation_protocol.evaluate_request(request)
    except Exception as exception:
        return EvaluationAttemptBatch(
            requests=(request,),
            failures=(
                EvaluationFailure[CandidateT].from_exception(
                    request=request,
                    exception=exception,
                ),
            ),
        )

    return EvaluationAttemptBatch(
        requests=(request,),
        outcomes=(
            EvaluationOutcome(
                record=record,
                evaluation_count=1,
            ),
        ),
    )


def evaluate_proposal_outcome(
    *,
    problem: Problem[BoundaryT, CandidateT, ProposalEvaluationRecordT],
    proposal: Proposal[CandidateT],
    proposal_evaluation_spec: ProposalEvaluationSpec | None = None,
) -> EvaluationOutcome[CandidateT, ProposalEvaluationRecordT]:
    """Execute one proposal through the request-first evaluation pipeline.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, ProposalEvaluationRecordT]
        Problem that validates candidates and evaluates requests.
    proposal : Proposal[CandidateT]
        Proposal to evaluate.
    proposal_evaluation_spec : ProposalEvaluationSpec | None, optional
        Optional proposal-side metadata attached to the synthesized request.

    Returns
    -------
    EvaluationOutcome[CandidateT, ProposalEvaluationRecordT]
        Canonical evaluation outcome for the proposal.
    """
    return evaluate_request_outcome(
        problem=problem,
        request=EvaluationRequest(
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
        ),
    )
