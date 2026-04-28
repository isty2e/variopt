"""Canonical request-local evaluation pipeline."""

from typing import TypeVar

from .artifacts import (
    EvaluationRequest,
    Proposal,
    ProposalEvaluationSpec,
    RequestAlignedEvaluationRecord,
)
from .outcomes import EvaluationOutcome
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
