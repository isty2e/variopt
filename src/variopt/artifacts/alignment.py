"""Logical alignment validation for request-owned evaluation attempts."""

from typing_extensions import TypeVar

from ..spaces import CandidateEquality
from ..spaces.equality import scalar_candidate_equality
from .attempts import EvaluationAttemptBatch, EvaluationFailure, EvaluationSuccess
from .requests import EvaluationRequest

CandidateT = TypeVar("CandidateT")
PayloadT = TypeVar("PayloadT")


def validate_aligned_attempts(
    requests: tuple[EvaluationRequest[CandidateT], ...],
    attempts: EvaluationAttemptBatch[CandidateT, PayloadT],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None = None,
) -> None:
    """Reject attempt batches that do not align with input requests.

    Parameters
    ----------
    requests : tuple[EvaluationRequest[CandidateT], ...]
        Canonical request slots submitted to an evaluator or kernel.
    attempts : EvaluationAttemptBatch[CandidateT, PayloadT]
        Dense attempt batch returned for those slots.
    candidate_equal : CandidateEquality[CandidateT] | None, optional
        Explicit candidate equality predicate used to validate request and
        refinement alignment.

    Raises
    ------
    TypeError
        If an explicit candidate equality predicate does not return ``bool``.
    ValueError
        If the attempt batch does not carry logically equivalent requests in
        the same slots.
    """
    if len(attempts.requests) != len(requests):
        msg = "attempt batch must contain exactly one slot per request"
        raise ValueError(msg)

    for expected_request, attempt_request, attempt in zip(
        requests,
        attempts.requests,
        attempts.attempts,
        strict=True,
    ):
        if type(attempt) is EvaluationFailure:
            attempts_match = _requests_match(
                attempt_request,
                expected_request,
                candidate_equal=candidate_equal,
            )
        elif type(attempt) is EvaluationSuccess:
            attempts_match = _success_matches_expected_request(
                attempt,
                expected_request,
                candidate_equal=candidate_equal,
            )
        else:
            attempts_match = False

        if attempts_match:
            continue

        msg = "attempt batch requests must align with input request order"
        raise ValueError(msg)

    if candidate_equal is None:
        return

    for success in attempts.successes:
        _ = EvaluationSuccess(
            request=success.request,
            payload=success.payload,
            evaluation_count=success.evaluation_count,
            refinement=success.refinement,
            kernel_diagnostics=success.kernel_diagnostics,
            candidate_equal=candidate_equal,
        )


def _requests_match(
    left_request: EvaluationRequest[CandidateT],
    right_request: EvaluationRequest[CandidateT],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> bool:
    if left_request.proposal_id != right_request.proposal_id:
        return False

    if left_request.proposal_evaluation_spec != right_request.proposal_evaluation_spec:
        return False

    return _candidates_match(
        left_request.candidate,
        right_request.candidate,
        candidate_equal=candidate_equal,
    )


def _success_matches_expected_request(
    success: EvaluationSuccess[CandidateT, PayloadT],
    expected_request: EvaluationRequest[CandidateT],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> bool:
    if _requests_match(
        success.request,
        expected_request,
        candidate_equal=candidate_equal,
    ):
        return True

    refinement = success.refinement
    if refinement is None:
        return False

    if success.request.proposal_id != expected_request.proposal_id:
        return False

    if (
        success.request.proposal_evaluation_spec
        != expected_request.proposal_evaluation_spec
    ):
        return False

    if not _candidates_match(
        refinement.source_candidate,
        expected_request.candidate,
        candidate_equal=candidate_equal,
    ):
        return False

    return _candidates_match(
        success.request.candidate,
        refinement.refined_candidate,
        candidate_equal=candidate_equal,
    )


def _candidates_match(
    left_candidate: CandidateT,
    right_candidate: CandidateT,
    *,
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> bool:
    if candidate_equal is None:
        return scalar_candidate_equality(left_candidate, right_candidate)

    return _require_bool_candidate_match(
        candidate_equal(left_candidate, right_candidate)
    )


def _require_bool_candidate_match(candidates_match: object) -> bool:
    if type(candidates_match) is not bool:
        msg = "candidate equality must return bool"
        raise TypeError(msg)

    return candidates_match
