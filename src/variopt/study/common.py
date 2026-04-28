"""Shared helpers for study orchestration."""

from typing_extensions import TypeVar

from ..artifacts import (
    EvaluationRequest,
    Observation,
    Proposal,
    ProposalEvaluationSpec,
    RequestAlignedEvaluationRecord,
)
from ..evaluators.async_evaluator.artifacts import CompletionGroup
from ..outcomes import EvaluationOutcome
from ..typevars import CandidateT

StudyEvaluationRecordT = TypeVar(
    "StudyEvaluationRecordT",
    bound=RequestAlignedEvaluationRecord,
)


def build_evaluation_requests(
    proposals: tuple[Proposal[CandidateT], ...],
    *,
    proposal_evaluation_specs: tuple[ProposalEvaluationSpec | None, ...] | None,
) -> tuple[EvaluationRequest[CandidateT], ...]:
    """Lower one proposal batch into canonical evaluation requests.

    Parameters
    ----------
    proposals : tuple[Proposal[CandidateT], ...]
        Proposals to evaluate.
    proposal_evaluation_specs : tuple[ProposalEvaluationSpec | None, ...] | None
        Optional evaluation specs aligned with ``proposals``.

    Returns
    -------
    tuple[EvaluationRequest[CandidateT], ...]
        Canonical evaluation requests aligned with the input proposal order.
    """
    if proposal_evaluation_specs is None:
        return tuple(
            EvaluationRequest(proposal=proposal)
            for proposal in proposals
        )

    return tuple(
        EvaluationRequest(
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
        )
        for proposal, proposal_evaluation_spec in zip(
            proposals,
            proposal_evaluation_specs,
            strict=True,
        )
    )


def store_completion_group(
    ordered_outcomes: list[
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT] | None
    ],
    completion_group: CompletionGroup[
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT]
    ],
    *,
    request_count: int,
) -> int:
    """Store one completion group and return its newly covered size.

    Parameters
    ----------
    ordered_outcomes : list[EvaluationOutcome[CandidateT, StudyEvaluationRecordT] | None]
        Mutable ordered outcome buffer for an exact-async batch.
    completion_group : CompletionGroup[EvaluationOutcome[CandidateT, StudyEvaluationRecordT]]
        Completion group returned by the async evaluator.
    request_count : int
        Logical request count for the batch.

    Returns
    -------
    int
        Number of outcomes written into ``ordered_outcomes``.

    Raises
    ------
    ValueError
        If ``completion_group`` exceeds batch bounds or overlaps an already
        stored outcome.
    """
    end_index = completion_group.start_index + len(completion_group.outcomes)
    if end_index > request_count:
        msg = "completion group exceeds logical batch bounds"
        raise ValueError(msg)

    for offset, outcome in enumerate(completion_group.outcomes):
        target_index = completion_group.start_index + offset
        if ordered_outcomes[target_index] is not None:
            msg = "completion groups must not overlap"
            raise ValueError(msg)

        ordered_outcomes[target_index] = outcome

    return len(completion_group.outcomes)


def finalize_ordered_outcomes(
    ordered_outcomes: list[EvaluationOutcome[CandidateT, StudyEvaluationRecordT] | None],
) -> tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]:
    """Return one fully populated ordered outcome tuple.

    Parameters
    ----------
    ordered_outcomes : list[EvaluationOutcome[CandidateT, StudyEvaluationRecordT] | None]
        Mutable ordered outcome buffer for an exact-async batch.

    Returns
    -------
    tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]
        Fully populated ordered outcome tuple.

    Raises
    ------
    RuntimeError
        If any outcome slot is still missing.
    """
    if any(outcome is None for outcome in ordered_outcomes):
        msg = "exact_async completion left missing request outcomes"
        raise RuntimeError(msg)

    return tuple(outcome for outcome in ordered_outcomes if outcome is not None)


def validate_aligned_outcomes(
    requests: tuple[EvaluationRequest[CandidateT], ...],
    outcomes: tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...],
) -> None:
    """Reject evaluator outcomes that do not align with input requests.

    Parameters
    ----------
    requests : tuple[EvaluationRequest[CandidateT], ...]
        Evaluation requests submitted to the evaluator.
    outcomes : tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]
        Evaluation outcomes returned by the evaluator.

    Raises
    ------
    ValueError
        If the evaluator returns the wrong number of outcomes or if any outcome
        record is not aligned with its input request.

    Notes
    -----
    ``Study`` only accepts request-aligned records in its canonical execution
    stack. Interaction-aware records belong to ``InteractionProblem`` and do
    not participate in this request-local study tier.
    """
    if len(outcomes) != len(requests):
        msg = "evaluator must return exactly one outcome per request"
        raise ValueError(msg)

    for request, outcome in zip(requests, outcomes, strict=True):
        if outcome.record.request != request:
            msg = "evaluator outcomes must align with input request order"
            raise ValueError(msg)


def trace_value_for_records(
    records: tuple[StudyEvaluationRecordT, ...],
) -> float | None:
    """Return one scalar trace value when the batch records are observations.

    Parameters
    ----------
    records : tuple[StudyEvaluationRecordT, ...]
        Batch records produced by a study step.

    Returns
    -------
    float | None
        Best observed objective value in the batch when every record is an
        :class:`Observation`, otherwise ``None``.
    """
    if len(records) == 0:
        return None

    best_batch_value: float | None = None
    best_batch_score: float | None = None
    for record in records:
        if not isinstance(record, Observation):
            return None
        if best_batch_score is None or record.score < best_batch_score:
            best_batch_score = record.score
            best_batch_value = record.value

    return best_batch_value
