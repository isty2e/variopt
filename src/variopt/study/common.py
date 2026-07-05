"""Shared helpers for study orchestration."""

from collections.abc import Sequence
from typing import Generic, Protocol, TypeGuard, runtime_checkable

from typing_extensions import TypeVar, override

from ..artifacts import (
    EvaluationRequest,
    Observation,
    Proposal,
    ProposalEvaluationSpec,
)
from ..artifacts.records import RequestAlignedEvaluationRecord
from ..evaluators.async_evaluator.artifacts import (
    CompletionGroup,
    EvaluationBatchHandle,
    EvaluationBatchResumeHandle,
)
from ..evaluators.async_evaluator.contracts import AsyncEvaluator
from ..evaluators.async_evaluator.sessions import (
    EvaluationBatchSession,
    ResumableBatchSession,
)
from ..evaluators.base import Evaluator
from ..outcomes import (
    EvaluationAttemptBatch,
    EvaluationOutcome,
    validate_outcome_refinement_alignment,
)
from ..problem import Problem
from ..spaces import CandidateEquality
from ..typevars import CandidateT

BoundaryT = TypeVar("BoundaryT")
CompletionT = TypeVar("CompletionT")
EvaluationT = TypeVar("EvaluationT")
StudyEvaluationRecordT = TypeVar(
    "StudyEvaluationRecordT",
    bound=RequestAlignedEvaluationRecord,
)


@runtime_checkable
class AttemptBatchEvaluator(Protocol[BoundaryT, CandidateT, StudyEvaluationRecordT]):
    """Evaluator capability that can return request-aligned attempt batches."""

    def evaluate_attempts(
        self,
        problem: Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]:
        """Execute requests and preserve success/failure attempt slots."""
        ...


@runtime_checkable
class AttemptBatchSessionEvaluator(
    Protocol[BoundaryT, CandidateT, StudyEvaluationRecordT]
):
    """Async evaluator capability that can stream attempt batches by slot."""

    def open_attempt_session(
        self,
        problem: Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationBatchSession[
        EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]
    ]:
        """Open a session that emits one-slot attempt batches."""
        ...


@runtime_checkable
class ResumableAttemptBatchSessionEvaluator(
    Protocol[CandidateT, StudyEvaluationRecordT]
):
    """Async evaluator capability that resumes native attempt-batch sessions."""

    def resume_attempt_session(
        self,
        handle: EvaluationBatchResumeHandle,
    ) -> EvaluationBatchSession[
        EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]
    ]:
        """Resume a session that emits one-slot attempt batches."""
        ...


def supports_attempt_batches(
    evaluator: Evaluator[
        Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT],
    ],
) -> TypeGuard[AttemptBatchEvaluator[BoundaryT, CandidateT, StudyEvaluationRecordT]]:
    """Return whether ``evaluator`` exposes dense attempt-batch evaluation."""
    return isinstance(evaluator, AttemptBatchEvaluator)


def supports_attempt_batch_sessions(
    evaluator: AsyncEvaluator[
        Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT],
    ],
) -> TypeGuard[
    AttemptBatchSessionEvaluator[BoundaryT, CandidateT, StudyEvaluationRecordT]
]:
    """Return whether ``evaluator`` exposes attempt-aware async sessions."""
    return isinstance(evaluator, AttemptBatchSessionEvaluator)


def supports_resumable_attempt_batch_sessions(
    evaluator: AsyncEvaluator[
        Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT],
    ],
) -> TypeGuard[
    ResumableAttemptBatchSessionEvaluator[CandidateT, StudyEvaluationRecordT]
]:
    """Return whether ``evaluator`` can resume native attempt-batch sessions."""
    return isinstance(evaluator, ResumableAttemptBatchSessionEvaluator)


class OutcomeToAttemptBatchSession(
    EvaluationBatchSession[EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]],
    Generic[CandidateT, StudyEvaluationRecordT],
):
    """Adapt an outcome-only async session into a success-only attempt session."""

    requests: tuple[EvaluationRequest[CandidateT], ...]
    outcome_session: EvaluationBatchSession[
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT]
    ]
    candidate_equal: CandidateEquality[CandidateT]

    def __init__(
        self,
        *,
        requests: tuple[EvaluationRequest[CandidateT], ...],
        outcome_session: EvaluationBatchSession[
            EvaluationOutcome[CandidateT, StudyEvaluationRecordT]
        ],
        candidate_equal: CandidateEquality[CandidateT],
    ) -> None:
        """Create one outcome-session adapter."""
        self.requests = requests
        self.outcome_session = outcome_session
        self.candidate_equal = candidate_equal

    @property
    @override
    def handle(self) -> EvaluationBatchHandle:
        """Return the wrapped evaluator-owned batch handle."""
        return self.outcome_session.handle

    @override
    def poll(
        self,
    ) -> Sequence[
        CompletionGroup[EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]]
    ]:
        """Return newly completed success-only attempt groups."""
        return tuple(
            self._attempt_completion_group(completion_group)
            for completion_group in self.outcome_session.poll()
        )

    @override
    def wait(
        self,
        *,
        timeout: float | None = None,
    ) -> Sequence[
        CompletionGroup[EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]]
    ]:
        """Wait for newly completed success-only attempt groups."""
        return tuple(
            self._attempt_completion_group(completion_group)
            for completion_group in self.outcome_session.wait(timeout=timeout)
        )

    @override
    def cancel(self) -> None:
        """Cancel the wrapped session."""
        self.outcome_session.cancel()

    def _attempt_completion_group(
        self,
        completion_group: CompletionGroup[
            EvaluationOutcome[CandidateT, StudyEvaluationRecordT]
        ],
    ) -> CompletionGroup[EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]]:
        end_index = completion_group.start_index + len(completion_group.outcomes)
        if end_index > len(self.requests):
            msg = "completion group exceeds logical batch bounds"
            raise ValueError(msg)

        group_requests = self.requests[completion_group.start_index : end_index]
        validate_aligned_outcomes(
            group_requests,
            completion_group.outcomes,
            candidate_equal=self.candidate_equal,
        )
        return CompletionGroup(
            start_index=completion_group.start_index,
            outcomes=tuple(
                EvaluationAttemptBatch(
                    requests=(request,),
                    outcomes=(outcome,),
                )
                for request, outcome in zip(
                    group_requests,
                    completion_group.outcomes,
                    strict=True,
                )
            ),
        )


class ResumableOutcomeToAttemptBatchSession(
    OutcomeToAttemptBatchSession[CandidateT, StudyEvaluationRecordT],
    ResumableBatchSession[EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]],
    Generic[CandidateT, StudyEvaluationRecordT],
):
    """Adapt a resumable outcome-only session into a resumable attempt session."""

    def __init__(
        self,
        *,
        requests: tuple[EvaluationRequest[CandidateT], ...],
        outcome_session: ResumableBatchSession[
            EvaluationOutcome[CandidateT, StudyEvaluationRecordT]
        ],
        candidate_equal: CandidateEquality[CandidateT],
    ) -> None:
        """Create one resumable outcome-session adapter."""
        super().__init__(
            requests=requests,
            outcome_session=outcome_session,
            candidate_equal=candidate_equal,
        )

    @override
    def suspend(self) -> EvaluationBatchResumeHandle:
        """Suspend the wrapped outcome session."""
        return require_resumable_batch_session(self.outcome_session).suspend()


def require_resumable_batch_session(
    batch_session: EvaluationBatchSession[EvaluationT],
) -> ResumableBatchSession[EvaluationT]:
    """Return ``batch_session`` when it advertises resumable capability."""
    if not isinstance(batch_session, ResumableBatchSession):
        msg = "resumable async evaluator returned a non-resumable batch session"
        raise TypeError(msg)
    return batch_session


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
    ordered_completions: list[CompletionT | None],
    completion_group: CompletionGroup[CompletionT],
    *,
    request_count: int,
) -> int:
    """Store one completion group and return its newly covered size.

    Parameters
    ----------
    ordered_completions : list[CompletionT | None]
        Mutable ordered completion buffer for an exact-async batch.
    completion_group : CompletionGroup[CompletionT]
        Completion group returned by the async evaluator.
    request_count : int
        Logical request count for the batch.

    Returns
    -------
    int
        Number of completions written into ``ordered_completions``.

    Raises
    ------
    ValueError
        If ``completion_group`` exceeds batch bounds or overlaps an already
        stored completion.
    """
    end_index = completion_group.start_index + len(completion_group.outcomes)
    if end_index > request_count:
        msg = "completion group exceeds logical batch bounds"
        raise ValueError(msg)

    for offset, completion in enumerate(completion_group.outcomes):
        target_index = completion_group.start_index + offset
        if ordered_completions[target_index] is not None:
            msg = "completion groups must not overlap"
            raise ValueError(msg)

        ordered_completions[target_index] = completion

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


def finalize_ordered_attempts(
    ordered_attempts: list[
        EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT] | None
    ],
) -> EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]:
    """Return one fully populated ordered attempt batch.

    Parameters
    ----------
    ordered_attempts : list[EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT] | None]
        Mutable ordered attempt-slot buffer for an exact-async batch.

    Returns
    -------
    EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]
        Dense attempt batch preserving logical request order.

    Raises
    ------
    RuntimeError
        If any attempt slot is still missing.
    """
    if any(attempt is None for attempt in ordered_attempts):
        msg = "exact_async completion left missing request attempts"
        raise RuntimeError(msg)

    return EvaluationAttemptBatch[
        CandidateT,
        StudyEvaluationRecordT,
    ].from_single_request_attempts(
        tuple(attempt for attempt in ordered_attempts if attempt is not None),
    )


def validate_aligned_outcomes(
    requests: tuple[EvaluationRequest[CandidateT], ...],
    outcomes: tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None = None,
) -> None:
    """Reject evaluator outcomes that do not align with input requests.

    Parameters
    ----------
    requests : tuple[EvaluationRequest[CandidateT], ...]
        Evaluation requests submitted to the evaluator.
    outcomes : tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]
        Evaluation outcomes returned by the evaluator.
    candidate_equal : CandidateEquality[CandidateT] | None, optional
        Explicit candidate equality predicate used to validate refinement
        alignment. When absent, strict scalar Python equality is used.

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
        validate_outcome_refinement_alignment(
            outcome,
            candidate_equal=candidate_equal,
        )


def validate_aligned_attempts(
    requests: tuple[EvaluationRequest[CandidateT], ...],
    attempts: EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None = None,
) -> None:
    """Reject attempt batches that do not align with input requests.

    Parameters
    ----------
    requests : tuple[EvaluationRequest[CandidateT], ...]
        Canonical request slots submitted to the evaluator or kernel.
    attempts : EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]
        Dense attempt batch returned for those slots.
    candidate_equal : CandidateEquality[CandidateT] | None, optional
        Explicit candidate equality predicate used to validate refinement
        alignment for successful attempts.

    Raises
    ------
    ValueError
        If the attempt batch does not carry logically equivalent requests in the
        same slots.
    """
    if len(attempts.requests) != len(requests):
        msg = "attempt batch must contain exactly one slot per request"
        raise ValueError(msg)

    for expected_request, actual_request in zip(
        requests,
        attempts.requests,
        strict=True,
    ):
        if not _requests_match(
            actual_request,
            expected_request,
            candidate_equal=candidate_equal,
        ):
            msg = "attempt batch requests must align with input request order"
            raise ValueError(msg)

    for outcome in attempts.outcomes:
        validate_outcome_refinement_alignment(
            outcome,
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


def _candidates_match(
    left_candidate: CandidateT,
    right_candidate: CandidateT,
    *,
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> bool:
    if candidate_equal is None:
        candidates_match = left_candidate == right_candidate
    else:
        candidates_match = candidate_equal(left_candidate, right_candidate)

    if type(candidates_match) is not bool:
        msg = "candidate equality must return bool"
        raise TypeError(msg)

    return candidates_match


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
