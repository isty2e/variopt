"""Shared helpers for study orchestration."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeGuard, runtime_checkable

from typing_extensions import TypeVar

from ..artifacts import (
    EvaluationAttemptBatch,
    EvaluationFailure,
    EvaluationRequest,
    EvaluationSuccess,
    Observation,
    ObservationPayload,
    Proposal,
    ProposalEvaluationSpec,
    RunReport,
    Trace,
    TraceEvent,
)
from ..artifacts.attempts import MaterializableEvaluationPayload
from ..artifacts.records import RequestAlignedEvaluationRecord
from ..evaluators.async_evaluator.artifacts import (
    CompletionGroup,
    EvaluationBatchResumeHandle,
)
from ..evaluators.async_evaluator.sessions import EvaluationBatchSession
from ..execution import ExecutionResources
from ..problem import Problem
from ..spaces import CandidateEquality
from ..spaces.equality import scalar_candidate_equality
from ..typevars import CandidateT, RunMethodStateT

BoundaryT = TypeVar("BoundaryT")
CompletionT = TypeVar("CompletionT")
StudyEvaluationPayload: TypeAlias = MaterializableEvaluationPayload[object]
StudyPayloadT = TypeVar(
    "StudyPayloadT",
    bound=StudyEvaluationPayload,
)
StudyRecordT = TypeVar(
    "StudyRecordT",
    bound=RequestAlignedEvaluationRecord[object],
)


@dataclass(frozen=True, slots=True)
class CheckpointSafeRunSnapshot(Generic[RunMethodStateT]):
    """Checkpoint-safe cut point into append-only run histories.

    Parameters
    ----------
    success_count : int
        Number of successful attempts included in the snapshot.
    failure_count : int
        Number of failed attempts included in the snapshot.
    trace_event_count : int
        Number of trace events included in the snapshot.
    evaluation_count : int
        Logical evaluation count aligned with the checkpoint-safe state.
    state : RunMethodStateT
        Run-method state at the checkpoint-safe boundary.
    """

    success_count: int
    failure_count: int
    trace_event_count: int
    evaluation_count: int
    state: RunMethodStateT

    def __post_init__(self) -> None:
        """Validate non-negative checkpoint cut points."""
        if self.success_count < 0:
            msg = "success_count must be non-negative"
            raise ValueError(msg)
        if self.failure_count < 0:
            msg = "failure_count must be non-negative"
            raise ValueError(msg)
        if self.trace_event_count < 0:
            msg = "trace_event_count must be non-negative"
            raise ValueError(msg)
        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)

    def to_report(
        self,
        *,
        successes: Sequence[EvaluationSuccess[CandidateT, StudyRecordT]],
        failures: Sequence[EvaluationFailure[CandidateT]],
        trace_events: Sequence[TraceEvent],
        candidate_equal: CandidateEquality[CandidateT],
    ) -> RunReport[CandidateT, StudyRecordT]:
        """Materialize a report from the captured history cut point.

        Parameters
        ----------
        successes : Sequence[EvaluationSuccess[CandidateT, StudyRecordT]]
            Append-only success history available at materialization time.
        failures : Sequence[EvaluationFailure[CandidateT]]
            Append-only failure history available at materialization time.
        trace_events : Sequence[TraceEvent]
            Append-only trace history available at materialization time.
        candidate_equal : CandidateEquality[CandidateT]
            Candidate equality predicate used to validate refinement alignment.

        Returns
        -------
        RunReport[CandidateT, StudyRecordT]
            Report projection aligned with this checkpoint-safe cut point.
        """
        return RunReport[CandidateT, StudyRecordT].from_successes(
            successes=tuple(successes[: self.success_count]),
            evaluation_count=self.evaluation_count,
            trace=Trace(events=tuple(trace_events[: self.trace_event_count])),
            failures=tuple(failures[: self.failure_count]),
            candidate_equal=candidate_equal,
        )


@runtime_checkable
class AttemptBatchEvaluator(Protocol[BoundaryT, CandidateT, StudyPayloadT]):
    """Evaluator capability that returns request-owned payload attempts."""

    def execution_resources(self) -> ExecutionResources:
        """Return execution resources owned by this evaluator."""
        ...

    def evaluate_attempts(
        self,
        problem: Problem[BoundaryT, CandidateT, StudyPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationAttemptBatch[CandidateT, StudyPayloadT]:
        """Execute requests and preserve success/failure attempt slots."""
        ...


@runtime_checkable
class AttemptBatchSessionEvaluator(
    Protocol[BoundaryT, CandidateT, StudyPayloadT]
):
    """Async evaluator capability that streams payload attempts by slot."""

    def execution_resources(self) -> ExecutionResources:
        """Return execution resources owned by this evaluator."""
        ...

    def open_attempt_session(
        self,
        problem: Problem[BoundaryT, CandidateT, StudyPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationBatchSession[
        EvaluationAttemptBatch[CandidateT, StudyPayloadT]
    ]:
        """Open a session that emits one-slot attempt batches."""
        ...


@runtime_checkable
class ResumableAttemptBatchSessionEvaluator(
    Protocol[BoundaryT, CandidateT, StudyPayloadT]
):
    """Async evaluator capability that opens and resumes attempt sessions."""

    def execution_resources(self) -> ExecutionResources:
        """Return execution resources owned by this evaluator."""
        ...

    def open_attempt_session(
        self,
        problem: Problem[BoundaryT, CandidateT, StudyPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationBatchSession[
        EvaluationAttemptBatch[CandidateT, StudyPayloadT]
    ]:
        """Open a session that emits one-slot attempt batches."""
        ...

    def resume_attempt_session(
        self,
        handle: EvaluationBatchResumeHandle,
    ) -> EvaluationBatchSession[
        EvaluationAttemptBatch[CandidateT, StudyPayloadT]
    ]:
        """Resume a session that emits one-slot attempt batches."""
        ...


StudyEvaluator: TypeAlias = (
    AttemptBatchEvaluator[BoundaryT, CandidateT, StudyPayloadT]
    | AttemptBatchSessionEvaluator[BoundaryT, CandidateT, StudyPayloadT]
)


def supports_attempt_batches(
    evaluator: StudyEvaluator[BoundaryT, CandidateT, StudyPayloadT],
) -> TypeGuard[AttemptBatchEvaluator[BoundaryT, CandidateT, StudyPayloadT]]:
    """Return whether ``evaluator`` exposes dense attempt-batch evaluation."""
    return isinstance(evaluator, AttemptBatchEvaluator)


def supports_attempt_batch_sessions(
    evaluator: StudyEvaluator[BoundaryT, CandidateT, StudyPayloadT],
) -> TypeGuard[
    AttemptBatchSessionEvaluator[BoundaryT, CandidateT, StudyPayloadT]
]:
    """Return whether ``evaluator`` exposes attempt-aware async sessions."""
    return isinstance(evaluator, AttemptBatchSessionEvaluator)


def supports_resumable_attempt_batch_sessions(
    evaluator: StudyEvaluator[BoundaryT, CandidateT, StudyPayloadT],
) -> TypeGuard[
    ResumableAttemptBatchSessionEvaluator[BoundaryT, CandidateT, StudyPayloadT]
]:
    """Return whether ``evaluator`` can resume native attempt-batch sessions."""
    return isinstance(evaluator, ResumableAttemptBatchSessionEvaluator)


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


def finalize_ordered_attempts(
    ordered_attempts: list[
        EvaluationAttemptBatch[CandidateT, StudyPayloadT] | None
    ],
) -> EvaluationAttemptBatch[CandidateT, StudyPayloadT]:
    """Return one fully populated ordered attempt batch.

    Parameters
    ----------
    ordered_attempts : list[EvaluationAttemptBatch[CandidateT, StudyPayloadT] | None]
        Mutable ordered attempt-slot buffer for an exact-async batch.

    Returns
    -------
    EvaluationAttemptBatch[CandidateT, StudyPayloadT]
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
        StudyPayloadT,
    ].from_single_request_attempts(
        tuple(attempt for attempt in ordered_attempts if attempt is not None),
    )


def validate_aligned_attempts(
    requests: tuple[EvaluationRequest[CandidateT], ...],
    attempts: EvaluationAttemptBatch[CandidateT, StudyPayloadT],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None = None,
) -> None:
    """Reject attempt batches that do not align with input requests.

    Parameters
    ----------
    requests : tuple[EvaluationRequest[CandidateT], ...]
        Canonical request slots submitted to the evaluator or kernel.
    attempts : EvaluationAttemptBatch[CandidateT, StudyPayloadT]
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


def validate_materialized_attempts(
    source_attempts: EvaluationAttemptBatch[CandidateT, StudyPayloadT],
    materialized_attempts: EvaluationAttemptBatch[CandidateT, StudyRecordT],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None = None,
) -> None:
    """Reject materialized batches that mutate attempt-slot semantics.

    Parameters
    ----------
    source_attempts : EvaluationAttemptBatch[CandidateT, StudyPayloadT]
        Pre-materialization evaluator/kernel attempts.
    materialized_attempts : EvaluationAttemptBatch[CandidateT, StudyRecordT]
        Post-materialization feedback attempts.
    candidate_equal : CandidateEquality[CandidateT] | None, optional
        Explicit candidate equality predicate used to revalidate refinement and
        request-aligned record payloads.

    Raises
    ------
    TypeError
        If a materialized success payload is not request-aligned.
    ValueError
        If the materializer drops, reorders, flips, or rewrites attempt slots or
        protected attempt metadata.
    """
    if len(materialized_attempts.attempts) != len(source_attempts.attempts):
        msg = "materialized attempt batch must preserve attempt slot count"
        raise ValueError(msg)

    for source_attempt, materialized_attempt in zip(
        source_attempts.attempts,
        materialized_attempts.attempts,
        strict=True,
    ):
        if type(source_attempt) is EvaluationFailure:
            _validate_materialized_failure_slot(source_attempt, materialized_attempt)
            continue

        if type(source_attempt) is EvaluationSuccess:
            _validate_materialized_success_slot(
                source_attempt,
                materialized_attempt,
                candidate_equal=candidate_equal,
            )
            continue

        msg = "source attempt batch contains an unknown attempt variant"
        raise TypeError(msg)


def _validate_materialized_failure_slot(
    source_failure: EvaluationFailure[CandidateT],
    materialized_attempt: EvaluationSuccess[CandidateT, StudyRecordT]
    | EvaluationFailure[CandidateT],
) -> None:
    if type(materialized_attempt) is not EvaluationFailure:
        msg = "materialized attempt batch must preserve failure slots"
        raise ValueError(msg)

    if materialized_attempt.request is not source_failure.request:
        msg = "materialized failure request must preserve source request identity"
        raise ValueError(msg)

    if (
        materialized_attempt.exception != source_failure.exception
        or materialized_attempt.evaluation_count != source_failure.evaluation_count
    ):
        msg = (
            "materialized failure metadata must preserve exception and "
            "evaluation_count"
        )
        raise ValueError(msg)


def _validate_materialized_success_slot(
    source_success: EvaluationSuccess[CandidateT, StudyPayloadT],
    materialized_attempt: EvaluationSuccess[CandidateT, StudyRecordT]
    | EvaluationFailure[CandidateT],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> None:
    if type(materialized_attempt) is not EvaluationSuccess:
        msg = "materialized attempt batch must preserve success slots"
        raise ValueError(msg)

    if materialized_attempt.request is not source_success.request:
        msg = "materialized success request must preserve source request identity"
        raise ValueError(msg)

    if materialized_attempt.evaluation_count != source_success.evaluation_count:
        msg = "materialized success metadata must preserve evaluation_count"
        raise ValueError(msg)

    if materialized_attempt.refinement is not source_success.refinement:
        msg = "materialized success metadata must preserve refinement"
        raise ValueError(msg)

    if materialized_attempt.kernel_diagnostics is not source_success.kernel_diagnostics:
        msg = "materialized success metadata must preserve kernel_diagnostics"
        raise ValueError(msg)

    if not _is_request_aligned_record_payload(materialized_attempt.payload):
        msg = "materialized success payload must be a request-aligned record"
        raise TypeError(msg)

    _ = EvaluationSuccess(
        request=materialized_attempt.request,
        payload=materialized_attempt.payload,
        evaluation_count=materialized_attempt.evaluation_count,
        refinement=materialized_attempt.refinement,
        kernel_diagnostics=materialized_attempt.kernel_diagnostics,
        candidate_equal=candidate_equal,
    )


def _is_request_aligned_record_payload(
    payload: object,
) -> TypeGuard[RequestAlignedEvaluationRecord[object]]:
    if not isinstance(payload, RequestAlignedEvaluationRecord):
        return False

    return type(payload.request) is EvaluationRequest


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
    success: EvaluationSuccess[CandidateT, StudyPayloadT],
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

    if success.request.proposal_evaluation_spec != expected_request.proposal_evaluation_spec:
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

    return _require_bool_candidate_match(candidate_equal(left_candidate, right_candidate))


def _require_bool_candidate_match(candidates_match: object) -> bool:
    """Return a candidate equality result after strict bool validation."""
    if type(candidates_match) is not bool:
        msg = "candidate equality must return bool"
        raise TypeError(msg)

    return candidates_match


def trace_value_for_records(
    records: tuple[StudyRecordT, ...],
) -> float | None:
    """Return one scalar trace value when the batch records are observations.

    Parameters
    ----------
    records : tuple[StudyRecordT, ...]
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
        if isinstance(record, ObservationPayload):
            score = record.score
            value = record.value
        elif isinstance(record, Observation):
            score = record.score
            value = record.value
        else:
            return None
        if best_batch_score is None or score < best_batch_score:
            best_batch_score = score
            best_batch_value = value

    return best_batch_value
