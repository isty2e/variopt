"""Tests for local-search diagnostic projection helpers."""

import pytest

from variopt import EvaluationAttemptBatch, EvaluationExceptionSnapshot
from variopt.algorithms.local_search.diagnostics import diagnostics_with_failed_attempts
from variopt.artifacts import (
    EvaluationFailure,
    EvaluationRequest,
    EvaluationSuccess,
    KernelDiagnostics,
    ObservationPayload,
    Proposal,
)


def test_diagnostics_with_failed_attempts_uses_fallback_when_base_is_missing() -> None:
    """Inner failed-attempt counts should survive even without base diagnostics."""
    request: EvaluationRequest[int] = EvaluationRequest(
        proposal=Proposal(candidate=4, proposal_id="p-1"),
    )
    failure = EvaluationFailure[int](
        request=request,
        exception=EvaluationExceptionSnapshot.from_exception(ValueError("bad trial")),
    )
    failed_attempt = EvaluationAttemptBatch[int](attempts=(failure,))
    fallback = KernelDiagnostics(
        backend="structured.local_search",
        method="test",
    )

    diagnostics = diagnostics_with_failed_attempts(
        None,
        (failed_attempt,),
        fallback_diagnostics=fallback,
    )

    assert diagnostics is not None
    assert diagnostics.backend == "structured.local_search"
    assert diagnostics.method == "test"
    assert diagnostics.failed_attempt_count == 1
    assert diagnostics.failed_evaluation_count == 1


def test_diagnostics_with_failed_attempts_omits_empty_fallback_diagnostics() -> None:
    """Do not create diagnostics objects when there is no diagnostic signal."""
    fallback = KernelDiagnostics(backend="structured.local_search")

    diagnostics = diagnostics_with_failed_attempts(
        None,
        (),
        fallback_diagnostics=fallback,
    )

    assert diagnostics is None


def test_diagnostics_with_failed_attempts_keeps_absent_owner_without_fallback() -> None:
    """Do not invent a diagnostics owner outside an opted-in kernel boundary."""
    request: EvaluationRequest[int] = EvaluationRequest(
        proposal=Proposal(candidate=4, proposal_id="p-1"),
    )
    failure = EvaluationFailure[int].from_exception(
        request=request,
        exception=ValueError("bad trial"),
    )
    failed_attempt = EvaluationAttemptBatch[int](attempts=(failure,))

    diagnostics = diagnostics_with_failed_attempts(None, (failed_attempt,))

    assert diagnostics is None


def test_diagnostics_with_failed_attempts_prefers_base_over_fallback() -> None:
    """Existing kernel diagnostics own identity; fallback only fills absence."""
    request: EvaluationRequest[int] = EvaluationRequest(
        proposal=Proposal(candidate=4, proposal_id="p-1"),
    )
    failure = EvaluationFailure[int].from_exception(
        request=request,
        exception=ValueError("bad trial"),
    )
    failed_attempt = EvaluationAttemptBatch[int](attempts=(failure,))
    base = KernelDiagnostics(backend="base", method="real")
    fallback = KernelDiagnostics(backend="fallback", method="synthetic")

    diagnostics = diagnostics_with_failed_attempts(
        base,
        (failed_attempt,),
        fallback_diagnostics=fallback,
    )

    assert diagnostics is not None
    assert diagnostics.backend == "base"
    assert diagnostics.method == "real"
    assert diagnostics.failed_attempt_count == 1
    assert diagnostics.failed_evaluation_count == 1


def test_diagnostics_with_failed_attempts_preserves_zero_cost_failure_signal() -> None:
    """A zero-cost failure is still a failed-attempt diagnostic event."""
    request: EvaluationRequest[int] = EvaluationRequest(
        proposal=Proposal(candidate=4, proposal_id="p-1"),
    )
    failure = EvaluationFailure[int](
        request=request,
        exception=EvaluationExceptionSnapshot.from_exception(ValueError("bad trial")),
        evaluation_count=0,
    )
    failed_attempt = EvaluationAttemptBatch[int](attempts=(failure,))

    diagnostics = diagnostics_with_failed_attempts(
        None,
        (failed_attempt,),
        fallback_diagnostics=KernelDiagnostics(backend="structured.local_search"),
    )

    assert diagnostics is not None
    assert diagnostics.failed_attempt_count == 1
    assert diagnostics.failed_evaluation_count == 0


def test_diagnostics_with_failed_attempts_sums_multiple_failure_batches() -> None:
    """Attempt count and logical evaluation cost are accumulated separately."""
    first_request: EvaluationRequest[int] = EvaluationRequest(
        proposal=Proposal(candidate=4, proposal_id="p-1"),
    )
    second_request: EvaluationRequest[int] = EvaluationRequest(
        proposal=Proposal(candidate=5, proposal_id="p-2"),
    )
    first_failure = EvaluationFailure[int].from_exception(
        request=first_request,
        exception=ValueError("bad trial"),
        evaluation_count=2,
    )
    second_failure = EvaluationFailure[int].from_exception(
        request=second_request,
        exception=RuntimeError("worse trial"),
        evaluation_count=3,
    )
    first_attempt = EvaluationAttemptBatch[int](attempts=(first_failure,))
    second_attempt = EvaluationAttemptBatch[int](attempts=(second_failure,))

    diagnostics = diagnostics_with_failed_attempts(
        KernelDiagnostics(backend="base"),
        (first_attempt, second_attempt),
    )

    assert diagnostics is not None
    assert diagnostics.failed_attempt_count == 2
    assert diagnostics.failed_evaluation_count == 5


def test_diagnostics_with_failed_attempts_rejects_success_slots() -> None:
    """The helper only summarizes failed inner attempts, never mixed slots."""
    request: EvaluationRequest[int] = EvaluationRequest(
        proposal=Proposal(candidate=4, proposal_id="p-1"),
    )
    success = EvaluationSuccess[int, ObservationPayload](
        request=request,
        payload=ObservationPayload(value=1.0, score=1.0),
    )
    mixed_attempt = EvaluationAttemptBatch[int](attempts=(success,))

    with pytest.raises(
        ValueError,
        match="failed_attempts must contain only failed one-request attempts",
    ):
        _ = diagnostics_with_failed_attempts(
            KernelDiagnostics(backend="base"),
            (mixed_attempt,),
        )
