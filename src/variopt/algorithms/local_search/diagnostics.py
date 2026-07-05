"""Diagnostic projection helpers shared by local-search kernels."""

from collections.abc import Sequence
from typing import TypeVar

from ...artifacts import EvaluationAttemptBatch, EvaluationFailure, KernelDiagnostics
from ...typevars import CandidateT

PayloadT = TypeVar("PayloadT")


def _merge_failed_attempts(
    failed_attempts: Sequence[EvaluationAttemptBatch[CandidateT, PayloadT]],
) -> EvaluationAttemptBatch[CandidateT, PayloadT]:
    """Merge episode-local failed attempts and reject non-failure slots."""
    if len(failed_attempts) == 0:
        return EvaluationAttemptBatch(attempts=())

    episode_failures = EvaluationAttemptBatch[
        CandidateT,
        PayloadT,
    ].from_single_request_attempts(tuple(failed_attempts))
    if len(episode_failures.failures) != episode_failures.attempt_count:
        msg = "failed_attempts must contain only failed one-request attempts"
        raise ValueError(msg)

    return episode_failures


def diagnostics_with_failed_attempts(
    diagnostics: KernelDiagnostics | None,
    failed_attempts: Sequence[EvaluationAttemptBatch[CandidateT, PayloadT]],
) -> KernelDiagnostics | None:
    """Annotate kernel diagnostics with inner failed-attempt accounting.

    Parameters
    ----------
    diagnostics : KernelDiagnostics | None
        Base diagnostics produced by the local-search episode.
    failed_attempts : Sequence[EvaluationAttemptBatch[CandidateT, PayloadT]]
        Episode-local failed one-request attempts. These are summarized in
        diagnostics rather than surfaced as additional top-level Study slots.

    Returns
    -------
    KernelDiagnostics | None
        Diagnostics with failed-attempt counts when ``diagnostics`` is present.
    """
    if diagnostics is None:
        return None

    episode_failures = _merge_failed_attempts(failed_attempts)
    failed_attempt_count = episode_failures.attempt_count
    failed_evaluation_count = episode_failures.evaluation_count
    if failed_attempt_count == 0 and failed_evaluation_count == 0:
        return diagnostics

    return diagnostics.with_failed_attempts(
        failed_attempt_count=failed_attempt_count,
        failed_evaluation_count=failed_evaluation_count,
    )


def top_level_failure_from_failed_attempts(
    failed_attempts: Sequence[EvaluationAttemptBatch[CandidateT, PayloadT]],
) -> EvaluationAttemptBatch[CandidateT, PayloadT]:
    """Project failed local-search episode attempts into one top-level slot.

    Parameters
    ----------
    failed_attempts : Sequence[EvaluationAttemptBatch[CandidateT, PayloadT]]
        Episode-local failed one-request attempts accumulated before the kernel
        gives up without a successful evaluation.

    Returns
    -------
    EvaluationAttemptBatch[CandidateT, PayloadT]
        Empty when no attempts exist, otherwise one representative top-level
        failure carrying the total failed evaluation cost.
    """
    episode_failures = _merge_failed_attempts(failed_attempts)
    if episode_failures.attempt_count == 0:
        return episode_failures

    representative_failure = episode_failures.failures[0]
    return EvaluationAttemptBatch[
        CandidateT,
        PayloadT,
    ](
        attempts=(
            EvaluationFailure(
                request=representative_failure.request,
                exception=representative_failure.exception,
                evaluation_count=episode_failures.evaluation_count,
            ),
        ),
    )
