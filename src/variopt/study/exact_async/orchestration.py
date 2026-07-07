"""Exact-async batch execution and study-session orchestration helpers."""

from typing_extensions import TypeVar

from ...artifacts import EvaluationAttemptBatch, EvaluationRequest
from ...evaluators.async_evaluator.sessions import ResumableBatchSession
from ...kernel import DirectKernel
from ...problem import Problem
from ...typevars import CandidateT, RunMethodStateT
from ..common import (
    StudyEvaluator,
    StudyPayloadT,
    StudyRecordT,
    build_evaluation_requests,
    finalize_ordered_attempts,
    store_completion_group,
    supports_attempt_batch_sessions,
    supports_attempt_batches,
    supports_resumable_attempt_batch_sessions,
)
from ..validation import require_resumable_async_evaluator
from .artifacts import StudyExactAsyncStepResumeHandle
from .contracts import StudyExactAsyncOwner
from .session import StudyExactAsyncStepSession
from .validation import validate_resumable_exact_async_request

BoundaryT = TypeVar("BoundaryT")


def evaluate_batch_exact_async(
    async_evaluator: StudyEvaluator[BoundaryT, CandidateT, StudyPayloadT],
    problem: Problem[BoundaryT, CandidateT, StudyPayloadT],
    requests: tuple[EvaluationRequest[CandidateT], ...],
) -> EvaluationAttemptBatch[CandidateT, StudyPayloadT]:
    """Execute one request batch through the exact-async evaluator path.

    Parameters
    ----------
    async_evaluator : StudyEvaluator[BoundaryT, CandidateT, StudyPayloadT]
        Evaluator exposing native attempt-batch capability for the request
        batch.
    problem : Problem[BoundaryT, CandidateT, StudyPayloadT]
        Problem passed to the evaluator.
    requests : tuple[EvaluationRequest[CandidateT], ...]
        Evaluation requests to execute.

    Returns
    -------
    EvaluationAttemptBatch[CandidateT, StudyPayloadT]
        Dense attempt batch aligned with ``requests``.
    """
    if supports_attempt_batch_sessions(async_evaluator):
        attempt_session = async_evaluator.open_attempt_session(problem, requests)
    elif supports_attempt_batches(async_evaluator):
        return async_evaluator.evaluate_attempts(problem, requests)
    else:
        msg = "exact_async evaluator must expose attempt-batch sessions"
        raise TypeError(msg)

    ordered_attempts: list[EvaluationAttemptBatch[CandidateT, StudyPayloadT] | None] = [
        None
    ] * attempt_session.handle.request_count
    completed_count = 0
    try:
        while completed_count < attempt_session.handle.request_count:
            attempt_completion_groups = tuple(attempt_session.wait())
            for attempt_completion_group in attempt_completion_groups:
                completed_count += store_completion_group(
                    ordered_attempts,
                    attempt_completion_group,
                    request_count=attempt_session.handle.request_count,
                )
    except BaseException:
        attempt_session.cancel()
        raise

    return finalize_ordered_attempts(ordered_attempts)


def open_exact_async_step_session(
    study: StudyExactAsyncOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    state: RunMethodStateT,
    batch_size: int = 1,
) -> StudyExactAsyncStepSession[
    BoundaryT,
    CandidateT,
    RunMethodStateT,
    StudyPayloadT,
    StudyRecordT,
]:
    """Open one resumable exact-async study step before tell assimilation.

    Parameters
    ----------
    study : StudyExactAsyncOwner[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Study-like owner exposing exact-async orchestration state.
    state : RunMethodStateT
        Run-method state to advance.
    batch_size : int, default=1
        Maximum number of proposals to request.

    Returns
    -------
    StudyExactAsyncStepSession[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Open resumable exact-async step session.

    Raises
    ------
    RuntimeError
        If the run method returns no proposals.
    ValueError
        If the run method returns more proposals than requested.
    TypeError
        If the evaluator does not return a resumable batch session.
    """
    validate_resumable_exact_async_request(
        study,
        state=state,
        batch_size=batch_size,
    )
    require_resumable_async_evaluator(study)
    proposals, post_ask_state = study.run_method.ask(state, batch_size=batch_size)
    if len(proposals) == 0:
        msg = "run_method returned no proposals"
        raise RuntimeError(msg)

    if len(proposals) > batch_size:
        msg = "run_method returned more proposals than requested"
        raise ValueError(msg)

    proposal_evaluation_specs = study.run_method.proposal_evaluation_specs(
        post_ask_state,
        proposals,
    )
    requests = build_evaluation_requests(
        proposals,
        proposal_evaluation_specs=proposal_evaluation_specs,
    )
    if not supports_resumable_attempt_batch_sessions(study.evaluator):
        msg = (
            "resumable exact_async evaluator must open and resume "
            "attempt-batch sessions"
        )
        raise TypeError(msg)
    batch_session = study.evaluator.open_attempt_session(
        study.problem,
        requests,
    )
    if not isinstance(batch_session, ResumableBatchSession):
        try:
            batch_session.cancel()
        except Exception:
            # Cleanup is best-effort; preserve the validation failure below.
            pass
        msg = "resumable async evaluator returned a non-resumable batch session"
        raise TypeError(msg)

    return StudyExactAsyncStepSession(
        study=study,
        requests=requests,
        post_ask_state=post_ask_state,
        batch_session=batch_session,
        candidate_equal=study.problem.space.candidates_equal,
    )


def resume_exact_async_step_session(
    study: StudyExactAsyncOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    handle: StudyExactAsyncStepResumeHandle[
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
    ],
) -> StudyExactAsyncStepSession[
    BoundaryT,
    CandidateT,
    RunMethodStateT,
    StudyPayloadT,
    StudyRecordT,
]:
    """Resume one suspended exact-async study step session.

    Parameters
    ----------
    study : StudyExactAsyncOwner[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Study-like owner exposing exact-async orchestration state.
    handle : StudyExactAsyncStepResumeHandle[CandidateT, RunMethodStateT, StudyPayloadT]
        Suspended step handle to resume.

    Returns
    -------
    StudyExactAsyncStepSession[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Resumed exact-async step session.

    Raises
    ------
    ValueError
        If the study is not using :class:`DirectKernel`.
    """
    require_resumable_async_evaluator(study)
    if not isinstance(study.kernel, DirectKernel):
        msg = (
            "study-level resumable exact_async orchestration currently "
            "requires DirectKernel"
        )
        raise ValueError(msg)

    if not supports_resumable_attempt_batch_sessions(study.evaluator):
        msg = (
            "resumable exact_async evaluator must open and resume "
            "attempt-batch sessions"
        )
        raise TypeError(msg)
    batch_session = study.evaluator.resume_attempt_session(
        handle.evaluator_handle,
    )
    return StudyExactAsyncStepSession(
        study=study,
        requests=handle.requests,
        post_ask_state=handle.post_ask_state,
        batch_session=batch_session,
        candidate_equal=study.problem.space.candidates_equal,
        ordered_attempts=list(handle.ordered_attempts),
    )
