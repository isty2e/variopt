"""Exact-async batch execution and study-session orchestration helpers."""

from typing_extensions import TypeVar

from ...artifacts import EvaluationAttemptBatch, EvaluationRequest
from ...artifacts.records import RequestAlignedEvaluationRecord
from ...evaluators.async_evaluator.contracts import AsyncEvaluator
from ...evaluators.async_evaluator.sessions import ResumableBatchSession
from ...kernel import DirectKernel
from ...outcomes import EvaluationOutcome
from ...problem import Problem
from ...typevars import CandidateT, RunMethodStateT
from ..common import (
    StudyEvaluationRecordT,
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
    async_evaluator: AsyncEvaluator[
        Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord],
    ],
    problem: Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
    requests: tuple[EvaluationRequest[CandidateT], ...],
) -> EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]:
    """Execute one request batch through the exact-async evaluator path.

    Parameters
    ----------
    async_evaluator : AsyncEvaluator[Problem[BoundaryT, CandidateT, StudyEvaluationRecordT], EvaluationRequest[CandidateT], EvaluationOutcome[CandidateT, StudyEvaluationRecordT]]
        Async evaluator used to execute the request batch.
    problem : Problem[BoundaryT, CandidateT, StudyEvaluationRecordT]
        Problem passed to the evaluator.
    requests : tuple[EvaluationRequest[CandidateT], ...]
        Evaluation requests to execute.

    Returns
    -------
    EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT]
        Dense attempt batch aligned with ``requests``.
    """
    if supports_attempt_batches(async_evaluator):
        return async_evaluator.evaluate_attempts(problem, requests)

    if supports_attempt_batch_sessions(async_evaluator):
        attempt_session = async_evaluator.open_attempt_session(problem, requests)
        ordered_attempts: list[
            EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT] | None
        ] = [None] * attempt_session.handle.request_count
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

    msg = "exact_async evaluator must expose attempt-batch evaluation"
    raise TypeError(msg)


def open_exact_async_step_session(
    study: StudyExactAsyncOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ],
    state: RunMethodStateT,
    batch_size: int = 1,
) -> StudyExactAsyncStepSession[
    BoundaryT,
    CandidateT,
    RunMethodStateT,
    StudyEvaluationRecordT,
]:
    """Open one resumable exact-async study step before tell assimilation.

    Parameters
    ----------
    study : StudyExactAsyncOwner[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
        Study-like owner exposing exact-async orchestration state.
    state : RunMethodStateT
        Run-method state to advance.
    batch_size : int, default=1
        Maximum number of proposals to request.

    Returns
    -------
    StudyExactAsyncStepSession[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
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
    resumable_evaluator = require_resumable_async_evaluator(study)
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
    if supports_attempt_batch_sessions(resumable_evaluator):
        batch_session = resumable_evaluator.open_attempt_session(
            study.problem,
            requests,
        )
    else:
        msg = "resumable exact_async evaluator must expose attempt-batch sessions"
        raise TypeError(msg)
    if not isinstance(batch_session, ResumableBatchSession):
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
        StudyEvaluationRecordT,
    ],
    handle: StudyExactAsyncStepResumeHandle[
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ],
) -> StudyExactAsyncStepSession[
    BoundaryT,
    CandidateT,
    RunMethodStateT,
    StudyEvaluationRecordT,
]:
    """Resume one suspended exact-async study step session.

    Parameters
    ----------
    study : StudyExactAsyncOwner[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
        Study-like owner exposing exact-async orchestration state.
    handle : StudyExactAsyncStepResumeHandle[CandidateT, RunMethodStateT, StudyEvaluationRecordT]
        Suspended step handle to resume.

    Returns
    -------
    StudyExactAsyncStepSession[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
        Resumed exact-async step session.

    Raises
    ------
    ValueError
        If the study is not using :class:`DirectKernel`.
    """
    resumable_evaluator = require_resumable_async_evaluator(study)
    if not isinstance(study.kernel, DirectKernel):
        msg = (
            "study-level resumable exact_async orchestration currently "
            "requires DirectKernel"
        )
        raise ValueError(msg)

    if supports_resumable_attempt_batch_sessions(resumable_evaluator):
        batch_session = resumable_evaluator.resume_attempt_session(
            handle.evaluator_handle,
        )
    else:
        msg = "resumable exact_async evaluator must resume attempt-batch sessions"
        raise TypeError(msg)
    return StudyExactAsyncStepSession(
        study=study,
        requests=handle.requests,
        post_ask_state=handle.post_ask_state,
        batch_session=batch_session,
        candidate_equal=study.problem.space.candidates_equal,
        ordered_attempts=list(handle.ordered_attempts),
    )
