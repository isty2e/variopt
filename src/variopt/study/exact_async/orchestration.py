"""Exact-async batch execution and study-session orchestration helpers."""

from typing_extensions import TypeVar

from ...artifacts import EvaluationRequest
from ...evaluators.async_evaluator.contracts import AsyncEvaluator
from ...evaluators.async_evaluator.sessions import ResumableBatchSession
from ...kernel import DirectKernel
from ...outcomes import EvaluationOutcome
from ...problem import Problem
from ...typevars import CandidateT, RunMethodStateT
from ..common import (
    StudyEvaluationRecordT,
    build_evaluation_requests,
    finalize_ordered_outcomes,
    store_completion_group,
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
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT],
    ],
    problem: Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
    requests: tuple[EvaluationRequest[CandidateT], ...],
) -> tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]:
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
    tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]
        Ordered evaluator outcomes aligned with ``requests``.
    """
    batch_session = async_evaluator.open_session(problem, requests)
    ordered_outcomes: list[
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT] | None
    ] = [None] * batch_session.handle.request_count
    completed_count = 0
    try:
        while completed_count < batch_session.handle.request_count:
            completion_groups = tuple(batch_session.wait())
            for completion_group in completion_groups:
                completed_count += store_completion_group(
                    ordered_outcomes,
                    completion_group,
                    request_count=batch_session.handle.request_count,
                )
    except BaseException:
        batch_session.cancel()
        raise

    return finalize_ordered_outcomes(ordered_outcomes)


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
    batch_session = resumable_evaluator.open_session(
        study.problem,
        requests,
    )
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

    batch_session = resumable_evaluator.resume_session(handle.evaluator_handle)
    return StudyExactAsyncStepSession(
        study=study,
        requests=handle.requests,
        post_ask_state=handle.post_ask_state,
        batch_session=batch_session,
        candidate_equal=study.problem.space.candidates_equal,
        ordered_outcomes=list(handle.ordered_outcomes),
    )
