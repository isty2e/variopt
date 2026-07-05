"""Exact-async study lifecycle and resume-handle artifacts."""

from dataclasses import dataclass
from typing import Generic, Literal

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ...artifacts import EvaluationAttemptBatch, EvaluationRequest
from ...evaluators.async_evaluator.artifacts import EvaluationBatchResumeHandle
from ...typevars import CandidateT, RunMethodStateT
from ..common import StudyPayloadT

StudyExactAsyncSessionLifecycle = Literal[
    "active",
    "completed",
    "failed",
    "cancelled",
    "suspended",
]


@dataclass(frozen=True, slots=True)
class StudyExactAsyncStepResumeHandle(FrozenGenericSlotsCompat,
    Generic[CandidateT, RunMethodStateT, StudyPayloadT]
):
    """Study-owned resume handle for one suspended exact-async step session.

    Parameters
    ----------
    evaluator_handle : EvaluationBatchResumeHandle
        Evaluator-owned resume handle for the suspended batch.
    requests : tuple[EvaluationRequest[CandidateT], ...]
        Requests issued for the suspended step.
    post_ask_state : RunMethodStateT
        Run-method state captured immediately after the corresponding ``ask``.
    ordered_attempts : tuple[EvaluationAttemptBatch[CandidateT, StudyPayloadT] | None, ...]
        Attempt slots aligned to ``requests``. Completed entries contain
        one-request attempt batches; unfinished entries are ``None``.
    """

    evaluator_handle: EvaluationBatchResumeHandle
    requests: tuple[EvaluationRequest[CandidateT], ...]
    post_ask_state: RunMethodStateT
    ordered_attempts: tuple[
        EvaluationAttemptBatch[CandidateT, StudyPayloadT] | None,
        ...,
    ]

    def __post_init__(self) -> None:
        """Validate suspended study-step payloads.

        Raises
        ------
        ValueError
            Raised when request counts or stored completion counts disagree
            with the evaluator handle.
        """
        if len(self.requests) != self.evaluator_handle.request_count:
            msg = "resume handle requests must align with evaluator_handle"
            raise ValueError(msg)

        if len(self.ordered_attempts) != self.evaluator_handle.request_count:
            msg = "ordered_attempts must align with evaluator_handle"
            raise ValueError(msg)

        completed_count = sum(
            attempt is not None for attempt in self.ordered_attempts
        )
        if completed_count != self.evaluator_handle.completed_count:
            msg = (
                "ordered_attempts completion count must match "
                "evaluator_handle.completed_count"
            )
            raise ValueError(msg)
