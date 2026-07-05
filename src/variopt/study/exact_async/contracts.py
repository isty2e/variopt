"""Owner protocols for exact-async study orchestration."""

from typing import Protocol

from typing_extensions import TypeVar

from ...artifacts import EvaluationRequest, Proposal
from ...evaluators.base import Evaluator
from ...kernel import Kernel, ProposalBatchQuery
from ...methods import RunMethod
from ...outcomes import EvaluationAttemptBatch, EvaluationOutcome
from ...problem import Problem
from ...typevars import CandidateT, RunMethodStateT
from ..common import StudyEvaluationRecordT

BoundaryT = TypeVar("BoundaryT")


class StudyRunMethodOwner(
    Protocol[CandidateT, RunMethodStateT, StudyEvaluationRecordT]
):
    """Subset of study state required to assimilate exact-async completions.

    Notes
    -----
    This protocol keeps exact-async assimilation helpers independent from the
    full ``Study`` surface.
    """

    @property
    def run_method(
        self,
    ) -> RunMethod[
        RunMethodStateT,
        Proposal[CandidateT],
        StudyEvaluationRecordT,
    ]:
        """Return the run method used to assimilate completed records."""
        ...


class StudyExactAsyncOwner(
    Protocol[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
):
    """Subset of study state required to open and resume exact-async sessions.

    Notes
    -----
    This protocol isolates the exact-async orchestration boundary from the full
    ``Study`` implementation.
    """

    @property
    def problem(
        self,
    ) -> Problem[BoundaryT, CandidateT, StudyEvaluationRecordT]:
        """Return the configured problem."""
        ...

    @property
    def run_method(
        self,
    ) -> RunMethod[
        RunMethodStateT,
        Proposal[CandidateT],
        StudyEvaluationRecordT,
    ]:
        """Return the configured run method."""
        ...

    @property
    def evaluator(
        self,
    ) -> Evaluator[
        Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT],
    ]:
        """Return the configured evaluator."""
        ...

    @property
    def kernel(
        self,
    ) -> Kernel[
        ProposalBatchQuery[BoundaryT, CandidateT, StudyEvaluationRecordT],
        EvaluationAttemptBatch[CandidateT, StudyEvaluationRecordT],
    ]:
        """Return the configured kernel."""
        ...
