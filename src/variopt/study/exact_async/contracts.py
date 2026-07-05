"""Owner protocols for exact-async study orchestration."""

from typing import Protocol

from typing_extensions import TypeVar

from ...artifacts import (
    EvaluationAttemptBatch,
    EvaluationAttemptMaterializer,
    Proposal,
)
from ...artifacts.records import RequestAlignedEvaluationRecord
from ...kernel import Kernel, ProposalBatchQuery
from ...methods import RunMethod
from ...problem import Problem
from ...typevars import CandidateT, RunMethodStateT
from ..common import (
    StudyEvaluationPayload,
    StudyEvaluator,
    StudyPayloadT,
    StudyRecordT,
)

AssimilationCandidateT = TypeVar("AssimilationCandidateT")
AssimilatorRecordT = TypeVar(
    "AssimilatorRecordT",
    bound=RequestAlignedEvaluationRecord[object],
    contravariant=True,
)
OwnerRunMethodRecordT = TypeVar(
    "OwnerRunMethodRecordT",
    bound=RequestAlignedEvaluationRecord[object],
)
OwnerPayloadT = TypeVar(
    "OwnerPayloadT",
    bound=StudyEvaluationPayload,
    contravariant=True,
)
BoundaryT = TypeVar("BoundaryT")


class AttemptBatchAssimilator(
    Protocol[RunMethodStateT, AssimilatorRecordT],
):
    """Capability that advances run-method state from materialized attempts."""

    def tell_attempts(
        self,
        state: RunMethodStateT,
        attempts: EvaluationAttemptBatch[
            AssimilationCandidateT,
            AssimilatorRecordT,
        ],
    ) -> RunMethodStateT:
        """Assimilate one dense request-aligned attempt batch."""
        ...


class StudyRunMethodOwner(
    Protocol[
        CandidateT,
        RunMethodStateT,
        OwnerPayloadT,
        OwnerRunMethodRecordT,
    ],
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
    ) -> AttemptBatchAssimilator[RunMethodStateT, OwnerRunMethodRecordT]:
        """Return the run method used to assimilate completed records."""
        ...

    @property
    def attempt_materializer(
        self,
    ) -> EvaluationAttemptMaterializer[
        CandidateT,
        OwnerPayloadT,
        OwnerRunMethodRecordT,
    ]:
        """Return the payload-to-record materializer for feedback attempts."""
        ...


class StudyExactAsyncOwner(
    Protocol[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ]
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
    ) -> Problem[BoundaryT, CandidateT, StudyPayloadT]:
        """Return the configured problem."""
        ...

    @property
    def run_method(
        self,
    ) -> RunMethod[
        RunMethodStateT,
        Proposal[CandidateT],
        StudyRecordT,
    ]:
        """Return the configured run method."""
        ...

    @property
    def evaluator(
        self,
    ) -> StudyEvaluator[BoundaryT, CandidateT, StudyPayloadT]:
        """Return the configured evaluator."""
        ...

    @property
    def kernel(
        self,
    ) -> Kernel[
        ProposalBatchQuery[BoundaryT, CandidateT, StudyPayloadT],
        EvaluationAttemptBatch[CandidateT, StudyPayloadT],
    ]:
        """Return the configured kernel."""
        ...

    @property
    def attempt_materializer(
        self,
    ) -> EvaluationAttemptMaterializer[
        CandidateT,
        StudyPayloadT,
        StudyRecordT,
    ]:
        """Return the payload-to-record materializer for feedback attempts."""
        ...
