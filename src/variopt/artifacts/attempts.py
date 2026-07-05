"""Evaluation-attempt artifact definitions."""

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from typing import Generic, Protocol, TypeAlias, TypeGuard, overload, runtime_checkable

from typing_extensions import Self, TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..spaces.equality import CandidateEquality, require_candidate_match
from ..typevars import CandidateT
from .diagnostics import KernelDiagnostics
from .records import (
    ObjectiveVectorPayload,
    ObjectiveVectorRecord,
    Observation,
    ObservationPayload,
    RequestAlignedEvaluationRecord,
)
from .refinement import CandidateRefinement, require_matching_refined_candidate
from .requests import EvaluationRequest, Proposal

PayloadT = TypeVar("PayloadT", default=ObservationPayload, covariant=True)
RecordPayloadT = TypeVar(
    "RecordPayloadT",
    bound=RequestAlignedEvaluationRecord[object],
)
ProjectedPayloadT = TypeVar("ProjectedPayloadT")
MaterializerPayloadT = TypeVar("MaterializerPayloadT", contravariant=True)
MaterializerRecordT = TypeVar(
    "MaterializerRecordT",
    bound=RequestAlignedEvaluationRecord[object],
    covariant=True,
)
ScalarObservationCandidateT = TypeVar("ScalarObservationCandidateT")
_ScalarObservationViewCandidateT = TypeVar(
    "_ScalarObservationViewCandidateT",
    covariant=True,
)
MaterializableEvaluationPayload: TypeAlias = (
    ObservationPayload | ObjectiveVectorPayload | RequestAlignedEvaluationRecord[CandidateT]
)


class _ScalarObservationView(Protocol[_ScalarObservationViewCandidateT]):
    """Structural scalar-observation view needed by success normalization."""

    @property
    def request(self) -> EvaluationRequest[_ScalarObservationViewCandidateT]:
        """Return the canonical request owned by the observation."""
        ...

    @property
    def value(self) -> float:
        """Return the raw scalar objective value."""
        ...

    @property
    def score(self) -> float:
        """Return the canonical minimization score."""
        ...

    @property
    def elapsed_seconds(self) -> float | None:
        """Return the optional wall-clock runtime."""
        ...


@runtime_checkable
class _RequestAlignedPayloadShape(Protocol):
    """Runtime-checkable shape for request-aligned compatibility payloads."""

    @property
    def request(self) -> object:
        """Return the payload's request slot."""
        ...

    @property
    def candidate(self) -> object:
        """Return the payload's evaluated candidate slot."""
        ...


@dataclass(frozen=True, slots=True)
class _UnvalidatedRefinementCandidate:
    """Sentinel for refinement pairs that have not been revalidated."""


_UNVALIDATED_REFINEMENT_CANDIDATE = _UnvalidatedRefinementCandidate()
ValidatedRefinementCandidate: TypeAlias = CandidateT | _UnvalidatedRefinementCandidate
EvaluationSuccessPickleState: TypeAlias = tuple[
    EvaluationRequest[CandidateT],
    PayloadT,
    int,
    CandidateRefinement[CandidateT] | None,
    KernelDiagnostics | None,
    bool,
    ValidatedRefinementCandidate[CandidateT],
    ValidatedRefinementCandidate[CandidateT],
    ValidatedRefinementCandidate[CandidateT],
    ValidatedRefinementCandidate[CandidateT],
]


@dataclass(frozen=True, slots=True)
class EvaluationExceptionSnapshot(FrozenGenericSlotsCompat):
    """Serializable summary of an exception raised during candidate evaluation.

    Parameters
    ----------
    exception_module : str
        Module that defines the exception type.
    exception_qualname : str
        Qualified name of the exception type inside ``exception_module``.
    message : str, default=""
        Stringified exception message.
    """

    exception_module: str
    exception_qualname: str
    message: str = ""

    @classmethod
    def from_exception(cls, exception: BaseException) -> Self:
        """Build a snapshot from a recordable user-code exception.

        Parameters
        ----------
        exception : BaseException
            Exception raised while evaluating a concrete candidate. The raw
            exception object is not stored. ``KeyboardInterrupt`` and
            ``SystemExit`` are rejected because they are not recordable
            evaluation failures.

        Returns
        -------
        Self
            JSON- and pickle-friendly exception summary.

        Raises
        ------
        TypeError
            If ``exception`` is not an ``Exception`` instance.
        """
        if not isinstance(exception, Exception):
            msg = "exception must be an Exception instance"
            raise TypeError(msg)

        exception_type = type(exception)
        return cls(
            exception_module=exception_type.__module__,
            exception_qualname=exception_type.__qualname__,
            message=str(exception),
        )

    def __post_init__(self) -> None:
        """Validate snapshot fields."""
        if type(self.exception_module) is not str:
            msg = "exception_module must be str"
            raise TypeError(msg)

        if self.exception_module == "":
            msg = "exception_module must not be empty"
            raise ValueError(msg)

        if type(self.exception_qualname) is not str:
            msg = "exception_qualname must be str"
            raise TypeError(msg)

        if self.exception_qualname == "":
            msg = "exception_qualname must not be empty"
            raise ValueError(msg)

        if type(self.message) is not str:
            msg = "message must be str"
            raise TypeError(msg)

    @property
    def exception_type(self) -> str:
        """Return the fully qualified exception type name.

        Returns
        -------
        str
            ``"<module>.<qualname>"`` for the captured exception type.
        """
        return f"{self.exception_module}.{self.exception_qualname}"


@dataclass(frozen=True, slots=True)
class EvaluationFailure(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Recorded failure for one concrete evaluation request.

    Parameters
    ----------
    request : EvaluationRequest[CandidateT]
        Canonical request whose candidate evaluation failed.
    exception : EvaluationExceptionSnapshot
        Serializable exception summary. Raw exception objects are intentionally
        excluded from this artifact.
    evaluation_count : int, default=1
        Logical evaluation cost consumed by the failed attempt.
    """

    request: EvaluationRequest[CandidateT]
    exception: EvaluationExceptionSnapshot
    evaluation_count: int = 1

    @classmethod
    def from_exception(
        cls,
        *,
        request: EvaluationRequest[CandidateT],
        exception: Exception,
        evaluation_count: int = 1,
    ) -> Self:
        """Build a failure artifact from a recordable user-code exception.

        Parameters
        ----------
        request : EvaluationRequest[CandidateT]
            Request being evaluated when ``exception`` was raised.
        exception : Exception
            Exception raised by user evaluation code.
        evaluation_count : int, default=1
            Logical evaluation cost consumed by the failed attempt.

        Returns
        -------
        Self
            Request-aligned failure artifact.
        """
        return cls(
            request=request,
            exception=EvaluationExceptionSnapshot.from_exception(exception),
            evaluation_count=evaluation_count,
        )

    def __post_init__(self) -> None:
        """Validate failure artifact invariants."""
        if type(self.request) is not EvaluationRequest:
            msg = "request must be an EvaluationRequest"
            raise TypeError(msg)

        if type(self.exception) is not EvaluationExceptionSnapshot:
            msg = "exception must be an EvaluationExceptionSnapshot"
            raise TypeError(msg)

        if type(self.evaluation_count) is not int:
            msg = "evaluation_count must be int"
            raise TypeError(msg)

        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)

    @property
    def candidate(self) -> CandidateT:
        """Return the candidate carried by the failed request.

        Returns
        -------
        CandidateT
            Candidate that failed during evaluation.
        """
        return self.request.candidate

    @property
    def proposal(self) -> Proposal[CandidateT]:
        """Return the proposal compatibility view.

        Returns
        -------
        Proposal[CandidateT]
            Proposal owned by :attr:`request`.
        """
        return self.request.proposal

    @property
    def proposal_id(self) -> str | None:
        """Return the optional proposal identifier for the failed request.

        Returns
        -------
        str | None
            Proposal identifier carried by the failed request, if present.
        """
        return self.request.proposal_id


@dataclass(frozen=True, slots=True, init=False)
class EvaluationSuccess(FrozenGenericSlotsCompat, Generic[CandidateT, PayloadT]):
    """Successful evaluation attempt for one concrete request.

    Parameters
    ----------
    request : EvaluationRequest[CandidateT]
        Canonical request whose candidate produced ``payload``.
    payload : PayloadT
        Request-free successful evaluation payload.
    evaluation_count : int, default=1
        Logical evaluation cost consumed by the successful attempt.
    refinement : CandidateRefinement[CandidateT] | None, optional
        Optional execution-side provenance for candidate refinement before
        evaluation. The refined candidate must match ``request.candidate``.
    kernel_diagnostics : KernelDiagnostics | None, optional
        Optional diagnostics emitted by the kernel episode that produced this
        success.
    candidate_equal : CandidateEquality[CandidateT] | None, optional
        Explicit candidate equality predicate used to validate refinement
        alignment when raw scalar equality is not the search-space contract.
    """

    request: EvaluationRequest[CandidateT]
    payload: PayloadT
    evaluation_count: int = 1
    refinement: CandidateRefinement[CandidateT] | None = None
    kernel_diagnostics: KernelDiagnostics | None = None
    # Keep this InitVar so dataclasses.replace(..., candidate_equal=...) can
    # revalidate after pickle intentionally strips non-serializable predicates.
    candidate_equal: InitVar[CandidateEquality[CandidateT] | None] = None
    _candidate_equal: CandidateEquality[CandidateT] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _candidate_equal_required: bool = field(
        default=False,
        repr=False,
        compare=False,
    )
    _validated_request_candidate: ValidatedRefinementCandidate[CandidateT] = field(
        default=_UNVALIDATED_REFINEMENT_CANDIDATE,
        repr=False,
        compare=False,
    )
    _validated_refined_candidate: ValidatedRefinementCandidate[CandidateT] = field(
        default=_UNVALIDATED_REFINEMENT_CANDIDATE,
        repr=False,
        compare=False,
    )
    _validated_payload_request_candidate: ValidatedRefinementCandidate[
        CandidateT
    ] = field(
        default=_UNVALIDATED_REFINEMENT_CANDIDATE,
        repr=False,
        compare=False,
    )
    _validated_refinement_source_candidate: ValidatedRefinementCandidate[
        CandidateT
    ] = field(
        default=_UNVALIDATED_REFINEMENT_CANDIDATE,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        *,
        request: EvaluationRequest[CandidateT],
        payload: PayloadT,
        evaluation_count: int = 1,
        refinement: CandidateRefinement[CandidateT] | None = None,
        kernel_diagnostics: KernelDiagnostics | None = None,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
        _candidate_equal: CandidateEquality[CandidateT] | None = None,
        _candidate_equal_required: bool = False,
        _validated_request_candidate: ValidatedRefinementCandidate[
            CandidateT
        ] = _UNVALIDATED_REFINEMENT_CANDIDATE,
        _validated_refined_candidate: ValidatedRefinementCandidate[
            CandidateT
        ] = _UNVALIDATED_REFINEMENT_CANDIDATE,
        _validated_payload_request_candidate: ValidatedRefinementCandidate[
            CandidateT
        ] = _UNVALIDATED_REFINEMENT_CANDIDATE,
        _validated_refinement_source_candidate: ValidatedRefinementCandidate[
            CandidateT
        ] = _UNVALIDATED_REFINEMENT_CANDIDATE,
    ) -> None:
        """Create one successful request-owned attempt artifact.

        Parameters
        ----------
        request : EvaluationRequest[CandidateT]
            Canonical request whose candidate produced ``payload``.
        payload : PayloadT
            Request-free successful evaluation payload.
        evaluation_count : int, default=1
            Logical evaluation cost consumed by the attempt.
        refinement : CandidateRefinement[CandidateT] | None, optional
            Optional refinement provenance to validate against ``request``.
        kernel_diagnostics : KernelDiagnostics | None, optional
            Optional kernel diagnostics for this successful attempt.
        candidate_equal : CandidateEquality[CandidateT] | None, optional
            Explicit equality predicate for refinement alignment.
        """
        effective_candidate_equal = candidate_equal
        if effective_candidate_equal is None:
            effective_candidate_equal = _candidate_equal

        object.__setattr__(self, "__orig_class__", None)
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "evaluation_count", evaluation_count)
        object.__setattr__(self, "refinement", refinement)
        object.__setattr__(self, "kernel_diagnostics", kernel_diagnostics)
        object.__setattr__(self, "_candidate_equal", effective_candidate_equal)
        object.__setattr__(
            self,
            "_candidate_equal_required",
            _candidate_equal_required or effective_candidate_equal is not None,
        )
        object.__setattr__(
            self,
            "_validated_request_candidate",
            _validated_request_candidate,
        )
        object.__setattr__(
            self,
            "_validated_refined_candidate",
            _validated_refined_candidate,
        )
        object.__setattr__(
            self,
            "_validated_payload_request_candidate",
            _validated_payload_request_candidate,
        )
        object.__setattr__(
            self,
            "_validated_refinement_source_candidate",
            _validated_refinement_source_candidate,
        )
        self._validate(candidate_equal=effective_candidate_equal)

    def _validate(
        self,
        *,
        candidate_equal: CandidateEquality[CandidateT] | None,
    ) -> None:
        """Validate request ownership, accounting, and refinement alignment."""
        if type(self.request) is not EvaluationRequest:
            msg = "request must be an EvaluationRequest"
            raise TypeError(msg)

        if type(self.evaluation_count) is not int:
            msg = "evaluation_count must be int"
            raise TypeError(msg)

        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)

        if (
            self.kernel_diagnostics is not None
            and type(self.kernel_diagnostics) is not KernelDiagnostics
        ):
            msg = "kernel_diagnostics must be a KernelDiagnostics"
            raise TypeError(msg)

        refinement = self.refinement
        if refinement is not None:
            if not _is_candidate_refinement(refinement):
                msg = "refinement must be a CandidateRefinement"
                raise TypeError(msg)

            if (
                not self._refinement_alignment_is_prevalidated()
                and candidate_equal is None
                and self._candidate_equal_required
            ):
                msg = (
                    "candidate_equal is required to revalidate refinement alignment "
                    "after changing an explicitly compared success"
                )
                raise TypeError(msg)

        payload = self.payload
        if _is_request_aligned_payload(payload):
            self._validate_record_payload_alignment(
                payload,
                candidate_equal=candidate_equal,
            )
        else:
            self._clear_payload_source_alignment()

        if refinement is None:
            object.__setattr__(
                self,
                "_validated_request_candidate",
                _UNVALIDATED_REFINEMENT_CANDIDATE,
            )
            object.__setattr__(
                self,
                "_validated_refined_candidate",
                _UNVALIDATED_REFINEMENT_CANDIDATE,
            )
            return

        if self._refinement_alignment_is_prevalidated():
            return

        require_matching_refined_candidate(
            record_candidate=self.request.candidate,
            refined_candidate=refinement.refined_candidate,
            mismatch_message=(
                "refinement refined_candidate must match the success request "
                "candidate"
            ),
            candidate_equal=candidate_equal,
        )
        object.__setattr__(
            self,
            "_validated_request_candidate",
            self.request.candidate,
        )
        object.__setattr__(
            self,
            "_validated_refined_candidate",
            refinement.refined_candidate,
        )

    def _refinement_alignment_is_prevalidated(self) -> bool:
        """Return whether the current request/refinement pair was validated."""
        return (
            self.refinement is not None
            and self._validated_request_candidate
            is not _UNVALIDATED_REFINEMENT_CANDIDATE
            and self._validated_refined_candidate
            is not _UNVALIDATED_REFINEMENT_CANDIDATE
            and self._validated_request_candidate is self.request.candidate
            and self._validated_refined_candidate is self.refinement.refined_candidate
        )

    def _payload_source_alignment_is_prevalidated(
        self,
        payload_request: EvaluationRequest[CandidateT],
    ) -> bool:
        """Return whether the current payload source pair was validated."""
        return (
            self.refinement is not None
            and self._validated_payload_request_candidate
            is not _UNVALIDATED_REFINEMENT_CANDIDATE
            and self._validated_refinement_source_candidate
            is not _UNVALIDATED_REFINEMENT_CANDIDATE
            and self._validated_payload_request_candidate is payload_request.candidate
            and self._validated_refinement_source_candidate
            is self.refinement.source_candidate
        )

    def _store_payload_source_alignment(
        self,
        payload_request: EvaluationRequest[CandidateT],
    ) -> None:
        """Cache successful payload-source refinement validation."""
        if self.refinement is None:
            self._clear_payload_source_alignment()
            return

        object.__setattr__(
            self,
            "_validated_payload_request_candidate",
            payload_request.candidate,
        )
        object.__setattr__(
            self,
            "_validated_refinement_source_candidate",
            self.refinement.source_candidate,
        )

    def _clear_payload_source_alignment(self) -> None:
        """Clear cached payload-source refinement validation."""
        object.__setattr__(
            self,
            "_validated_payload_request_candidate",
            _UNVALIDATED_REFINEMENT_CANDIDATE,
        )
        object.__setattr__(
            self,
            "_validated_refinement_source_candidate",
            _UNVALIDATED_REFINEMENT_CANDIDATE,
        )

    def _validate_record_payload_alignment(
        self,
        payload: object,
        *,
        candidate_equal: CandidateEquality[CandidateT] | None,
    ) -> None:
        """Require a request-aligned payload to belong to this success request."""
        if not _is_request_aligned_payload(payload):
            msg = "success payload must be request-aligned"
            raise TypeError(msg)

        if payload.candidate is not self.request.candidate:
            msg = "success payload candidate must match the success request candidate"
            raise ValueError(msg)

        if payload.request is self.request:
            self._clear_payload_source_alignment()
            return

        if not _is_evaluation_request_in_candidate_domain(
            payload.request,
            self.request.candidate,
        ):
            msg = "success payload request must match the success request"
            raise ValueError(msg)

        if not self._payload_request_metadata_matches(payload.request):
            msg = "success payload request must match the success request"
            raise ValueError(msg)

        self._validate_payload_request_refinement_source(
            payload.request,
            candidate_equal=candidate_equal,
        )

    def _payload_request_metadata_matches(
        self,
        payload_request: EvaluationRequest[CandidateT],
    ) -> bool:
        """Return whether a payload source request matches this request metadata."""
        if payload_request.proposal_id != self.request.proposal_id:
            return False

        if payload_request.proposal_evaluation_spec != self.request.proposal_evaluation_spec:
            return False

        return True

    def _validate_payload_request_refinement_source(
        self,
        payload_request: EvaluationRequest[CandidateT],
        *,
        candidate_equal: CandidateEquality[CandidateT] | None,
    ) -> None:
        """Require payload source requests to match this refinement source."""
        refinement = self.refinement
        if refinement is None:
            self._clear_payload_source_alignment()
            return

        if self._payload_source_alignment_is_prevalidated(payload_request):
            return

        if candidate_equal is None and self._candidate_equal_required:
            msg = (
                "candidate_equal is required to revalidate success payload "
                "refinement source after changing an explicitly compared success"
            )
            raise TypeError(msg)

        require_candidate_match(
            left_candidate=refinement.source_candidate,
            right_candidate=payload_request.candidate,
            mismatch_message=(
                "success payload request candidate must match refinement source "
                "candidate"
            ),
            candidate_equal=candidate_equal,
        )
        self._store_payload_source_alignment(payload_request)

    @staticmethod
    def from_scalar_observation(
        *,
        observation: Observation[ScalarObservationCandidateT],
        request: EvaluationRequest[ScalarObservationCandidateT] | None = None,
        evaluation_count: int = 1,
        refinement: CandidateRefinement[ScalarObservationCandidateT] | None = None,
        kernel_diagnostics: KernelDiagnostics | None = None,
        candidate_equal: CandidateEquality[ScalarObservationCandidateT] | None = None,
    ) -> "EvaluationSuccess[ScalarObservationCandidateT, ObservationPayload]":
        """Normalize a scalar observation into a request-owned success.

        Parameters
        ----------
        observation : Observation[CandidateT]
            Legacy scalar observation whose scalar value fields become the
            request-free payload.
        request : EvaluationRequest[CandidateT] | None, optional
            Canonical request to own the resulting success. When omitted, a
            observation's canonical request is reused.
        evaluation_count : int, default=1
            Logical evaluation cost consumed by the successful attempt.
        refinement : CandidateRefinement[CandidateT] | None, optional
            Optional execution-side provenance for candidate refinement.
        kernel_diagnostics : KernelDiagnostics | None, optional
            Optional diagnostics emitted by the producing kernel episode.
        candidate_equal : CandidateEquality[CandidateT] | None, optional
            Explicit equality predicate used to validate request and refinement
            alignment.

        Returns
        -------
        EvaluationSuccess[CandidateT, ObservationPayload]
            Canonical success carrying a request-free scalar payload.
        """
        _require_scalar_observation(observation)
        return _success_from_scalar_observation(
            observation=observation,
            request=request,
            evaluation_count=evaluation_count,
            refinement=refinement,
            kernel_diagnostics=kernel_diagnostics,
            candidate_equal=candidate_equal,
        )

    @property
    def candidate(self) -> CandidateT:
        """Return the candidate carried by the successful request.

        Returns
        -------
        CandidateT
            Candidate that produced :attr:`payload`.
        """
        return self.request.candidate

    @property
    def proposal(self) -> Proposal[CandidateT]:
        """Return the proposal compatibility view.

        Returns
        -------
        Proposal[CandidateT]
            Proposal owned by :attr:`request`.
        """
        return self.request.proposal

    @property
    def proposal_id(self) -> str | None:
        """Return the optional proposal identifier for the successful request.

        Returns
        -------
        str | None
            Proposal identifier carried by the successful request, if present.
        """
        return self.request.proposal_id

    def scalar_observation(self) -> Observation[CandidateT]:
        """Project a scalar success into an observation compatibility record.

        Returns
        -------
        Observation[CandidateT]
            Request-aligned scalar observation derived from either an
            ``ObservationPayload`` or a legacy ``Observation`` payload.

        Raises
        ------
        TypeError
            If the success payload is not scalar.
        """
        payload = self.payload
        if isinstance(payload, Observation):
            return Observation(
                request=self.request,
                candidate=self.request.candidate,
                value=payload.value,
                score=payload.score,
                elapsed_seconds=payload.elapsed_seconds,
            )
        if isinstance(payload, ObservationPayload):
            return Observation(
                request=self.request,
                candidate=self.request.candidate,
                value=payload.value,
                score=payload.score,
                elapsed_seconds=payload.elapsed_seconds,
            )

        msg = "success payload is not scalar"
        raise TypeError(msg)

    def with_payload(
        self,
        payload: ProjectedPayloadT,
        *,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
    ) -> "EvaluationSuccess[CandidateT, ProjectedPayloadT]":
        """Return this success with a different payload representation.

        Parameters
        ----------
        payload : ProjectedPayloadT
            Replacement payload that still describes the same successful
            request. Metadata such as evaluation cost, refinement provenance,
            diagnostics, and cached refinement-validation state is preserved.
        candidate_equal : CandidateEquality[CandidateT] | None, optional
            Equality predicate used when the replacement payload requires
            revalidating candidate alignment after a pickle round-trip stripped
            the stored predicate.

        Returns
        -------
        EvaluationSuccess[CandidateT, ProjectedPayloadT]
            Success carrying ``payload`` with the original request and metadata.
        """
        return EvaluationSuccess(
            request=self.request,
            payload=payload,
            evaluation_count=self.evaluation_count,
            refinement=self.refinement,
            kernel_diagnostics=self.kernel_diagnostics,
            candidate_equal=candidate_equal,
            _candidate_equal=self._candidate_equal,
            _candidate_equal_required=self._candidate_equal_required,
            _validated_request_candidate=self._validated_request_candidate,
            _validated_refined_candidate=self._validated_refined_candidate,
            _validated_payload_request_candidate=(
                self._validated_payload_request_candidate
            ),
            _validated_refinement_source_candidate=(
                self._validated_refinement_source_candidate
            ),
        )

    def _pickle_state(self) -> EvaluationSuccessPickleState[CandidateT, PayloadT]:
        """Return pickle state without serializing candidate equality callables."""
        return (
            self.request,
            self.payload,
            self.evaluation_count,
            self.refinement,
            self.kernel_diagnostics,
            self._candidate_equal_required,
            self._validated_request_candidate,
            self._validated_refined_candidate,
            self._validated_payload_request_candidate,
            self._validated_refinement_source_candidate,
        )

    def _restore_pickle_state(
        self,
        state: EvaluationSuccessPickleState[CandidateT, PayloadT],
    ) -> None:
        """Restore pickle state emitted by :meth:`_pickle_state`."""
        (
            request,
            payload,
            evaluation_count,
            refinement,
            kernel_diagnostics,
            candidate_equal_required,
            validated_request_candidate,
            validated_refined_candidate,
            validated_payload_request_candidate,
            validated_refinement_source_candidate,
        ) = state
        object.__setattr__(self, "__orig_class__", None)
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "evaluation_count", evaluation_count)
        object.__setattr__(self, "refinement", refinement)
        object.__setattr__(self, "kernel_diagnostics", kernel_diagnostics)
        object.__setattr__(self, "_candidate_equal", None)
        object.__setattr__(self, "_candidate_equal_required", candidate_equal_required)
        object.__setattr__(
            self,
            "_validated_request_candidate",
            validated_request_candidate,
        )
        object.__setattr__(
            self,
            "_validated_refined_candidate",
            validated_refined_candidate,
        )
        object.__setattr__(
            self,
            "_validated_payload_request_candidate",
            validated_payload_request_candidate,
        )
        object.__setattr__(
            self,
            "_validated_refinement_source_candidate",
            validated_refinement_source_candidate,
        )
        self._validate(candidate_equal=None)


def _success_from_scalar_observation(
    *,
    observation: _ScalarObservationView[ScalarObservationCandidateT],
    request: EvaluationRequest[ScalarObservationCandidateT] | None = None,
    evaluation_count: int = 1,
    refinement: CandidateRefinement[ScalarObservationCandidateT] | None = None,
    kernel_diagnostics: KernelDiagnostics | None = None,
    candidate_equal: CandidateEquality[ScalarObservationCandidateT] | None = None,
) -> EvaluationSuccess[ScalarObservationCandidateT, ObservationPayload]:
    """Normalize a scalar observation into a request-owned success."""
    scalar_payload = ObservationPayload(
        value=observation.value,
        score=observation.score,
        elapsed_seconds=observation.elapsed_seconds,
    )
    if request is None:
        success_request = observation.request
        return EvaluationSuccess(
            request=success_request,
            payload=scalar_payload,
            evaluation_count=evaluation_count,
            refinement=refinement,
            kernel_diagnostics=kernel_diagnostics,
            candidate_equal=candidate_equal,
        )

    require_candidate_match(
        left_candidate=request.candidate,
        right_candidate=observation.request.candidate,
        mismatch_message=(
            "request candidate must match the scalar observation candidate"
        ),
        candidate_equal=candidate_equal,
    )
    return EvaluationSuccess(
        request=request,
        payload=scalar_payload,
        evaluation_count=evaluation_count,
        refinement=refinement,
        kernel_diagnostics=kernel_diagnostics,
        candidate_equal=candidate_equal,
    )


def _require_scalar_observation(observation: object) -> None:
    """Require the public scalar-observation factory input to be an Observation."""
    if type(observation) is not Observation:
        msg = "observation must be an Observation"
        raise TypeError(msg)


setattr(EvaluationSuccess, "__getstate__", EvaluationSuccess.__dict__["_pickle_state"])
setattr(
    EvaluationSuccess,
    "__setstate__",
    EvaluationSuccess.__dict__["_restore_pickle_state"],
)


EvaluationAttempt: TypeAlias = (
    EvaluationSuccess[CandidateT, PayloadT] | EvaluationFailure[CandidateT]
)


def _is_evaluation_success(
    attempt: EvaluationAttempt[CandidateT, PayloadT],
) -> TypeGuard[EvaluationSuccess[CandidateT, PayloadT]]:
    return type(attempt) is EvaluationSuccess


def _is_evaluation_failure(
    attempt: EvaluationAttempt[CandidateT, PayloadT],
) -> TypeGuard[EvaluationFailure[CandidateT]]:
    return type(attempt) is EvaluationFailure


def _is_candidate_refinement(
    refinement: CandidateRefinement[CandidateT] | None,
) -> TypeGuard[CandidateRefinement[CandidateT]]:
    return type(refinement) is CandidateRefinement


def _is_request_aligned_payload(
    payload: object,
) -> TypeGuard[RequestAlignedEvaluationRecord[object]]:
    if not isinstance(payload, _RequestAlignedPayloadShape):
        return False

    return type(payload.request) is EvaluationRequest


def _is_materializable_record_payload(
    payload: object,
    success: EvaluationSuccess[CandidateT, object],
) -> TypeGuard[RequestAlignedEvaluationRecord[CandidateT]]:
    if not _is_request_aligned_payload(payload):
        return False

    if payload.candidate is not success.request.candidate:
        return False

    if payload.request is not success.request:
        if payload.request.proposal_id != success.request.proposal_id:
            return False

        if payload.request.proposal_evaluation_spec != success.request.proposal_evaluation_spec:
            return False

    refinement = success.refinement
    if refinement is None:
        return payload.request.candidate is success.request.candidate

    return payload.request.candidate is refinement.source_candidate


def _is_evaluation_request_in_candidate_domain(
    request: object,
    candidate: CandidateT,
) -> TypeGuard[EvaluationRequest[CandidateT]]:
    """Narrow an erased request generic after caller-side candidate alignment."""
    _ = candidate
    return type(request) is EvaluationRequest


def materialize_success_record(
    success: EvaluationSuccess[CandidateT, object],
) -> RequestAlignedEvaluationRecord[object]:
    """Project one successful attempt into a request-aligned record.

    Parameters
    ----------
    success : EvaluationSuccess[CandidateT, PayloadT]
        Canonical successful attempt to project.

    Returns
    -------
    RequestAlignedEvaluationRecord
        Request-aligned record suitable for run-method feedback and reports.

    Raises
    ------
    TypeError
        If the payload cannot be projected into a request-aligned record.
    """
    payload = success.payload
    refinement = success.refinement
    projection_proposal = success.request.proposal
    if refinement is not None:
        projection_proposal = Proposal(
            candidate=refinement.source_candidate,
            proposal_id=success.proposal_id,
        )

    if type(payload) is ObservationPayload:
        return Observation(
            proposal=projection_proposal,
            proposal_evaluation_spec=success.request.proposal_evaluation_spec,
            candidate=success.request.candidate,
            value=payload.value,
            score=payload.score,
            elapsed_seconds=payload.elapsed_seconds,
        )

    if _is_materializable_record_payload(payload, success):
        return payload

    if type(payload) is Observation:
        return Observation(
            proposal=projection_proposal,
            proposal_evaluation_spec=success.request.proposal_evaluation_spec,
            candidate=success.request.candidate,
            value=payload.value,
            score=payload.score,
            elapsed_seconds=payload.elapsed_seconds,
        )

    if type(payload) is ObjectiveVectorPayload:
        return ObjectiveVectorRecord(
            proposal=projection_proposal,
            proposal_evaluation_spec=success.request.proposal_evaluation_spec,
            candidate=success.request.candidate,
            objective_values=payload.objective_values,
            objective_scores=payload.objective_scores,
            elapsed_seconds=payload.elapsed_seconds,
        )

    if type(payload) is ObjectiveVectorRecord:
        return ObjectiveVectorRecord(
            proposal=projection_proposal,
            proposal_evaluation_spec=success.request.proposal_evaluation_spec,
            candidate=success.request.candidate,
            objective_values=payload.objective_values,
            objective_scores=payload.objective_scores,
            elapsed_seconds=payload.elapsed_seconds,
        )

    msg = "success payload cannot be materialized as a request-aligned record"
    raise TypeError(msg)


@overload
def materialize_success_records(
    successes: Sequence[EvaluationSuccess[CandidateT, ObservationPayload]],
) -> tuple[Observation[CandidateT], ...]: ...


@overload
def materialize_success_records(
    successes: Sequence[EvaluationSuccess[CandidateT, ObjectiveVectorPayload]],
) -> tuple[ObjectiveVectorRecord[CandidateT], ...]: ...


@overload
def materialize_success_records(
    successes: Sequence[
        EvaluationSuccess[CandidateT, RequestAlignedEvaluationRecord[CandidateT]]
    ],
) -> tuple[RequestAlignedEvaluationRecord[CandidateT], ...]: ...


@overload
def materialize_success_records(
    successes: Sequence[EvaluationSuccess[CandidateT, RecordPayloadT]],
) -> tuple[RecordPayloadT, ...]: ...


def materialize_success_records(
    successes: Sequence[EvaluationSuccess[CandidateT, object]],
) -> tuple[RequestAlignedEvaluationRecord[object], ...]:
    """Project successful attempts into request-aligned records."""
    return tuple(materialize_success_record(success) for success in successes)


@dataclass(frozen=True, slots=True, init=False)
class EvaluationAttemptBatch(FrozenGenericSlotsCompat, Generic[CandidateT, PayloadT]):
    """Ordered request-slot batch of successful and failed attempts.

    Parameters
    ----------
    attempts : Sequence[EvaluationAttempt[CandidateT, PayloadT]]
        Attempt slots in the same order as the evaluated request batch. Each
        slot is exactly one :class:`EvaluationSuccess` or
        :class:`EvaluationFailure`.
    """

    attempts: tuple[EvaluationAttempt[CandidateT, PayloadT], ...]
    _requests_cache: tuple[EvaluationRequest[CandidateT], ...] | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )
    _successes_cache: tuple[EvaluationSuccess[CandidateT, PayloadT], ...] | None = (
        field(
            init=False,
            repr=False,
            compare=False,
            default=None,
        )
    )
    _failures_cache: tuple[EvaluationFailure[CandidateT], ...] | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )
    _success_indices_cache: tuple[int, ...] | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )
    _failure_indices_cache: tuple[int, ...] | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )
    _payloads_cache: tuple[PayloadT, ...] | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )
    _evaluation_count_cache: int | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )
    _has_failures_cache: bool | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __init__(
        self,
        *,
        attempts: Sequence[EvaluationAttempt[CandidateT, PayloadT]],
    ) -> None:
        """Create one ordered attempt batch.

        Parameters
        ----------
        attempts : Sequence[EvaluationAttempt[CandidateT, PayloadT]]
            Ordered attempt slots. Empty batches are valid and represent an
            empty request batch.
        """
        object.__setattr__(self, "__orig_class__", None)
        object.__setattr__(self, "attempts", tuple(attempts))
        object.__setattr__(self, "_requests_cache", None)
        object.__setattr__(self, "_successes_cache", None)
        object.__setattr__(self, "_failures_cache", None)
        object.__setattr__(self, "_success_indices_cache", None)
        object.__setattr__(self, "_failure_indices_cache", None)
        object.__setattr__(self, "_payloads_cache", None)
        object.__setattr__(self, "_evaluation_count_cache", None)
        object.__setattr__(self, "_has_failures_cache", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate that each slot is one closed attempt variant."""
        for attempt in self.attempts:
            if not _is_evaluation_success(attempt) and not _is_evaluation_failure(
                attempt
            ):
                msg = "attempts must contain EvaluationSuccess or EvaluationFailure"
                raise TypeError(msg)

    @classmethod
    def from_single_request_attempts(
        cls,
        attempts: Sequence["EvaluationAttemptBatch[CandidateT, PayloadT]"],
    ) -> "EvaluationAttemptBatch[CandidateT, PayloadT]":
        """Merge one-slot attempt batches into one ordered attempt batch.

        Parameters
        ----------
        attempts : Sequence[EvaluationAttemptBatch[CandidateT, PayloadT]]
            Attempt batches that each represent exactly one request slot.

        Returns
        -------
        EvaluationAttemptBatch[CandidateT, PayloadT]
            Ordered aggregate preserving the input attempt order.

        Raises
        ------
        TypeError
            If ``attempts`` contains non-``EvaluationAttemptBatch`` values.
        ValueError
            If any input batch contains more or fewer than one request slot.
        """
        merged_attempts: list[EvaluationAttempt[CandidateT, PayloadT]] = []
        for attempt_batch in attempts:
            if type(attempt_batch) is not EvaluationAttemptBatch:
                msg = "attempts must contain EvaluationAttemptBatch values"
                raise TypeError(msg)
            if attempt_batch.attempt_count != 1:
                msg = "each merged attempt must contain exactly one request"
                raise ValueError(msg)
            merged_attempts.append(attempt_batch.attempts[0])

        return cls(attempts=tuple(merged_attempts))

    @classmethod
    def concatenate(
        cls,
        batches: Sequence["EvaluationAttemptBatch[CandidateT, PayloadT]"],
    ) -> "EvaluationAttemptBatch[CandidateT, PayloadT]":
        """Concatenate ordered attempt batches.

        Parameters
        ----------
        batches : Sequence[EvaluationAttemptBatch[CandidateT, PayloadT]]
            Attempt batches to concatenate in order.

        Returns
        -------
        EvaluationAttemptBatch[CandidateT, PayloadT]
            Ordered aggregate preserving each batch's local slot order.

        Raises
        ------
        TypeError
            If ``batches`` contains non-``EvaluationAttemptBatch`` values.
        """
        attempts: list[EvaluationAttempt[CandidateT, PayloadT]] = []
        for batch in batches:
            if type(batch) is not EvaluationAttemptBatch:
                msg = "batches must contain EvaluationAttemptBatch values"
                raise TypeError(msg)
            attempts.extend(batch.attempts)

        return cls(attempts=tuple(attempts))

    @property
    def attempt_count(self) -> int:
        """Return the number of request slots in the batch.

        Returns
        -------
        int
            Number of evaluation attempts represented by this batch.
        """
        return len(self.attempts)

    @property
    def requests(self) -> tuple[EvaluationRequest[CandidateT], ...]:
        """Return requests in attempt-slot order.

        Returns
        -------
        tuple[EvaluationRequest[CandidateT], ...]
            Requests owned by each attempt slot.
        """
        cached_requests = self._requests_cache
        if cached_requests is None:
            cached_requests = tuple(attempt.request for attempt in self.attempts)
            object.__setattr__(self, "_requests_cache", cached_requests)
        return cached_requests

    @property
    def successes(self) -> tuple[EvaluationSuccess[CandidateT, PayloadT], ...]:
        """Return successful attempts in slot order.

        Returns
        -------
        tuple[EvaluationSuccess[CandidateT, PayloadT], ...]
            Successful attempts only.
        """
        cached_successes = self._successes_cache
        if cached_successes is None:
            cached_successes = tuple(
                attempt
                for attempt in self.attempts
                if _is_evaluation_success(attempt)
            )
            object.__setattr__(self, "_successes_cache", cached_successes)
        return cached_successes

    @property
    def failures(self) -> tuple[EvaluationFailure[CandidateT], ...]:
        """Return failed attempts in slot order.

        Returns
        -------
        tuple[EvaluationFailure[CandidateT], ...]
            Failed attempts only.
        """
        cached_failures = self._failures_cache
        if cached_failures is None:
            cached_failures = tuple(
                attempt
                for attempt in self.attempts
                if _is_evaluation_failure(attempt)
            )
            object.__setattr__(self, "_failures_cache", cached_failures)
        return cached_failures

    @property
    def success_indices(self) -> tuple[int, ...]:
        """Return request-slot indices for successful attempts.

        Returns
        -------
        tuple[int, ...]
            Indices whose attempt slot is an :class:`EvaluationSuccess`.
        """
        cached_indices = self._success_indices_cache
        if cached_indices is None:
            cached_indices = tuple(
                index
                for index, attempt in enumerate(self.attempts)
                if _is_evaluation_success(attempt)
            )
            object.__setattr__(self, "_success_indices_cache", cached_indices)
        return cached_indices

    @property
    def failure_indices(self) -> tuple[int, ...]:
        """Return request-slot indices for failed attempts.

        Returns
        -------
        tuple[int, ...]
            Indices whose attempt slot is an :class:`EvaluationFailure`.
        """
        cached_indices = self._failure_indices_cache
        if cached_indices is None:
            cached_indices = tuple(
                index
                for index, attempt in enumerate(self.attempts)
                if _is_evaluation_failure(attempt)
            )
            object.__setattr__(self, "_failure_indices_cache", cached_indices)
        return cached_indices

    @property
    def payloads(self) -> tuple[PayloadT, ...]:
        """Return successful payloads in slot order.

        Returns
        -------
        tuple[PayloadT, ...]
            Request-free payloads carried by successful attempts.
        """
        cached_payloads = self._payloads_cache
        if cached_payloads is None:
            cached_payloads = tuple(success.payload for success in self.successes)
            object.__setattr__(self, "_payloads_cache", cached_payloads)
        return cached_payloads

    @property
    def evaluation_count(self) -> int:
        """Return total logical evaluation cost for all attempts.

        Returns
        -------
        int
            Sum of success and failure evaluation counts.
        """
        cached_evaluation_count = self._evaluation_count_cache
        if cached_evaluation_count is None:
            cached_evaluation_count = sum(
                attempt.evaluation_count for attempt in self.attempts
            )
            object.__setattr__(
                self,
                "_evaluation_count_cache",
                cached_evaluation_count,
            )
        return cached_evaluation_count

    @property
    def has_failures(self) -> bool:
        """Return whether the batch contains failed attempts.

        Returns
        -------
        bool
            ``True`` when at least one attempt failed.
        """
        cached_has_failures = self._has_failures_cache
        if cached_has_failures is None:
            cached_has_failures = any(
                _is_evaluation_failure(attempt) for attempt in self.attempts
            )
            object.__setattr__(self, "_has_failures_cache", cached_has_failures)
        return cached_has_failures

    def single_success_or_none(
        self,
    ) -> EvaluationSuccess[CandidateT, PayloadT] | None:
        """Return the successful attempt from a one-slot batch.

        Returns
        -------
        EvaluationSuccess[CandidateT, PayloadT] | None
            The single successful attempt, or ``None`` when the only slot
            failed.

        Raises
        ------
        ValueError
            If the batch does not represent exactly one request slot.
        RuntimeError
            If the one-slot invariant is internally inconsistent.
        """
        if self.attempt_count != 1:
            msg = "single success view requires exactly one request"
            raise ValueError(msg)

        attempt = self.attempts[0]
        if _is_evaluation_success(attempt):
            return attempt
        if _is_evaluation_failure(attempt):
            return None

        msg = "single request attempt must contain one success or one failure"
        raise RuntimeError(msg)


@overload
def materialize_attempt_batch_records(
    attempts: EvaluationAttemptBatch[CandidateT, ObservationPayload],
) -> EvaluationAttemptBatch[CandidateT, Observation[CandidateT]]: ...


@overload
def materialize_attempt_batch_records(
    attempts: EvaluationAttemptBatch[CandidateT, ObjectiveVectorPayload],
) -> EvaluationAttemptBatch[CandidateT, ObjectiveVectorRecord[CandidateT]]: ...


@overload
def materialize_attempt_batch_records(
    attempts: EvaluationAttemptBatch[
        CandidateT,
        RequestAlignedEvaluationRecord[CandidateT],
    ],
) -> EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord[CandidateT]]: ...


@overload
def materialize_attempt_batch_records(
    attempts: EvaluationAttemptBatch[CandidateT, RecordPayloadT],
) -> EvaluationAttemptBatch[CandidateT, RecordPayloadT]: ...


def materialize_attempt_batch_records(
    attempts: EvaluationAttemptBatch[CandidateT, object],
) -> EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord[object]]:
    """Project successful attempt payloads into request-aligned records.

    Parameters
    ----------
    attempts : EvaluationAttemptBatch[CandidateT, MaterializableEvaluationPayload[CandidateT]]
        Ordered attempt batch whose successes carry request-free scalar/vector
        payloads or existing request-aligned compatibility records.

    Returns
    -------
    EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord]
        Attempt batch with the same request slots and failures, but with every
        success payload materialized for run-method feedback and terminal
        record surfaces.
    """
    materialized_attempts: list[
        EvaluationSuccess[CandidateT, RequestAlignedEvaluationRecord[object]]
        | EvaluationFailure[CandidateT]
    ] = []
    for attempt in attempts.attempts:
        if _is_evaluation_failure(attempt):
            materialized_attempts.append(attempt)
            continue
        if _is_evaluation_success(attempt):
            materialized_attempts.append(
                attempt.with_payload(materialize_success_record(attempt))
            )
            continue

        msg = "attempt batch contains an unknown attempt variant"
        raise TypeError(msg)

    return EvaluationAttemptBatch(attempts=tuple(materialized_attempts))


class EvaluationAttemptMaterializer(
    Protocol[CandidateT, MaterializerPayloadT, MaterializerRecordT]
):
    """Typed boundary that projects evaluator payload attempts into records.

    Notes
    -----
    This protocol owns the associated-type relation that the standalone
    materialization overloads cannot express for generic ``Study`` execution:
    one evaluator payload family maps to one run-method feedback record family.
    """

    def materialize_attempts(
        self,
        attempts: EvaluationAttemptBatch[CandidateT, MaterializerPayloadT],
    ) -> EvaluationAttemptBatch[CandidateT, MaterializerRecordT]:
        """Project one ordered attempt batch into feedback records.

        Parameters
        ----------
        attempts : EvaluationAttemptBatch[CandidateT, MaterializerPayloadT]
            Evaluator or kernel attempt batch carrying request-owned payloads.

        Returns
        -------
        EvaluationAttemptBatch[CandidateT, MaterializerRecordT]
            Attempt batch with identical request/failure slots and materialized
            success records.
        """
        ...


@dataclass(frozen=True, slots=True)
class DefaultEvaluationAttemptMaterializer(
    Generic[CandidateT],
):
    """Default materializer for built-in scalar, vector, and record payloads."""

    @overload
    def materialize_attempts(
        self,
        attempts: EvaluationAttemptBatch[CandidateT, ObservationPayload],
    ) -> EvaluationAttemptBatch[CandidateT, Observation[CandidateT]]: ...

    @overload
    def materialize_attempts(
        self,
        attempts: EvaluationAttemptBatch[CandidateT, ObjectiveVectorPayload],
    ) -> EvaluationAttemptBatch[CandidateT, ObjectiveVectorRecord[CandidateT]]: ...

    @overload
    def materialize_attempts(
        self,
        attempts: EvaluationAttemptBatch[
            CandidateT,
            RequestAlignedEvaluationRecord[CandidateT],
        ],
    ) -> EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord[CandidateT]]: ...

    def materialize_attempts(
        self,
        attempts: (
            EvaluationAttemptBatch[CandidateT, ObservationPayload]
            | EvaluationAttemptBatch[CandidateT, ObjectiveVectorPayload]
            | EvaluationAttemptBatch[
                CandidateT,
                RequestAlignedEvaluationRecord[CandidateT],
            ]
        ),
    ) -> EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord[object]]:
        """Project one ordered attempt batch with the default artifact rules.

        Parameters
        ----------
        attempts : EvaluationAttemptBatch[CandidateT, MaterializableEvaluationPayload[CandidateT]]
            Attempt batch whose successes carry built-in scalar/vector payloads
            or already request-aligned records.

        Returns
        -------
        EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord[object]]
            Request-aligned record attempts suitable for run-method feedback.
        """
        return materialize_attempt_batch_records(attempts)
