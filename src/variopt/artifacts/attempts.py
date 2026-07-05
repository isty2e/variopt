"""Evaluation-attempt artifact definitions."""

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from typing import Generic, TypeAlias, TypeGuard, TypeVar

from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..spaces.equality import CandidateEquality
from ..typevars import CandidateT
from .refinement import CandidateRefinement, require_matching_refined_candidate
from .requests import EvaluationRequest, Proposal

PayloadT = TypeVar("PayloadT")


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
    bool,
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
    candidate_equal : CandidateEquality[CandidateT] | None, optional
        Explicit candidate equality predicate used to validate refinement
        alignment when raw scalar equality is not the search-space contract.
    """

    request: EvaluationRequest[CandidateT]
    payload: PayloadT
    evaluation_count: int = 1
    refinement: CandidateRefinement[CandidateT] | None = None
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

    def __init__(
        self,
        *,
        request: EvaluationRequest[CandidateT],
        payload: PayloadT,
        evaluation_count: int = 1,
        refinement: CandidateRefinement[CandidateT] | None = None,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
        _candidate_equal: CandidateEquality[CandidateT] | None = None,
        _candidate_equal_required: bool = False,
        _validated_request_candidate: ValidatedRefinementCandidate[
            CandidateT
        ] = _UNVALIDATED_REFINEMENT_CANDIDATE,
        _validated_refined_candidate: ValidatedRefinementCandidate[
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

        refinement = self.refinement
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

        if not _is_candidate_refinement(refinement):
            msg = "refinement must be a CandidateRefinement"
            raise TypeError(msg)

        if self._refinement_alignment_is_prevalidated():
            return

        if candidate_equal is None and self._candidate_equal_required:
            msg = (
                "candidate_equal is required to revalidate refinement alignment "
                "after changing an explicitly compared success"
            )
            raise TypeError(msg)

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

    def __post_init__(
        self,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
    ) -> None:
        """Validate success metadata after dataclass construction."""
        effective_candidate_equal = candidate_equal
        if effective_candidate_equal is None:
            effective_candidate_equal = self._candidate_equal
        self._validate(candidate_equal=effective_candidate_equal)

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

    def _pickle_state(self) -> EvaluationSuccessPickleState[CandidateT, PayloadT]:
        """Return pickle state without serializing candidate equality callables."""
        return (
            self.request,
            self.payload,
            self.evaluation_count,
            self.refinement,
            self._candidate_equal_required,
            self._validated_request_candidate,
            self._validated_refined_candidate,
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
            candidate_equal_required,
            validated_request_candidate,
            validated_refined_candidate,
        ) = state
        object.__setattr__(self, "__orig_class__", None)
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "evaluation_count", evaluation_count)
        object.__setattr__(self, "refinement", refinement)
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
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate that each slot is one closed attempt variant."""
        for attempt in self.attempts:
            if not _is_evaluation_success(attempt) and not _is_evaluation_failure(
                attempt
            ):
                msg = "attempts must contain EvaluationSuccess or EvaluationFailure"
                raise TypeError(msg)

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
        return tuple(attempt.request for attempt in self.attempts)

    @property
    def successes(self) -> tuple[EvaluationSuccess[CandidateT, PayloadT], ...]:
        """Return successful attempts in slot order.

        Returns
        -------
        tuple[EvaluationSuccess[CandidateT, PayloadT], ...]
            Successful attempts only.
        """
        return tuple(
            attempt
            for attempt in self.attempts
            if _is_evaluation_success(attempt)
        )

    @property
    def failures(self) -> tuple[EvaluationFailure[CandidateT], ...]:
        """Return failed attempts in slot order.

        Returns
        -------
        tuple[EvaluationFailure[CandidateT], ...]
            Failed attempts only.
        """
        return tuple(
            attempt
            for attempt in self.attempts
            if _is_evaluation_failure(attempt)
        )

    @property
    def success_indices(self) -> tuple[int, ...]:
        """Return request-slot indices for successful attempts.

        Returns
        -------
        tuple[int, ...]
            Indices whose attempt slot is an :class:`EvaluationSuccess`.
        """
        return tuple(
            index
            for index, attempt in enumerate(self.attempts)
            if _is_evaluation_success(attempt)
        )

    @property
    def failure_indices(self) -> tuple[int, ...]:
        """Return request-slot indices for failed attempts.

        Returns
        -------
        tuple[int, ...]
            Indices whose attempt slot is an :class:`EvaluationFailure`.
        """
        return tuple(
            index
            for index, attempt in enumerate(self.attempts)
            if _is_evaluation_failure(attempt)
        )

    @property
    def payloads(self) -> tuple[PayloadT, ...]:
        """Return successful payloads in slot order.

        Returns
        -------
        tuple[PayloadT, ...]
            Request-free payloads carried by successful attempts.
        """
        return tuple(success.payload for success in self.successes)

    @property
    def evaluation_count(self) -> int:
        """Return total logical evaluation cost for all attempts.

        Returns
        -------
        int
            Sum of success and failure evaluation counts.
        """
        return sum(attempt.evaluation_count for attempt in self.attempts)

    @property
    def has_failures(self) -> bool:
        """Return whether the batch contains failed attempts.

        Returns
        -------
        bool
            ``True`` when at least one attempt failed.
        """
        return len(self.failures) > 0

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
