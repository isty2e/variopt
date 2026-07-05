"""Execution-side evaluation attempt artifacts."""

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field, fields
from typing import Generic, cast

from typing_extensions import TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .artifacts.attempts import EvaluationExceptionSnapshot, EvaluationFailure
from .artifacts.records import Observation, RequestAlignedEvaluationRecord
from .artifacts.refinement import (
    CandidateRefinement,
    require_matching_refined_candidate,
)
from .artifacts.requests import EvaluationRequest
from .kernel import KernelDiagnostics
from .spaces import CandidateEquality
from .typevars import CandidateT

OutcomeRecordT = TypeVar(
    "OutcomeRecordT",
    bound=RequestAlignedEvaluationRecord,
    default=Observation[CandidateT],
)

__all__ = [
    "CandidateRefinement",
    "EvaluationAttemptBatch",
    "EvaluationExceptionSnapshot",
    "EvaluationFailure",
    "EvaluationOutcome",
]

_UNVALIDATED_REFINEMENT_CANDIDATE = object()


def _normalize_attempt_indices(
    indices: Sequence[int] | None,
    *,
    attempt_count: int,
    request_count: int,
    label: str,
) -> tuple[int, ...]:
    if indices is None:
        if attempt_count == 0:
            return ()
        if attempt_count == request_count:
            return tuple(range(request_count))
        msg = f"{label}_indices are required unless {label}s cover every request"
        raise ValueError(msg)

    index_tuple = tuple(indices)
    if len(index_tuple) != attempt_count:
        msg = f"{label}_indices must align with {label}s"
        raise ValueError(msg)

    for index in index_tuple:
        if type(index) is not int:
            msg = f"{label}_indices must contain int values"
            raise TypeError(msg)
        if index < 0 or index >= request_count:
            msg = f"{label}_indices must be within the request range"
            raise ValueError(msg)

    if any(
        left_index >= right_index
        for left_index, right_index in zip(
            index_tuple,
            index_tuple[1:],
        )
    ):
        msg = f"{label}_indices must be strictly increasing"
        raise ValueError(msg)

    return index_tuple


def validate_outcome_refinement_alignment(
    outcome: "EvaluationOutcome[CandidateT, OutcomeRecordT]",
    *,
    candidate_equal: CandidateEquality[CandidateT] | None = None,
) -> None:
    """Validate that outcome refinement provenance matches its record.

    Parameters
    ----------
    outcome : EvaluationOutcome[CandidateT, OutcomeRecordT]
        Outcome whose record/refinement alignment should be checked.
    candidate_equal : CandidateEquality[CandidateT] | None, optional
        Explicit candidate equality predicate. When absent, strict scalar Python
        equality is used.

    Raises
    ------
    TypeError
        If candidate equality is not scalar, or if an explicit predicate does
        not return ``bool``.
    ValueError
        If the refined candidate does not match the outcome record candidate.
    """
    if outcome.refinement is None:
        return

    require_matching_refined_candidate(
        record_candidate=cast(CandidateT, outcome.record.candidate),
        refined_candidate=outcome.refinement.refined_candidate,
        mismatch_message=(
            "refinement refined_candidate must match the outcome "
            "record candidate"
        ),
        candidate_equal=candidate_equal,
    )


@dataclass(frozen=True, slots=True, init=False)
class EvaluationOutcome(FrozenGenericSlotsCompat, Generic[CandidateT, OutcomeRecordT]):
    """Executed evaluation outcome with explicit execution accounting.

    Parameters
    ----------
    record : OutcomeRecordT | None, optional
        Canonical request-aligned record produced by the evaluator or kernel.
    observation : Observation[CandidateT] | None, optional
        Scalar compatibility alias for ``record`` at the API boundary.
    evaluation_count : int, default=1
        Logical evaluation cost associated with the outcome.
    kernel_diagnostics : KernelDiagnostics | None, optional
        Optional execution-side diagnostics emitted by the kernel.
    refinement : CandidateRefinement[CandidateT] | None, optional
        Optional execution-side provenance for candidate refinement before
        evaluation.

    Notes
    -----
    Exactly one of ``record`` or ``observation`` must be supplied. ``record``
    remains the canonical internal contract; ``observation`` is the scalar
    compatibility alias for request-local scalar studies.
    """

    record: OutcomeRecordT
    evaluation_count: int = 1
    kernel_diagnostics: KernelDiagnostics | None = None
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
    _validated_record_candidate: object = field(
        default=_UNVALIDATED_REFINEMENT_CANDIDATE,
        repr=False,
        compare=False,
    )
    _validated_refined_candidate: object = field(
        default=_UNVALIDATED_REFINEMENT_CANDIDATE,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        *,
        record: OutcomeRecordT | None = None,
        observation: Observation[CandidateT] | None = None,
        evaluation_count: int = 1,
        kernel_diagnostics: KernelDiagnostics | None = None,
        refinement: CandidateRefinement[CandidateT] | None = None,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
        _candidate_equal: CandidateEquality[CandidateT] | None = None,
        _candidate_equal_required: bool = False,
        _validated_record_candidate: object = _UNVALIDATED_REFINEMENT_CANDIDATE,
        _validated_refined_candidate: object = _UNVALIDATED_REFINEMENT_CANDIDATE,
    ) -> None:
        """Create one canonical evaluation outcome.

        Parameters
        ----------
        record : OutcomeRecordT | None, optional
            Canonical request-aligned evaluation record.
        observation : Observation[CandidateT] | None, optional
            Scalar compatibility alias for ``record``.
        evaluation_count : int, default=1
            Logical evaluation cost associated with the outcome.
        kernel_diagnostics : KernelDiagnostics | None, optional
            Optional kernel-side diagnostics.
        refinement : CandidateRefinement[CandidateT] | None, optional
            Optional candidate-refinement provenance.
        candidate_equal : CandidateEquality[CandidateT] | None, optional
            Explicit candidate equality predicate used to validate refinement
            alignment. When absent, strict scalar Python equality is used.

        Raises
        ------
        ValueError
            If neither or both of ``record`` and ``observation`` are provided.
        RuntimeError
            If record normalization fails unexpectedly.
        """
        if (record is None) == (observation is None):
            msg = "exactly one of record or observation must be provided"
            raise ValueError(msg)

        if record is not None:
            normalized_record = record
        elif observation is not None:
            normalized_record = cast(OutcomeRecordT, observation)
        else:
            msg = "evaluation record normalization failed"
            raise RuntimeError(msg)

        effective_candidate_equal = candidate_equal
        if effective_candidate_equal is None:
            effective_candidate_equal = _candidate_equal
        candidate_equal_required = (
            _candidate_equal_required or effective_candidate_equal is not None
        )

        object.__setattr__(self, "__orig_class__", None)
        object.__setattr__(self, "record", normalized_record)
        object.__setattr__(self, "evaluation_count", evaluation_count)
        object.__setattr__(self, "kernel_diagnostics", kernel_diagnostics)
        object.__setattr__(self, "refinement", refinement)
        object.__setattr__(self, "_candidate_equal", effective_candidate_equal)
        object.__setattr__(
            self,
            "_candidate_equal_required",
            candidate_equal_required,
        )
        object.__setattr__(
            self,
            "_validated_record_candidate",
            _validated_record_candidate,
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
        """Validate outcome accounting metadata.

        Raises
        ------
        ValueError
            If ``evaluation_count`` is negative.
        """
        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)

        if self.refinement is None:
            object.__setattr__(
                self,
                "_validated_record_candidate",
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

        if candidate_equal is None and self._candidate_equal_required:
            msg = (
                "candidate_equal is required to revalidate refinement alignment "
                "after changing an explicitly compared outcome"
            )
            raise TypeError(msg)

        validate_outcome_refinement_alignment(
            self,
            candidate_equal=candidate_equal,
        )
        object.__setattr__(
            self,
            "_validated_record_candidate",
            self.record.candidate,
        )
        object.__setattr__(
            self,
            "_validated_refined_candidate",
            self.refinement.refined_candidate,
        )

    def _refinement_alignment_is_prevalidated(self) -> bool:
        """Return whether the current refinement pair was already checked."""
        return (
            self.refinement is not None
            and self._validated_record_candidate is not _UNVALIDATED_REFINEMENT_CANDIDATE
            and self._validated_refined_candidate is not _UNVALIDATED_REFINEMENT_CANDIDATE
            and self._validated_record_candidate is self.record.candidate
            and self._validated_refined_candidate is self.refinement.refined_candidate
        )

    def __post_init__(
        self,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
    ) -> None:
        """Validate outcome accounting metadata after dataclass construction."""
        effective_candidate_equal = candidate_equal
        if effective_candidate_equal is None:
            effective_candidate_equal = self._candidate_equal
        self._validate(candidate_equal=effective_candidate_equal)

    @property
    def observation(self) -> Observation[CandidateT]:
        """Return the scalar observation compatibility view.

        Returns
        -------
        Observation[CandidateT]
            Scalar observation carried by this outcome.

        Raises
        ------
        TypeError
            If the outcome record is not a scalar
            :class:`~variopt.artifacts.Observation`.

        Notes
        -----
        Prefer :attr:`record` in canonical internal code.
        """
        if not isinstance(self.record, Observation):
            msg = "evaluation outcome does not carry a scalar Observation"
            raise TypeError(msg)
        return cast(Observation[CandidateT], self.record)


@dataclass(frozen=True, slots=True, init=False)
class EvaluationAttemptBatch(
    FrozenGenericSlotsCompat,
    Generic[CandidateT, OutcomeRecordT],
):
    """Dense request-aligned batch of successful and failed evaluation attempts.

    Parameters
    ----------
    requests : Sequence[EvaluationRequest[CandidateT]]
        Ordered requests that define the attempt slots.
    outcomes : Sequence[EvaluationOutcome[CandidateT, OutcomeRecordT]], default=()
        Successful attempts.
    outcome_indices : Sequence[int] | None, optional
        Request-slot indices aligned one-to-one with ``outcomes``. When omitted,
        ``outcomes`` must cover every request.
    failures : Sequence[EvaluationFailure[CandidateT]], default=()
        Recorded user-code evaluation failures.
    failure_indices : Sequence[int] | None, optional
        Request-slot indices aligned one-to-one with ``failures``. When omitted,
        ``failures`` must cover every request.

    Notes
    -----
    Every request slot must be represented exactly once as either an outcome or
    a failure. Alignment is identity-based on the canonical request object, not
    candidate equality.
    """

    requests: tuple[EvaluationRequest[CandidateT], ...]
    outcome_indices: tuple[int, ...]
    outcomes: tuple[EvaluationOutcome[CandidateT, OutcomeRecordT], ...]
    failure_indices: tuple[int, ...]
    failures: tuple[EvaluationFailure[CandidateT], ...]

    def __init__(
        self,
        *,
        requests: Sequence[EvaluationRequest[CandidateT]],
        outcomes: Sequence[EvaluationOutcome[CandidateT, OutcomeRecordT]] = (),
        outcome_indices: Sequence[int] | None = None,
        failures: Sequence[EvaluationFailure[CandidateT]] = (),
        failure_indices: Sequence[int] | None = None,
    ) -> None:
        """Create one dense attempt batch.

        Parameters
        ----------
        requests : Sequence[EvaluationRequest[CandidateT]]
            Ordered requests that define the attempt slots.
        outcomes : Sequence[EvaluationOutcome[CandidateT, OutcomeRecordT]], default=()
            Successful attempts.
        outcome_indices : Sequence[int] | None, optional
            Request-slot indices aligned one-to-one with ``outcomes``.
        failures : Sequence[EvaluationFailure[CandidateT]], default=()
            Recorded user-code evaluation failures.
        failure_indices : Sequence[int] | None, optional
            Request-slot indices aligned one-to-one with ``failures``.
        """
        request_tuple = tuple(requests)
        outcome_tuple = tuple(outcomes)
        failure_tuple = tuple(failures)
        outcome_index_tuple = _normalize_attempt_indices(
            outcome_indices,
            attempt_count=len(outcome_tuple),
            request_count=len(request_tuple),
            label="outcome",
        )
        failure_index_tuple = _normalize_attempt_indices(
            failure_indices,
            attempt_count=len(failure_tuple),
            request_count=len(request_tuple),
            label="failure",
        )

        object.__setattr__(self, "__orig_class__", None)
        object.__setattr__(self, "requests", request_tuple)
        object.__setattr__(self, "outcome_indices", outcome_index_tuple)
        object.__setattr__(self, "outcomes", outcome_tuple)
        object.__setattr__(self, "failure_indices", failure_index_tuple)
        object.__setattr__(self, "failures", failure_tuple)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate request-slot coverage and attempt alignment."""
        for request in self.requests:
            if type(request) is not EvaluationRequest:
                msg = "requests must contain EvaluationRequest values"
                raise TypeError(msg)

        for outcome in self.outcomes:
            if type(outcome) is not EvaluationOutcome:
                msg = "outcomes must contain EvaluationOutcome values"
                raise TypeError(msg)

        for failure in self.failures:
            if type(failure) is not EvaluationFailure:
                msg = "failures must contain EvaluationFailure values"
                raise TypeError(msg)

        occupied_indices = self.outcome_indices + self.failure_indices
        occupied_index_set = set(occupied_indices)
        if len(occupied_index_set) != len(occupied_indices):
            msg = "attempt indices must be unique"
            raise ValueError(msg)

        expected_indices = set(range(len(self.requests)))
        if occupied_index_set != expected_indices:
            msg = "attempt indices must cover every request exactly once"
            raise ValueError(msg)

        for outcome_index, outcome in zip(
            self.outcome_indices,
            self.outcomes,
            strict=True,
        ):
            if outcome.record.request is not self.requests[outcome_index]:
                msg = "outcome record request must match its attempt slot"
                raise ValueError(msg)

        for failure_index, failure in zip(
            self.failure_indices,
            self.failures,
            strict=True,
        ):
            if failure.request is not self.requests[failure_index]:
                msg = "failure request must match its attempt slot"
                raise ValueError(msg)

    @classmethod
    def from_single_request_attempts(
        cls,
        attempts: Sequence["EvaluationAttemptBatch[CandidateT, OutcomeRecordT]"],
    ) -> "EvaluationAttemptBatch[CandidateT, OutcomeRecordT]":
        """Merge one-request attempt batches into one dense batch.

        Parameters
        ----------
        attempts : Sequence[EvaluationAttemptBatch[CandidateT, OutcomeRecordT]]
            Attempt batches that each represent exactly one request slot.

        Returns
        -------
        EvaluationAttemptBatch[CandidateT, OutcomeRecordT]
            Dense aggregate preserving the input attempt order.

        Raises
        ------
        TypeError
            If ``attempts`` contains non-``EvaluationAttemptBatch`` values.
        ValueError
            If any attempt contains more or fewer than one request slot.
        """
        requests: list[EvaluationRequest[CandidateT]] = []
        outcomes: list[EvaluationOutcome[CandidateT, OutcomeRecordT]] = []
        outcome_indices: list[int] = []
        failures: list[EvaluationFailure[CandidateT]] = []
        failure_indices: list[int] = []

        for attempt_index, attempt in enumerate(attempts):
            if type(attempt) is not EvaluationAttemptBatch:
                msg = "attempts must contain EvaluationAttemptBatch values"
                raise TypeError(msg)
            if attempt.attempt_count != 1:
                msg = "each merged attempt must contain exactly one request"
                raise ValueError(msg)

            requests.append(attempt.requests[0])
            if len(attempt.outcomes) == 1:
                outcome_indices.append(attempt_index)
                outcomes.append(attempt.outcomes[0])
                continue

            if len(attempt.failures) == 1:
                failure_indices.append(attempt_index)
                failures.append(attempt.failures[0])
                continue

            msg = "single-request attempt must contain one outcome or one failure"
            raise RuntimeError(msg)

        return cls(
            requests=tuple(requests),
            outcomes=tuple(outcomes),
            outcome_indices=tuple(outcome_indices),
            failures=tuple(failures),
            failure_indices=tuple(failure_indices),
        )

    @classmethod
    def concatenate(
        cls,
        batches: Sequence["EvaluationAttemptBatch[CandidateT, OutcomeRecordT]"],
    ) -> "EvaluationAttemptBatch[CandidateT, OutcomeRecordT]":
        """Concatenate dense attempt batches while preserving local slot order.

        Parameters
        ----------
        batches : Sequence[EvaluationAttemptBatch[CandidateT, OutcomeRecordT]]
            Attempt batches to concatenate in order.

        Returns
        -------
        EvaluationAttemptBatch[CandidateT, OutcomeRecordT]
            Dense aggregate with outcome and failure indices rebased by each
            batch's request-slot offset.

        Raises
        ------
        TypeError
            If ``batches`` contains non-``EvaluationAttemptBatch`` values.
        """
        requests: list[EvaluationRequest[CandidateT]] = []
        outcomes: list[EvaluationOutcome[CandidateT, OutcomeRecordT]] = []
        outcome_indices: list[int] = []
        failures: list[EvaluationFailure[CandidateT]] = []
        failure_indices: list[int] = []

        for batch in batches:
            if type(batch) is not EvaluationAttemptBatch:
                msg = "batches must contain EvaluationAttemptBatch values"
                raise TypeError(msg)

            slot_offset = len(requests)
            requests.extend(batch.requests)
            outcomes.extend(batch.outcomes)
            outcome_indices.extend(
                slot_offset + outcome_index
                for outcome_index in batch.outcome_indices
            )
            failures.extend(batch.failures)
            failure_indices.extend(
                slot_offset + failure_index
                for failure_index in batch.failure_indices
            )

        return cls(
            requests=tuple(requests),
            outcomes=tuple(outcomes),
            outcome_indices=tuple(outcome_indices),
            failures=tuple(failures),
            failure_indices=tuple(failure_indices),
        )

    @property
    def attempt_count(self) -> int:
        """Return the number of request slots in the batch.

        Returns
        -------
        int
            Number of evaluation attempts represented by this batch.
        """
        return len(self.requests)

    @property
    def records(self) -> tuple[OutcomeRecordT, ...]:
        """Return successful records in outcome order.

        Returns
        -------
        tuple[OutcomeRecordT, ...]
            Records carried by successful outcomes only.
        """
        return tuple(outcome.record for outcome in self.outcomes)

    @property
    def outcome_requests(self) -> tuple[EvaluationRequest[CandidateT], ...]:
        """Return successful request slots in outcome order.

        Returns
        -------
        tuple[EvaluationRequest[CandidateT], ...]
            Canonical requests for successful attempts, aligned one-to-one with
            :attr:`outcomes`.
        """
        return tuple(self.requests[index] for index in self.outcome_indices)

    @property
    def failure_requests(self) -> tuple[EvaluationRequest[CandidateT], ...]:
        """Return failed request slots in failure order.

        Returns
        -------
        tuple[EvaluationRequest[CandidateT], ...]
            Canonical requests for failed attempts, aligned one-to-one with
            :attr:`failures`.
        """
        return tuple(self.requests[index] for index in self.failure_indices)

    @property
    def evaluation_count(self) -> int:
        """Return total logical evaluation cost for the batch.

        Returns
        -------
        int
            Sum of successful and failed attempt costs.
        """
        return sum(outcome.evaluation_count for outcome in self.outcomes) + sum(
            failure.evaluation_count for failure in self.failures
        )

    @property
    def has_failures(self) -> bool:
        """Return whether the batch contains recorded failures.

        Returns
        -------
        bool
            ``True`` when at least one attempt failed.
        """
        return len(self.failures) > 0

    def single_outcome_or_none(
        self,
    ) -> EvaluationOutcome[CandidateT, OutcomeRecordT] | None:
        """Return the successful outcome from a one-slot attempt batch.

        Returns
        -------
        EvaluationOutcome[CandidateT, OutcomeRecordT] | None
            The single successful outcome, or ``None`` when the single attempt
            slot is a recorded failure.

        Raises
        ------
        ValueError
            If the batch does not represent exactly one request slot.
        RuntimeError
            If the one-slot invariant is internally inconsistent.
        """
        if self.attempt_count != 1:
            msg = "single outcome view requires exactly one request"
            raise ValueError(msg)
        if len(self.outcomes) == 1:
            return self.outcomes[0]
        if len(self.failures) == 1:
            return None

        msg = "single request attempt must contain one outcome or one failure"
        raise RuntimeError(msg)


def evaluation_outcome_getstate(
    self: EvaluationOutcome[CandidateT, OutcomeRecordT],
) -> list[object | None]:
    state: list[object | None] = []
    for dataclass_field in fields(self):
        if dataclass_field.name == "_candidate_equal":
            state.append(None)
            continue
        state.append(getattr(self, dataclass_field.name, None))
    return state


def evaluation_outcome_setstate(
    self: EvaluationOutcome[CandidateT, OutcomeRecordT],
    state: list[object | None],
) -> None:
    restored_names: set[str] = set()
    dataclass_fields = fields(self)
    for dataclass_field, value in zip(dataclass_fields, state):
        restored_names.add(dataclass_field.name)
        if dataclass_field.name == "_candidate_equal":
            object.__setattr__(self, dataclass_field.name, None)
            continue
        object.__setattr__(self, dataclass_field.name, value)

    for dataclass_field in dataclass_fields:
        if dataclass_field.name in restored_names:
            continue
        if dataclass_field.name == "_candidate_equal":
            object.__setattr__(self, dataclass_field.name, None)
        elif dataclass_field.name == "_candidate_equal_required":
            object.__setattr__(self, dataclass_field.name, False)
        elif dataclass_field.name in {
            "_validated_record_candidate",
            "_validated_refined_candidate",
        }:
            object.__setattr__(
                self,
                dataclass_field.name,
                _UNVALIDATED_REFINEMENT_CANDIDATE,
            )

setattr(EvaluationOutcome, "__getstate__", evaluation_outcome_getstate)
setattr(EvaluationOutcome, "__setstate__", evaluation_outcome_setstate)

del evaluation_outcome_getstate
del evaluation_outcome_setstate
