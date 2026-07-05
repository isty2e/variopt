"""Execution-side evaluation attempt artifacts."""

from dataclasses import InitVar, dataclass, field, fields
from typing import Generic, cast

from typing_extensions import TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .artifacts import KernelDiagnostics
from .artifacts.attempts import EvaluationExceptionSnapshot, EvaluationFailure
from .artifacts.records import Observation, RequestAlignedEvaluationRecord
from .artifacts.refinement import (
    CandidateRefinement,
    require_matching_refined_candidate,
)
from .spaces import CandidateEquality
from .typevars import CandidateT

OutcomeRecordT = TypeVar(
    "OutcomeRecordT",
    bound=RequestAlignedEvaluationRecord,
    default=Observation[CandidateT],
)

__all__ = [
    "CandidateRefinement",
    "EvaluationExceptionSnapshot",
    "EvaluationFailure",
    "EvaluationOutcome",
]

_UNVALIDATED_REFINEMENT_CANDIDATE = object()


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
