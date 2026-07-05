"""Terminal-surface artifact definitions."""

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field, fields
from typing import Generic, TypeVar, cast

from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..spaces import CandidateEquality
from ..typevars import CandidateT
from .attempts import EvaluationFailure
from .records import ObjectiveVectorRecord, Observation, RequestAlignedEvaluationRecord
from .refinement import CandidateRefinement, require_matching_refined_candidate

RunRecordT = TypeVar("RunRecordT", bound=RequestAlignedEvaluationRecord)


def _refinement_validation_pairs(
    *,
    records: Sequence[RequestAlignedEvaluationRecord],
    refinements: tuple[CandidateRefinement[CandidateT] | None, ...],
) -> tuple[tuple[CandidateT, CandidateT], ...]:
    validation_pairs: list[tuple[CandidateT, CandidateT]] = []
    for record, refinement in zip(records, refinements, strict=True):
        if refinement is None:
            continue

        validation_pairs.append(
            (
                cast(CandidateT, record.candidate),
                refinement.refined_candidate,
            ),
        )

    return tuple(validation_pairs)


def _refinement_pairs_are_prevalidated(
    *,
    current_pairs: tuple[tuple[CandidateT, CandidateT], ...],
    validated_pairs: tuple[tuple[CandidateT, CandidateT], ...],
) -> bool:
    if len(current_pairs) != len(validated_pairs):
        return False

    return all(
        current_record_candidate is validated_record_candidate
        and current_refined_candidate is validated_refined_candidate
        for (
            current_record_candidate,
            current_refined_candidate,
        ), (
            validated_record_candidate,
            validated_refined_candidate,
        ) in zip(current_pairs, validated_pairs, strict=True)
    )


def _normalize_refinements(
    *,
    records: Sequence[RequestAlignedEvaluationRecord],
    refinements: Sequence[CandidateRefinement[CandidateT] | None],
    record_label: str,
    candidate_equal: CandidateEquality[CandidateT] | None,
    candidate_equal_required: bool = False,
    validated_refinement_pairs: tuple[tuple[CandidateT, CandidateT], ...] = (),
) -> tuple[
    tuple[CandidateRefinement[CandidateT] | None, ...],
    tuple[tuple[CandidateT, CandidateT], ...],
]:
    refinement_tuple = tuple(refinements)
    if refinement_tuple == ():
        return (), ()

    if len(refinement_tuple) != len(records):
        msg = f"refinements must be empty or align with {record_label}"
        raise ValueError(msg)

    if all(refinement is None for refinement in refinement_tuple):
        return (), ()

    current_refinement_pairs = _refinement_validation_pairs(
        records=records,
        refinements=refinement_tuple,
    )
    if _refinement_pairs_are_prevalidated(
        current_pairs=current_refinement_pairs,
        validated_pairs=validated_refinement_pairs,
    ):
        return refinement_tuple, current_refinement_pairs

    if candidate_equal is None and candidate_equal_required:
        msg = (
            "candidate_equal is required to revalidate refinement alignment after "
            "changing an explicitly compared terminal surface"
        )
        raise TypeError(msg)

    for record, refinement in zip(records, refinement_tuple, strict=True):
        if refinement is None:
            continue

        require_matching_refined_candidate(
            record_candidate=cast(CandidateT, record.candidate),
            refined_candidate=refinement.refined_candidate,
            mismatch_message=(
                "refinement refined_candidate must match the aligned "
                f"{record_label} candidate"
            ),
            candidate_equal=candidate_equal,
        )

    return refinement_tuple, current_refinement_pairs


def _normalize_terminal_refinements(
    *,
    records: Sequence[RequestAlignedEvaluationRecord],
    refinements: Sequence[CandidateRefinement[CandidateT] | None],
    record_label: str,
    candidate_equal: CandidateEquality[CandidateT] | None,
    carried_candidate_equal: CandidateEquality[CandidateT] | None,
    candidate_equal_required: bool,
    validated_refinement_pairs: tuple[tuple[CandidateT, CandidateT], ...],
) -> tuple[
    CandidateEquality[CandidateT] | None,
    bool,
    tuple[CandidateRefinement[CandidateT] | None, ...],
    tuple[tuple[CandidateT, CandidateT], ...],
]:
    effective_candidate_equal = candidate_equal
    if effective_candidate_equal is None:
        effective_candidate_equal = carried_candidate_equal

    next_candidate_equal_required = (
        candidate_equal_required or effective_candidate_equal is not None
    )
    effective_validated_pairs = validated_refinement_pairs
    if candidate_equal is not None:
        effective_validated_pairs = ()

    normalized_refinements, next_validated_pairs = _normalize_refinements(
        records=records,
        refinements=refinements,
        record_label=record_label,
        candidate_equal=effective_candidate_equal,
        candidate_equal_required=next_candidate_equal_required,
        validated_refinement_pairs=effective_validated_pairs,
    )

    return (
        effective_candidate_equal,
        next_candidate_equal_required,
        normalized_refinements,
        next_validated_pairs,
    )


def _optional_refinement_tuple(
    refinements: Sequence[CandidateRefinement[CandidateT] | None] | None,
) -> tuple[CandidateRefinement[CandidateT] | None, ...]:
    if refinements is None:
        return ()

    return tuple(refinements)


def _optional_failure_tuple(
    failures: Sequence[EvaluationFailure[CandidateT]] | None,
) -> tuple[EvaluationFailure[CandidateT], ...]:
    if failures is None:
        return ()

    return tuple(failures)


def _terminal_failure_evaluation_count(
    failures: Sequence[EvaluationFailure[CandidateT]],
) -> int:
    for failure in failures:
        if type(failure) is not EvaluationFailure:
            msg = "failures must contain EvaluationFailure values"
            raise TypeError(msg)

    return sum(failure.evaluation_count for failure in failures)


def _initialize_dataclass_fields(
    instance: FrozenGenericSlotsCompat,
    *,
    field_values: Mapping[str, object | None],
) -> None:
    dataclass_field_names = {dataclass_field.name for dataclass_field in fields(instance)}
    value_names = set(field_values)
    missing_names = dataclass_field_names - value_names
    extra_names = value_names - dataclass_field_names
    if missing_names or extra_names:
        msg = (
            "prevalidated surface construction field mismatch: "
            f"missing={sorted(missing_names)!r}, extra={sorted(extra_names)!r}"
        )
        raise RuntimeError(msg)

    for field_name, value in field_values.items():
        object.__setattr__(instance, field_name, value)


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Immutable diagnostics event emitted during a run.

    Parameters
    ----------
    kind : str
        Stable event category.
    message : str
        Human-readable event detail.
    proposal_id : str | None, optional
        Optional proposal identifier associated with the event.
    value : float | None, optional
        Optional scalar payload carried with the event.
    """

    kind: str
    message: str
    proposal_id: str | None = None
    value: float | None = None

    def __post_init__(self) -> None:
        """Validate trace-event metadata.

        Raises
        ------
        ValueError
            If ``kind`` or ``message`` is empty, or if ``proposal_id`` is an
            empty string.
        """
        if self.kind == "":
            msg = "kind must not be empty"
            raise ValueError(msg)

        if self.message == "":
            msg = "message must not be empty"
            raise ValueError(msg)

        if self.proposal_id == "":
            msg = "proposal_id must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Trace:
    """Immutable append-only diagnostics trace.

    Parameters
    ----------
    events : tuple[TraceEvent, ...], default=()
        Ordered events recorded during a run.
    """

    events: tuple[TraceEvent, ...] = ()

    def append(self, event: TraceEvent) -> Self:
        """Return a new trace containing one additional event.

        Parameters
        ----------
        event : TraceEvent
            Event to append.

        Returns
        -------
        Self
            New trace with ``event`` appended to the end.
        """
        return type(self)(events=self.events + (event,))


@dataclass(frozen=True, slots=True)
class RunReport(FrozenGenericSlotsCompat, Generic[CandidateT, RunRecordT]):
    """Terminal report for one completed study run.

    Parameters
    ----------
    records : tuple[RunRecordT, ...]
        Full ordered record history produced by the run.
    evaluation_count : int
        Total logical evaluation cost accrued during the run.
    trace : Trace, default=Trace()
        Diagnostics trace captured during execution.
    refinements : tuple[CandidateRefinement[CandidateT] | None, ...], default=()
        Optional record-aligned refinement provenance. An empty tuple means no
        refinement metadata was recorded for the run.
    failures : tuple[EvaluationFailure[CandidateT], ...], default=()
        Recorded failed evaluation attempts. Successful records remain in
        ``records`` only.
    """

    records: tuple[RunRecordT, ...]
    evaluation_count: int
    trace: Trace = field(default_factory=Trace)
    refinements: tuple[CandidateRefinement[CandidateT] | None, ...] = ()
    failures: tuple[EvaluationFailure[CandidateT], ...] = ()
    candidate_equal: InitVar[CandidateEquality[CandidateT] | None] = None
    _candidate_equal: CandidateEquality[CandidateT] | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _candidate_equal_required: bool = field(
        default=False,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _validated_refinement_pairs: tuple[tuple[CandidateT, CandidateT], ...] = field(
        default=(),
        repr=False,
        compare=False,
        kw_only=True,
    )

    def __post_init__(
        self,
        candidate_equal: CandidateEquality[CandidateT] | None,
    ) -> None:
        """Validate report accounting invariants.

        Raises
        ------
        ValueError
            If ``evaluation_count`` is negative, smaller than ``len(records)``,
            if non-empty refinement metadata is not aligned with records, or
            if a refinement's evaluated candidate disagrees with the aligned
            record candidate.
        TypeError
            If candidate equality does not produce a scalar truth value.
        """
        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)

        failure_evaluation_count = _terminal_failure_evaluation_count(self.failures)
        if self.evaluation_count < len(self.records) + failure_evaluation_count:
            msg = "evaluation_count must be at least the number of records"
            raise ValueError(msg)

        (
            effective_candidate_equal,
            candidate_equal_required,
            normalized_refinements,
            validated_refinement_pairs,
        ) = _normalize_terminal_refinements(
            records=self.records,
            refinements=self.refinements,
            record_label="records",
            candidate_equal=candidate_equal,
            carried_candidate_equal=self._candidate_equal,
            candidate_equal_required=self._candidate_equal_required,
            validated_refinement_pairs=self._validated_refinement_pairs,
        )
        object.__setattr__(self, "refinements", normalized_refinements)
        object.__setattr__(self, "_candidate_equal", effective_candidate_equal)
        object.__setattr__(self, "_candidate_equal_required", candidate_equal_required)
        object.__setattr__(
            self,
            "_validated_refinement_pairs",
            validated_refinement_pairs,
        )

    @classmethod
    def from_records(
        cls,
        records: Sequence[RunRecordT],
        evaluation_count: int | None = None,
        trace: Trace | None = None,
        refinements: Sequence[CandidateRefinement[CandidateT] | None] | None = None,
        failures: Sequence[EvaluationFailure[CandidateT]] | None = None,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
    ) -> Self:
        """Build a terminal report from an arbitrary record sequence.

        Parameters
        ----------
        records : Sequence[RunRecordT]
            Ordered record history to store in the report.
        evaluation_count : int | None, optional
            Optional logical evaluation cost. Defaults to ``len(records)``.
        trace : Trace | None, optional
            Optional diagnostics trace. Defaults to an empty trace.
        refinements : Sequence[CandidateRefinement[CandidateT] | None] | None, optional
            Optional record-aligned refinement provenance. ``None`` keeps the
            compact no-metadata sentinel. Aligned all-``None`` metadata is
            canonicalized to the same sentinel.
        failures : Sequence[EvaluationFailure[CandidateT]] | None, optional
            Recorded failed evaluation attempts.
        candidate_equal : CandidateEquality[CandidateT] | None, optional
            Explicit candidate equality predicate used to validate refinement
            alignment. When absent, strict scalar Python equality is used.

        Returns
        -------
        Self
            Canonical run report over ``records``.
        """
        record_tuple = tuple(records)
        normalized_trace = Trace() if trace is None else trace
        refinement_tuple = _optional_refinement_tuple(refinements)
        failure_tuple = _optional_failure_tuple(failures)
        failure_evaluation_count = _terminal_failure_evaluation_count(failure_tuple)
        normalized_evaluation_count = len(record_tuple) + failure_evaluation_count
        if evaluation_count is not None:
            normalized_evaluation_count = evaluation_count

        return cls(
            records=record_tuple,
            evaluation_count=normalized_evaluation_count,
            trace=normalized_trace,
            refinements=refinement_tuple,
            failures=failure_tuple,
            candidate_equal=candidate_equal,
        )


@dataclass(frozen=True, slots=True)
class RunResult(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Terminal scalar summary of a completed study run.

    Parameters
    ----------
    best_observation : Observation[CandidateT] | None
        Best scalar observation found during the run, if any.
    observations : tuple[Observation[CandidateT], ...]
        Full ordered scalar observation history.
    evaluation_count : int
        Total logical evaluation cost accrued during the run.
    trace : Trace, default=Trace()
        Diagnostics trace captured during execution.
    refinements : tuple[CandidateRefinement[CandidateT] | None, ...], default=()
        Optional observation-aligned refinement provenance. An empty tuple
        means no refinement metadata was recorded for the run.
    failures : tuple[EvaluationFailure[CandidateT], ...], default=()
        Recorded failed evaluation attempts. Successful scalar observations
        remain in ``observations`` only.
    """

    best_observation: Observation[CandidateT] | None
    observations: tuple[Observation[CandidateT], ...]
    evaluation_count: int
    trace: Trace = field(default_factory=Trace)
    refinements: tuple[CandidateRefinement[CandidateT] | None, ...] = ()
    failures: tuple[EvaluationFailure[CandidateT], ...] = ()
    candidate_equal: InitVar[CandidateEquality[CandidateT] | None] = None
    _candidate_equal: CandidateEquality[CandidateT] | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _candidate_equal_required: bool = field(
        default=False,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _validated_refinement_pairs: tuple[tuple[CandidateT, CandidateT], ...] = field(
        default=(),
        repr=False,
        compare=False,
        kw_only=True,
    )

    def __post_init__(
        self,
        candidate_equal: CandidateEquality[CandidateT] | None,
    ) -> None:
        """Validate scalar run-summary invariants.

        Raises
        ------
        ValueError
            If accounting is inconsistent or if ``best_observation`` does not
            match the minimum-score element of ``observations``.
        """
        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)

        failure_evaluation_count = _terminal_failure_evaluation_count(self.failures)
        if self.evaluation_count < len(self.observations) + failure_evaluation_count:
            msg = "evaluation_count must be at least the number of observations"
            raise ValueError(msg)

        if self.best_observation is None and self.observations:
            msg = "best_observation must be set when observations are present"
            raise ValueError(msg)

        if (
            self.best_observation is not None
            and self.best_observation not in self.observations
        ):
            msg = "best_observation must come from observations"
            raise ValueError(msg)

        if self.best_observation is not None and any(
            observation.score < self.best_observation.score
            for observation in self.observations
        ):
            msg = "best_observation must have the minimal observation score"
            raise ValueError(msg)

        (
            effective_candidate_equal,
            candidate_equal_required,
            normalized_refinements,
            validated_refinement_pairs,
        ) = _normalize_terminal_refinements(
            records=self.observations,
            refinements=self.refinements,
            record_label="observations",
            candidate_equal=candidate_equal,
            carried_candidate_equal=self._candidate_equal,
            candidate_equal_required=self._candidate_equal_required,
            validated_refinement_pairs=self._validated_refinement_pairs,
        )
        object.__setattr__(self, "refinements", normalized_refinements)
        object.__setattr__(self, "_candidate_equal", effective_candidate_equal)
        object.__setattr__(self, "_candidate_equal_required", candidate_equal_required)
        object.__setattr__(
            self,
            "_validated_refinement_pairs",
            validated_refinement_pairs,
        )

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[Observation[CandidateT]],
        evaluation_count: int | None = None,
        trace: Trace | None = None,
        refinements: Sequence[CandidateRefinement[CandidateT] | None] | None = None,
        failures: Sequence[EvaluationFailure[CandidateT]] | None = None,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
    ) -> Self:
        """Build a scalar run summary from an observation history.

        Parameters
        ----------
        observations : Sequence[Observation[CandidateT]]
            Ordered scalar observation history.
        evaluation_count : int | None, optional
            Optional logical evaluation cost. Defaults to ``len(observations)``.
        trace : Trace | None, optional
            Optional diagnostics trace. Defaults to an empty trace.
        refinements : Sequence[CandidateRefinement[CandidateT] | None] | None, optional
            Optional observation-aligned refinement provenance. ``None`` keeps
            the compact no-metadata sentinel.
        failures : Sequence[EvaluationFailure[CandidateT]] | None, optional
            Recorded failed evaluation attempts.
        candidate_equal : CandidateEquality[CandidateT] | None, optional
            Explicit candidate equality predicate used to validate refinement
            alignment. When absent, strict scalar Python equality is used.

        Returns
        -------
        Self
            Terminal scalar summary ordered by canonical minimization score.
        """
        observation_tuple = tuple(observations)
        normalized_trace = Trace() if trace is None else trace
        refinement_tuple = _optional_refinement_tuple(refinements)
        failure_tuple = _optional_failure_tuple(failures)
        failure_evaluation_count = _terminal_failure_evaluation_count(failure_tuple)
        normalized_evaluation_count = len(observation_tuple) + failure_evaluation_count
        if evaluation_count is not None:
            normalized_evaluation_count = evaluation_count

        if not observation_tuple:
            return cls(
                best_observation=None,
                observations=(),
                evaluation_count=normalized_evaluation_count,
                trace=normalized_trace,
                refinements=refinement_tuple,
                failures=failure_tuple,
                candidate_equal=candidate_equal,
            )

        best_observation = min(
            observation_tuple,
            key=lambda observation: observation.score,
        )

        return cls(
            best_observation=best_observation,
            observations=observation_tuple,
            evaluation_count=normalized_evaluation_count,
            trace=normalized_trace,
            refinements=refinement_tuple,
            failures=failure_tuple,
            candidate_equal=candidate_equal,
        )


def dominates_objective_scores(
    *,
    left_scores: tuple[float, ...],
    right_scores: tuple[float, ...],
) -> bool:
    """Return whether one score vector dominates another.

    Parameters
    ----------
    left_scores : tuple[float, ...]
        Candidate dominating score vector in canonical minimization form.
    right_scores : tuple[float, ...]
        Candidate dominated score vector in canonical minimization form.

    Returns
    -------
    bool
        ``True`` when ``left_scores`` is component-wise no worse than
        ``right_scores`` and strictly better in at least one component.

    Raises
    ------
    ValueError
        If the two vectors do not share the same dimension.
    """
    if len(left_scores) != len(right_scores):
        msg = "objective score vectors must have the same dimension"
        raise ValueError(msg)

    strictly_better = False
    for left_score, right_score in zip(left_scores, right_scores, strict=True):
        if left_score > right_score:
            return False
        if left_score < right_score:
            strictly_better = True

    return strictly_better


def collect_nondominated_records(
    records: tuple[ObjectiveVectorRecord[CandidateT], ...],
) -> tuple[ObjectiveVectorRecord[CandidateT], ...]:
    """Return the stable nondominated frontier of vector-valued records.

    Parameters
    ----------
    records : tuple[ObjectiveVectorRecord[CandidateT], ...]
        Candidate vector-valued records in the desired output stability order.

    Returns
    -------
    tuple[ObjectiveVectorRecord[CandidateT], ...]
        Nondominated subset of ``records`` in stable input order.
    """
    nondominated_records: list[ObjectiveVectorRecord[CandidateT]] = []
    for candidate_record in records:
        if any(
            dominates_objective_scores(
                left_scores=other_record.objective_scores,
                right_scores=candidate_record.objective_scores,
            )
            for other_record in records
            if other_record is not candidate_record
        ):
            continue
        nondominated_records.append(candidate_record)

    return tuple(nondominated_records)


@dataclass(frozen=True, slots=True)
class NondominatedRunSurface(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Terminal multi-objective surface over vector-valued records.

    Parameters
    ----------
    nondominated_records : tuple[ObjectiveVectorRecord[CandidateT], ...]
        Stable nondominated frontier of ``records``.
    records : tuple[ObjectiveVectorRecord[CandidateT], ...]
        Full ordered vector-valued record history.
    evaluation_count : int
        Total logical evaluation cost accrued during the run.
    trace : Trace, default=Trace()
        Diagnostics trace captured during execution.
    refinements : tuple[CandidateRefinement[CandidateT] | None, ...], default=()
        Optional record-aligned refinement provenance. An empty tuple means no
        refinement metadata was recorded for the run.
    failures : tuple[EvaluationFailure[CandidateT], ...], default=()
        Recorded failed evaluation attempts. Successful vector records remain
        in ``records`` only.
    """

    nondominated_records: tuple[ObjectiveVectorRecord[CandidateT], ...]
    records: tuple[ObjectiveVectorRecord[CandidateT], ...]
    evaluation_count: int
    trace: Trace = field(default_factory=Trace)
    refinements: tuple[CandidateRefinement[CandidateT] | None, ...] = ()
    failures: tuple[EvaluationFailure[CandidateT], ...] = ()
    candidate_equal: InitVar[CandidateEquality[CandidateT] | None] = None
    _candidate_equal: CandidateEquality[CandidateT] | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _candidate_equal_required: bool = field(
        default=False,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _validated_refinement_pairs: tuple[tuple[CandidateT, CandidateT], ...] = field(
        default=(),
        repr=False,
        compare=False,
        kw_only=True,
    )
    _validated_frontier_source_records: tuple[ObjectiveVectorRecord[CandidateT], ...] = (
        field(
            default=(),
            init=False,
            repr=False,
            compare=False,
        )
    )
    _validated_frontier_records: tuple[ObjectiveVectorRecord[CandidateT], ...] = field(
        default=(),
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def _from_prevalidated_frontier(
        cls,
        *,
        nondominated_records: tuple[ObjectiveVectorRecord[CandidateT], ...],
        records: tuple[ObjectiveVectorRecord[CandidateT], ...],
        evaluation_count: int,
        trace: Trace,
        refinements: tuple[CandidateRefinement[CandidateT] | None, ...],
        failures: tuple[EvaluationFailure[CandidateT], ...],
        candidate_equal: CandidateEquality[CandidateT] | None,
    ) -> Self:
        surface = cls.__new__(cls)
        _initialize_dataclass_fields(
            surface,
            field_values={
                "__orig_class__": None,
                "nondominated_records": nondominated_records,
                "records": records,
                "evaluation_count": evaluation_count,
                "trace": trace,
                "refinements": refinements,
                "failures": failures,
                "_candidate_equal": None,
                "_candidate_equal_required": False,
                "_validated_refinement_pairs": (),
                "_validated_frontier_source_records": records,
                "_validated_frontier_records": nondominated_records,
            },
        )
        surface.__post_init__(candidate_equal)
        return surface

    def __post_init__(
        self,
        candidate_equal: CandidateEquality[CandidateT] | None,
    ) -> None:
        """Validate multi-objective surface invariants.

        Raises
        ------
        ValueError
            If accounting is inconsistent, record dimensions disagree, the
            frontier does not come from ``records``, or
            ``nondominated_records`` is not the stable nondominated frontier.
        """
        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)

        failure_evaluation_count = _terminal_failure_evaluation_count(self.failures)
        if self.evaluation_count < len(self.records) + failure_evaluation_count:
            msg = "evaluation_count must be at least the number of records"
            raise ValueError(msg)

        if len({len(record.objective_scores) for record in self.records}) > 1:
            msg = "all objective score vectors must share one dimension"
            raise ValueError(msg)

        frontier_is_prevalidated = (
            self.records is self._validated_frontier_source_records
            and self.nondominated_records is self._validated_frontier_records
        )
        if not frontier_is_prevalidated and (
            self.nondominated_records != collect_nondominated_records(self.records)
        ):
            msg = (
                "nondominated_records must equal the stable nondominated frontier "
                "of records"
            )
            raise ValueError(msg)

        (
            effective_candidate_equal,
            candidate_equal_required,
            normalized_refinements,
            validated_refinement_pairs,
        ) = _normalize_terminal_refinements(
            records=self.records,
            refinements=self.refinements,
            record_label="records",
            candidate_equal=candidate_equal,
            carried_candidate_equal=self._candidate_equal,
            candidate_equal_required=self._candidate_equal_required,
            validated_refinement_pairs=self._validated_refinement_pairs,
        )
        object.__setattr__(self, "refinements", normalized_refinements)
        object.__setattr__(self, "_candidate_equal", effective_candidate_equal)
        object.__setattr__(self, "_candidate_equal_required", candidate_equal_required)
        object.__setattr__(
            self,
            "_validated_refinement_pairs",
            validated_refinement_pairs,
        )

    @classmethod
    def from_records(
        cls,
        records: Sequence[ObjectiveVectorRecord[CandidateT]],
        evaluation_count: int | None = None,
        trace: Trace | None = None,
        refinements: Sequence[CandidateRefinement[CandidateT] | None] | None = None,
        failures: Sequence[EvaluationFailure[CandidateT]] | None = None,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
    ) -> Self:
        """Build a nondominated surface from vector-valued records.

        Parameters
        ----------
        records : Sequence[ObjectiveVectorRecord[CandidateT]]
            Full ordered vector-valued record history.
        evaluation_count : int | None, optional
            Optional logical evaluation cost. Defaults to ``len(records)``.
        trace : Trace | None, optional
            Optional diagnostics trace. Defaults to an empty trace.
        refinements : Sequence[CandidateRefinement[CandidateT] | None] | None, optional
            Optional record-aligned refinement provenance. ``None`` keeps the
            compact no-metadata sentinel.
        failures : Sequence[EvaluationFailure[CandidateT]] | None, optional
            Recorded failed evaluation attempts.
        candidate_equal : CandidateEquality[CandidateT] | None, optional
            Explicit candidate equality predicate used to validate refinement
            alignment. When absent, strict scalar Python equality is used.

        Returns
        -------
        Self
            Terminal surface whose frontier is the stable nondominated subset
            of ``records``.
        """
        record_tuple = tuple(records)
        normalized_trace = Trace() if trace is None else trace
        refinement_tuple = _optional_refinement_tuple(refinements)
        failure_tuple = _optional_failure_tuple(failures)
        failure_evaluation_count = _terminal_failure_evaluation_count(failure_tuple)
        normalized_evaluation_count = len(record_tuple) + failure_evaluation_count
        if evaluation_count is not None:
            normalized_evaluation_count = evaluation_count

        nondominated_records = collect_nondominated_records(record_tuple)
        return cls._from_prevalidated_frontier(
            nondominated_records=nondominated_records,
            records=record_tuple,
            evaluation_count=normalized_evaluation_count,
            trace=normalized_trace,
            refinements=refinement_tuple,
            failures=failure_tuple,
            candidate_equal=candidate_equal,
        )

    @classmethod
    def from_report(
        cls,
        report: RunReport[CandidateT, ObjectiveVectorRecord[CandidateT]],
        candidate_equal: CandidateEquality[CandidateT] | None = None,
    ) -> Self:
        """Materialize a nondominated surface from a run report.

        Parameters
        ----------
        report : RunReport[CandidateT, ObjectiveVectorRecord[CandidateT]]
            Run report carrying vector-valued evaluation records.
        candidate_equal : CandidateEquality[CandidateT] | None, optional
            Explicit candidate equality predicate used to validate refinement
            alignment. When absent, strict scalar Python equality is used.

        Returns
        -------
        Self
            Terminal nondominated surface derived from ``report``.
        """
        return cls.from_records(
            records=report.records,
            evaluation_count=report.evaluation_count,
            trace=report.trace,
            refinements=report.refinements,
            failures=report.failures,
            candidate_equal=candidate_equal,
        )


def terminal_surface_getstate(self: FrozenGenericSlotsCompat) -> list[object | None]:
    state: list[object | None] = []
    for dataclass_field in fields(self):
        if dataclass_field.name == "_candidate_equal":
            state.append(None)
            continue
        if dataclass_field.name in {
            "_validated_frontier_source_records",
            "_validated_frontier_records",
        }:
            state.append(())
            continue
        state.append(getattr(self, dataclass_field.name, None))
    return state


def terminal_surface_setstate(
    self: FrozenGenericSlotsCompat,
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
        elif dataclass_field.name in {"failures", "_validated_refinement_pairs"}:
            object.__setattr__(self, dataclass_field.name, ())
        elif dataclass_field.name in {
            "_validated_frontier_source_records",
            "_validated_frontier_records",
        }:
            object.__setattr__(self, dataclass_field.name, ())
        else:
            object.__setattr__(self, dataclass_field.name, None)


setattr(RunReport, "__getstate__", terminal_surface_getstate)
setattr(RunReport, "__setstate__", terminal_surface_setstate)
setattr(RunResult, "__getstate__", terminal_surface_getstate)
setattr(RunResult, "__setstate__", terminal_surface_setstate)
setattr(NondominatedRunSurface, "__getstate__", terminal_surface_getstate)
setattr(NondominatedRunSurface, "__setstate__", terminal_surface_setstate)

del terminal_surface_getstate
del terminal_surface_setstate
