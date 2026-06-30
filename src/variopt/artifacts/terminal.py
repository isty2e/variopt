"""Terminal-surface artifact definitions."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..typevars import CandidateT
from .records import ObjectiveVectorRecord, Observation, RequestAlignedEvaluationRecord
from .refinement import CandidateRefinement

RunRecordT = TypeVar("RunRecordT", bound=RequestAlignedEvaluationRecord)


def _normalize_refinements(
    *,
    records: Sequence[RequestAlignedEvaluationRecord],
    refinements: Sequence[CandidateRefinement[CandidateT] | None],
    record_label: str,
) -> tuple[CandidateRefinement[CandidateT] | None, ...]:
    refinement_tuple = tuple(refinements)
    if refinement_tuple == ():
        return ()

    if len(refinement_tuple) != len(records):
        msg = f"refinements must be empty or align with {record_label}"
        raise ValueError(msg)

    if all(refinement is None for refinement in refinement_tuple):
        return ()

    for record, refinement in zip(records, refinement_tuple, strict=True):
        if refinement is None:
            continue

        try:
            candidates_match = bool(
                record.candidate == refinement.refined_candidate
            )
        except ValueError as error:
            msg = "candidate equality must produce a scalar truth value"
            raise TypeError(msg) from error

        if not candidates_match:
            msg = (
                "refinement refined_candidate must match the aligned "
                f"{record_label} candidate"
            )
            raise ValueError(msg)

    return refinement_tuple


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
    """

    records: tuple[RunRecordT, ...]
    evaluation_count: int
    trace: Trace = field(default_factory=Trace)
    refinements: tuple[CandidateRefinement[CandidateT] | None, ...] = ()

    def __post_init__(self) -> None:
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

        if self.evaluation_count < len(self.records):
            msg = "evaluation_count must be at least the number of records"
            raise ValueError(msg)

        object.__setattr__(
            self,
            "refinements",
            _normalize_refinements(
                records=self.records,
                refinements=self.refinements,
                record_label="records",
            ),
        )

    @classmethod
    def from_records(
        cls,
        records: Sequence[RunRecordT],
        evaluation_count: int | None = None,
        trace: Trace | None = None,
        refinements: Sequence[CandidateRefinement[CandidateT] | None] | None = None,
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

        Returns
        -------
        Self
            Canonical run report over ``records``.
        """
        record_tuple = tuple(records)
        normalized_trace = Trace() if trace is None else trace
        refinement_tuple = () if refinements is None else tuple(refinements)
        normalized_evaluation_count = len(record_tuple)
        if evaluation_count is not None:
            normalized_evaluation_count = evaluation_count

        return cls(
            records=record_tuple,
            evaluation_count=normalized_evaluation_count,
            trace=normalized_trace,
            refinements=refinement_tuple,
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
    """

    best_observation: Observation[CandidateT] | None
    observations: tuple[Observation[CandidateT], ...]
    evaluation_count: int
    trace: Trace = field(default_factory=Trace)
    refinements: tuple[CandidateRefinement[CandidateT] | None, ...] = ()

    def __post_init__(self) -> None:
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

        if self.evaluation_count < len(self.observations):
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

        object.__setattr__(
            self,
            "refinements",
            _normalize_refinements(
                records=self.observations,
                refinements=self.refinements,
                record_label="observations",
            ),
        )

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[Observation[CandidateT]],
        evaluation_count: int | None = None,
        trace: Trace | None = None,
        refinements: Sequence[CandidateRefinement[CandidateT] | None] | None = None,
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

        Returns
        -------
        Self
            Terminal scalar summary ordered by canonical minimization score.
        """
        observation_tuple = tuple(observations)
        normalized_trace = Trace() if trace is None else trace
        refinement_tuple = () if refinements is None else tuple(refinements)
        normalized_evaluation_count = len(observation_tuple)
        if evaluation_count is not None:
            normalized_evaluation_count = evaluation_count

        if not observation_tuple:
            return cls(
                best_observation=None,
                observations=(),
                evaluation_count=normalized_evaluation_count,
                trace=normalized_trace,
                refinements=refinement_tuple,
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

    return all(
        left_score <= right_score
        for left_score, right_score in zip(left_scores, right_scores, strict=True)
    ) and any(
        left_score < right_score
        for left_score, right_score in zip(left_scores, right_scores, strict=True)
    )


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
    """

    nondominated_records: tuple[ObjectiveVectorRecord[CandidateT], ...]
    records: tuple[ObjectiveVectorRecord[CandidateT], ...]
    evaluation_count: int
    trace: Trace = field(default_factory=Trace)
    refinements: tuple[CandidateRefinement[CandidateT] | None, ...] = ()

    def __post_init__(self) -> None:
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

        if self.evaluation_count < len(self.records):
            msg = "evaluation_count must be at least the number of records"
            raise ValueError(msg)

        if len({len(record.objective_scores) for record in self.records}) > 1:
            msg = "all objective score vectors must share one dimension"
            raise ValueError(msg)

        if any(record not in self.records for record in self.nondominated_records):
            msg = "nondominated_records must come from records"
            raise ValueError(msg)

        expected_frontier = collect_nondominated_records(self.records)
        if self.nondominated_records != expected_frontier:
            msg = (
                "nondominated_records must equal the stable nondominated frontier "
                "of records"
            )
            raise ValueError(msg)

        object.__setattr__(
            self,
            "refinements",
            _normalize_refinements(
                records=self.records,
                refinements=self.refinements,
                record_label="records",
            ),
        )

    @classmethod
    def from_records(
        cls,
        records: Sequence[ObjectiveVectorRecord[CandidateT]],
        evaluation_count: int | None = None,
        trace: Trace | None = None,
        refinements: Sequence[CandidateRefinement[CandidateT] | None] | None = None,
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

        Returns
        -------
        Self
            Terminal surface whose frontier is the stable nondominated subset
            of ``records``.
        """
        record_tuple = tuple(records)
        normalized_trace = Trace() if trace is None else trace
        refinement_tuple = () if refinements is None else tuple(refinements)
        normalized_evaluation_count = len(record_tuple)
        if evaluation_count is not None:
            normalized_evaluation_count = evaluation_count

        return cls(
            nondominated_records=collect_nondominated_records(record_tuple),
            records=record_tuple,
            evaluation_count=normalized_evaluation_count,
            trace=normalized_trace,
            refinements=refinement_tuple,
        )

    @classmethod
    def from_report(
        cls,
        report: RunReport[CandidateT, ObjectiveVectorRecord[CandidateT]],
    ) -> Self:
        """Materialize a nondominated surface from a run report.

        Parameters
        ----------
        report : RunReport[CandidateT, ObjectiveVectorRecord[CandidateT]]
            Run report carrying vector-valued evaluation records.

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
        )
