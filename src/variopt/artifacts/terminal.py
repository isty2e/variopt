"""Terminal-surface artifact definitions."""

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field, fields, replace
from typing import Generic, TypeVar

from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..spaces import CandidateEquality
from ..spaces.equality import require_candidate_match
from ..typevars import CandidateT
from .attempts import (
    EvaluationFailure,
    EvaluationSuccess,
    materialize_success_records,
)
from .records import (
    ObjectiveVectorPayload,
    ObjectiveVectorRecord,
    Observation,
    ObservationPayload,
    RequestAlignedEvaluationRecord,
)
from .refinement import CandidateRefinement
from .requests import EvaluationRequest, Proposal

RunRecordT = TypeVar("RunRecordT", bound=RequestAlignedEvaluationRecord[object])
SuccessPayloadT = TypeVar("SuccessPayloadT")
TerminalRecordCandidateT = TypeVar("TerminalRecordCandidateT")


def _candidate_matches(
    *,
    left_candidate: TerminalRecordCandidateT,
    right_candidate: TerminalRecordCandidateT,
    candidate_equal: CandidateEquality[TerminalRecordCandidateT] | None,
) -> bool:
    mismatch_message = "candidate mismatch"
    try:
        require_candidate_match(
            left_candidate=left_candidate,
            right_candidate=right_candidate,
            mismatch_message=mismatch_message,
            candidate_equal=candidate_equal,
        )
    except ValueError as exception:
        if str(exception) != mismatch_message:
            raise
        return False
    return True


def _record_request_metadata_matches(
    left_record: RequestAlignedEvaluationRecord[CandidateT],
    right_record: RequestAlignedEvaluationRecord[CandidateT],
) -> bool:
    if left_record.request.proposal_id != right_record.request.proposal_id:
        return False

    return (
        left_record.request.proposal_evaluation_spec
        == right_record.request.proposal_evaluation_spec
    )


def _record_candidate_projection_matches(
    left_record: RequestAlignedEvaluationRecord[CandidateT],
    right_record: RequestAlignedEvaluationRecord[CandidateT],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> bool:
    if not _candidate_matches(
        left_candidate=left_record.candidate,
        right_candidate=right_record.candidate,
        candidate_equal=candidate_equal,
    ):
        return False

    return _candidate_matches(
        left_candidate=left_record.request.candidate,
        right_candidate=right_record.request.candidate,
        candidate_equal=candidate_equal,
    )


def _observation_matches_success(
    observation: Observation[CandidateT],
    success: EvaluationSuccess[CandidateT, ObservationPayload],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> bool:
    projected_observation = _observation_from_success(success)
    if not _record_request_metadata_matches(observation, projected_observation):
        return False

    if not _record_candidate_projection_matches(
        observation,
        projected_observation,
        candidate_equal=candidate_equal,
    ):
        return False

    return (
        observation.value == projected_observation.value
        and observation.score == projected_observation.score
        and observation.elapsed_seconds == projected_observation.elapsed_seconds
    )


def _objective_vector_record_matches_success(
    record: ObjectiveVectorRecord[CandidateT],
    success: EvaluationSuccess[CandidateT, ObjectiveVectorPayload],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> bool:
    projected_record = _vector_record_from_success(success)
    if not _record_request_metadata_matches(record, projected_record):
        return False

    if not _record_candidate_projection_matches(
        record,
        projected_record,
        candidate_equal=candidate_equal,
    ):
        return False

    return (
        record.objective_values == projected_record.objective_values
        and record.objective_scores == projected_record.objective_scores
        and record.elapsed_seconds == projected_record.elapsed_seconds
    )


def _success_matching_observation(
    observation: Observation[CandidateT],
    successes: Sequence[EvaluationSuccess[CandidateT, ObservationPayload]],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> EvaluationSuccess[CandidateT, ObservationPayload]:
    for success in successes:
        if _observation_matches_success(
            observation,
            success,
            candidate_equal=candidate_equal,
        ):
            return success

    msg = "best_observation must come from observations"
    raise ValueError(msg)


def _successes_matching_vector_records(
    records: Sequence[ObjectiveVectorRecord[CandidateT]],
    successes: Sequence[EvaluationSuccess[CandidateT, ObjectiveVectorPayload]],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> tuple[EvaluationSuccess[CandidateT, ObjectiveVectorPayload], ...]:
    matched_successes: list[EvaluationSuccess[CandidateT, ObjectiveVectorPayload]] = []
    for record in records:
        for success in successes:
            if _objective_vector_record_matches_success(
                record,
                success,
                candidate_equal=candidate_equal,
            ):
                matched_successes.append(success)
                break
        else:
            msg = (
                "nondominated_records must equal the stable nondominated frontier "
                "of records"
            )
            raise ValueError(msg)

    return tuple(matched_successes)


def _record_success_request(
    record: RequestAlignedEvaluationRecord[TerminalRecordCandidateT],
    *,
    refinement: CandidateRefinement[TerminalRecordCandidateT] | None,
    candidate_equal: CandidateEquality[TerminalRecordCandidateT] | None,
) -> EvaluationRequest[TerminalRecordCandidateT]:
    source_request = record.request
    record_candidate = record.candidate
    if source_request.candidate is record_candidate:
        return source_request

    if _candidate_matches(
        left_candidate=source_request.candidate,
        right_candidate=record_candidate,
        candidate_equal=candidate_equal,
    ):
        return EvaluationRequest(
            proposal=Proposal(
                candidate=record_candidate,
                proposal_id=source_request.proposal_id,
            ),
            proposal_evaluation_spec=source_request.proposal_evaluation_spec,
        )

    if refinement is None:
        msg = (
            "refinement is required when record request candidate differs from "
            "the evaluated candidate"
        )
        raise ValueError(msg)

    return EvaluationRequest(
        proposal=Proposal(
            candidate=record_candidate,
            proposal_id=source_request.proposal_id,
        ),
        proposal_evaluation_spec=source_request.proposal_evaluation_spec,
    )


def _successes_from_records(
    *,
    records: Sequence[RequestAlignedEvaluationRecord[CandidateT]],
    refinements: Sequence[CandidateRefinement[CandidateT] | None],
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> tuple[EvaluationSuccess[CandidateT, RequestAlignedEvaluationRecord[CandidateT]], ...]:
    record_tuple = tuple(records)
    refinement_tuple = tuple(refinements)
    if refinement_tuple and len(refinement_tuple) != len(record_tuple):
        msg = "refinements must be empty or align with records"
        raise ValueError(msg)

    if refinement_tuple == ():
        refinement_tuple = tuple(None for _record in record_tuple)

    return tuple(
        EvaluationSuccess(
            request=_record_success_request(
                record,
                refinement=refinement,
                candidate_equal=candidate_equal,
            ),
            payload=record,
            refinement=refinement,
            candidate_equal=candidate_equal,
        )
        for record, refinement in zip(record_tuple, refinement_tuple, strict=True)
    )


def _successes_from_observations(
    *,
    observations: Sequence[Observation[CandidateT]],
    refinements: Sequence[CandidateRefinement[CandidateT] | None],
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> tuple[EvaluationSuccess[CandidateT, ObservationPayload], ...]:
    observation_tuple = tuple(observations)
    refinement_tuple = tuple(refinements)
    if refinement_tuple and len(refinement_tuple) != len(observation_tuple):
        msg = "refinements must be empty or align with observations"
        raise ValueError(msg)

    if refinement_tuple == ():
        refinement_tuple = tuple(None for _observation in observation_tuple)

    return tuple(
        EvaluationSuccess(
            request=_record_success_request(
                observation,
                refinement=refinement,
                candidate_equal=candidate_equal,
            ),
            payload=ObservationPayload(
                value=observation.value,
                score=observation.score,
                elapsed_seconds=observation.elapsed_seconds,
            ),
            refinement=refinement,
            candidate_equal=candidate_equal,
        )
        for observation, refinement in zip(
            observation_tuple,
            refinement_tuple,
            strict=True,
        )
    )


def _successes_from_vector_records(
    *,
    records: Sequence[ObjectiveVectorRecord[CandidateT]],
    refinements: Sequence[CandidateRefinement[CandidateT] | None],
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> tuple[EvaluationSuccess[CandidateT, ObjectiveVectorPayload], ...]:
    record_tuple = tuple(records)
    refinement_tuple = tuple(refinements)
    if refinement_tuple and len(refinement_tuple) != len(record_tuple):
        msg = "refinements must be empty or align with records"
        raise ValueError(msg)

    if refinement_tuple == ():
        refinement_tuple = tuple(None for _record in record_tuple)

    return tuple(
        EvaluationSuccess(
            request=_record_success_request(
                record,
                refinement=refinement,
                candidate_equal=candidate_equal,
            ),
            payload=ObjectiveVectorPayload(
                objective_values=record.objective_values,
                objective_scores=record.objective_scores,
                elapsed_seconds=record.elapsed_seconds,
            ),
            refinement=refinement,
            candidate_equal=candidate_equal,
        )
        for record, refinement in zip(record_tuple, refinement_tuple, strict=True)
    )


def _normalize_successes(
    *,
    successes: Sequence[EvaluationSuccess[CandidateT, SuccessPayloadT]],
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> tuple[EvaluationSuccess[CandidateT, SuccessPayloadT], ...]:
    success_tuple = tuple(successes)
    for success in success_tuple:
        if type(success) is not EvaluationSuccess:
            msg = "successes must contain EvaluationSuccess values"
            raise TypeError(msg)

    if candidate_equal is None:
        return success_tuple

    return tuple(
        EvaluationSuccess(
            request=success.request,
            payload=success.payload,
            evaluation_count=success.evaluation_count,
            refinement=success.refinement,
            kernel_diagnostics=success.kernel_diagnostics,
            candidate_equal=candidate_equal,
        )
        for success in success_tuple
    )


def _success_refinements(
    successes: Sequence[EvaluationSuccess[CandidateT, SuccessPayloadT]],
) -> tuple[CandidateRefinement[CandidateT] | None, ...]:
    refinements = tuple(success.refinement for success in successes)
    if all(refinement is None for refinement in refinements):
        return ()
    return refinements


def _successes_with_refinements(
    *,
    successes: Sequence[EvaluationSuccess[CandidateT, SuccessPayloadT]],
    refinements: Sequence[CandidateRefinement[CandidateT] | None],
    candidate_equal: CandidateEquality[CandidateT] | None,
) -> tuple[EvaluationSuccess[CandidateT, SuccessPayloadT], ...]:
    success_tuple = tuple(successes)
    refinement_tuple = tuple(refinements)
    if len(refinement_tuple) != len(success_tuple):
        msg = "refinements must align with successes"
        raise ValueError(msg)

    if candidate_equal is None:
        return tuple(
            replace(success, refinement=refinement)
            for success, refinement in zip(
                success_tuple,
                refinement_tuple,
                strict=True,
            )
        )

    return tuple(
        replace(
            success,
            refinement=refinement,
            candidate_equal=candidate_equal,
        )
        for success, refinement in zip(
            success_tuple,
            refinement_tuple,
            strict=True,
        )
    )


def _success_evaluation_count(
    successes: Sequence[EvaluationSuccess[CandidateT, SuccessPayloadT]],
) -> int:
    return sum(success.evaluation_count for success in successes)


def _projection_proposal_for_success(
    success: EvaluationSuccess[CandidateT, SuccessPayloadT],
) -> Proposal[CandidateT]:
    refinement = success.refinement
    if refinement is None:
        return success.request.proposal

    return Proposal(
        candidate=refinement.source_candidate,
        proposal_id=success.proposal_id,
    )


def _observation_from_success(
    success: EvaluationSuccess[CandidateT, ObservationPayload],
) -> Observation[CandidateT]:
    payload = success.payload
    return Observation(
        proposal=_projection_proposal_for_success(success),
        proposal_evaluation_spec=success.request.proposal_evaluation_spec,
        candidate=success.request.candidate,
        value=payload.value,
        score=payload.score,
        elapsed_seconds=payload.elapsed_seconds,
    )


def _vector_record_from_success(
    success: EvaluationSuccess[CandidateT, ObjectiveVectorPayload],
) -> ObjectiveVectorRecord[CandidateT]:
    payload = success.payload
    return ObjectiveVectorRecord(
        proposal=_projection_proposal_for_success(success),
        proposal_evaluation_spec=success.request.proposal_evaluation_spec,
        candidate=success.request.candidate,
        objective_values=payload.objective_values,
        objective_scores=payload.objective_scores,
        elapsed_seconds=payload.elapsed_seconds,
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
        failure.__post_init__()

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


@dataclass(frozen=True, slots=True, init=False)
class Trace:
    """Immutable append-only diagnostics trace.

    Parameters
    ----------
    events : Sequence[TraceEvent], default=()
        Ordered events recorded during a run.
    """

    events: tuple[TraceEvent, ...] = ()

    def __init__(self, events: Sequence[TraceEvent] = ()) -> None:
        """Create a trace with a canonical immutable event sequence."""
        object.__setattr__(self, "events", tuple(events))

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


@dataclass(frozen=True, slots=True, init=False)
class RunReport(FrozenGenericSlotsCompat, Generic[CandidateT, RunRecordT]):
    """Terminal report for one completed study run.

    Parameters
    ----------
    successes : tuple[EvaluationSuccess[CandidateT, RunRecordT], ...]
        Full ordered successful attempt history produced by the run.
    evaluation_count : int
        Total logical evaluation cost accrued during the run.
    trace : Trace, default=Trace()
        Diagnostics trace captured during execution.
    failures : tuple[EvaluationFailure[CandidateT], ...], default=()
        Recorded failed evaluation attempts. Successful payloads remain in
        ``successes`` only.
    """

    successes: tuple[EvaluationSuccess[CandidateT, RunRecordT], ...]
    evaluation_count: int
    trace: Trace = field(default_factory=Trace)
    failures: tuple[EvaluationFailure[CandidateT], ...] = ()
    candidate_equal: InitVar[CandidateEquality[CandidateT] | None] = None

    def __init__(
        self,
        *,
        successes: Sequence[EvaluationSuccess[CandidateT, RunRecordT]] | None = None,
        evaluation_count: int,
        trace: Trace | None = None,
        failures: Sequence[EvaluationFailure[CandidateT]] | None = None,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
        refinements: Sequence[CandidateRefinement[CandidateT] | None] | None = None,
    ) -> None:
        """Create one terminal report from request-owned successes."""
        normalized_successes: tuple[EvaluationSuccess[CandidateT, RunRecordT], ...]
        if successes is None:
            normalized_successes = ()
        else:
            normalized_successes = tuple(successes)
            if refinements is not None:
                normalized_successes = _successes_with_refinements(
                    successes=normalized_successes,
                    refinements=refinements,
                    candidate_equal=candidate_equal,
                )

        object.__setattr__(self, "__orig_class__", None)
        object.__setattr__(self, "successes", normalized_successes)
        object.__setattr__(self, "evaluation_count", evaluation_count)
        object.__setattr__(self, "trace", Trace() if trace is None else trace)
        object.__setattr__(self, "failures", _optional_failure_tuple(failures))
        self.__post_init__(candidate_equal)

    def __post_init__(
        self,
        candidate_equal: CandidateEquality[CandidateT] | None,
    ) -> None:
        """Validate report accounting invariants.

        Raises
        ------
        ValueError
            If ``evaluation_count`` is negative or smaller than the reported
            successful and failed attempt costs.
        """
        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)

        normalized_successes = _normalize_successes(
            successes=self.successes,
            candidate_equal=candidate_equal,
        )
        object.__setattr__(self, "successes", normalized_successes)

        failure_evaluation_count = _terminal_failure_evaluation_count(self.failures)
        success_evaluation_count = _success_evaluation_count(normalized_successes)
        if self.evaluation_count < success_evaluation_count + failure_evaluation_count:
            msg = "evaluation_count must be at least the terminal attempt cost"
            raise ValueError(msg)

    @property
    def records(self) -> tuple[RunRecordT, ...]:
        """Return successful attempts as request-aligned record projections."""
        return materialize_success_records(self.successes)

    @property
    def refinements(self) -> tuple[CandidateRefinement[CandidateT] | None, ...]:
        """Return success-aligned refinement provenance."""
        return _success_refinements(self.successes)

    @classmethod
    def from_successes(
        cls,
        successes: Sequence[EvaluationSuccess[CandidateT, RunRecordT]],
        evaluation_count: int | None = None,
        trace: Trace | None = None,
        failures: Sequence[EvaluationFailure[CandidateT]] | None = None,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
    ) -> Self:
        """Build a terminal report from request-owned successes."""
        success_tuple = tuple(successes)
        normalized_trace = Trace() if trace is None else trace
        failure_tuple = _optional_failure_tuple(failures)
        failure_evaluation_count = _terminal_failure_evaluation_count(failure_tuple)
        normalized_evaluation_count = (
            _success_evaluation_count(success_tuple) + failure_evaluation_count
        )
        if evaluation_count is not None:
            normalized_evaluation_count = evaluation_count

        return cls(
            successes=success_tuple,
            evaluation_count=normalized_evaluation_count,
            trace=normalized_trace,
            failures=failure_tuple,
            candidate_equal=candidate_equal,
        )

    @classmethod
    def from_records(
        cls,
        records: Sequence[RequestAlignedEvaluationRecord[CandidateT]],
        evaluation_count: int | None = None,
        trace: Trace | None = None,
        refinements: Sequence[CandidateRefinement[CandidateT] | None] | None = None,
        failures: Sequence[EvaluationFailure[CandidateT]] | None = None,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
    ) -> "RunReport[CandidateT, RequestAlignedEvaluationRecord[CandidateT]]":
        """Build a terminal report from an arbitrary record sequence.

        Parameters
        ----------
        records : Sequence[RequestAlignedEvaluationRecord[CandidateT]]
            Ordered request-aligned record history to store in the report.
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
            Canonical run report over request-owned successes.
        """
        normalized_trace = Trace() if trace is None else trace
        refinement_tuple = _optional_refinement_tuple(refinements)
        failure_tuple = _optional_failure_tuple(failures)
        failure_evaluation_count = _terminal_failure_evaluation_count(failure_tuple)
        successes = _successes_from_records(
            records=records,
            refinements=refinement_tuple,
            candidate_equal=candidate_equal,
        )
        normalized_evaluation_count = (
            _success_evaluation_count(successes) + failure_evaluation_count
        )
        if evaluation_count is not None:
            normalized_evaluation_count = evaluation_count

        _ = cls
        return RunReport[CandidateT, RequestAlignedEvaluationRecord[CandidateT]](
            successes=successes,
            evaluation_count=normalized_evaluation_count,
            trace=normalized_trace,
            failures=failure_tuple,
            candidate_equal=candidate_equal,
        )


@dataclass(frozen=True, slots=True, init=False)
class RunResult(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Terminal scalar summary of a completed study run.

    Parameters
    ----------
    best_success : EvaluationSuccess[CandidateT, ObservationPayload] | None
        Best scalar success found during the run, if any.
    successes : tuple[EvaluationSuccess[CandidateT, ObservationPayload], ...]
        Full ordered scalar success history.
    evaluation_count : int
        Total logical evaluation cost accrued during the run.
    trace : Trace, default=Trace()
        Diagnostics trace captured during execution.
    failures : tuple[EvaluationFailure[CandidateT], ...], default=()
        Recorded failed evaluation attempts. Successful scalar payloads remain
        in ``successes`` only.
    """

    best_success: EvaluationSuccess[CandidateT, ObservationPayload] | None
    successes: tuple[EvaluationSuccess[CandidateT, ObservationPayload], ...]
    evaluation_count: int
    trace: Trace = field(default_factory=Trace)
    failures: tuple[EvaluationFailure[CandidateT], ...] = ()
    candidate_equal: InitVar[CandidateEquality[CandidateT] | None] = None

    def __init__(
        self,
        *,
        best_success: EvaluationSuccess[CandidateT, ObservationPayload] | None = None,
        successes: Sequence[EvaluationSuccess[CandidateT, ObservationPayload]]
        | None = None,
        evaluation_count: int,
        trace: Trace | None = None,
        failures: Sequence[EvaluationFailure[CandidateT]] | None = None,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
        best_observation: Observation[CandidateT] | None = None,
        observations: Sequence[Observation[CandidateT]] | None = None,
        refinements: Sequence[CandidateRefinement[CandidateT] | None] | None = None,
    ) -> None:
        """Create one scalar result, normalizing observations to successes."""
        if best_success is not None and best_observation is not None:
            msg = "provide either best_success or best_observation, not both"
            raise ValueError(msg)
        normalized_successes: tuple[
            EvaluationSuccess[CandidateT, ObservationPayload],
            ...,
        ]
        normalized_best_success = best_success
        post_init_candidate_equal = candidate_equal
        if observations is not None:
            normalized_successes = _successes_from_observations(
                observations=observations,
                refinements=_optional_refinement_tuple(refinements),
                candidate_equal=candidate_equal,
            )
            post_init_candidate_equal = None
            if best_observation is not None:
                normalized_best_success = _success_matching_observation(
                    best_observation,
                    normalized_successes,
                    candidate_equal=candidate_equal,
                )
            elif normalized_successes:
                normalized_best_success = min(
                    normalized_successes,
                    key=lambda success: success.payload.score,
                )
            else:
                normalized_best_success = None
        elif successes is not None:
            normalized_successes = tuple(successes)
            if refinements is not None:
                normalized_successes = _successes_with_refinements(
                    successes=normalized_successes,
                    refinements=refinements,
                    candidate_equal=candidate_equal,
                )
                if normalized_successes:
                    normalized_best_success = min(
                        normalized_successes,
                        key=lambda success: success.payload.score,
                    )
                else:
                    normalized_best_success = None
        else:
            normalized_successes = ()

        object.__setattr__(self, "__orig_class__", None)
        object.__setattr__(self, "best_success", normalized_best_success)
        object.__setattr__(self, "successes", normalized_successes)
        object.__setattr__(self, "evaluation_count", evaluation_count)
        object.__setattr__(self, "trace", Trace() if trace is None else trace)
        object.__setattr__(self, "failures", _optional_failure_tuple(failures))
        self.__post_init__(post_init_candidate_equal)

    def __post_init__(
        self,
        candidate_equal: CandidateEquality[CandidateT] | None,
    ) -> None:
        """Validate scalar run-summary invariants.

        Raises
        ------
        ValueError
            If accounting is inconsistent or if ``best_success`` does not match
            the minimum-score element of ``successes``.
        """
        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)

        normalized_successes = _normalize_successes(
            successes=self.successes,
            candidate_equal=candidate_equal,
        )
        object.__setattr__(self, "successes", normalized_successes)
        normalized_best_success = self.best_success
        if normalized_best_success is not None and candidate_equal is not None:
            normalized_best_success = EvaluationSuccess(
                request=normalized_best_success.request,
                payload=normalized_best_success.payload,
                evaluation_count=normalized_best_success.evaluation_count,
                refinement=normalized_best_success.refinement,
                kernel_diagnostics=normalized_best_success.kernel_diagnostics,
                candidate_equal=candidate_equal,
            )
            object.__setattr__(self, "best_success", normalized_best_success)

        failure_evaluation_count = _terminal_failure_evaluation_count(self.failures)
        success_evaluation_count = _success_evaluation_count(normalized_successes)
        if self.evaluation_count < success_evaluation_count + failure_evaluation_count:
            msg = "evaluation_count must be at least the terminal attempt cost"
            raise ValueError(msg)

        if self.best_success is None and normalized_successes:
            msg = "best_success must be set when successes are present"
            raise ValueError(msg)

        if self.best_success is not None and not any(
            success is self.best_success for success in normalized_successes
        ):
            if self.best_success not in normalized_successes:
                msg = "best_success must come from successes"
                raise ValueError(msg)

        if self.best_success is not None and any(
            success.payload.score < self.best_success.payload.score
            for success in normalized_successes
        ):
            msg = "best_success must have the minimal success score"
            raise ValueError(msg)

    @property
    def observations(self) -> tuple[Observation[CandidateT], ...]:
        """Return scalar observations projected from request-owned successes."""
        return tuple(_observation_from_success(success) for success in self.successes)

    @property
    def best_observation(self) -> Observation[CandidateT] | None:
        """Return the best scalar observation projection, if any."""
        if self.best_success is None:
            return None
        return _observation_from_success(self.best_success)

    @property
    def refinements(self) -> tuple[CandidateRefinement[CandidateT] | None, ...]:
        """Return success-aligned refinement provenance."""
        return _success_refinements(self.successes)

    @classmethod
    def from_successes(
        cls,
        successes: Sequence[EvaluationSuccess[CandidateT, ObservationPayload]],
        evaluation_count: int | None = None,
        trace: Trace | None = None,
        failures: Sequence[EvaluationFailure[CandidateT]] | None = None,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
    ) -> Self:
        """Build a scalar run summary from request-owned scalar successes."""
        success_tuple = tuple(successes)
        normalized_trace = Trace() if trace is None else trace
        failure_tuple = _optional_failure_tuple(failures)
        failure_evaluation_count = _terminal_failure_evaluation_count(failure_tuple)
        normalized_evaluation_count = (
            _success_evaluation_count(success_tuple) + failure_evaluation_count
        )
        if evaluation_count is not None:
            normalized_evaluation_count = evaluation_count

        best_success: EvaluationSuccess[CandidateT, ObservationPayload] | None = None
        if success_tuple:
            best_success = min(
                success_tuple,
                key=lambda success: success.payload.score,
            )

        return cls(
            best_success=best_success,
            successes=success_tuple,
            evaluation_count=normalized_evaluation_count,
            trace=normalized_trace,
            failures=failure_tuple,
            candidate_equal=candidate_equal,
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
            Terminal scalar summary ordered by successful attempt slot.
        """
        normalized_trace = Trace() if trace is None else trace
        refinement_tuple = _optional_refinement_tuple(refinements)
        failure_tuple = _optional_failure_tuple(failures)
        failure_evaluation_count = _terminal_failure_evaluation_count(failure_tuple)
        successes = _successes_from_observations(
            observations=observations,
            refinements=refinement_tuple,
            candidate_equal=candidate_equal,
        )
        normalized_evaluation_count = (
            _success_evaluation_count(successes) + failure_evaluation_count
        )
        if evaluation_count is not None:
            normalized_evaluation_count = evaluation_count

        return cls.from_successes(
            successes=successes,
            evaluation_count=normalized_evaluation_count,
            trace=normalized_trace,
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


def collect_nondominated_successes(
    successes: tuple[EvaluationSuccess[CandidateT, ObjectiveVectorPayload], ...],
) -> tuple[EvaluationSuccess[CandidateT, ObjectiveVectorPayload], ...]:
    """Return the stable nondominated frontier of vector-valued successes."""
    nondominated_successes: list[
        EvaluationSuccess[CandidateT, ObjectiveVectorPayload]
    ] = []
    for candidate_success in successes:
        if any(
            dominates_objective_scores(
                left_scores=other_success.payload.objective_scores,
                right_scores=candidate_success.payload.objective_scores,
            )
            for other_success in successes
            if other_success is not candidate_success
        ):
            continue
        nondominated_successes.append(candidate_success)

    return tuple(nondominated_successes)


@dataclass(frozen=True, slots=True, init=False)
class NondominatedRunSurface(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Terminal multi-objective surface over vector-valued successes.

    Parameters
    ----------
    nondominated_successes : tuple[EvaluationSuccess[CandidateT, ObjectiveVectorPayload], ...]
        Stable nondominated frontier of ``successes``.
    successes : tuple[EvaluationSuccess[CandidateT, ObjectiveVectorPayload], ...]
        Full ordered vector-valued success history.
    evaluation_count : int
        Total logical evaluation cost accrued during the run.
    trace : Trace, default=Trace()
        Diagnostics trace captured during execution.
    failures : tuple[EvaluationFailure[CandidateT], ...], default=()
        Recorded failed evaluation attempts. Successful vector payloads remain
        in ``successes`` only.
    """

    nondominated_successes: tuple[
        EvaluationSuccess[CandidateT, ObjectiveVectorPayload],
        ...,
    ]
    successes: tuple[EvaluationSuccess[CandidateT, ObjectiveVectorPayload], ...]
    evaluation_count: int
    trace: Trace = field(default_factory=Trace)
    failures: tuple[EvaluationFailure[CandidateT], ...] = ()
    candidate_equal: InitVar[CandidateEquality[CandidateT] | None] = None
    _validated_frontier_source_successes: tuple[
        EvaluationSuccess[CandidateT, ObjectiveVectorPayload],
        ...,
    ] = (
        field(
            default=(),
            init=False,
            repr=False,
            compare=False,
        )
    )
    _validated_frontier_successes: tuple[
        EvaluationSuccess[CandidateT, ObjectiveVectorPayload],
        ...,
    ] = field(default=(), init=False, repr=False, compare=False)

    def __init__(
        self,
        *,
        nondominated_successes: Sequence[
            EvaluationSuccess[CandidateT, ObjectiveVectorPayload]
        ]
        | None = None,
        successes: Sequence[EvaluationSuccess[CandidateT, ObjectiveVectorPayload]]
        | None = None,
        evaluation_count: int,
        trace: Trace | None = None,
        failures: Sequence[EvaluationFailure[CandidateT]] | None = None,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
        nondominated_records: Sequence[ObjectiveVectorRecord[CandidateT]] | None = None,
        records: Sequence[ObjectiveVectorRecord[CandidateT]] | None = None,
        refinements: Sequence[CandidateRefinement[CandidateT] | None] | None = None,
    ) -> None:
        """Create one vector surface, normalizing records to successes."""
        normalized_successes: tuple[
            EvaluationSuccess[CandidateT, ObjectiveVectorPayload],
            ...,
        ]
        normalized_nondominated_successes: tuple[
            EvaluationSuccess[CandidateT, ObjectiveVectorPayload],
            ...,
        ]
        post_init_candidate_equal = candidate_equal
        if records is not None:
            normalized_successes = _successes_from_vector_records(
                records=records,
                refinements=_optional_refinement_tuple(refinements),
                candidate_equal=candidate_equal,
            )
            post_init_candidate_equal = None
        elif successes is not None:
            normalized_successes = tuple(successes)
            if refinements is not None:
                normalized_successes = _successes_with_refinements(
                    successes=normalized_successes,
                    refinements=refinements,
                    candidate_equal=candidate_equal,
                )
                post_init_candidate_equal = None
        else:
            normalized_successes = ()

        if nondominated_records is not None:
            normalized_nondominated_successes = _successes_matching_vector_records(
                nondominated_records,
                normalized_successes,
                candidate_equal=candidate_equal,
            )
        elif records is not None:
            normalized_nondominated_successes = collect_nondominated_successes(
                normalized_successes
            )
        elif refinements is not None and successes is not None:
            normalized_nondominated_successes = collect_nondominated_successes(
                normalized_successes
            )
        elif nondominated_successes is not None:
            normalized_nondominated_successes = tuple(nondominated_successes)
        else:
            normalized_nondominated_successes = ()

        object.__setattr__(self, "__orig_class__", None)
        object.__setattr__(
            self,
            "nondominated_successes",
            normalized_nondominated_successes,
        )
        object.__setattr__(self, "successes", normalized_successes)
        object.__setattr__(self, "evaluation_count", evaluation_count)
        object.__setattr__(self, "trace", Trace() if trace is None else trace)
        object.__setattr__(self, "failures", _optional_failure_tuple(failures))
        object.__setattr__(self, "_validated_frontier_source_successes", ())
        object.__setattr__(self, "_validated_frontier_successes", ())
        self.__post_init__(post_init_candidate_equal)

    @classmethod
    def _from_prevalidated_frontier(
        cls,
        *,
        nondominated_successes: tuple[
            EvaluationSuccess[CandidateT, ObjectiveVectorPayload],
            ...,
        ],
        successes: tuple[EvaluationSuccess[CandidateT, ObjectiveVectorPayload], ...],
        evaluation_count: int,
        trace: Trace,
        failures: tuple[EvaluationFailure[CandidateT], ...],
        candidate_equal: CandidateEquality[CandidateT] | None,
    ) -> Self:
        surface = cls.__new__(cls)
        _initialize_dataclass_fields(
            surface,
            field_values={
                "__orig_class__": None,
                "nondominated_successes": nondominated_successes,
                "successes": successes,
                "evaluation_count": evaluation_count,
                "trace": trace,
                "failures": failures,
                "_validated_frontier_source_successes": successes,
                "_validated_frontier_successes": nondominated_successes,
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
            If accounting is inconsistent, payload dimensions disagree, the
            frontier does not come from ``successes``, or
            ``nondominated_successes`` is not the stable nondominated frontier.
        """
        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)

        normalized_successes = _normalize_successes(
            successes=self.successes,
            candidate_equal=candidate_equal,
        )
        normalized_nondominated_successes = _normalize_successes(
            successes=self.nondominated_successes,
            candidate_equal=candidate_equal,
        )
        object.__setattr__(self, "successes", normalized_successes)
        object.__setattr__(
            self,
            "nondominated_successes",
            normalized_nondominated_successes,
        )

        failure_evaluation_count = _terminal_failure_evaluation_count(self.failures)
        success_evaluation_count = _success_evaluation_count(normalized_successes)
        if self.evaluation_count < success_evaluation_count + failure_evaluation_count:
            msg = "evaluation_count must be at least the terminal attempt cost"
            raise ValueError(msg)

        if (
            len(
                {
                    len(success.payload.objective_scores)
                    for success in normalized_successes
                }
            )
            > 1
        ):
            msg = "all objective score vectors must share one dimension"
            raise ValueError(msg)

        frontier_is_prevalidated = (
            normalized_successes is self._validated_frontier_source_successes
            and normalized_nondominated_successes
            is self._validated_frontier_successes
        )
        if not frontier_is_prevalidated and (
            normalized_nondominated_successes
            != collect_nondominated_successes(normalized_successes)
        ):
            msg = (
                "nondominated_successes must equal the stable nondominated frontier "
                "of successes"
            )
            raise ValueError(msg)

    @property
    def records(self) -> tuple[ObjectiveVectorRecord[CandidateT], ...]:
        """Return vector records projected from request-owned successes."""
        return tuple(_vector_record_from_success(success) for success in self.successes)

    @property
    def nondominated_records(self) -> tuple[ObjectiveVectorRecord[CandidateT], ...]:
        """Return nondominated vector record projections."""
        return tuple(
            _vector_record_from_success(success)
            for success in self.nondominated_successes
        )

    @property
    def refinements(self) -> tuple[CandidateRefinement[CandidateT] | None, ...]:
        """Return success-aligned refinement provenance."""
        return _success_refinements(self.successes)

    @classmethod
    def from_successes(
        cls,
        successes: Sequence[EvaluationSuccess[CandidateT, ObjectiveVectorPayload]],
        evaluation_count: int | None = None,
        trace: Trace | None = None,
        failures: Sequence[EvaluationFailure[CandidateT]] | None = None,
        candidate_equal: CandidateEquality[CandidateT] | None = None,
    ) -> Self:
        """Build a nondominated surface from request-owned vector successes."""
        success_tuple = tuple(successes)
        normalized_trace = Trace() if trace is None else trace
        failure_tuple = _optional_failure_tuple(failures)
        failure_evaluation_count = _terminal_failure_evaluation_count(failure_tuple)
        normalized_evaluation_count = (
            _success_evaluation_count(success_tuple) + failure_evaluation_count
        )
        if evaluation_count is not None:
            normalized_evaluation_count = evaluation_count

        nondominated_successes = collect_nondominated_successes(success_tuple)
        return cls._from_prevalidated_frontier(
            nondominated_successes=nondominated_successes,
            successes=success_tuple,
            evaluation_count=normalized_evaluation_count,
            trace=normalized_trace,
            failures=failure_tuple,
            candidate_equal=candidate_equal,
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
        normalized_trace = Trace() if trace is None else trace
        refinement_tuple = _optional_refinement_tuple(refinements)
        failure_tuple = _optional_failure_tuple(failures)
        failure_evaluation_count = _terminal_failure_evaluation_count(failure_tuple)
        successes = _successes_from_vector_records(
            records=records,
            refinements=refinement_tuple,
            candidate_equal=candidate_equal,
        )
        normalized_evaluation_count = (
            _success_evaluation_count(successes) + failure_evaluation_count
        )
        if evaluation_count is not None:
            normalized_evaluation_count = evaluation_count

        return cls.from_successes(
            successes=successes,
            evaluation_count=normalized_evaluation_count,
            trace=normalized_trace,
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
        vector_successes: list[EvaluationSuccess[CandidateT, ObjectiveVectorPayload]] = []
        for success in report.successes:
            payload = success.payload
            vector_successes.append(
                EvaluationSuccess(
                    request=_record_success_request(
                        payload,
                        refinement=success.refinement,
                        candidate_equal=candidate_equal,
                    ),
                    payload=ObjectiveVectorPayload(
                        objective_values=payload.objective_values,
                        objective_scores=payload.objective_scores,
                        elapsed_seconds=payload.elapsed_seconds,
                    ),
                    evaluation_count=success.evaluation_count,
                    refinement=success.refinement,
                    kernel_diagnostics=success.kernel_diagnostics,
                    candidate_equal=candidate_equal,
                )
            )

        return cls.from_successes(
            successes=tuple(vector_successes),
            evaluation_count=report.evaluation_count,
            trace=report.trace,
            failures=report.failures,
            candidate_equal=candidate_equal,
        )


_TerminalSurfaceStateOwner = (
    RunReport[object, RequestAlignedEvaluationRecord[object]]
    | RunResult[object]
    | NondominatedRunSurface[object]
)


def run_report_getstate(
    self: RunReport[CandidateT, RunRecordT],
) -> list[object | None]:
    return [
        None,
        self.successes,
        self.evaluation_count,
        self.trace,
        self.failures,
    ]


def run_result_getstate(self: RunResult[CandidateT]) -> list[object | None]:
    return [
        None,
        self.best_success,
        self.successes,
        self.evaluation_count,
        self.trace,
        self.failures,
    ]


def nondominated_run_surface_getstate(
    self: NondominatedRunSurface[CandidateT],
) -> list[object | None]:
    return [
        None,
        self.nondominated_successes,
        self.successes,
        self.evaluation_count,
        self.trace,
        self.failures,
        (),
        (),
    ]


def terminal_surface_setstate(
    self: _TerminalSurfaceStateOwner,
    state: list[object | None],
) -> None:
    dataclass_fields = fields(self)
    if len(state) != len(dataclass_fields):
        msg = (
            "terminal artifact pickle state field count mismatch: "
            f"expected {len(dataclass_fields)}, got {len(state)}"
        )
        raise TypeError(msg)

    for dataclass_field, value in zip(dataclass_fields, state, strict=True):
        object.__setattr__(self, dataclass_field.name, value)

    if isinstance(self, RunReport):
        self.__post_init__(None)
    elif isinstance(self, RunResult):
        self.__post_init__(None)
    else:
        object.__setattr__(self, "_validated_frontier_source_successes", ())
        object.__setattr__(self, "_validated_frontier_successes", ())
        self.__post_init__(None)


setattr(RunReport, "__getstate__", run_report_getstate)
setattr(RunReport, "__setstate__", terminal_surface_setstate)
setattr(RunResult, "__getstate__", run_result_getstate)
setattr(RunResult, "__setstate__", terminal_surface_setstate)
setattr(NondominatedRunSurface, "__getstate__", nondominated_run_surface_getstate)
setattr(NondominatedRunSurface, "__setstate__", terminal_surface_setstate)

del run_report_getstate
del run_result_getstate
del nondominated_run_surface_getstate
del terminal_surface_setstate
