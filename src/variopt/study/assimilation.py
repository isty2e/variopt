"""Shared study attempt-assimilation artifacts."""

from dataclasses import dataclass, field
from typing import Generic, NoReturn

from typing_extensions import TypeVar

from ..artifacts import (
    EvaluationAttemptBatch,
    EvaluationAttemptMaterializer,
    EvaluationFailure,
    EvaluationSuccess,
    RunReport,
    Trace,
    TraceEvent,
    materialize_success_records,
)
from ..artifacts.records import RequestAlignedEvaluationRecord
from ..spaces import CandidateEquality
from ..typevars import CandidateT, RunMethodStateT
from .common import (
    CheckpointSafeRunSnapshot,
    StudyPayloadT,
    trace_value_for_records,
    validate_materialized_attempts,
)
from .failures import (
    RunExecutionFailed,
    build_checkpoint_safe_report_or_raise_cause,
    build_run_report_or_raise_cause,
)

AssimilationRecordT = TypeVar(
    "AssimilationRecordT",
    bound=RequestAlignedEvaluationRecord[object],
)


@dataclass(frozen=True, slots=True)
class StudyAssimilatedStep(Generic[CandidateT, AssimilationRecordT]):
    """Record-aligned feedback projection for one completed study step.

    Parameters
    ----------
    attempts : EvaluationAttemptBatch[CandidateT, AssimilationRecordT]
        Feedback attempts passed to the run method.
    records : tuple[AssimilationRecordT, ...]
        Successful request-aligned records in attempt order.
    trace_event : TraceEvent
        Trace event derived from the completed attempt batch.
    evaluation_count : int
        Run-report evaluation-count delta owned by this step.
    """

    attempts: EvaluationAttemptBatch[CandidateT, AssimilationRecordT]
    records: tuple[AssimilationRecordT, ...]
    trace_event: TraceEvent
    evaluation_count: int

    @classmethod
    def from_attempts(
        cls,
        attempts: EvaluationAttemptBatch[CandidateT, AssimilationRecordT],
        *,
        evaluation_count: int,
    ) -> "StudyAssimilatedStep[CandidateT, AssimilationRecordT]":
        """Build one feedback projection from materialized attempts.

        Parameters
        ----------
        attempts : EvaluationAttemptBatch[CandidateT, AssimilationRecordT]
            Materialized feedback attempts.
        evaluation_count : int
            Evaluation-count delta to attach to this step.

        Returns
        -------
        StudyAssimilatedStep[CandidateT, AssimilationRecordT]
            Step projection containing records, trace, and accounting.
        """
        records = materialize_success_records(attempts.successes)
        return cls(
            attempts=attempts,
            records=records,
            trace_event=TraceEvent(
                kind="study.step",
                message=(
                    f"completed {attempts.attempt_count} attempt(s): "
                    f"{len(records)} succeeded, "
                    f"{len(attempts.failures)} failed"
                ),
                value=trace_value_for_records(records),
            ),
            evaluation_count=evaluation_count,
        )


@dataclass(slots=True)
class StudyRunHistory(Generic[CandidateT, RunMethodStateT, AssimilationRecordT]):
    """Append-only run history used to project reports and checkpoint cuts.

    Parameters
    ----------
    successes : list[EvaluationSuccess[CandidateT, AssimilationRecordT]]
        Successful feedback attempts accumulated by completed steps.
    failures : list[EvaluationFailure[CandidateT]]
        Recorded user-code evaluation failures accumulated by completed steps.
    trace_events : list[TraceEvent]
        Buffered trace events accumulated by completed steps.
    evaluation_count : int, default=0
        Run-report evaluation count accumulated by completed steps.
    """

    successes: list[EvaluationSuccess[CandidateT, AssimilationRecordT]] = field(
        default_factory=list,
    )
    failures: list[EvaluationFailure[CandidateT]] = field(default_factory=list)
    trace_events: list[TraceEvent] = field(default_factory=list)
    evaluation_count: int = 0

    def append_step(
        self,
        step: StudyAssimilatedStep[CandidateT, AssimilationRecordT],
    ) -> None:
        """Append one completed step projection to the run history.

        Parameters
        ----------
        step : StudyAssimilatedStep[CandidateT, AssimilationRecordT]
            Completed step projection to append.
        """
        self.successes.extend(step.attempts.successes)
        self.failures.extend(step.attempts.failures)
        self.trace_events.append(step.trace_event)
        self.evaluation_count += step.evaluation_count

    def checkpoint_snapshot(
        self,
        state: RunMethodStateT,
    ) -> CheckpointSafeRunSnapshot[RunMethodStateT]:
        """Return a checkpoint-safe cut aligned with the current history.

        Parameters
        ----------
        state : RunMethodStateT
            Run-method state aligned with this history.

        Returns
        -------
        CheckpointSafeRunSnapshot[RunMethodStateT]
            Snapshot of the current history lengths and state.
        """
        return CheckpointSafeRunSnapshot(
            success_count=len(self.successes),
            failure_count=len(self.failures),
            trace_event_count=len(self.trace_events),
            evaluation_count=self.evaluation_count,
            state=state,
        )

    def to_report(
        self,
        *,
        candidate_equal: CandidateEquality[CandidateT],
        evaluation_count: int | None = None,
    ) -> RunReport[CandidateT, AssimilationRecordT]:
        """Build a terminal report from the current history.

        Parameters
        ----------
        candidate_equal : CandidateEquality[CandidateT]
            Candidate equality predicate used to validate refinement alignment.
        evaluation_count : int | None, default=None
            Optional accounting override for failures that consume budget before a
            step can be appended.

        Returns
        -------
        RunReport[CandidateT, AssimilationRecordT]
            Terminal run report.
        """
        resolved_evaluation_count = (
            self.evaluation_count if evaluation_count is None else evaluation_count
        )
        return RunReport[CandidateT, AssimilationRecordT].from_successes(
            successes=tuple(self.successes),
            evaluation_count=resolved_evaluation_count,
            trace=Trace(events=tuple(self.trace_events)),
            failures=tuple(self.failures),
            candidate_equal=candidate_equal,
        )

    def checkpoint_report(
        self,
        snapshot: CheckpointSafeRunSnapshot[RunMethodStateT],
        *,
        candidate_equal: CandidateEquality[CandidateT],
    ) -> RunReport[CandidateT, AssimilationRecordT]:
        """Build a report projected to ``snapshot``.

        Parameters
        ----------
        snapshot : CheckpointSafeRunSnapshot[RunMethodStateT]
            Checkpoint-safe history cut to project.
        candidate_equal : CandidateEquality[CandidateT]
            Candidate equality predicate used to validate refinement alignment.

        Returns
        -------
        RunReport[CandidateT, AssimilationRecordT]
            Checkpoint-safe report projection.
        """
        return snapshot.to_report(
            successes=self.successes,
            failures=self.failures,
            trace_events=self.trace_events,
            candidate_equal=candidate_equal,
        )

    def raise_run_execution_failed(
        self,
        *,
        cause: Exception,
        state: RunMethodStateT,
        safe_snapshot: CheckpointSafeRunSnapshot[RunMethodStateT] | None,
        candidate_equal: CandidateEquality[CandidateT],
        evaluation_count: int | None = None,
    ) -> NoReturn:
        """Raise a hard run failure with partial and checkpoint-safe reports.

        Parameters
        ----------
        cause : Exception
            Hard failure to wrap.
        state : RunMethodStateT
            Run-method state at the failure boundary.
        safe_snapshot : CheckpointSafeRunSnapshot[RunMethodStateT] | None
            Latest checkpoint-safe snapshot, if one is available.
        candidate_equal : CandidateEquality[CandidateT]
            Candidate equality predicate used to validate refinement alignment.
        evaluation_count : int | None, default=None
            Optional partial-report accounting override.

        Raises
        ------
        RunExecutionFailed
            Always raised with partial and optional checkpoint-safe projections.
        """
        checkpoint_safe_report: RunReport[CandidateT, AssimilationRecordT] | None = None
        checkpoint_safe_state: RunMethodStateT | None = None
        if safe_snapshot is not None:
            checkpoint_safe_report = build_checkpoint_safe_report_or_raise_cause(
                cause=cause,
                snapshot=safe_snapshot,
                successes=self.successes,
                failures=self.failures,
                trace_events=self.trace_events,
                candidate_equal=candidate_equal,
            )
            checkpoint_safe_state = safe_snapshot.state

        resolved_evaluation_count = (
            self.evaluation_count if evaluation_count is None else evaluation_count
        )
        raise RunExecutionFailed[
            CandidateT,
            RunMethodStateT,
            AssimilationRecordT,
        ](
            partial_report=build_run_report_or_raise_cause(
                cause=cause,
                successes=self.successes,
                failures=self.failures,
                trace=Trace(events=tuple(self.trace_events)),
                evaluation_count=resolved_evaluation_count,
                candidate_equal=candidate_equal,
            ),
            partial_state=state,
            checkpoint_safe_report=checkpoint_safe_report,
            checkpoint_safe_state=checkpoint_safe_state,
            cause=cause,
        ) from cause


def materialize_feedback_attempts(
    source_attempts: EvaluationAttemptBatch[CandidateT, StudyPayloadT],
    materializer: EvaluationAttemptMaterializer[
        CandidateT,
        StudyPayloadT,
        AssimilationRecordT,
    ],
    *,
    candidate_equal: CandidateEquality[CandidateT],
) -> EvaluationAttemptBatch[CandidateT, AssimilationRecordT]:
    """Materialize evaluator attempts into run-method feedback attempts.

    Parameters
    ----------
    source_attempts : EvaluationAttemptBatch[CandidateT, StudyPayloadT]
        Evaluator or kernel attempts carrying request-free payloads.
    materializer : EvaluationAttemptMaterializer[CandidateT, StudyPayloadT, AssimilationRecordT]
        Materializer that converts successful payloads to request-aligned records.
    candidate_equal : CandidateEquality[CandidateT]
        Candidate equality predicate used to validate refinement alignment.

    Returns
    -------
    EvaluationAttemptBatch[CandidateT, AssimilationRecordT]
        Materialized feedback attempts preserving source slot semantics.
    """
    feedback_attempts = materializer.materialize_attempts(source_attempts)
    validate_materialized_attempts(
        source_attempts,
        feedback_attempts,
        candidate_equal=candidate_equal,
    )
    return feedback_attempts
