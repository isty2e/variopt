"""Artifact nouns for structured local-search runtime episodes."""

from dataclasses import dataclass
from typing import Generic

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....artifacts import Observation
from .....kernel import KernelStatus
from .....outcomes import EvaluationAttemptBatch, EvaluationOutcome
from ..neighborhood import StructuredCandidateT


@dataclass(frozen=True, slots=True)
class StructuredImprovementScanResult(
    FrozenGenericSlotsCompat,
    Generic[StructuredCandidateT],
):
    """Result of scanning a bounded structured neighborhood for improvement.

    Parameters
    ----------
    improved_outcome : EvaluationOutcome[StructuredCandidateT] | None
        First improving outcome found during the scan, if any.
    evaluation_count : int
        Logical evaluation cost consumed by the scan.
    failed_attempts : tuple[EvaluationAttemptBatch[StructuredCandidateT], ...], default=()
        Failed one-request evaluator attempts encountered during the scan.
    budget_exhausted : bool, default=False
        Whether the scan stopped because no evaluation budget remained.
    """

    improved_outcome: EvaluationOutcome[StructuredCandidateT] | None
    evaluation_count: int
    failed_attempts: tuple[EvaluationAttemptBatch[StructuredCandidateT], ...] = ()
    budget_exhausted: bool = False


@dataclass(frozen=True, slots=True)
class StructuredVariableNeighborhoodStageAttempt(
    FrozenGenericSlotsCompat,
    Generic[StructuredCandidateT],
):
    """One attempted neighborhood stage inside a variable-neighborhood episode.

    Parameters
    ----------
    improved_outcome : EvaluationOutcome[StructuredCandidateT] | None
        Improved outcome found during the stage, if any.
    evaluation_count : int
        Number of evaluations consumed by the stage.
    terminal_status : KernelStatus
        Terminal kernel status after the stage.
    terminal_message : str
        Human-readable terminal status message.
    failed_attempts : tuple[EvaluationAttemptBatch[StructuredCandidateT], ...], default=()
        Failed one-request evaluator attempts encountered during the stage.
    budget_exhausted : bool, default=False
        Whether the stage stopped because no evaluation budget remained.
    """

    improved_outcome: EvaluationOutcome[StructuredCandidateT] | None
    evaluation_count: int
    terminal_status: KernelStatus
    terminal_message: str
    failed_attempts: tuple[EvaluationAttemptBatch[StructuredCandidateT], ...] = ()
    budget_exhausted: bool = False


@dataclass(frozen=True, slots=True)
class StructuredLocalImprovementResult(
    FrozenGenericSlotsCompat, Generic[StructuredCandidateT]
):
    """One completed inner local-improvement episode over a fixed incumbent.

    Parameters
    ----------
    record : Observation[StructuredCandidateT] | None
        Final observation returned by the local-improvement episode, or
        ``None`` when the episode produced no successful evaluation.
    evaluation_count : int
        Total evaluations consumed by the episode.
    completed_steps : int
        Number of completed neighborhood steps.
    converged : bool
        Whether the episode converged without finding further improvements.
    failed_attempts : tuple[EvaluationAttemptBatch[StructuredCandidateT], ...], default=()
        Failed one-request evaluator attempts encountered during the episode.
    budget_exhausted : bool, default=False
        Whether the episode stopped because no evaluation budget remained.
    """

    record: Observation[StructuredCandidateT] | None
    evaluation_count: int
    completed_steps: int
    converged: bool
    failed_attempts: tuple[EvaluationAttemptBatch[StructuredCandidateT], ...] = ()
    budget_exhausted: bool = False
