"""Artifact nouns for structured local-search runtime episodes."""

from dataclasses import dataclass
from typing import Generic

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....artifacts import Observation
from .....kernel import KernelStatus
from .....outcomes import EvaluationOutcome
from ..neighborhood import StructuredCandidateT


@dataclass(frozen=True, slots=True)
class StructuredVariableNeighborhoodStageAttempt(FrozenGenericSlotsCompat,
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
    """

    improved_outcome: EvaluationOutcome[StructuredCandidateT] | None
    evaluation_count: int
    terminal_status: KernelStatus
    terminal_message: str


@dataclass(frozen=True, slots=True)
class StructuredLocalImprovementResult(FrozenGenericSlotsCompat, Generic[StructuredCandidateT]):
    """One completed inner local-improvement episode over a fixed incumbent.

    Parameters
    ----------
    record : Observation[StructuredCandidateT]
        Final observation returned by the local-improvement episode.
    evaluation_count : int
        Total evaluations consumed by the episode.
    completed_steps : int
        Number of completed neighborhood steps.
    converged : bool
        Whether the episode converged without finding further improvements.
    """

    record: Observation[StructuredCandidateT]
    evaluation_count: int
    completed_steps: int
    converged: bool
