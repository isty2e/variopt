"""Private immutable state for the native differential-evolution optimizer."""

from dataclasses import dataclass
from typing import Generic

import numpy as np

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....artifacts import Proposal
from ....randomness import RandomStateSnapshot
from ....typevars import CandidateT


@dataclass(frozen=True, slots=True)
class DEPopulationMember(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Evaluated candidate stored in DE population state.

    Parameters
    ----------
    candidate : CandidateT
        Candidate stored in the population.
    value : float
        Raw objective value for the candidate.
    score : float
        Canonical minimization-form score for the candidate.
    """

    candidate: CandidateT
    value: float
    score: float

    def __post_init__(self) -> None:
        """Reject invalid population member values."""
        if not np.isfinite(self.value):
            msg = "value must be finite"
            raise ValueError(msg)

        if not np.isfinite(self.score):
            msg = "score must be finite"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DEPendingEvaluation(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Proposal metadata kept between ``ask`` and ``tell``.

    Parameters
    ----------
    proposal : Proposal[CandidateT]
        Proposal issued to the evaluator.
    target_index : int | None, default=None
        Optional population slot targeted by the proposal.
    """

    proposal: Proposal[CandidateT]
    target_index: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid pending-evaluation metadata."""
        if self.target_index is not None and self.target_index < 0:
            msg = "target_index must be non-negative when present"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DEObservedEvaluation(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Observed candidate buffered until one full DE population is available.

    Parameters
    ----------
    member : DEPopulationMember[CandidateT]
        Evaluated population member.
    target_index : int | None, default=None
        Optional population slot targeted by the evaluation.
    """

    member: DEPopulationMember[CandidateT]
    target_index: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid observed-evaluation metadata."""
        if self.target_index is not None and self.target_index < 0:
            msg = "target_index must be non-negative when present"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DEOptimizerState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Explicit immutable optimizer state for native differential evolution.

    Parameters
    ----------
    random_state : RandomStateSnapshot
        Immutable random-state snapshot.
    proposal_index : int, default=0
        Monotone proposal counter.
    generation_index : int, default=0
        Monotone generation counter.
    population : tuple[DEPopulationMember[CandidateT], ...], default=()
        Current DE population.
    queued_evaluations : tuple[DEPendingEvaluation[CandidateT], ...], default=()
        Pending evaluations not yet issued to the evaluator.
    pending_evaluations : tuple[DEPendingEvaluation[CandidateT], ...], default=()
        Evaluations issued to the evaluator and awaiting observation.
    buffered_evaluations : tuple[DEObservedEvaluation[CandidateT], ...], default=()
        Observed evaluations buffered until a full population update can
        commit.
    """

    random_state: RandomStateSnapshot
    proposal_index: int = 0
    generation_index: int = 0
    population: tuple[DEPopulationMember[CandidateT], ...] = ()
    queued_evaluations: tuple[DEPendingEvaluation[CandidateT], ...] = ()
    pending_evaluations: tuple[DEPendingEvaluation[CandidateT], ...] = ()
    buffered_evaluations: tuple[DEObservedEvaluation[CandidateT], ...] = ()

    def __post_init__(self) -> None:
        """Reject invalid DE state counters."""
        if self.proposal_index < 0:
            msg = "proposal_index must be non-negative"
            raise ValueError(msg)

        if self.generation_index < 0:
            msg = "generation_index must be non-negative"
            raise ValueError(msg)
