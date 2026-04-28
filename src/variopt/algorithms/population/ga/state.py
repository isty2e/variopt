"""Private immutable state for the native genetic algorithm optimizer."""

from dataclasses import dataclass
from typing import Generic

import numpy as np

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....artifacts import Proposal
from ....randomness import RandomStateSnapshot
from ....typevars import CandidateT


@dataclass(frozen=True, slots=True)
class GAPopulationMember(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Evaluated candidate stored in GA population state.

    Parameters
    ----------
    candidate : CandidateT
        Candidate value stored in the population.
    value : float
        Objective value associated with the candidate.
    score : float
        Normalized score used for GA selection pressure.
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
class GAOptimizerState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Explicit immutable optimizer state for the native GA.

    Parameters
    ----------
    random_state : RandomStateSnapshot
        Captured random-state snapshot for deterministic continuation.
    proposal_index : int, default=0
        Monotone proposal identifier counter.
    generation_index : int, default=0
        Monotone generation counter.
    population : tuple[GAPopulationMember[CandidateT], ...], default=()
        Current evaluated population.
    queued_proposals : tuple[Proposal[CandidateT], ...], default=()
        Proposals ready to be issued.
    pending_proposals : tuple[Proposal[CandidateT], ...], default=()
        Issued proposals awaiting evaluation.
    buffered_members : tuple[GAPopulationMember[CandidateT], ...], default=()
        Newly evaluated members buffered until the next generation update.
    """

    random_state: RandomStateSnapshot
    proposal_index: int = 0
    generation_index: int = 0
    population: tuple[GAPopulationMember[CandidateT], ...] = ()
    queued_proposals: tuple[Proposal[CandidateT], ...] = ()
    pending_proposals: tuple[Proposal[CandidateT], ...] = ()
    buffered_members: tuple[GAPopulationMember[CandidateT], ...] = ()

    def __post_init__(self) -> None:
        """Reject invalid GA state counters."""
        if self.proposal_index < 0:
            msg = "proposal_index must be non-negative"
            raise ValueError(msg)

        if self.generation_index < 0:
            msg = "generation_index must be non-negative"
            raise ValueError(msg)
