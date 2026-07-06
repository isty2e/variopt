"""Canonical immutable state for internal generational GA variants."""

from dataclasses import dataclass
from enum import Enum
from typing import Generic

import numpy as np

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....artifacts import Proposal
from ....randomness import RandomStateSnapshot
from ....typevars import CandidateT


class GenerationalGAVariant(Enum):
    """Closed identity of a generational GA optimizer variant."""

    NATIVE = "ga"
    CLEARING = "clearing_ga"
    SPECIES = "species_ga"
    RESTRICTED_TOURNAMENT = "restricted_tournament_ga"


@dataclass(frozen=True, slots=True)
class GenerationalGAPopulationMember(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Evaluated candidate stored in a generational GA population.

    Parameters
    ----------
    candidate : CandidateT
        Candidate value stored in the population.
    value : float
        Objective value associated with the candidate.
    score : float
        Normalized score used by minimization-oriented selection pressure.
    """

    candidate: CandidateT
    value: float
    score: float

    def __post_init__(self) -> None:
        """Reject non-finite objective accounting values.

        Raises
        ------
        ValueError
            If ``value`` or ``score`` is not finite.
        """
        if not np.isfinite(self.value):
            msg = "value must be finite"
            raise ValueError(msg)

        if not np.isfinite(self.score):
            msg = "score must be finite"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class GenerationalGAOptimizerState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Explicit immutable optimizer state for generational GA variants.

    Parameters
    ----------
    variant : GenerationalGAVariant
        Optimizer variant that owns this run-method state.
    random_state : RandomStateSnapshot
        Captured random-state snapshot for deterministic continuation.
    proposal_index : int, default=0
        Monotone proposal identifier counter.
    generation_index : int, default=0
        Monotone generation counter.
    population : tuple[GenerationalGAPopulationMember[CandidateT], ...], default=()
        Current evaluated population.
    queued_proposals : tuple[Proposal[CandidateT], ...], default=()
        Generated proposals waiting to be issued.
    pending_proposals : tuple[Proposal[CandidateT], ...], default=()
        Issued proposals awaiting aligned observations.
    buffered_members : tuple[GenerationalGAPopulationMember[CandidateT], ...], default=()
        Newly evaluated members buffered until a generation can be committed.
    """

    variant: GenerationalGAVariant
    random_state: RandomStateSnapshot
    proposal_index: int = 0
    generation_index: int = 0
    population: tuple[GenerationalGAPopulationMember[CandidateT], ...] = ()
    queued_proposals: tuple[Proposal[CandidateT], ...] = ()
    pending_proposals: tuple[Proposal[CandidateT], ...] = ()
    buffered_members: tuple[GenerationalGAPopulationMember[CandidateT], ...] = ()

    def __post_init__(self) -> None:
        """Reject invalid lifecycle counters.

        Raises
        ------
        TypeError
            If ``variant`` is not a generational GA variant marker.
        ValueError
            If ``proposal_index`` or ``generation_index`` is negative.
        """
        if type(self.variant) is not GenerationalGAVariant:
            msg = "variant must be a GenerationalGAVariant"
            raise TypeError(msg)

        if self.proposal_index < 0:
            msg = "proposal_index must be non-negative"
            raise ValueError(msg)

        if self.generation_index < 0:
            msg = "generation_index must be non-negative"
            raise ValueError(msg)
