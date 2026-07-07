"""Canonical immutable state for internal generational GA variants."""

from dataclasses import dataclass, field
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
class GenerationalGAMemberBuffer(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Immutable append-optimized buffer for split generational GA batches.

    Parameters
    ----------
    member_count : int, default=0
        Total number of buffered population members across this node and all
        previous nodes.
    latest_batch : tuple[GenerationalGAPopulationMember[CandidateT], ...], default=()
        Most recently appended non-empty batch.
    previous : GenerationalGAMemberBuffer[CandidateT] | None, default=None
        Earlier immutable buffer node.
    """

    member_count: int = 0
    latest_batch: tuple[GenerationalGAPopulationMember[CandidateT], ...] = ()
    previous: "GenerationalGAMemberBuffer[CandidateT] | None" = None

    def __post_init__(self) -> None:
        """Reject inconsistent buffered-member accounting.

        Raises
        ------
        ValueError
            If the count does not match the linked batches or an empty
            non-root batch is present.
        """
        if self.member_count < 0:
            msg = "member_count must be non-negative"
            raise ValueError(msg)

        previous_count = 0 if self.previous is None else self.previous.member_count
        expected_count = previous_count + len(self.latest_batch)
        if self.member_count != expected_count:
            msg = "member_count must match buffered batch lengths"
            raise ValueError(msg)

        if self.previous is not None and len(self.latest_batch) == 0:
            msg = "latest_batch must not be empty when previous is present"
            raise ValueError(msg)

    def append(
        self,
        members: tuple[GenerationalGAPopulationMember[CandidateT], ...],
    ) -> "GenerationalGAMemberBuffer[CandidateT]":
        """Return a buffer with ``members`` appended as one immutable batch.

        Parameters
        ----------
        members : tuple[GenerationalGAPopulationMember[CandidateT], ...]
            New members to append.

        Returns
        -------
        GenerationalGAMemberBuffer[CandidateT]
            Buffer containing the existing members followed by ``members``.
        """
        if len(members) == 0:
            return self

        return GenerationalGAMemberBuffer(
            member_count=self.member_count + len(members),
            latest_batch=members,
            previous=self,
        )

    def materialize(
        self,
    ) -> tuple[GenerationalGAPopulationMember[CandidateT], ...]:
        """Return buffered members in original append order.

        Returns
        -------
        tuple[GenerationalGAPopulationMember[CandidateT], ...]
            Materialized member sequence.
        """
        batches: list[tuple[GenerationalGAPopulationMember[CandidateT], ...]] = []
        node: GenerationalGAMemberBuffer[CandidateT] | None = self
        while node is not None:
            if len(node.latest_batch) > 0:
                batches.append(node.latest_batch)
            node = node.previous

        return tuple(member for batch in reversed(batches) for member in batch)


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
        Generated proposals for the current generation.
    queued_proposal_index : int, default=0
        Start index of the not-yet-issued suffix in ``queued_proposals``.
    pending_proposals : tuple[Proposal[CandidateT], ...], default=()
        Issued proposals awaiting aligned observations.
    buffered_member_buffer : GenerationalGAMemberBuffer[CandidateT], optional
        Newly evaluated members buffered until a generation can be committed.
    """

    variant: GenerationalGAVariant
    random_state: RandomStateSnapshot
    proposal_index: int = 0
    generation_index: int = 0
    population: tuple[GenerationalGAPopulationMember[CandidateT], ...] = ()
    queued_proposals: tuple[Proposal[CandidateT], ...] = ()
    queued_proposal_index: int = 0
    pending_proposals: tuple[Proposal[CandidateT], ...] = ()
    buffered_member_buffer: GenerationalGAMemberBuffer[CandidateT] = field(
        default_factory=GenerationalGAMemberBuffer,
    )

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

        if self.queued_proposal_index < 0:
            msg = "queued_proposal_index must be non-negative"
            raise ValueError(msg)

        if len(self.queued_proposals) == 0 and self.queued_proposal_index != 0:
            msg = "queued_proposal_index must be zero when queued_proposals is empty"
            raise ValueError(msg)

        if self.queued_proposal_index > len(self.queued_proposals):
            msg = "queued_proposal_index must not exceed queued_proposals length"
            raise ValueError(msg)
