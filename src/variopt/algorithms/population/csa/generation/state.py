"""CSA generation queue and runtime-state definitions."""

from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Generic

import numpy as np
from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....artifacts import Observation
from .....randomness import random_state_permutation_indices
from .....typevars import CandidateT
from .proposal.state.attribution import PlannedProposalAttribution


@dataclass(frozen=True, slots=True)
class GeneratedCandidate(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """One generated child together with optional proposal attribution.

    Parameters
    ----------
    candidate : CandidateT
        Generated child candidate.
    planned_attribution : PlannedProposalAttribution | None, default=None
        Optional proposal-attribution payload carried into the evaluation step.
    """

    candidate: CandidateT
    planned_attribution: PlannedProposalAttribution | None = None


@dataclass(frozen=True, slots=True)
class GenerationQueue(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Immutable queue of generated candidates for one CSA child pool.

    Parameters
    ----------
    candidates : tuple[GeneratedCandidate[CandidateT], ...], default=()
        Backing tuple of generated candidates. Entries before ``head_index``
        have already been issued.
    head_index : int, default=0
        Index of the next candidate to issue.
    """

    candidates: tuple[GeneratedCandidate[CandidateT], ...] = ()
    head_index: int = 0

    def __post_init__(self) -> None:
        """Validate queue bounds and normalize the backing candidate tuple.

        Raises
        ------
        ValueError
            If ``head_index`` falls outside the backing tuple bounds.
        """
        candidates = tuple(self.candidates)
        object.__setattr__(self, "candidates", candidates)
        if self.head_index < 0 or self.head_index > len(candidates):
            msg = "head_index must be between zero and the candidate count"
            raise ValueError(msg)

    @property
    def is_empty(self) -> bool:
        """Return whether the queue is empty."""
        return self.head_index == len(self.candidates)

    @classmethod
    def from_candidates(
        cls,
        candidates: Sequence[GeneratedCandidate[CandidateT]],
        *,
        shuffle: bool,
        random_state: np.random.RandomState,
    ) -> Self:
        """Build a queue from generated candidates, optionally shuffled.

        Parameters
        ----------
        candidates : Sequence[GeneratedCandidate[CandidateT]]
            Generated candidates to enqueue.
        shuffle : bool
            Whether to shuffle the queue before returning it.
        random_state : np.random.RandomState
            Random state used when ``shuffle`` is enabled.

        Returns
        -------
        Self
            Generation queue containing the supplied candidates.
        """
        candidate_tuple = tuple(candidates)
        if not shuffle or len(candidate_tuple) <= 1:
            return cls(candidates=candidate_tuple)

        indices = random_state_permutation_indices(random_state, len(candidate_tuple))
        return cls(
            candidates=tuple(candidate_tuple[index] for index in indices),
        )

    def dequeue(self) -> tuple[GeneratedCandidate[CandidateT], Self]:
        """Return the next candidate and the remaining queue.

        Returns
        -------
        tuple[GeneratedCandidate[CandidateT], Self]
            Head candidate and the queue containing the remaining candidates.

        Raises
        ------
        RuntimeError
            If the queue is empty.
        """
        if self.is_empty:
            msg = "cannot dequeue from an empty generation queue"
            raise RuntimeError(msg)

        return (
            self.candidates[self.head_index],
            type(self)(
                candidates=self.candidates,
                head_index=self.head_index + 1,
            ),
        )


@dataclass(frozen=True, slots=True)
class GenerationRuntimeState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Runtime-only state for one CSA child pool across ask/tell boundaries.

    Parameters
    ----------
    queue : GenerationQueue[CandidateT], default=GenerationQueue()
        Remaining generated candidates waiting to become proposals.
    pending_proposal_ids : frozenset[str], default=frozenset()
        Proposal identifiers issued from the current pool but not yet observed.
    buffered_observations : tuple[Observation[CandidateT], ...], default=()
        Observations buffered until the entire child pool is complete.
    """

    queue: GenerationQueue[CandidateT] = field(
        default_factory=lambda: GenerationQueue(),
    )
    pending_proposal_ids: frozenset[str] = frozenset()
    buffered_observations: tuple[Observation[CandidateT], ...] = ()

    @property
    def is_active(self) -> bool:
        """Return whether a child pool is currently in flight."""
        return (
            not self.queue.is_empty
            or len(self.pending_proposal_ids) > 0
            or len(self.buffered_observations) > 0
        )

    @property
    def ready_to_commit(self) -> bool:
        """Return whether all generated children have been observed."""
        return (
            self.queue.is_empty
            and len(self.pending_proposal_ids) == 0
            and len(self.buffered_observations) > 0
        )

    def begin(self, queue: GenerationQueue[CandidateT]) -> Self:
        """Return a runtime that starts tracking a newly generated child pool.

        Parameters
        ----------
        queue : GenerationQueue[CandidateT]
            Newly generated child queue to track.

        Returns
        -------
        Self
            Fresh active runtime bound to ``queue``.

        Raises
        ------
        RuntimeError
            If another child pool is already active.
        ValueError
            If ``queue`` is empty.
        """
        if self.is_active:
            msg = "cannot begin a new child pool while another pool is active"
            raise RuntimeError(msg)

        if queue.is_empty:
            msg = "cannot begin a child pool from an empty queue"
            raise ValueError(msg)

        return type(self)(queue=queue)

    def dequeue_candidate(self) -> tuple[GeneratedCandidate[CandidateT], Self]:
        """Return the next queued candidate and the updated runtime.

        Returns
        -------
        tuple[GeneratedCandidate[CandidateT], Self]
            Head candidate and runtime state with the remaining queue.
        """
        candidate, next_queue = self.queue.dequeue()
        return candidate, type(self)(
            queue=next_queue,
            pending_proposal_ids=self.pending_proposal_ids,
            buffered_observations=self.buffered_observations,
        )

    def register_proposal(self, proposal_id: str) -> Self:
        """Return a runtime that tracks an issued proposal from the queue.

        Parameters
        ----------
        proposal_id : str
            Proposal identifier issued for the current generation pool.

        Returns
        -------
        Self
            Runtime state with ``proposal_id`` added to the pending set.
        """
        return type(self)(
            queue=self.queue,
            pending_proposal_ids=self.pending_proposal_ids | {proposal_id},
            buffered_observations=self.buffered_observations,
        )

    def buffer_observations(
        self,
        observations: Sequence[Observation[CandidateT]],
    ) -> Self:
        """Return a runtime that records observed children from the active pool.

        Parameters
        ----------
        observations : Sequence[Observation[CandidateT]]
            Observations returned for proposals issued from the active pool.

        Returns
        -------
        Self
            Runtime state with completed proposal identifiers removed and
            observations appended to the buffer.
        """
        consumed_ids = {
            observation.proposal.proposal_id
            for observation in observations
            if observation.proposal.proposal_id is not None
        }
        return type(self)(
            queue=self.queue,
            pending_proposal_ids=self.pending_proposal_ids - consumed_ids,
            buffered_observations=self.buffered_observations + tuple(observations),
        )

    def consume_failed_proposals(self, proposal_ids: AbstractSet[str]) -> Self:
        """Return a runtime with failed proposal ids removed from the pool.

        Parameters
        ----------
        proposal_ids : collections.abc.Set[str]
            Proposal identifiers whose evaluation attempts failed.

        Returns
        -------
        Self
            Runtime state with the failed proposal identifiers removed while
            preserving queued children and successful buffered observations.
        """
        if not proposal_ids:
            return self

        return type(self)(
            queue=self.queue,
            pending_proposal_ids=self.pending_proposal_ids - proposal_ids,
            buffered_observations=self.buffered_observations,
        )

    def release_buffer(self) -> tuple[tuple[Observation[CandidateT], ...], Self]:
        """Return buffered observations and reset the runtime to idle.

        Returns
        -------
        tuple[tuple[Observation[CandidateT], ...], Self]
            Buffered observations together with a reset idle runtime.

        Raises
        ------
        RuntimeError
            If the child pool has not completed yet.
        """
        if not self.ready_to_commit:
            msg = "cannot release a generation buffer before the pool is complete"
            raise RuntimeError(msg)

        return self.buffered_observations, type(self)()
