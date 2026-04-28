"""CSA generation queue and runtime-state definitions."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic

import numpy as np
from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....artifacts import Observation
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
        Generated candidates waiting to be issued as proposals.
    """

    candidates: tuple[GeneratedCandidate[CandidateT], ...] = ()

    @property
    def is_empty(self) -> bool:
        """Return whether the queue is empty."""
        return len(self.candidates) == 0

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

        indices = list(range(len(candidate_tuple)))
        random_state.shuffle(indices)
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

        return self.candidates[0], type(self)(candidates=self.candidates[1:])


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
