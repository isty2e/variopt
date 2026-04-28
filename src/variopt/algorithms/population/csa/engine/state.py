"""Canonical CSA engine-state aggregates."""

from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from typing import Generic

from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....artifacts import Proposal
from .....json_types import JSONDict, JSONValue
from .....randomness import RandomStateSnapshot
from .....typevars import CandidateT
from ..banking.clustering.policy import CSAClusteringPolicy
from ..banking.growth.policy import CSABankGrowthPolicy
from ..generation.proposal.policy import CSAProposalPolicy
from ..generation.proposal.state.aggregate import CSAProposalState
from ..generation.state import GenerationRuntimeState
from ..progression.state import CSAProgressionState
from ..scoring.acceptance import CSAAcceptancePolicy
from ..scoring.model import CSAScoreModel
from ..selection.state import SeedSelectionState
from ..trace.events.state import CSAEventTraceState
from .banking_state import CSABankingState
from .scoring_state import CSAScoringState

_CSA_ENGINE_STATE_FORMAT = "variopt.csa_engine_state"
_CSA_ENGINE_STATE_VERSION = 1


@dataclass(frozen=True, slots=True)
class CSAPendingProposals(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Immutable registry of proposals currently outstanding in the engine.

    Parameters
    ----------
    proposals : tuple[Proposal[CandidateT], ...], default=()
        Outstanding proposals keyed by proposal id.
    """

    proposals: tuple[Proposal[CandidateT], ...] = ()

    def __post_init__(self) -> None:
        """Reject duplicate or unkeyed pending proposals."""
        seen_ids: set[str] = set()
        for proposal in self.proposals:
            proposal_id = proposal.proposal_id
            if proposal_id is None:
                msg = "pending proposals must carry proposal ids"
                raise ValueError(msg)

            if proposal_id in seen_ids:
                msg = "pending proposals must have distinct proposal ids"
                raise ValueError(msg)

            seen_ids.add(proposal_id)

    @property
    def is_empty(self) -> bool:
        """Return whether the registry is empty."""
        return len(self.proposals) == 0

    def get(self, proposal_id: str) -> Proposal[CandidateT] | None:
        """Return the proposal matching one proposal id, if present.

        Parameters
        ----------
        proposal_id : str
            Proposal identifier to look up.

        Returns
        -------
        Proposal[CandidateT] | None
            Matching pending proposal, or ``None`` when no such proposal is
            registered.
        """
        for proposal in self.proposals:
            if proposal.proposal_id == proposal_id:
                return proposal

        return None

    def add(self, proposal: Proposal[CandidateT]) -> Self:
        """Return a registry with one additional pending proposal.

        Parameters
        ----------
        proposal : Proposal[CandidateT]
            Proposal to register as pending.

        Returns
        -------
        Self
            Registry with ``proposal`` appended.

        Raises
        ------
        ValueError
            If ``proposal`` lacks a proposal id or duplicates an existing one.
        """
        proposal_id = proposal.proposal_id
        if proposal_id is None:
            msg = "pending proposals must carry proposal ids"
            raise ValueError(msg)

        if self.get(proposal_id) is not None:
            msg = "pending proposals must have distinct proposal ids"
            raise ValueError(msg)

        return type(self)(proposals=self.proposals + (proposal,))

    def remove_many(self, proposal_ids: AbstractSet[str]) -> Self:
        """Return a registry with the given proposal ids removed.

        Parameters
        ----------
        proposal_ids : collections.abc.Set[str]
            Proposal ids to remove.

        Returns
        -------
        Self
            Registry with all matching proposals removed.
        """
        if not proposal_ids:
            return self

        return type(self)(
            proposals=tuple(
                proposal
                for proposal in self.proposals
                if proposal.proposal_id not in proposal_ids
            ),
        )


@dataclass(frozen=True, slots=True)
class CSAEngineState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Canonical internal state for the full CSA execution engine.

    Parameters
    ----------
    random_state : RandomStateSnapshot
        Canonical RNG snapshot for the engine.
    banking_state : CSABankingState[CandidateT]
        Bank and bank-adjacent runtime state.
    progression_state : CSAProgressionState
        Cutoff and run-boundary progression state.
    selection_state : SeedSelectionState
        Seed-selection runtime state.
    generation_state : GenerationRuntimeState[CandidateT]
        In-flight generation runtime state.
    proposal_state : CSAProposalState
        Proposal adaptation state.
    scoring_state : CSAScoringState[CandidateT]
        Score-model runtime state.
    pending_proposals : CSAPendingProposals[CandidateT]
        Outstanding proposals waiting for observations.
    trace_state : CSAEventTraceState[CandidateT] | None, default=None
        Optional trace reducer state.
    proposal_index : int, default=0
        Monotone counter used for proposal id allocation.
    """

    random_state: RandomStateSnapshot
    banking_state: CSABankingState[CandidateT]
    progression_state: CSAProgressionState
    selection_state: SeedSelectionState
    generation_state: GenerationRuntimeState[CandidateT]
    proposal_state: CSAProposalState
    scoring_state: CSAScoringState[CandidateT]
    pending_proposals: CSAPendingProposals[CandidateT]
    trace_state: CSAEventTraceState[CandidateT] | None = None
    proposal_index: int = 0

    def __post_init__(self) -> None:
        """Reject invalid engine-state counters."""
        if self.proposal_index < 0:
            msg = "proposal_index must be non-negative"
            raise ValueError(msg)

    def to_dict(
        self,
        *,
        candidate_to_dict: Callable[[CandidateT], JSONValue],
    ) -> JSONDict:
        """Return a JSON-safe checkpoint snapshot for a safe-boundary engine state.

        Parameters
        ----------
        candidate_to_dict : Callable[[CandidateT], JSONValue]
            Callback that converts canonical candidates into JSON-safe values.

        Returns
        -------
        JSONDict
            Versioned JSON-safe checkpoint snapshot.

        Raises
        ------
        ValueError
            If the engine is not at a safe checkpoint boundary.
        """
        if not self.pending_proposals.is_empty:
            msg = "CSA checkpoints require the pending proposal registry to be empty"
            raise ValueError(msg)
        if self.generation_state.is_active:
            msg = "CSA checkpoints require generation runtime to be idle"
            raise ValueError(msg)

        return {
            "format": _CSA_ENGINE_STATE_FORMAT,
            "version": _CSA_ENGINE_STATE_VERSION,
            "random_state": self.random_state.to_dict(),
            "banking_state": self.banking_state.to_dict(
                candidate_to_dict=candidate_to_dict,
            ),
            "progression_state": self.progression_state.to_dict(),
            "selection_state": self.selection_state.to_dict(),
            "proposal_state": self.proposal_state.to_dict(),
            "scoring_state": self.scoring_state.to_dict(),
            "proposal_index": self.proposal_index,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        *,
        candidate_from_dict: Callable[[JSONValue], CandidateT],
        growth_policy: CSABankGrowthPolicy,
        clustering_policy: CSAClusteringPolicy,
        proposal_policy: CSAProposalPolicy,
        acceptance_policy: CSAAcceptancePolicy,
        score_model: CSAScoreModel[CandidateT],
    ) -> "CSAEngineState[CandidateT]":
        """Build a safe-boundary engine state from a JSON-safe checkpoint snapshot.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            Versioned JSON-safe checkpoint snapshot.
        candidate_from_dict : Callable[[JSONValue], CandidateT]
            Callback that reconstructs canonical candidates from JSON-safe
            values.
        growth_policy : CSABankGrowthPolicy
            Growth policy that owns the reconstructed banking state.
        clustering_policy : CSAClusteringPolicy
            Clustering policy that owns the reconstructed banking state.
        proposal_policy : CSAProposalPolicy
            Proposal policy that owns the reconstructed proposal state.
        acceptance_policy : CSAAcceptancePolicy
            Acceptance policy that owns the reconstructed scoring state.
        score_model : CSAScoreModel[CandidateT]
            Score model that owns the reconstructed scoring state.

        Returns
        -------
        CSAEngineState[CandidateT]
            Reconstructed safe-boundary engine state.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        ValueError
            If the snapshot format or version is unsupported.
        """
        format_name = data.get("format")
        version = data.get("version")
        raw_random_state = data.get("random_state")
        raw_banking_state = data.get("banking_state")
        raw_progression_state = data.get("progression_state")
        raw_selection_state = data.get("selection_state")
        raw_proposal_state = data.get("proposal_state")
        raw_scoring_state = data.get("scoring_state")
        proposal_index = data.get("proposal_index")
        if format_name != _CSA_ENGINE_STATE_FORMAT:
            msg = "unsupported CSA checkpoint format"
            raise ValueError(msg)
        if version != _CSA_ENGINE_STATE_VERSION:
            msg = "unsupported CSA checkpoint version"
            raise ValueError(msg)
        if not isinstance(raw_random_state, dict):
            msg = "CSA checkpoint snapshot requires random_state mapping"
            raise TypeError(msg)
        if not isinstance(raw_banking_state, dict):
            msg = "CSA checkpoint snapshot requires banking_state mapping"
            raise TypeError(msg)
        if not isinstance(raw_progression_state, dict):
            msg = "CSA checkpoint snapshot requires progression_state mapping"
            raise TypeError(msg)
        if not isinstance(raw_selection_state, dict):
            msg = "CSA checkpoint snapshot requires selection_state mapping"
            raise TypeError(msg)
        if not isinstance(raw_proposal_state, dict):
            msg = "CSA checkpoint snapshot requires proposal_state mapping"
            raise TypeError(msg)
        if not isinstance(raw_scoring_state, dict):
            msg = "CSA checkpoint snapshot requires scoring_state mapping"
            raise TypeError(msg)
        if not isinstance(proposal_index, int):
            msg = "CSA checkpoint snapshot requires integer proposal_index"
            raise TypeError(msg)

        return cls(
            random_state=RandomStateSnapshot.from_dict(raw_random_state),
            banking_state=CSABankingState[CandidateT].from_dict(
                raw_banking_state,
                candidate_from_dict=candidate_from_dict,
                growth_policy=growth_policy,
                clustering_policy=clustering_policy,
            ),
            progression_state=CSAProgressionState.from_dict(raw_progression_state),
            selection_state=SeedSelectionState.from_dict(raw_selection_state),
            generation_state=GenerationRuntimeState[CandidateT](),
            proposal_state=CSAProposalState.from_dict(
                raw_proposal_state,
                policy=proposal_policy,
            ),
            scoring_state=CSAScoringState[CandidateT].from_dict(
                raw_scoring_state,
                acceptance_policy=acceptance_policy,
                score_model=score_model,
            ),
            pending_proposals=CSAPendingProposals[CandidateT](),
            trace_state=None,
            proposal_index=proposal_index,
        )

    def allocate_proposal_id(self, *, prefix: str = "csa-") -> tuple[str, Self]:
        """Return one new proposal id together with the advanced engine state.

        Parameters
        ----------
        prefix : str, default=\"csa-\"
            Prefix used while formatting the proposal id.

        Returns
        -------
        tuple[str, Self]
            Allocated proposal id and engine state with the counter advanced.
        """
        proposal_id = f"{prefix}{self.proposal_index}"
        return proposal_id, replace(
            self,
            proposal_index=self.proposal_index + 1,
        )

    def replace_random_state(self, random_state: RandomStateSnapshot) -> Self:
        """Return a copy with one replacement RNG snapshot.

        Parameters
        ----------
        random_state : RandomStateSnapshot
            Replacement RNG snapshot.

        Returns
        -------
        Self
            Engine state with ``random_state`` replaced.
        """
        return replace(self, random_state=random_state)

    def issue_proposal(
        self,
        proposal: Proposal[CandidateT],
        *,
        tracks_generation: bool,
    ) -> Self:
        """Return an engine state that records one issued proposal.

        Parameters
        ----------
        proposal : Proposal[CandidateT]
            Proposal being issued to the evaluator boundary.
        tracks_generation : bool
            Whether the proposal should also be registered in generation
            runtime state.

        Returns
        -------
        Self
            Engine state with the proposal added to pending proposals and, when
            requested, generation tracking.
        """
        next_generation_state = self.generation_state
        if tracks_generation:
            proposal_id = proposal.proposal_id
            assert proposal_id is not None
            next_generation_state = self.generation_state.register_proposal(proposal_id)

        next_pending_proposals = CSAPendingProposals[CandidateT](
            proposals=self.pending_proposals.proposals + (proposal,),
        )

        return replace(
            self,
            pending_proposals=next_pending_proposals,
            generation_state=next_generation_state,
        )

    def consume_pending_proposals(self, proposal_ids: AbstractSet[str]) -> Self:
        """Return an engine state with some pending proposals removed.

        Parameters
        ----------
        proposal_ids : collections.abc.Set[str]
            Proposal ids to remove from the pending registry.

        Returns
        -------
        Self
            Engine state with the selected proposals removed.
        """
        return replace(
            self,
            pending_proposals=self.pending_proposals.remove_many(proposal_ids),
        )
