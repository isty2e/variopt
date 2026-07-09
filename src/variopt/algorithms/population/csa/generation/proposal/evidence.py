"""Outcome-aligned evidence nouns for CSA proposal adaptation."""

from dataclasses import dataclass
from typing import Generic

from ......artifacts import Observation
from ......spaces import LeafPath
from ......typevars import CandidateT
from ...banking.update.transition import CSABankTransition
from .state.attribution import ProposalAttribution


@dataclass(frozen=True, slots=True)
class CSAProposalEvaluation(Generic[CandidateT]):
    """Successful CSA feedback preserved until bank reduction completes.

    Parameters
    ----------
    observation : Observation[CandidateT]
        Request-aligned objective observation.
    evaluation_count : int
        Logical evaluation cost reported by the successful attempt.
    refinement_changed_leaf_paths : tuple[LeafPath, ...] | None, default=None
        Explicit changed paths from refinement provenance. ``None`` means no
        explicit refinement metadata was supplied; an empty tuple is an
        explicit no-change refinement.
    """

    observation: Observation[CandidateT]
    evaluation_count: int
    refinement_changed_leaf_paths: tuple[LeafPath, ...] | None = None

    def __post_init__(self) -> None:
        """Normalize explicit paths and validate logical evaluation cost."""
        if type(self.evaluation_count) is not int:
            msg = "evaluation_count must be an int"
            raise TypeError(msg)
        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)
        if self.refinement_changed_leaf_paths is not None:
            object.__setattr__(
                self,
                "refinement_changed_leaf_paths",
                tuple(tuple(path) for path in self.refinement_changed_leaf_paths),
            )

    @classmethod
    def from_observation(
        cls,
        observation: Observation[CandidateT],
    ) -> "CSAProposalEvaluation[CandidateT]":
        """Build direct-tell feedback with one logical evaluation."""
        return cls(observation=observation, evaluation_count=1)


@dataclass(frozen=True, slots=True)
class CSAProposalOutcomeEvidence(Generic[CandidateT]):
    """Adaptive provenance joined with evaluation and bank-transition facts.

    Parameters
    ----------
    attribution : ProposalAttribution
        Adaptive generator provenance captured when the proposal was issued.
    evaluation : CSAProposalEvaluation[CandidateT]
        Successful evaluation facts preserved through the tell boundary.
    bank_transition : CSABankTransition
        Canonical conclusive bank transition for the proposal.
    """

    attribution: ProposalAttribution
    evaluation: CSAProposalEvaluation[CandidateT]
    bank_transition: CSABankTransition

    def __post_init__(self) -> None:
        """Require all joined facts to identify the same proposal."""
        observation_proposal_id = self.evaluation.observation.proposal.proposal_id
        if observation_proposal_id is None:
            msg = "proposal outcome evidence requires an observation proposal id"
            raise ValueError(msg)
        if self.attribution.proposal_id != observation_proposal_id:
            msg = "proposal attribution must align with evaluation proposal id"
            raise ValueError(msg)
        if self.bank_transition.proposal_id != observation_proposal_id:
            msg = "bank transition must align with evaluation proposal id"
            raise ValueError(msg)
