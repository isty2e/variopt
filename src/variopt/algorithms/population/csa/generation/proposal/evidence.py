"""Outcome-aligned evidence nouns for CSA proposal adaptation."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Literal

from ......artifacts import Observation
from ......spaces import LeafPath
from ......typevars import CandidateT
from ...banking.update.transition import CSABankTransition
from .state.attribution import ProposalAttribution

CSAProposalLeafCreditSource = Literal["mutation", "local_displacement"]


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


@dataclass(frozen=True, slots=True)
class CSAProposalLeafCredit:
    """One bounded association between pipeline credit and a structured leaf.

    Parameters
    ----------
    source : CSAProposalLeafCreditSource
        Pipeline stage that associated the outcome with the leaf.
    path : LeafPath
        Canonical structured leaf path receiving the association credit.
    credit : float
        Non-negative share of one outcome's pipeline credit.
    """

    source: CSAProposalLeafCreditSource
    path: LeafPath
    credit: float

    def __post_init__(self) -> None:
        """Normalize the path and reject invalid credit records."""
        if self.source not in ("mutation", "local_displacement"):
            msg = "source must identify a canonical proposal pipeline stage"
            raise ValueError(msg)
        object.__setattr__(self, "path", tuple(self.path))
        if type(self.credit) is not float:
            msg = "credit must be a float"
            raise TypeError(msg)
        if not 0.0 <= self.credit <= 1.0:
            msg = "credit must lie within [0, 1]"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CSAProposalCredit(Generic[CandidateT]):
    """Scale-invariant adaptation credit for one completed proposal outcome.

    A proposal earns credit only when it survives the conclusive batch-level
    bank transition. Logical evaluation cost discounts that binary success.
    Zero-cost outcomes use a unit denominator so that credit stays finite and
    bounded.

    Parameters
    ----------
    outcome_evidence : CSAProposalOutcomeEvidence[CandidateT]
        Joined proposal, evaluation, and conclusive bank-transition evidence.
    """

    outcome_evidence: CSAProposalOutcomeEvidence[CandidateT]

    @property
    def proposal_id(self) -> str:
        """Return the stable proposal identifier."""
        return self.outcome_evidence.attribution.proposal_id

    @property
    def logical_cost(self) -> int:
        """Return the bounded-credit logical cost denominator."""
        return max(1, self.outcome_evidence.evaluation.evaluation_count)

    @property
    def pipeline_credit(self) -> float:
        """Return final bank-survival credit per logical evaluation cost."""
        if not self.outcome_evidence.bank_transition.survived_batch:
            return 0.0
        return 1 / self.logical_cost

    def leaf_association_credits(
        self,
        *,
        local_displacement_leaf_paths: Sequence[LeafPath] = (),
    ) -> tuple[CSAProposalLeafCredit, ...]:
        """Distribute pipeline credit across distinct leaf-stage associations.

        Parameters
        ----------
        local_displacement_leaf_paths : Sequence[LeafPath], default=()
            Leaves changed by local post-processing after proposal generation.

        Returns
        -------
        tuple[CSAProposalLeafCredit, ...]
            Mutation associations followed by local-displacement associations.
            Their total credit never exceeds :attr:`pipeline_credit`.
        """
        mutation_paths = tuple(
            dict.fromkeys(
                tuple(path)
                for path in self.outcome_evidence.attribution.mutated_leaf_paths
            ),
        )
        local_displacement_paths = tuple(
            dict.fromkeys(tuple(path) for path in local_displacement_leaf_paths),
        )
        association_count = len(mutation_paths) + len(local_displacement_paths)
        if association_count == 0:
            return ()

        association_credit = self.pipeline_credit / float(association_count)
        return tuple(
            CSAProposalLeafCredit(
                source="mutation",
                path=path,
                credit=association_credit,
            )
            for path in mutation_paths
        ) + tuple(
            CSAProposalLeafCredit(
                source="local_displacement",
                path=path,
                credit=association_credit,
            )
            for path in local_displacement_paths
        )


def derive_proposal_credits(
    outcome_evidence: Sequence[CSAProposalOutcomeEvidence[CandidateT]],
) -> tuple[CSAProposalCredit[CandidateT], ...]:
    """Return proposal credits in canonical proposal-id order.

    Parameters
    ----------
    outcome_evidence : Sequence[CSAProposalOutcomeEvidence[CandidateT]]
        Completed evidence from one conclusive generation-level bank update.

    Returns
    -------
    tuple[CSAProposalCredit[CandidateT], ...]
        Derived credits ordered independently of async completion order.

    Raises
    ------
    ValueError
        If a proposal identifier occurs more than once in the generation.
    """
    ordered_evidence = sorted(
        outcome_evidence,
        key=lambda evidence: evidence.attribution.proposal_id,
    )
    proposal_ids = tuple(
        evidence.attribution.proposal_id for evidence in ordered_evidence
    )
    if len(set(proposal_ids)) != len(proposal_ids):
        msg = "proposal credit evidence must contain distinct proposal ids"
        raise ValueError(msg)
    return tuple(
        CSAProposalCredit(outcome_evidence=evidence) for evidence in ordered_evidence
    )
