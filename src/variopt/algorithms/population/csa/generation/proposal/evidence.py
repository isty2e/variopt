"""Outcome-aligned evidence nouns for CSA proposal adaptation."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Generic, Literal

from ......artifacts import Observation
from ......spaces import LeafPath
from ......typevars import CandidateT
from ...banking.update.transition import CSABankTransition
from .state.attribution import ProposalAttribution

CSAProposalLeafAssociationSource = Literal["mutation", "local_displacement"]


def _normalize_leaf_association_source(value: str) -> CSAProposalLeafAssociationSource:
    if value == "mutation":
        return "mutation"
    if value == "local_displacement":
        return "local_displacement"
    msg = "source must identify a canonical proposal pipeline stage"
    raise ValueError(msg)


def _normalize_survival_efficiency_share(value: int | float) -> float:
    if isinstance(value, bool):
        msg = "survival_efficiency_share must be numeric"
        raise TypeError(msg)
    try:
        normalized_value = float(value)
    except (TypeError, ValueError) as error:
        msg = "survival_efficiency_share must be numeric"
        raise TypeError(msg) from error
    if not isfinite(normalized_value):
        msg = "survival_efficiency_share must be finite"
        raise ValueError(msg)
    if not 0.0 <= normalized_value <= 1.0:
        msg = "survival_efficiency_share must lie within [0, 1]"
        raise ValueError(msg)
    return normalized_value


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
class CSAProposalLeafSignal:
    """One bounded association between survival efficiency and a structured leaf.

    Parameters
    ----------
    source : CSAProposalLeafAssociationSource
        Pipeline stage that associated the outcome with the leaf.
    path : LeafPath
        Canonical structured leaf path receiving the association signal.
    survival_efficiency_share : float
        Non-negative share of one outcome's survival efficiency.
    """

    source: CSAProposalLeafAssociationSource
    path: LeafPath
    survival_efficiency_share: float

    def __post_init__(self) -> None:
        """Normalize the path and reject an invalid efficiency share."""
        object.__setattr__(
            self,
            "source",
            _normalize_leaf_association_source(self.source),
        )
        object.__setattr__(self, "path", tuple(self.path))
        object.__setattr__(
            self,
            "survival_efficiency_share",
            _normalize_survival_efficiency_share(self.survival_efficiency_share),
        )


@dataclass(frozen=True, slots=True)
class CSAProposalAdaptationSignal(Generic[CandidateT]):
    """Scale-invariant adaptation signal for one completed proposal outcome.

    A proposal has positive survival efficiency only when it survives the
    conclusive batch-level bank transition. Logical evaluation cost discounts
    that binary success. Zero-cost outcomes use a unit denominator so the signal
    stays finite and bounded.

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
        """Return the bounded signal's logical-cost denominator."""
        return max(1, self.outcome_evidence.evaluation.evaluation_count)

    @property
    def survival_efficiency(self) -> float:
        """Return final-bank survival per logical evaluation cost."""
        if not self.outcome_evidence.bank_transition.survived_batch:
            return 0.0
        return 1 / self.logical_cost

    def leaf_association_signals(
        self,
        *,
        local_displacement_leaf_paths: Sequence[LeafPath] = (),
    ) -> tuple[CSAProposalLeafSignal, ...]:
        """Distribute survival efficiency across leaf-stage associations.

        Parameters
        ----------
        local_displacement_leaf_paths : Sequence[LeafPath], default=()
            Leaves changed by local post-processing after proposal generation.

        Returns
        -------
        tuple[CSAProposalLeafSignal, ...]
            Mutation associations followed by local-displacement associations.
            Their total share never exceeds :attr:`survival_efficiency`.
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

        survival_efficiency_share = self.survival_efficiency / float(association_count)
        return tuple(
            CSAProposalLeafSignal(
                source="mutation",
                path=path,
                survival_efficiency_share=survival_efficiency_share,
            )
            for path in mutation_paths
        ) + tuple(
            CSAProposalLeafSignal(
                source="local_displacement",
                path=path,
                survival_efficiency_share=survival_efficiency_share,
            )
            for path in local_displacement_paths
        )


def derive_proposal_adaptation_signals(
    outcome_evidence: Sequence[CSAProposalOutcomeEvidence[CandidateT]],
) -> tuple[CSAProposalAdaptationSignal[CandidateT], ...]:
    """Return adaptation signals in canonical proposal-id order.

    Parameters
    ----------
    outcome_evidence : Sequence[CSAProposalOutcomeEvidence[CandidateT]]
        Completed evidence from one conclusive generation-level bank update.

    Returns
    -------
    tuple[CSAProposalAdaptationSignal[CandidateT], ...]
        Derived signals ordered independently of async completion order.

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
        msg = "proposal adaptation evidence must contain distinct proposal ids"
        raise ValueError(msg)
    return tuple(
        CSAProposalAdaptationSignal(outcome_evidence=evidence)
        for evidence in ordered_evidence
    )
