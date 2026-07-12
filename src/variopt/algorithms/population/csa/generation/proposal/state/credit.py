"""Canonical generation-level credit summaries for CSA proposal state."""

from dataclasses import dataclass
from math import fsum, isfinite

from .......spaces import LeafPath
from .attribution import NumericSubspaceDisplacement


@dataclass(frozen=True, slots=True)
class ProposalFamilyCreditSummary:
    """Generation-level credit summarized for one proposal family.

    Parameters
    ----------
    family_key : str
        Canonical proposal-family identifier.
    observation_count : int
        Number of family outcomes represented by the summary.
    total_credit : float
        Sum of bounded pipeline credits for those outcomes.
    """

    family_key: str
    observation_count: int
    total_credit: float

    def __post_init__(self) -> None:
        """Reject malformed family summaries."""
        if self.family_key == "":
            msg = "family_key must not be empty"
            raise ValueError(msg)
        if type(self.observation_count) is not int:
            msg = "observation_count must be an int"
            raise TypeError(msg)
        if self.observation_count <= 0:
            msg = "observation_count must be positive"
            raise ValueError(msg)
        if type(self.total_credit) is not float:
            msg = "total_credit must be a float"
            raise TypeError(msg)
        if not isfinite(self.total_credit):
            msg = "total_credit must be finite"
            raise ValueError(msg)
        if not 0.0 <= self.total_credit <= self.observation_count:
            msg = "total_credit must be bounded by observation_count"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ProposalLeafCreditSummary:
    """Generation-level credit summarized for one leaf-stage association.

    Parameters
    ----------
    path : LeafPath
        Canonical structured leaf path.
    observation_count : int
        Number of associated outcomes represented by the summary.
    total_credit : float
        Sum of bounded association credits for those outcomes.
    """

    path: LeafPath
    observation_count: int
    total_credit: float

    def __post_init__(self) -> None:
        """Normalize the path and reject malformed leaf summaries."""
        object.__setattr__(self, "path", tuple(self.path))
        if type(self.observation_count) is not int:
            msg = "observation_count must be an int"
            raise TypeError(msg)
        if self.observation_count <= 0:
            msg = "observation_count must be positive"
            raise ValueError(msg)
        if type(self.total_credit) is not float:
            msg = "total_credit must be a float"
            raise TypeError(msg)
        if not isfinite(self.total_credit):
            msg = "total_credit must be finite"
            raise ValueError(msg)
        if not 0.0 <= self.total_credit <= self.observation_count:
            msg = "total_credit must be bounded by observation_count"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ProposalNumericDisplacementCredit:
    """One numeric displacement weighted by canonical pipeline credit.

    Parameters
    ----------
    displacement : NumericSubspaceDisplacement
        Numeric displacement vector inferred from one completed proposal.
    credit : float
        Bounded pipeline credit used as the covariance sample weight.
    """

    displacement: NumericSubspaceDisplacement
    credit: float

    def __post_init__(self) -> None:
        """Reject malformed displacement credit."""
        if type(self.credit) is not float:
            msg = "credit must be a float"
            raise TypeError(msg)
        if not 0.0 <= self.credit <= 1.0:
            msg = "credit must lie within [0, 1]"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ProposalGenerationCreditBatch:
    """Canonical adaptation input for one completed proposal generation.

    Parameters
    ----------
    evidence_count : int
        Number of adaptive outcomes represented by the generation.
    family_summaries : tuple[ProposalFamilyCreditSummary, ...], default=()
        Family-level credit summaries in canonical key order.
    mutation_leaf_summaries : tuple[ProposalLeafCreditSummary, ...], default=()
        Mutation-stage leaf summaries in canonical path order.
    local_displacement_leaf_summaries : tuple[ProposalLeafCreditSummary, ...], default=()
        Local-displacement leaf summaries in canonical path order.
    numeric_displacement_credits : tuple[ProposalNumericDisplacementCredit, ...], default=()
        Successful displacement samples in canonical proposal order.
    """

    evidence_count: int
    family_summaries: tuple[ProposalFamilyCreditSummary, ...] = ()
    mutation_leaf_summaries: tuple[ProposalLeafCreditSummary, ...] = ()
    local_displacement_leaf_summaries: tuple[ProposalLeafCreditSummary, ...] = ()
    numeric_displacement_credits: tuple[ProposalNumericDisplacementCredit, ...] = ()

    def __post_init__(self) -> None:
        """Normalize tuple fields and reject inconsistent generation batches."""
        object.__setattr__(self, "family_summaries", tuple(self.family_summaries))
        object.__setattr__(
            self,
            "mutation_leaf_summaries",
            tuple(self.mutation_leaf_summaries),
        )
        object.__setattr__(
            self,
            "local_displacement_leaf_summaries",
            tuple(self.local_displacement_leaf_summaries),
        )
        object.__setattr__(
            self,
            "numeric_displacement_credits",
            tuple(self.numeric_displacement_credits),
        )
        if type(self.evidence_count) is not int:
            msg = "evidence_count must be an int"
            raise TypeError(msg)
        if self.evidence_count <= 0:
            msg = "evidence_count must be positive"
            raise ValueError(msg)

        family_keys = tuple(summary.family_key for summary in self.family_summaries)
        if len(set(family_keys)) != len(family_keys):
            msg = "family_summaries must use distinct family keys"
            raise ValueError(msg)
        mutation_paths = tuple(summary.path for summary in self.mutation_leaf_summaries)
        if len(set(mutation_paths)) != len(mutation_paths):
            msg = "mutation_leaf_summaries must use distinct paths"
            raise ValueError(msg)
        local_paths = tuple(
            summary.path for summary in self.local_displacement_leaf_summaries
        )
        if len(set(local_paths)) != len(local_paths):
            msg = "local_displacement_leaf_summaries must use distinct paths"
            raise ValueError(msg)

        all_summaries = (
            *self.family_summaries,
            *self.mutation_leaf_summaries,
            *self.local_displacement_leaf_summaries,
        )
        if any(
            summary.observation_count > self.evidence_count for summary in all_summaries
        ):
            msg = "credit summary observations must not exceed evidence_count"
            raise ValueError(msg)
        if (
            fsum(summary.total_credit for summary in self.family_summaries)
            > self.evidence_count
        ):
            msg = "family credit must be bounded by evidence_count"
            raise ValueError(msg)
        if (
            fsum(
                summary.total_credit
                for summary in (
                    *self.mutation_leaf_summaries,
                    *self.local_displacement_leaf_summaries,
                )
            )
            > self.evidence_count
        ):
            msg = "leaf association credit must be bounded by evidence_count"
            raise ValueError(msg)
        if len(self.numeric_displacement_credits) > self.evidence_count:
            msg = "numeric displacement count must not exceed evidence_count"
            raise ValueError(msg)
