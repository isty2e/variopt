"""Canonical generation-level evidence for CSA proposal adaptation."""

from dataclasses import dataclass
from math import fsum, isfinite

from .......spaces import LeafPath
from .attribution import NumericSubspaceDisplacement


@dataclass(frozen=True, slots=True)
class ProposalFamilyAdaptationSummary:
    """Generation-level adaptation summary for one proposal family.

    Parameters
    ----------
    family_key : str
        Canonical proposal-family identifier.
    observation_count : int
        Number of family outcomes represented by the summary.
    total_survival_efficiency : float
        Sum of bounded pipeline efficiencies for those outcomes.
    """

    family_key: str
    observation_count: int
    total_survival_efficiency: float

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
        if type(self.total_survival_efficiency) is not float:
            msg = "total_survival_efficiency must be a float"
            raise TypeError(msg)
        if not isfinite(self.total_survival_efficiency):
            msg = "total_survival_efficiency must be finite"
            raise ValueError(msg)
        if not 0.0 <= self.total_survival_efficiency <= self.observation_count:
            msg = "total_survival_efficiency must be bounded by observation_count"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ProposalLeafAdaptationSummary:
    """Generation-level adaptation summary for one leaf-stage association.

    Parameters
    ----------
    path : LeafPath
        Canonical structured leaf path.
    observation_count : int
        Number of associated outcomes represented by the summary.
    total_survival_efficiency : float
        Sum of bounded association efficiencies for those outcomes.
    """

    path: LeafPath
    observation_count: int
    total_survival_efficiency: float

    def __post_init__(self) -> None:
        """Normalize the path and reject malformed leaf summaries."""
        object.__setattr__(self, "path", tuple(self.path))
        if type(self.observation_count) is not int:
            msg = "observation_count must be an int"
            raise TypeError(msg)
        if self.observation_count <= 0:
            msg = "observation_count must be positive"
            raise ValueError(msg)
        if type(self.total_survival_efficiency) is not float:
            msg = "total_survival_efficiency must be a float"
            raise TypeError(msg)
        if not isfinite(self.total_survival_efficiency):
            msg = "total_survival_efficiency must be finite"
            raise ValueError(msg)
        if not 0.0 <= self.total_survival_efficiency <= self.observation_count:
            msg = "total_survival_efficiency must be bounded by observation_count"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ProposalNumericDisplacementEvidence:
    """One numeric displacement weighted by survival efficiency.

    Parameters
    ----------
    displacement : NumericSubspaceDisplacement
        Numeric displacement vector inferred from one completed proposal.
    survival_efficiency : float
        Bounded proposal survival efficiency used as the covariance sample weight.
    """

    displacement: NumericSubspaceDisplacement
    survival_efficiency: float

    def __post_init__(self) -> None:
        """Reject malformed displacement evidence."""
        if type(self.survival_efficiency) is not float:
            msg = "survival_efficiency must be a float"
            raise TypeError(msg)
        if not 0.0 < self.survival_efficiency <= 1.0:
            msg = "survival_efficiency must lie within (0, 1]"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ProposalGenerationAdaptationEvidence:
    """Canonical adaptation input for one completed proposal generation.

    Parameters
    ----------
    evidence_count : int
        Number of adaptive outcomes represented by the generation.
    family_summaries : tuple[ProposalFamilyAdaptationSummary, ...], default=()
        Family-level adaptation summaries in canonical key order.
    mutation_leaf_summaries : tuple[ProposalLeafAdaptationSummary, ...], default=()
        Mutation-stage leaf summaries in canonical path order.
    local_displacement_leaf_summaries : tuple[ProposalLeafAdaptationSummary, ...], default=()
        Local-displacement leaf summaries in canonical path order.
    numeric_displacement_evidence : tuple[ProposalNumericDisplacementEvidence, ...], default=()
        Successful displacement samples in canonical proposal order.
    """

    evidence_count: int
    family_summaries: tuple[ProposalFamilyAdaptationSummary, ...] = ()
    mutation_leaf_summaries: tuple[ProposalLeafAdaptationSummary, ...] = ()
    local_displacement_leaf_summaries: tuple[ProposalLeafAdaptationSummary, ...] = ()
    numeric_displacement_evidence: tuple[ProposalNumericDisplacementEvidence, ...] = ()

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
            "numeric_displacement_evidence",
            tuple(self.numeric_displacement_evidence),
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
            msg = "adaptation summary observations must not exceed evidence_count"
            raise ValueError(msg)
        if (
            sum(summary.observation_count for summary in self.family_summaries)
            > self.evidence_count
        ):
            msg = "family observations must be bounded by evidence_count"
            raise ValueError(msg)
        if (
            fsum(
                summary.total_survival_efficiency
                for summary in (
                    *self.mutation_leaf_summaries,
                    *self.local_displacement_leaf_summaries,
                )
            )
            > self.evidence_count
        ):
            msg = "leaf survival-efficiency shares must be bounded by evidence_count"
            raise ValueError(msg)
        if len(self.numeric_displacement_evidence) > self.evidence_count:
            msg = "numeric displacement count must not exceed evidence_count"
            raise ValueError(msg)
