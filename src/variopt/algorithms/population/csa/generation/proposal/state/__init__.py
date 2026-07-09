"""Canonical state facade for CSA proposal adaptation."""

from .aggregate import CSAProposalState
from .attribution import (
    AdaptiveProposalGeneratorKind,
    NonAdaptiveProposalAttribution,
    NonAdaptiveProposalReason,
    NumericSubspaceAttribution,
    NumericSubspaceDisplacement,
    PlannedNonAdaptiveProposalAttribution,
    PlannedProposalAttribution,
    PlannedProposalProvenance,
    ProposalAttribution,
    ProposalProvenance,
)
from .stats import (
    ProposalFamilyStat,
    ProposalLeafStat,
    ProposalNumericSubspaceCovarianceStat,
)

__all__ = [
    "AdaptiveProposalGeneratorKind",
    "CSAProposalState",
    "NonAdaptiveProposalAttribution",
    "NonAdaptiveProposalReason",
    "NumericSubspaceAttribution",
    "NumericSubspaceDisplacement",
    "PlannedNonAdaptiveProposalAttribution",
    "PlannedProposalAttribution",
    "PlannedProposalProvenance",
    "ProposalAttribution",
    "ProposalProvenance",
    "ProposalFamilyStat",
    "ProposalLeafStat",
    "ProposalNumericSubspaceCovarianceStat",
]
