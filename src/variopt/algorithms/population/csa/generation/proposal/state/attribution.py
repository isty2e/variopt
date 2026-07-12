"""Immutable attribution nouns for CSA proposal adaptation."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias

from typing_extensions import Self

from .......spaces import LeafPath

NonAdaptiveProposalReason = Literal[
    "space_sample",
    "refresh_sample",
    "regular",
    "initial",
]
AdaptiveProposalGeneratorKind = Literal["mutation", "passthrough"]
ADAPTIVE_PROPOSAL_GENERATOR_KINDS: tuple[str, ...] = (
    "mutation",
    "passthrough",
)
NON_ADAPTIVE_PROPOSAL_REASONS: tuple[str, ...] = (
    "space_sample",
    "refresh_sample",
    "regular",
    "initial",
)


@dataclass(frozen=True, slots=True)
class NumericSubspaceAttribution:
    """Immutable numeric-subspace metadata captured during proposal generation.

    Parameters
    ----------
    leaf_paths : tuple[LeafPath, ...]
        Numeric leaf paths covered by the attribution.
    source_coordinates : tuple[float, ...]
        Coordinate vector of the source candidate in numeric-subspace order.
    """

    leaf_paths: tuple[LeafPath, ...]
    source_coordinates: tuple[float, ...]

    def __post_init__(self) -> None:
        """Normalize and validate numeric-subspace attribution."""
        normalized_leaf_paths = tuple(tuple(path) for path in self.leaf_paths)
        normalized_source_coordinates = tuple(
            float(coordinate) for coordinate in self.source_coordinates
        )
        object.__setattr__(self, "leaf_paths", normalized_leaf_paths)
        object.__setattr__(self, "source_coordinates", normalized_source_coordinates)
        if len(normalized_leaf_paths) == 0:
            msg = "numeric subspace attribution requires at least one leaf path"
            raise ValueError(msg)
        if len(set(normalized_leaf_paths)) != len(normalized_leaf_paths):
            msg = "numeric subspace attribution requires distinct leaf paths"
            raise ValueError(msg)
        if len(normalized_leaf_paths) != len(normalized_source_coordinates):
            msg = "numeric subspace attribution dimensions must match"
            raise ValueError(msg)
        if any(not isfinite(value) for value in normalized_source_coordinates):
            msg = "numeric subspace source coordinates must be finite"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class NumericSubspaceDisplacement:
    """Observed displacement over one numeric structured leaf family.

    Parameters
    ----------
    leaf_paths : tuple[LeafPath, ...]
        Numeric leaf paths covered by the displacement.
    displacement_coordinates : tuple[float, ...]
        Coordinate deltas observed in numeric-subspace order.
    """

    leaf_paths: tuple[LeafPath, ...]
    displacement_coordinates: tuple[float, ...]

    def __post_init__(self) -> None:
        """Normalize and validate numeric-subspace displacement."""
        normalized_leaf_paths = tuple(tuple(path) for path in self.leaf_paths)
        normalized_displacement_coordinates = tuple(
            float(coordinate) for coordinate in self.displacement_coordinates
        )
        object.__setattr__(self, "leaf_paths", normalized_leaf_paths)
        object.__setattr__(
            self,
            "displacement_coordinates",
            normalized_displacement_coordinates,
        )
        if len(normalized_leaf_paths) == 0:
            msg = "numeric subspace displacement requires at least one leaf path"
            raise ValueError(msg)
        if len(set(normalized_leaf_paths)) != len(normalized_leaf_paths):
            msg = "numeric subspace displacement requires distinct leaf paths"
            raise ValueError(msg)
        if len(normalized_leaf_paths) != len(normalized_displacement_coordinates):
            msg = "numeric subspace displacement dimensions must match"
            raise ValueError(msg)
        if any(not isfinite(value) for value in normalized_displacement_coordinates):
            msg = "numeric subspace displacement coordinates must be finite"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PlannedProposalAttribution:
    """Pre-proposal-id attribution captured during proposal generation.

    Parameters
    ----------
    proposal_family_key : str | None, default=None
        Optional proposal-family identifier.
    mutated_leaf_paths : tuple[LeafPath, ...], default=()
        Leaf paths explicitly mutated by the operator.
    numeric_subspace_attribution : NumericSubspaceAttribution | None, default=None
        Optional numeric-subspace attribution payload.
    generator_kind : AdaptiveProposalGeneratorKind, default="mutation"
        Whether generation changed the source candidate or passed it through.
    """

    proposal_family_key: str | None = None
    mutated_leaf_paths: tuple[LeafPath, ...] = ()
    numeric_subspace_attribution: NumericSubspaceAttribution | None = None
    generator_kind: AdaptiveProposalGeneratorKind = "mutation"

    def __post_init__(self) -> None:
        """Normalize and validate immutable planned-attribution fields."""
        object.__setattr__(self, "mutated_leaf_paths", tuple(self.mutated_leaf_paths))
        if self.generator_kind not in ADAPTIVE_PROPOSAL_GENERATOR_KINDS:
            msg = "generator_kind must identify a canonical adaptive generator path"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PlannedNonAdaptiveProposalAttribution:
    """Explicit pre-id classification for a non-adaptive proposal.

    Parameters
    ----------
    reason : NonAdaptiveProposalReason
        Canonical reason the proposal does not participate in adaptation.
    """

    reason: NonAdaptiveProposalReason

    def __post_init__(self) -> None:
        """Reject non-canonical non-adaptive classifications."""
        if self.reason not in NON_ADAPTIVE_PROPOSAL_REASONS:
            msg = "reason must identify a canonical non-adaptive proposal path"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ProposalAttribution:
    """In-flight attribution for one proposal-side adaptation update.

    Parameters
    ----------
    proposal_id : str
        Issued proposal identifier.
    proposal_family_key : str | None, default=None
        Optional proposal-family identifier.
    mutated_leaf_paths : tuple[LeafPath, ...], default=()
        Leaf paths explicitly mutated by the operator.
    numeric_subspace_attribution : NumericSubspaceAttribution | None, default=None
        Optional numeric-subspace attribution payload.
    generator_kind : AdaptiveProposalGeneratorKind, default="mutation"
        Whether generation changed the source candidate or passed it through.
    """

    proposal_id: str
    proposal_family_key: str | None = None
    mutated_leaf_paths: tuple[LeafPath, ...] = ()
    numeric_subspace_attribution: NumericSubspaceAttribution | None = None
    generator_kind: AdaptiveProposalGeneratorKind = "mutation"

    def __post_init__(self) -> None:
        """Normalize immutable attribution fields and reject invalid ids."""
        if self.proposal_id == "":
            msg = "proposal_id must not be empty"
            raise ValueError(msg)
        if self.generator_kind not in ADAPTIVE_PROPOSAL_GENERATOR_KINDS:
            msg = "generator_kind must identify a canonical adaptive generator path"
            raise ValueError(msg)

        object.__setattr__(self, "mutated_leaf_paths", tuple(self.mutated_leaf_paths))

    @classmethod
    def from_planned(
        cls,
        *,
        proposal_id: str,
        attribution: PlannedProposalAttribution,
    ) -> Self:
        """Return a pending attribution bound to one issued proposal id.

        Parameters
        ----------
        proposal_id : str
            Issued proposal identifier.
        attribution : PlannedProposalAttribution
            Planned attribution captured during proposal generation.

        Returns
        -------
        Self
            Proposal attribution bound to ``proposal_id``.
        """
        return cls(
            proposal_id=proposal_id,
            proposal_family_key=attribution.proposal_family_key,
            mutated_leaf_paths=attribution.mutated_leaf_paths,
            numeric_subspace_attribution=attribution.numeric_subspace_attribution,
            generator_kind=attribution.generator_kind,
        )


@dataclass(frozen=True, slots=True)
class NonAdaptiveProposalAttribution:
    """In-flight classification for one intentionally non-adaptive proposal.

    Parameters
    ----------
    proposal_id : str
        Issued proposal identifier.
    reason : NonAdaptiveProposalReason
        Canonical reason the proposal does not participate in adaptation.
    """

    proposal_id: str
    reason: NonAdaptiveProposalReason

    def __post_init__(self) -> None:
        """Reject empty identifiers and non-canonical reasons."""
        if self.proposal_id == "":
            msg = "proposal_id must not be empty"
            raise ValueError(msg)
        if self.reason not in NON_ADAPTIVE_PROPOSAL_REASONS:
            msg = "reason must identify a canonical non-adaptive proposal path"
            raise ValueError(msg)

    @classmethod
    def from_planned(
        cls,
        *,
        proposal_id: str,
        attribution: PlannedNonAdaptiveProposalAttribution,
    ) -> Self:
        """Return a pending classification bound to one issued proposal id."""
        return cls(proposal_id=proposal_id, reason=attribution.reason)


PlannedProposalProvenance: TypeAlias = (
    PlannedProposalAttribution | PlannedNonAdaptiveProposalAttribution
)
ProposalProvenance: TypeAlias = ProposalAttribution | NonAdaptiveProposalAttribution


def bind_proposal_provenance(
    *,
    proposal_id: str,
    provenance: PlannedProposalProvenance,
) -> ProposalProvenance:
    """Bind one explicit planned provenance variant to an issued proposal id."""
    if type(provenance) is PlannedProposalAttribution:
        return ProposalAttribution.from_planned(
            proposal_id=proposal_id,
            attribution=provenance,
        )
    if type(provenance) is PlannedNonAdaptiveProposalAttribution:
        return NonAdaptiveProposalAttribution.from_planned(
            proposal_id=proposal_id,
            attribution=provenance,
        )

    msg = "provenance must be a planned proposal provenance variant"
    raise TypeError(msg)
