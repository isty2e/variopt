"""Immutable attribution nouns for CSA proposal adaptation."""

from dataclasses import dataclass

from typing_extensions import Self

from .......spaces import LeafPath


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
            float(coordinate)
            for coordinate in self.source_coordinates
        )
        object.__setattr__(self, "leaf_paths", normalized_leaf_paths)
        object.__setattr__(self, "source_coordinates", normalized_source_coordinates)
        if len(normalized_leaf_paths) == 0:
            msg = "numeric subspace attribution requires at least one leaf path"
            raise ValueError(msg)
        if len(normalized_leaf_paths) != len(normalized_source_coordinates):
            msg = "numeric subspace attribution dimensions must match"
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
            float(coordinate)
            for coordinate in self.displacement_coordinates
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
        if len(normalized_leaf_paths) != len(normalized_displacement_coordinates):
            msg = "numeric subspace displacement dimensions must match"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PlannedProposalAttribution:
    """Pre-proposal-id attribution captured during proposal generation.

    Parameters
    ----------
    source_score : float
        Score of the source candidate used for proposal generation.
    proposal_family_key : str | None, default=None
        Optional proposal-family identifier.
    mutated_leaf_paths : tuple[LeafPath, ...], default=()
        Leaf paths explicitly mutated by the operator.
    numeric_subspace_attribution : NumericSubspaceAttribution | None, default=None
        Optional numeric-subspace attribution payload.
    """

    source_score: float
    proposal_family_key: str | None = None
    mutated_leaf_paths: tuple[LeafPath, ...] = ()
    numeric_subspace_attribution: NumericSubspaceAttribution | None = None

    def __post_init__(self) -> None:
        """Normalize immutable planned-attribution fields."""
        object.__setattr__(self, "mutated_leaf_paths", tuple(self.mutated_leaf_paths))


@dataclass(frozen=True, slots=True)
class ProposalAttribution:
    """In-flight attribution for one proposal-side adaptation update.

    Parameters
    ----------
    proposal_id : str
        Issued proposal identifier.
    source_score : float
        Score of the source candidate used for proposal generation.
    proposal_family_key : str | None, default=None
        Optional proposal-family identifier.
    mutated_leaf_paths : tuple[LeafPath, ...], default=()
        Leaf paths explicitly mutated by the operator.
    numeric_subspace_attribution : NumericSubspaceAttribution | None, default=None
        Optional numeric-subspace attribution payload.
    """

    proposal_id: str
    source_score: float
    proposal_family_key: str | None = None
    mutated_leaf_paths: tuple[LeafPath, ...] = ()
    numeric_subspace_attribution: NumericSubspaceAttribution | None = None

    def __post_init__(self) -> None:
        """Normalize immutable attribution fields and reject invalid ids."""
        if self.proposal_id == "":
            msg = "proposal_id must not be empty"
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
            source_score=attribution.source_score,
            proposal_family_key=attribution.proposal_family_key,
            mutated_leaf_paths=attribution.mutated_leaf_paths,
            numeric_subspace_attribution=attribution.numeric_subspace_attribution,
        )
