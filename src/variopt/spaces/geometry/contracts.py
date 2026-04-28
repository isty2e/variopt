"""Geometry contracts for structured search-space diversity helpers."""

from typing import Protocol, runtime_checkable

from ..types import SpaceCandidateValue
from .parts import StructuredDistanceParts


@runtime_checkable
class StructuredSpaceGeometry(Protocol):
    """Canonical geometry contract over structured candidate values.

    Notes
    -----
    Implementations expose the structured distance decomposition used by
    diversity metrics and geometry-aware algorithms.
    """

    def distance_parts(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> StructuredDistanceParts:
        """Return canonical structured distance parts.

        Parameters
        ----------
        left : SpaceCandidateValue
            Left structured candidate.
        right : SpaceCandidateValue
            Right structured candidate.

        Returns
        -------
        StructuredDistanceParts
            Structured distance decomposition between ``left`` and ``right``.
        """
        ...


@runtime_checkable
class CompiledStructuredGeometryProvider(Protocol):
    """Optional sidecar protocol for third-party compiled structured geometry.

    Notes
    -----
    This protocol is intentionally separate from ``StructuredSearchSpace``.
    Implementing it lets one structured space opt into compiled geometry
    without making compiled runtime concerns part of the core space-spec
    contract.
    """

    def compile_structured_geometry(self) -> StructuredSpaceGeometry | None:
        """Return one compiled geometry realization for this space, if any."""
        ...
