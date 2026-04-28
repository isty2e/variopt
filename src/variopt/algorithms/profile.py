"""Generic algorithm-profile abstractions."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ResolvedProfileT = TypeVar("ResolvedProfileT")


class AlgorithmProfile(ABC, Generic[ResolvedProfileT]):
    """Boundary-level algorithm configuration that resolves once into canonical form.

    Notes
    -----
    Profiles exist at the public boundary and resolve into a canonical internal
    configuration before optimization begins.
    """

    @abstractmethod
    def resolve(self) -> ResolvedProfileT:
        """Return the canonical internal configuration implied by this profile."""
