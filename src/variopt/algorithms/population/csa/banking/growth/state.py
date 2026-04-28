"""CSA bank-growth state definitions."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic

from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ......json_types import JSONDict, JSONValue
from ......typevars import CandidateT
from .policy import CSABankGrowthPolicy


@dataclass(frozen=True, slots=True)
class CSABankGrowthState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Canonical state for CSA adaptive bank growth and shrink behavior.

    Parameters
    ----------
    policy : CSABankGrowthPolicy
        Growth policy that governs stage transitions.
    active_energy_gap_limit : float
        Currently active energy-gap threshold used by the growth logic.
    generation_growth_count : int, default=0
        Number of growth events emitted in the current generation.
    """

    policy: CSABankGrowthPolicy
    active_energy_gap_limit: float
    generation_growth_count: int = 0

    def __post_init__(self) -> None:
        """Reject invalid states."""
        if self.active_energy_gap_limit < 0.0:
            msg = "active_energy_gap_limit must be non-negative"
            raise ValueError(msg)

        if self.generation_growth_count < 0:
            msg = "generation_growth_count must be non-negative"
            raise ValueError(msg)

    @classmethod
    def from_policy(cls, policy: CSABankGrowthPolicy) -> Self:
        """Build the initial growth state for one policy.

        Parameters
        ----------
        policy : CSABankGrowthPolicy
            Growth policy that defines the initial energy-gap limit.

        Returns
        -------
        Self
            Initial growth state implied by ``policy``.
        """
        return cls(
            policy=policy,
            active_energy_gap_limit=policy.initial_energy_gap_limit,
            generation_growth_count=0,
        )

    @property
    def enabled(self) -> bool:
        """Return whether adaptive growth is active."""
        return self.policy.enabled

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the growth state.

        Returns
        -------
        JSONDict
            JSON-safe growth-state snapshot.
        """
        return {
            "active_energy_gap_limit": self.active_energy_gap_limit,
            "generation_growth_count": self.generation_growth_count,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        *,
        policy: CSABankGrowthPolicy,
    ) -> "CSABankGrowthState[CandidateT]":
        """Build a growth state from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe growth-state snapshot.
        policy : CSABankGrowthPolicy
            Growth policy that owns the reconstructed state.

        Returns
        -------
        CSABankGrowthState[CandidateT]
            Reconstructed growth state.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        active_energy_gap_limit = data.get("active_energy_gap_limit")
        generation_growth_count = data.get("generation_growth_count")
        if not isinstance(active_energy_gap_limit, (int, float)):
            msg = "growth-state snapshot requires numeric active_energy_gap_limit"
            raise TypeError(msg)
        if not isinstance(generation_growth_count, int):
            msg = "growth-state snapshot requires integer generation_growth_count"
            raise TypeError(msg)
        return cls(
            policy=policy,
            active_energy_gap_limit=float(active_energy_gap_limit),
            generation_growth_count=generation_growth_count,
        )
