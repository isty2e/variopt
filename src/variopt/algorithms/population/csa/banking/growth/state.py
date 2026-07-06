"""CSA bank-growth state definitions."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Generic

from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ......json_types import (
    JSONDict,
    JSONValue,
    require_json_field,
    require_json_finite_float,
    require_json_int,
)
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
        if isinstance(self.active_energy_gap_limit, bool):
            msg = "active_energy_gap_limit must be numeric"
            raise TypeError(msg)
        if not isfinite(self.active_energy_gap_limit):
            msg = "active_energy_gap_limit must be finite"
            raise ValueError(msg)

        if self.active_energy_gap_limit < 0.0:
            msg = "active_energy_gap_limit must be non-negative"
            raise ValueError(msg)

        if type(self.generation_growth_count) is not int:
            msg = "generation_growth_count must be an integer"
            raise TypeError(msg)
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
        active_energy_gap_limit = require_json_finite_float(
            require_json_field(data, "active_energy_gap_limit"),
            field_name="active_energy_gap_limit",
        )
        generation_growth_count = require_json_int(
            require_json_field(data, "generation_growth_count"),
            field_name="generation_growth_count",
        )
        return cls(
            policy=policy,
            active_energy_gap_limit=active_energy_gap_limit,
            generation_growth_count=generation_growth_count,
        )
