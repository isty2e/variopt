"""CSA acceptance state definitions."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, isfinite

import numpy as np
from typing_extensions import Self

from .....json_types import (
    JSONDict,
    JSONValue,
    require_json_field,
    require_json_finite_float,
)
from .acceptance import CSAAcceptancePolicy


@dataclass(frozen=True, slots=True)
class CSAAcceptanceState:
    """Canonical state for CSA acceptance scheduling.

    Parameters
    ----------
    policy : CSAAcceptancePolicy
        Acceptance policy governing temperature decay and Boltzmann scaling.
    temperature : float
        Current acceptance temperature.
    """

    policy: CSAAcceptancePolicy
    temperature: float

    @classmethod
    def from_policy(cls, policy: CSAAcceptancePolicy) -> Self:
        """Build the initial state implied by one acceptance policy.

        Parameters
        ----------
        policy : CSAAcceptancePolicy
            Acceptance policy defining the initial temperature.

        Returns
        -------
        Self
            Initial acceptance state.
        """
        return cls(
            policy=policy,
            temperature=policy.initial_temperature,
        )

    def __post_init__(self) -> None:
        """Reject invalid state temperatures."""
        if not isfinite(self.temperature) or self.temperature < 0.0:
            msg = "temperature must be a finite non-negative float"
            raise ValueError(msg)

    @property
    def beta(self) -> float | None:
        """Return the inverse-temperature coefficient used by Boltzmann acceptance."""
        if self.temperature <= 0.0:
            return None

        return 1.0 / (self.temperature * self.policy.boltzmann_constant)

    @property
    def requires_random_state(self) -> bool:
        """Return whether this acceptance state may consume randomness."""
        return self.beta is not None

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the acceptance state.

        Returns
        -------
        JSONDict
            JSON-safe acceptance-state snapshot.
        """
        return {"temperature": self.temperature}

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        *,
        policy: CSAAcceptancePolicy,
    ) -> Self:
        """Build an acceptance state from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe acceptance-state snapshot.
        policy : CSAAcceptancePolicy
            Acceptance policy that owns the reconstructed state.

        Returns
        -------
        Self
            Reconstructed acceptance state.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        return cls(
            policy=policy,
            temperature=require_json_finite_float(
                require_json_field(data, "temperature"),
                field_name="temperature",
            ),
        )

    def should_accept(
        self,
        *,
        trial_score: float,
        reference_score: float,
        random_state: np.random.RandomState | None = None,
    ) -> bool:
        """Return whether CSA should accept the trial score.

        Parameters
        ----------
        trial_score : float
            Score of the proposed trial candidate.
        reference_score : float
            Score of the incumbent or comparison candidate.
        random_state : np.random.RandomState | None, optional
            Random state required when probabilistic acceptance is active.

        Returns
        -------
        bool
            Whether the trial should be accepted.

        Raises
        ------
        ValueError
            If probabilistic acceptance is active but ``random_state`` is not
            provided.
        """
        if trial_score < reference_score:
            return True

        beta = self.beta
        if beta is None:
            return False

        if random_state is None:
            msg = "random_state is required when probabilistic acceptance is active"
            raise ValueError(msg)

        return bool(
            exp(beta * (reference_score - trial_score)) > random_state.random_sample()
        )

    def advance(self) -> Self:
        """Return the next acceptance state after one CSA iteration."""
        next_temperature = self.temperature
        if self.policy.recover:
            next_temperature /= self.policy.reduction_factor
        else:
            next_temperature *= self.policy.reduction_factor

        if next_temperature < self.policy.minimum_temperature:
            next_temperature = self.policy.minimum_temperature

        return type(self)(
            policy=self.policy,
            temperature=next_temperature,
        )
