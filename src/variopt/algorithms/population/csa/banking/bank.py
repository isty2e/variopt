"""Immutable bank state for CSA-lite."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Generic

import numpy as np

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....json_types import (
    JSONDict,
    JSONValue,
    require_json_finite_float,
    require_json_int,
)
from .....typevars import CandidateT


@dataclass(frozen=True, slots=True)
class BankEntry(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Evaluated candidate admitted into a CSA bank.

    Parameters
    ----------
    candidate : CandidateT
        Candidate stored in the bank.
    value : float
        Objective value for the candidate.
    proposal_id : str | None, default=None
        Optional proposal identifier associated with the candidate.
    """

    candidate: CandidateT
    value: float
    proposal_id: str | None = None

    def __post_init__(self) -> None:
        """Reject values that cannot be represented in strict JSON."""
        if isinstance(self.value, bool):
            msg = "value must be numeric"
            raise TypeError(msg)
        if not isfinite(self.value):
            msg = "value must be finite"
            raise ValueError(msg)

    def to_dict(
        self,
        *,
        candidate_to_dict: Callable[[CandidateT], JSONValue],
    ) -> JSONDict:
        """Return a JSON-safe mapping for one bank entry.

        Parameters
        ----------
        candidate_to_dict : Callable[[CandidateT], JSONValue]
            Callback that converts canonical candidates into JSON-safe values.

        Returns
        -------
        JSONDict
            JSON-safe bank-entry snapshot.
        """
        return {
            "candidate": candidate_to_dict(self.candidate),
            "value": self.value,
            "proposal_id": self.proposal_id,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        *,
        candidate_from_dict: Callable[[JSONValue], CandidateT],
    ) -> "BankEntry[CandidateT]":
        """Build a bank entry from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe bank-entry snapshot.
        candidate_from_dict : Callable[[JSONValue], CandidateT]
            Callback that reconstructs canonical candidates from JSON-safe
            values.

        Returns
        -------
        BankEntry[CandidateT]
            Reconstructed bank entry.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        value = data.get("value")
        proposal_id = data.get("proposal_id")
        finite_value = require_json_finite_float(value, field_name="value")
        if proposal_id is not None and not isinstance(proposal_id, str):
            msg = "bank entry snapshot requires proposal_id to be a string or null"
            raise TypeError(msg)
        return cls(
            candidate=candidate_from_dict(data.get("candidate")),
            value=finite_value,
            proposal_id=proposal_id,
        )


@dataclass(frozen=True, slots=True)
class Bank(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Persistent bank of evaluated entries for CSA-lite.

    Parameters
    ----------
    capacity : int
        Maximum number of bank entries.
    entries : tuple[BankEntry[CandidateT], ...], default=()
        Current bank entries in bank order.
    """

    capacity: int
    entries: tuple[BankEntry[CandidateT], ...] = ()

    def __post_init__(self) -> None:
        """Reject invalid bank shapes."""
        if self.capacity <= 0:
            msg = "capacity must be positive"
            raise ValueError(msg)

        if len(self.entries) > self.capacity:
            msg = "entries must not exceed capacity"
            raise ValueError(msg)

    def to_dict(
        self,
        *,
        candidate_to_dict: Callable[[CandidateT], JSONValue],
    ) -> JSONDict:
        """Return a JSON-safe mapping for the bank.

        Parameters
        ----------
        candidate_to_dict : Callable[[CandidateT], JSONValue]
            Callback that converts canonical candidates into JSON-safe values.

        Returns
        -------
        JSONDict
            JSON-safe bank snapshot.
        """
        return {
            "capacity": self.capacity,
            "entries": [
                entry.to_dict(candidate_to_dict=candidate_to_dict)
                for entry in self.entries
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        *,
        candidate_from_dict: Callable[[JSONValue], CandidateT],
    ) -> "Bank[CandidateT]":
        """Build a bank from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe bank snapshot.
        candidate_from_dict : Callable[[JSONValue], CandidateT]
            Callback that reconstructs canonical candidates from JSON-safe
            values.

        Returns
        -------
        Bank[CandidateT]
            Reconstructed bank snapshot.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        capacity = require_json_int(data.get("capacity"), field_name="capacity")
        raw_entries = data.get("entries")
        if not isinstance(raw_entries, list):
            msg = "bank snapshot requires entry list"
            raise TypeError(msg)
        entries: list[BankEntry[CandidateT]] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                msg = "bank snapshot entries must be mappings"
                raise TypeError(msg)
            entries.append(
                BankEntry[CandidateT].from_dict(
                    raw_entry,
                    candidate_from_dict=candidate_from_dict,
                ),
            )
        return cls(
            capacity=capacity,
            entries=tuple(entries),
        )

    @property
    def is_full(self) -> bool:
        """Return whether the bank has reached capacity."""
        return len(self.entries) >= self.capacity

    def select_parents(
        self,
        arity: int,
        random_state: np.random.RandomState,
    ) -> tuple[CandidateT, ...]:
        """Select parent candidates uniformly without replacement.

        Parameters
        ----------
        arity : int
            Number of parent candidates to select.
        random_state : np.random.RandomState
            Random state used for sampling.

        Returns
        -------
        tuple[CandidateT, ...]
            Parent candidates sampled uniformly without replacement.

        Raises
        ------
        ValueError
            If ``arity`` is non-positive or exceeds the number of bank entries.
        """
        if arity <= 0:
            msg = "arity must be positive"
            raise ValueError(msg)

        if len(self.entries) < arity:
            msg = "bank does not contain enough entries for the requested arity"
            raise ValueError(msg)

        indices = list(range(len(self.entries)))
        random_state.shuffle(indices)
        return tuple(self.entries[index].candidate for index in indices[:arity])
